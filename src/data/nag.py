import h5py
import torch
import numpy as np
from time import time
from typing import Dict, List, Tuple, Union, Any

from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter_sum, scatter_mean

import src
from src.data.tensor_holder import TensorHolderMixIn
from src.data import Data, Batch
from src.utils import (
    tensor_idx,
    is_arange,
    has_duplicates,
    sparse_sample,
    from_flat_tensor,
    check_incremental_keys,
    fill_list_with_string_indexing)

import logging
log = logging.getLogger(__name__)

__all__ = ['NAG', 'NAGBatch']


class NAG(TensorHolderMixIn):
    """Holder for a Nested Acyclic Graph, containing a list of
    nested partitions of the same point cloud.

    By default, the level indices are the absolute level indices.
    The first absolute level of a NAG (`nag[0]`) always refers to the
    atomic level. The last absolute level of a NAG is the last loaded
    level in the NAG.

    In other words, the level 0 is not always loaded
    in the NAG, for instance when using a `nano` model, which does not
    process atomic data and therefore avoids loading atomic data.
    However, the level `NAG.end_i_level` is loaded by definition.
    """

    _data_serialization_prefix = 'level_'
    _start_i_level_serialization_key = 'start_i_level'

    def __init__(self, data_list: List[Data], start_i_level: int = 0):
        assert len(data_list) > 0, \
            "The NAG must have at least 1 level of hierarchy. Please " \
            "provide a minimum of 1 Data object."
        self._list = list(data_list)
        self.start_i_level = start_i_level
        if src.is_debug_enabled():
            self.debug()

    def __iter__(self):
        for i in self.level_range:
            yield self[i]

    def get_sub_size(
            self,
            high: int,
            low: int = 0
    ) -> List[torch.Tensor]:
        """Compute the number of `low`-level elements contained in each
        `high`-level superpoint.

        The sizes are computed in a bottom-up approach using the
        `super_index` at each level, starting from the `low`-level
        sizes. The `low` sizes are initialized using the following
        rules:
          - if a 'node_size' attribute can be found at level `low`, use
            it to initialize the size of `low` elements
          - if `self[low + 1].sub` exists, use it to initialize the size
            of `low` elements
          - if `low >= self.start_i_level` initialize the size of all
            `low` elements to 1

        :param high: int
            Level for which we want to compute the node size
        :param low: int
            Level whose elements we want to count. `low=-1` is accepted
            when the level-0 of the `NAG` has a `sub` attribute
            (i.e. level-0 points are themselves segments of `-1` level
            absent from the NAG object)
        :return:
        """
        assert self.start_i_level-1 <= low < high < self.absolute_num_levels
        assert self.start_i_level <= low or self[self.start_i_level].is_super, \
            "To get the sub sizes from a level not loaded in the NAG, " \

        # Sizes are computed in a bottom-up fashion. Note this scatter
        # operation assumes all levels of hierarchy use dense,
        # consecutive indices which are consistent between levels
        if (
                low >= self.start_i_level
                and getattr(self[low], 'node_size', None) is not None):
            sub_sizes = scatter_sum(
                self[low].node_size, self[low].super_index, dim=0)
        elif getattr(self[low + 1], 'sub', None) is not None:
            sub_sizes = self[low + 1].sub.sizes
        elif low >= self.start_i_level:
            sub_sizes = torch.bincount(self[low].super_index)
        else:
            raise ValueError(
                f"Cannot infer the size of level {low=} element sizes")

        for i in range(low + 1, high):
            sub_sizes = scatter_sum(sub_sizes, self[i].super_index, dim=0)

        return sub_sizes

    def get_super_index(
            self,
            high: int,
            low: int = 0
    ) -> List[torch.Tensor]:
        """Compute the super_index linking the points at level 'low'
        with superpoints at level 'high'.

        Note1:`low = self.start_i_level - 1`  is accepted if
        level-[low+1] is loaded and has a 'sub' attribute.

        Note2: `high = end_i_level + 1` is accepted if level-[high-1]
        is loaded and has a 'super_index' attribute.
        """
        assert self.start_i_level - 1 <= low < high <= self.end_i_level + 1
        assert self.start_i_level <= low or self[low].is_super
        assert high < self.end_i_level + 1 or self[high].is_sub

        low = -1 if low < 0 else low

        super_index = self[0].sub.to_super_index() if low < 0 \
            else self[low].super_index

        for i in range(low + 1, high):
            super_index = self[i].super_index[super_index]

        return super_index

    @property
    def num_levels(self):
        """Number of levels of hierarchy in the nested graph that are
        currently loaded.
        """
        return len(self._list)

    @property
    def num_points(self):
        """Number of points/nodes at each level of the hierarchy.

        :return: list of int, containing at index i the number of point
            at level i. For non-loaded lower levels, gives 0.
        """
        num_points = []
        for i in range(self.absolute_num_levels):
            if i < self.start_i_level:
                num_points.append(0)
            else :
                num_points.append(self[i].num_points)

        return num_points

    @property
    def level_ratios(self) -> Dict:
        """Ratios of number of nodes between consecutive partition
        levels. This can be useful for investigating how much each
        partition level 'compresses' the previous one.
        """
        return {
            f"|P_{i}| / |P_{i+1}|": f"{self.num_points[i] / self.num_points[i + 1]:0.1f}"
            for i in range(self.start_i_level, self.absolute_num_levels - 1)}

    def to_list(self) -> List['Data']:
        """Return the Data list"""
        return self._list

    def _items(self) -> Dict:
        """Return a dictionary containing all Tensor attributes."""
        items = {self._start_i_level_serialization_key: self.start_i_level}
        for relative_i_level, absolute_i_level in enumerate(self.level_range):
            items[f'{self._data_serialization_prefix}{absolute_i_level}'] \
                = self._list[relative_i_level]
        return items

    @classmethod
    def from_flat_tensor(
            cls,
            tensors_dict: Dict[Any, torch.Tensor],
            flat_dict: Dict
    ) -> 'NAG':
        """Reconstruct a flattened `NAG` object from tensors as
        produced by `to_flat_tensor()`.

        :param tensors_dict: Dict[Any, Tensor]
            Dictionary of flat 1D tensors created by the
            `to_flat_tensor` utility function
        :param flat_dict: Dict
            Dictionary holding metadata for reconstructing the object
            from `tensors_dict`
        """
        # Load all the items stored in the flat_dict, assuming these
        # relate to the object at hand
        item_dict = from_flat_tensor(tensors_dict, flat_dict)

        # Check expected keys are in the item_dict
        assert cls._start_i_level_serialization_key in item_dict.keys()
        start_i_level = item_dict[cls._start_i_level_serialization_key]
        prefix = cls._data_serialization_prefix
        num_levels, max_inc, all_inc_used = check_incremental_keys(
            item_dict, prefix=prefix)
        data_keys = [f"{prefix}{i}" for i in range(start_i_level, start_i_level + num_levels + 1)]
        assert all_inc_used

        # Instantiate the object with the restored attributes
        nag = cls([item_dict[k] for k in data_keys], start_i_level)

        # Restore additional attributes if any
        for key, item in item_dict.items():
            if key in data_keys:
                continue
            setattr(nag, key, item)

        return nag

    def forget_batching(self):
        """Change the class of the object to a NAG and remove any
        attributes characterizing BAGBatch objects.
        """
        return NAG([data.forget_batching() for data in self._list])

    def __setattr__(self, key, value):
        """Set Data levels as '{_data_serialization_prefix}{i}'
        attributes.
        """
        prefix = self._data_serialization_prefix
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            relative_i_level = int(key[len(prefix):]) - self.start_i_level
            try:
                self._list[relative_i_level] = value
                return
            except IndexError:
                raise AttributeError(f"No Data at index {int(key[len(prefix):])}")
        super().__setattr__(key, value)

    def __getattr__(self, key):
        """Recover Data levels as '{_data_serialization_prefix}{i}'
        attributes.
        """
        prefix = self._data_serialization_prefix
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            relative_i_level = int(key[len(prefix):]) - self.start_i_level
            try:
                return self._list[relative_i_level]
            except IndexError:
                raise AttributeError(f"No value at index {int(key[len(prefix):])}")
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{key}'")

    def __getitem__(self, idx: Union[int, slice]) -> Union['NAG', 'Data']:
        """Return a Data object from the hierarchy, according to the
        absolute level index. (If you want to access to the Data object
        with the relative index, use NAG._list[idx] instead.)

        The index 'idx' is the index of the hierarchy level to return :
        - nag[0] tries to return the atom-level.
            It raises an error if the NAG does not hold the atom-level.
        - nag[1] tries to return the first segment-level.

        Negative indexing is also supported,
        e.g.:
        - nag[-1] returns the last level of the hierarchy,
        - nag[-nag.num_levels] TRIES to return the atom-level
            (whereas the negative indexing of lists would never raise an error).

        :param idx: int, slice
            The hierarchy level to return.

        """
        if isinstance(idx, int):
            idx = idx % self.num_levels if idx < 0 else idx

            self.assert_level_in_nag(idx)
            assert idx in self.level_range, \
                f"Index {idx} is out of range. NAG has levels {self.level_range}"

            return self._list[idx - self.start_i_level]

        else :
            selected_levels = list(range(self.absolute_num_levels))[idx]

            assert all([self.start_i_level <= i <= self.end_i_level \
                        for i in selected_levels]), \
                "Some indices are out of range. NAG has levels {self.level_range}"

            # NB : idx.start < start_i_level is incompatible at that line, thanks to
            # the previous assert
            start = (idx.start if idx.start is not None else 0 ) - self.start_i_level

            stop = (idx.stop if idx.stop is not None else self.num_levels) - self.start_i_level
            step = idx.step

            return self.__class__(
                self._list[slice(start, stop, step)], selected_levels[0])


    def select(
            self,
            i_level: int,
            idx: Union[int, List[int], torch.Tensor, np.ndarray]
    ) -> 'NAG':
        """Indexing mechanism on the NAG.

        Returns a new copy of the indexed NAG, with updated clusters.
        Supports int, torch and numpy indexing.

        Contrary to indexing 'Data' objects in isolation, this will
        maintain cluster indices compatibility across all levels of the
        hierarchy.

        Note that cluster indices in 'idx' must be duplicate-free.
        Indeed, duplicates would create ambiguous situations or lower
        and higher hierarchy level updates.

        :param i_level: int
            The hierarchy level to index from.
        :param idx: int, np.NDArray, torch.Tensor
            Index to select nodes of the chosen hierarchy. Must be
            duplicate-free
        """
        assert isinstance(i_level, int)
        self.assert_level_in_nag(i_level)

        # Check whether the indexing is actually effective. If not, all
        # indexing-related behavior can be skipped for simplicity
        idx = tensor_idx(idx, device=self.device)
        num_nodes = self.num_points[i_level]
        no_indexing_required = idx is None or is_arange(idx, num_nodes)
        if no_indexing_required:
            return self.clone()

        # Make sure idx contains no duplicate entries
        if src.is_debug_enabled():
            assert not has_duplicates(idx), \
                "Duplicate indices are not supported. This would cause " \
                "ambiguities in edges and super- and sub- indices."

        # Prepare the output Data list
        data_list = [None] * self.absolute_num_levels

        # Select the nodes at level 'i_level' and update edges, subpoint
        # and superpoint indices accordingly. The returned 'out_sub' and
        # 'out_super' will help us update the lower and higher hierarchy
        # levels iteratively
        data_list[i_level], out_sub, out_super = self[i_level].select(
            idx, update_sub=True, update_super=True)

        # Iteratively update lower hierarchy levels
        for i in range(i_level - 1, self.start_i_level - 1, -1):
            # Unpack the 'out_sub' from the previous above level
            (idx_sub, sub_super) = out_sub

            # Select points but do not update 'super_index', it will be
            # directly provided by the above-level's 'sub_super'
            data_list[i], out_sub, _ = self[i].select(
                idx_sub, update_sub=True, update_super=False)

            # Directly update the 'super_index' using 'sub_super' from
            # the above level
            data_list[i].super_index = sub_super

        # Iteratively update higher hierarchy levels
        for i in range(i_level + 1, self.absolute_num_levels):
            # Unpack the 'out_super' from the previous below level
            (idx_super, super_sub) = out_super

            # Select points but do not update 'sub', it will be directly
            # provided by the above-level's 'super_sub'
            data_list[i], _, out_super = self[i].select(
                idx_super, update_sub=False, update_super=True)

            # Directly update the 'sub' using 'super_sub' from the above
            # level
            data_list[i].sub = super_sub

        # The higher level InstanceData needs to be informed of the
        # select operation. Otherwise, instance labels of higher levels
        # will still keep track of potentially removed level-0 points.
        # To this end, we recompute the instance labels with a bottom-up
        # approach
        for k in ['obj', 'obj_pred']:
            if k in self[self.start_i_level].keys and self[self.start_i_level][k] is not None:
                for i in range(self.start_i_level , self.absolute_num_levels - 1):
                    data_list[i + 1][k] = data_list[i][k].merge(
                        data_list[i].super_index)

        # Create a new NAG with the list of indexed Data
        nag = NAG(data_list[self.start_i_level:], start_i_level = self.start_i_level)

        return nag

    def save(
            self,
            path: str,
            y_to_csr: bool = True,
            pos_dtype: torch.dtype = torch.float,
            fp_dtype: torch.dtype = torch.float,
            rgb_to_byte: bool = True):
        """Save NAG to HDF5 file.

        :param path:
        :param y_to_csr: bool
            Convert 'y' to CSR format before saving. Only applies if
            'y' is a 2D histogram
        :param pos_dtype: torch dtype
            Data type to which 'pos' should be cast before saving. The
            reason for this separate treatment of 'pos' is that global
            coordinates may be too large and casting to 'fp_dtype' may
            result in hurtful precision loss
        :param fp_dtype: torch dtype
            Data type to which floating point tensors should be cast
            before saving
        :param rgb_to_byte: bool
            Whether to cast the 'rgb' and 'mean_rgb' attributes to byte
            (uint8) before saving.
        """
        with h5py.File(path, 'w') as f:
            f.attrs[self._start_i_level_serialization_key] = self.start_i_level
            for absolute_i_level, data in self.enumerate_with_absolute_index:
                g = f.create_group(f'{self._data_serialization_prefix}{absolute_i_level}')
                data.save(
                    g,
                    y_to_csr=y_to_csr,
                    pos_dtype=pos_dtype,
                    fp_dtype=fp_dtype,
                    rgb_to_byte=rgb_to_byte)

    @classmethod
    def load(
            cls,
            path: str,
            low: int = 0,
            high: int = -1,
            idx: Union[int, List, np.ndarray, torch.Tensor] = None,
            keys_idx: List[str] = None,
            keys_low: List[str] = None,
            keys: List[str] = None,
            update_sub: bool = True,
            update_super: bool = True,
            non_fp_to_long: bool = False,
            rgb_to_float: bool = False,
            verbose: bool = False
    ) -> 'NAG':
        """Load NAG from an HDF5 file. See `NAG.save` for writing such
        file. Options allow reading only part of the data.

        NB: if relevant, a NAGBatch object will be returned.

        :param path: str
            Path the file
        :param low: int
            Lowest partition level to read
        :param high: int
            Highest partition level to read
        :param idx: list, array, tensor, slice
            Index or boolean mask used to select from low
        :param keys_idx: list(str)
            Keys on which the indexing should be applied
        :param keys_low: list(str)
            Keys to read for low-level. If None, all keys will be read
        :param keys: list(str)
            Keys to read. If None, all keys will be read
        :param update_sub: bool
            See NAG.select and Data.select
        :param update_super:
            See NAG.select and Data.select
        :param non_fp_to_long: bool
            By default `save_tensor()` cast all non-float tensors to the
            smallest integer dtype before saving. This allows saving
            memory and I/O bandwidth. Upon reading, `non_fp_to_long`
            rules whether these should be cast back to int64 or kept in
            this "compressed" dtype. One good reason for not doing so is
            to accelerate data loading and device transfer. To cast the
            tensors to int64 later on in the pipeline, use the `NAGCast`
            and `Cast` transforms
        :param rgb_to_float: bool
            Whether to cast the 'rgb' and 'mean_rgb' attributes to float
            (float32) before loading.
        :param verbose: bool
        :return:
        """
        keys_low = keys if keys_low is None and keys is not None else keys_low

        data_list = []
        with h5py.File(path, 'r') as f:

            saved_start_i_level = int(
                f.attrs.get(cls._start_i_level_serialization_key, 0))
            assert low >= saved_start_i_level, \
                "Trying to load low levels that are not saved in the file"

            # If we try to load a file that doesn't have any `level_` key, 
            # it means that we are trying to load a Data object instead 
            # of a NAG object
            if not any ([cls._data_serialization_prefix in k for k in f.keys()]):
                return Data.load(
                    f,
                    idx=idx,
                    keys_idx=keys_idx,
                    keys=keys_low,
                    update_sub=update_sub,
                    non_fp_to_long=non_fp_to_long,
                    rgb_to_float=rgb_to_float,
                    verbose=verbose)
                
            # Initialize partition levels min and max to read from the
            # file. This functionality is especially intended for
            # loading levels 1 and above when we want to avoid loading
            # the memory-costly level-0 points
            high = saved_start_i_level + len(f) - 1 if high < 0 else high
            assert high <= saved_start_i_level + len(f) - 1, \
                "Trying to load high levels that are not saved in the file"

            # Make sure all required partition levels are present in the
            # file
            assert all([
                f'{cls._data_serialization_prefix}{i}' in f.keys()
                for i in range(low, high + 1)])

            # Apply index selection on the low only, if required. For
            # all subsequent levels, only keys selection is available
            for absolute_i_level in range(low, high + 1):
                start = time()
                #relative_i_level = i_level - saved_start_i_level
                if absolute_i_level == low:
                    data = Data.load(
                        f[f'{cls._data_serialization_prefix}{absolute_i_level}'],
                        idx=idx,
                        keys_idx=keys_idx,
                        keys=keys_low,
                        update_sub=update_sub,
                        non_fp_to_long=non_fp_to_long,
                        rgb_to_float=rgb_to_float,
                        verbose=verbose)
                else:
                    data = Data.load(
                        f[f'{cls._data_serialization_prefix}{absolute_i_level}'],
                        keys=keys,
                        update_sub=False,
                        non_fp_to_long=non_fp_to_long,
                        rgb_to_float=rgb_to_float,
                        verbose=verbose)

                data_list.append(data)
                if verbose:
                    print(f'{cls.__name__}.load lvl-{absolute_i_level:<13} : 'f'{time() - start:0.3f}s\n')

        # Check if the returned actually corresponds to a NAGBatch
        # object rather than a simple NAG object
        if isinstance(data_list[0], Batch) and idx is None:
            cls = NAGBatch
        else:
            cls = NAG

        # If no indexing was required, we can simply return a NAG with
        # the loaded Data list
        if idx is None:
            return cls(data_list, start_i_level = low)

        # If the low-level Data was indexed but there are no levels
        # above, we only need to update the indices held by super_index,
        # if any
        if len(data_list) == 1:
            if data_list[0].is_sub:
                data_list[0].super_index = consecutive_cluster(
                    data_list[0].super_index)[0]
            return cls(data_list, start_i_level = low)

        # In case the lowest level was indexed, we need to update the
        # above levels too. Unfortunately, this is probably because we
        # do not want to load the whole low-level partition, so we
        # artificially create a Data object to simulate it, just to be
        # able to leverage the convenient NAG.select method.
        # NB: this may be a little slow for the CPU-based DataLoader
        # operations at train time, so we will prefer setting
        # update_super=False in this situation and do the necessary
        # later on GPU
        if update_super:
            return cls._cat_select(data_list[0], data_list[1:], low, idx=idx)

        # In the case where update_super is not required but the low
        # level was indexed, we cannot combine the leve-0 and level-1+
        # Data into a NAG, because the indexing might have broken index
        # consistency between the levels. So we return the elements in a
        # NAG._cat_select-friendly way, for later update
        return data_list[0], data_list[1:], low, idx

    @classmethod
    def _cat_select(
            cls,
            data: 'Data',
            data_list: List['Data'],
            start_i_level: int,
            idx: Union[int, List, np.ndarray, torch.Tensor] = None,
    ) -> 'NAG':
        """Does part of what `Data.select()` does but in an ugly way.
        This is mostly intended for the `DataLoader` to be able to load
        `NAG` and sample level-0 points on CPU in reasonable time and
        finish the `update_sub`, `update_super` work on GPU later on if
        need be...

        :param data: Data object for level-0 points
        :param data_list: list of Data objects for level-1+ points
        :param idx: optional, indexing that has been applied on level-0
            data and guides higher levels updating (see `NAG.select()`
            and `Data.select()` with `update_super=True`)
        :return:
        """
        assert isinstance(data, Data)
        assert isinstance(data_list, list)
        assert len(data_list) > 0
        assert isinstance(data_list[0], Data)

        # In case only the lowest level was indexed, we need to update
        # the above levels too. So we artificially create a Data object
        # to simulate it, just to be able to leverage the convenient
        # `NAG.select` method.
        # NB: this may be a little slow for the CPU-based DataLoader
        # operations at train time, so it may be preferable to run
        # `cat_select` on GPU
        # TODO: this may break if 'sub' is not present. Also, need to
        #  thoroughly test this mechanism, because not 100% sure it
        #  still behaves as originally expected, and the expected
        #  speedup of indexing hdf5 files for cpu-time selection still
        #  needs to be investigated
        raise NotImplementedError(
            "cat_select has not been sufficiently tested and cannot be used "
            "for now. In particular, rather than relying on cat_select, "
            "NAG.load needs to be improved to behave exactly like NAG.select")
        fake_super_index = data_list[0].sub.to_super_index()
        fake_x = torch.empty_like(fake_super_index)
        data_fake = Data(x=fake_x, super_index=fake_super_index)
        nag = cls([data_fake] + data_list, start_i_level)
        nag = nag.select(0, idx)
        data.super_index = nag[0].super_index
        nag._list[0] = data

        return nag

    def debug(self):
        """Sanity checks."""
        assert self.num_levels > 0
        for i, d in self.enumerate_with_absolute_index:
            assert isinstance(d, Data)
            if i > 0:
                assert d.is_super
                assert d.num_points == self[i - 1].num_super
            if i < self.absolute_num_levels - 1:
                assert d.is_sub
                assert d.num_points == self[i + 1].num_sub
            d.debug()

    def get_sampling(
            self,
            high: int = 1,
            low: int = 0,
            n_max: int = 32,
            n_min: int = 1,
            mask: Union[int, List, np.ndarray, torch.Tensor] = None,
            return_pointers: bool = False
    ) -> Union['torch.Tensor', Tuple['torch.Tensor', 'torch.Tensor']]:
        """Compute indices to sample elements at `low`-level, based on
        which segment they belong to at `high`-level.

        The sampling operation is run without replacement and each
        segment is sampled at least `n_min` and at most `n_max` times,
        within the limits allowed by its actual size.

        Optionally, a `mask` can be passed to filter out some
        `low`-level points.

        :param high: int
            Partition level of the segments we want to sample. By
            default, `high=1` to sample the level-1 segments
        :param low: int
            Partition level we will sample from, guided by the `high`
            segments. By default, `low=0` to sample the level-0 points.
            `low=self.start_i_level - 1` is accepted when level-"self.start_i_level"
            has a `sub` attribute (i.e. level-"self.start_i_level" points are
            themselves segments of `self.start_i_level - 1` level absent from the NAG object).
        :param n_max: int
            Maximum number of `low`-level elements to sample in each
            `high`-level segment
        :param n_min: int
            Minimum number of `low`-level elements to sample in each
            `high`-level segment, within the limits of its size (i.e. no
            oversampling)
        :param mask: list, np.ndarray, torch.Tensor
            Indicates a subset of `low`-level elements to consider. This
            allows ignoring
        :param return_pointers: bool
            Whether pointers should be returned along with sampling
            indices. These indicate which of the output `low`-level
            sampling indices belong to which `high`-level segment
        """
        super_index = self.get_super_index(high, low=low)
        return sparse_sample(
            super_index,
            n_max=n_max,
            n_min=n_min,
            mask=mask,
            return_pointers=return_pointers)

    def __repr__(self):
        info = [
            f"{key}={getattr(self, key)}"
            for key in ['num_levels', 'num_points', 'device']]

        if self.num_levels > 1:
            info.append(f"ratios={self.level_ratios}")

        return f"{self.__class__.__name__}({', '.join(info)})"

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            if src.is_debug_enabled():
                print(f'{self.__class__.__name__}.equal: classes differ')
            return False
        if self.num_levels != other.num_levels:
            if src.is_debug_enabled():
                print(f'{self.__class__.__name__}.equal: num_levels differ')
            return False
        for d1, d2 in zip(self, other):
            if not d1.equal(d2):
                if src.is_debug_enabled():
                    print(f'{self.__class__.__name__}.equal: data differ')
                return False
        return True

    def show(self, **kwargs):
        """See `src.visualization.show`."""
        # Local import to avoid import loop errors
        from src.visualization import show
        return show(self, **kwargs)

    @property
    def end_i_level(self):
        """Index of the last loaded level of the hierarchy"""
        return self.start_i_level + self.num_levels - 1

    @property
    def has_atoms(self):
        return self.start_i_level == 0

    @property
    def first_level(self):
        """Return the first available level of the hierarchy."""
        return self[self.start_i_level]

    @property
    def absolute_num_levels(self):
        """Return the number of levels of the hierarchy,
        including the first levels that have not been loaded in the NAG.
        """
        return self.num_levels + self.start_i_level

    @property
    def level_range(self):
        """Return the range of hierarchy levels which have been loaded
        in the NAG.

        The 0-level is the atom-level
        The 1-level is the first segment-level.
        The 2-level is the second segment-level, etc.
        """
        return range(self.start_i_level, self.end_i_level + 1)

    @property
    def enumerate_with_absolute_index(self):
        """
        Provides an iterator that enumerates the `NAG` with its absolute
        level index.

        Returns:
            iterator: An iterator of tuples (absolute_index, level), where:
            - `absolute_index` is an integer starting from `self.start_i_level`,
            - `level` is a `Data` object.
        """

        return enumerate(self, start = self.start_i_level)

    def assert_level_in_nag(self, level: int):
        assert self.start_i_level <= level <= self.end_i_level, \
            f"Level {level} is out of range. NAG has levels {self.level_range}"

    def apply_data_transform(self, transform, i_level = None):
        """Apply a list of transforms to all levels of the hierarchy or the same
        transform to all the available levels of the hierarchy.

        :param transform: list of length `nag.absolute_num_levels` or a single
            transform.
            When a list is provided, the i-th transform is applied to
            the i-th level of the hierarchy. A `None` value in the list
            will skip the transform application to the corresponding level.


        :param i_level: int or None. The value indicates the level to apply the
            transform to. To use that mechanism, the `transform` parameter must
            be a single transform, not a list.
        """
        assert not( (i_level is not None) and (isinstance(transform, list)) ), \
            ("Applying a transform to a specific level needs a single "
             "transform in transform.")

        if isinstance(transform, list):
            transform_list = transform
            assert len(transform_list) == self.absolute_num_levels
        elif i_level is not None:
            transform_list = [None] * self.absolute_num_levels
            transform_list[i_level] = transform
        else :
            transform_list = [transform] * self.absolute_num_levels

            for i in range(len(transform_list)):
                if i < self.start_i_level or i > self.end_i_level:
                    transform_list[i] = None


        for relative_i_level, absolute_i_level in enumerate(self.level_range):
            transform = transform_list[absolute_i_level]
            if transform is not None:
                self.assert_level_in_nag(absolute_i_level)
                self._list[relative_i_level] = transform(self._list[relative_i_level])

    def add_keys_to(self,
                          
                          level: int,
                          keys: List[str],
                          to: str = 'x',
                          strict: bool = True,
                          delete_after: bool = False) -> None:
        """Get attributes from their keys and concatenate them to x.

        :param level: int 
            Level at which to remove attributes.
        :param keys: str or list(str)
            The feature concatenated to 'to'
        :param to: str
            Destination attribute where the features in 'keys' will be
            concatenated
        :param strict: bool
            Whether we want to raise an error if a key is not found
        :param delete_after: bool
            Whether the Data attributes should be removed once added to 'to'
        """
        level_keys = fill_list_with_string_indexing(
            level=level,
            default=[],
            value=keys,
            output_length=self.absolute_num_levels,
            start_index=self.start_i_level)
        
        for absolute_i_level, key in enumerate(level_keys):
            if len(key) == 0:
                continue
            self[absolute_i_level].add_keys_to(keys=key, 
                                                    to=to, 
                                                    strict=strict,
                                                    delete_after=delete_after)

class NAGBatch(NAG):
    """Wrapper for NAG batching."""

    def __init__(self, batch_list: List[Batch], start_i_level: int = 0):
        assert all([isinstance(b, Batch) for b in batch_list]), \
            f"Expected a list of Batch objects as input."
        super().__init__(batch_list, start_i_level)

    @classmethod
    def from_nag_list(cls, nag_list: List['NAG']) -> 'NAGBatch':
        # TODO: seems sluggish, need to investigate. Might be due to too
        #  many level-0 points. The bottleneck is in the level-0
        #  Batch.from_data_list, the 'cat' operation seems to be
        #  dominating
        assert isinstance(nag_list, list)
        assert len(nag_list) > 0
        assert all(isinstance(x, NAG) for x in nag_list)

        start_i_level_list = [n.start_i_level for n in nag_list]
        assert all([start_i_level_list[0] == i for i in start_i_level_list[1:]]), \
            ("When batching, the NAG objects must hold the same levels in the "
             "hierarchy. \nFor instance, you cannot batch a NAG that includes "
             "the atomic level with one that does not.")

        batch_list = [
            Batch.from_data_list(l)
            for l in zip(*[n._list for n in nag_list])]

        return cls(batch_list, start_i_level_list[0])

    def to_nag_list(self, strict: bool = False) -> List['NAG']:
        return [
            NAG(dl, self.start_i_level)
            for dl in zip(*[b.to_data_list(strict=strict) for b in self])]
