import h5py
import torch
import numpy as np
from time import time
from typing import (
    List,
    Tuple,
    Dict,
    Union,
    Any)

import src
from src.data.tensor_holder import TensorHolderMixIn
from src.utils import (
    tensor_idx,
    is_arange,
    is_sorted,
    indices_to_pointers,
    sizes_to_pointers,
    fast_repeat,
    save_tensor,
    load_tensor,
    from_flat_tensor,
    check_incremental_keys)


__all__ = ['CSRData', 'CSRBatch']


# TODO : all CSR reasoning could maybe be ported to PyG.utils.sparse and
#  torch_sparse.SparseTensor ? Would be good to refactor everything by
#  leveraging established dependencies. Still, need to look into the
#  features these offer and make sure they cover all our needs for the
#  project. In particular, we need efficient fancy indexing and batching
#  of sparse tensors.
#  One thing is for sure, as of Nov 2024, torch.sparse still does not do
#  the job:
#    - CSR tensors cannot be fancy-indexed
#    - batches of CSR tensors must contain the same number of elements
#  BUT if we move away from CSR in favor of COO:
#    - COO tensors do support index_select()
#    - COO tensors support cat(), stack(), vstack(), hstack()
#  Meanwhile, torch_sparse.SparseTensor seems to support CSR and COO,
#  fancy indexing, and concatenation. BUT, although feasible, some
#  basic elementwise operations on the values like addition or
#  multiplication requires non-trivial syntax

class CSRData(TensorHolderMixIn):
    """Implements the CSRData format and associated mechanisms in Torch.

    When defining a subclass A of CSRData, it is recommended to create
    an associated CSRBatch subclass by doing the following:
        - ABatch inherits from (A, CSRBatch)
        - A.get_base_class() returns A
        - A.get_batch_class() returns ABatch
    """

    _pointer_serialization_key = 'pointers'
    _iiv_serialization_key = 'is_index_value'
    _value_serialization_prefix = 'value_'

    def __init__(
            self,
            pointers: torch.Tensor,
            *args,
            dense: bool = False,
            is_index_value: List[bool] = None):
        """Initialize the pointers and values.

        Values are passed as args and stored in a list. They are
        expected to all have the same size and support torch tensor
        indexing (i.e. they can be torch tensor or CSRData objects
        themselves).

        If `dense=True`, pointers are treated as a dense tensor of
        indices to be converted into pointer indices.

        Optionally, a list of booleans `is_index_value` can be passed.
        It must be the same size as *args and indicates, for each value,
        whether it holds elements that should be treated as indices when
        stacking CSRData objects into a CSRBatch. If so, the indices
        will be updated wrt the cumulative size of the batched values.
        """
        if dense:
            self.pointers, order = indices_to_pointers(pointers)
            args = [a[order] for a in args]
        else:
            self.pointers = pointers
        self.values = [*args] if len(args) > 0 else None
        if is_index_value is None or is_index_value == []:
            self.is_index_value = torch.zeros(
                self.num_values, dtype=torch.bool, device=pointers.device)
        else:
            self.is_index_value = torch.as_tensor(
                is_index_value, dtype=torch.bool, device=pointers.device)
        if src.is_debug_enabled():
            self.debug()

    def debug(self):
        # assert self.num_groups >= 1, \
        #     "pointer indices must cover at least one group."
        assert self.pointers[0] == 0, \
            "The first pointer element must always be 0."
        assert torch.all(self.sizes >= 0), \
            "pointer indices must be increasing."

        if self.values is not None:
            assert isinstance(self.values, list), \
                "Values must be held in a list."
            assert all([len(v) == self.num_items for v in self.values]), \
                "All value objects must have the same size."
            assert len(self.values[0]) == self.num_items, \
                "pointers must cover the entire range of values."
            for v in self.values:
                if isinstance(v, CSRData):
                    v.debug()

        if self.values is not None and self.is_index_value is not None:
            assert (isinstance(self.is_index_value, torch.Tensor)
                    and self.is_index_value.dtype == torch.bool), \
                "is_index_value must be a tensor of booleans."
            assert self.is_index_value.dtype == torch.bool, \
                "is_index_value must be an tensor of booleans."
            assert self.is_index_value.ndim == 1, \
                "is_index_value must be a 1D tensor."
            assert self.is_index_value.shape[0] == self.num_values, \
                "is_index_value size must match the number of value tensors."

    def _items(self) -> Dict:
        """Return a dictionary containing all attributes."""
        items = {
            self._pointer_serialization_key: self.pointers,
            self._iiv_serialization_key: self.is_index_value}
        for i in range(self.num_values):
            items[f'{self._value_serialization_prefix}{i}'] = self.values[i]
        return items

    def __setattr__(self, key, value):
        """Set values as '{_value_serialization_prefix}{i}' attributes.
        """
        prefix = self._value_serialization_prefix
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            index = int(key[len(prefix):])
            try:
                self.values[index] = value
                return
            except IndexError:
                raise AttributeError(f"No value at index {index}")
        super().__setattr__(key, value)

    @classmethod
    def from_flat_tensor(
            cls,
            tensors_dict: Dict[Any, torch.Tensor],
            flat_dict: Dict
    ) -> 'CSRData':
        """Reconstruct a flattened `CSRData` object from tensors as
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
        ptr_key = cls._pointer_serialization_key
        iiv_key = cls._iiv_serialization_key
        assert ptr_key in item_dict.keys()
        assert iiv_key in item_dict.keys()

        prefix = cls._value_serialization_prefix
        num_values, max_inc, all_inc_used = check_incremental_keys(
            item_dict, prefix=prefix)
        value_keys = [f"{prefix}{i}" for i in range(num_values)]
        assert all_inc_used

        # Instantiate the object with the restored attributes
        csr = cls(
            item_dict[ptr_key],
            *[item_dict[k] for k in value_keys],
            dense=False,
            is_index_value=item_dict[iiv_key])

        # Blindly assign any leftover attribute still in `item_dict`.
        # This will cover, for instance, the `__sizes__` attribute used
        # by `CSRBatch`
        used_keys = [ptr_key, iiv_key] + value_keys
        for key, item in item_dict.items():
            if key in used_keys:
                continue
            setattr(csr, key, item)

        return csr

    def forget_batching(self):
        """Change the class of the object to the base class and remove
        any attributes characterizing batch objects.
        """
        return self.get_base_class()(
            self.pointers,
            *self.values,
            dense=False,
            is_index_value=self.is_index_value)

    def __getattr__(self, key):
        """Recover values as '{_value_serialization_prefix}{i}'
        attributes.
        """
        prefix = self._value_serialization_prefix
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            index = int(key[len(prefix):])
            try:
                return self.values[index]
            except IndexError:
                raise AttributeError(f"No value at index {index}")
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{key}'")

    @property
    def num_groups(self):
        return self.pointers.shape[0] - 1

    @property
    def num_values(self):
        return len(self.values) if self.values is not None else 0

    @property
    def num_items(self):
        return self.pointers[-1]

    @property
    def sizes(self) -> torch.Tensor:
        """Returns the size of each group (i.e. the pointer jumps).
        """
        return self.pointers[1:] - self.pointers[:-1]

    @property
    def indices(self) -> torch.Tensor:
        """Returns the dense indices corresponding to the pointers.
        """
        return fast_repeat(
            torch.arange(self.num_groups, device=self.device), self.sizes)

    @classmethod
    def get_base_class(cls) -> type:
        """Helps `self.from_list()` and `self.to_list()` identify which
        classes to use for batch collation and un-collation.
        """
        return CSRData

    @classmethod
    def get_batch_class(cls) -> type:
        """Helps `self.from_list()` and `self.to_list()` identify which
        classes to use for batch collation and un-collation.
        """
        return CSRBatch

    def reindex_groups(
            self,
            group_indices: torch.Tensor,
            order: torch.Tensor = None,
            num_groups: int = None
    ) -> 'CSRData':
        """Returns a copy of self with modified pointers to account for
        new groups. Affects the num_groups and the order of groups.
        Injects 0-length pointers where need be.

        By default, pointers are implicitly linked to the group indices
        in range(0, self.num_groups).

        Here we provide new group_indices for the existing pointers,
        with group_indices[i] corresponding to the position of existing
        group i in the new tensor. The indices missing from
        group_indices account for empty groups to be injected.

        The num_groups specifies the number of groups in the new tensor.
        If not provided, it is inferred from the size of group_indices.
        """
        if order is None:
            order = torch.argsort(group_indices)
        csr_new = self[order].insert_empty_groups(
            group_indices[order], num_groups=num_groups)
        return csr_new

    def insert_empty_groups(
            self,
            group_indices: torch.Tensor,
            num_groups: int = None
    ) -> 'CSRData':
        """Method called when in-place reindexing groups.

        The group_indices are assumed to be sorted and group_indices[i]
        corresponds to the position of existing group i in the new
        tensor. The indices missing from group_indices correspond to
        empty groups to be injected.

        The num_groups specifies the number of groups in the new tensor.
        If not provided, it is inferred from the size of group_indices.
        """
        assert self.num_groups == group_indices.shape[0], \
            "New group indices must correspond to the existing number " \
            "of groups"
        assert is_sorted(group_indices), "New group indices must be sorted."

        if num_groups is not None:
            num_groups = max(group_indices.max() + 1, num_groups)
        else:
            num_groups = group_indices.max() + 1

        starts = torch.cat([
            torch.tensor([-1], dtype=torch.long, device=self.device),
            group_indices.to(self.device)])
        ends = torch.cat([
            group_indices.to(self.device),
            torch.tensor([num_groups], dtype=torch.long, device=self.device)])
        repeats = ends - starts
        self.pointers = self.pointers.repeat_interleave(repeats)

        return self

    @staticmethod
    def index_select_pointers(
            pointers: torch.Tensor,
            indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Index selection of pointers.

        Returns a new pointer tensor with updated pointers, along with
        an index tensor to be used to update any values tensor
        associated with the input pointers.
        """
        assert indices.max() <= pointers.shape[0] - 2
        device = pointers.device

        # Create the new pointers
        pointers_new = torch.cat([
            torch.zeros(1, dtype=pointers.dtype, device=device),
            torch.cumsum(pointers[indices + 1] - pointers[indices], 0)])

        # Create the indexing tensor to select and order values.
        # Simply, we could have used a list of slices but we want to
        # avoid for loops and list concatenations to benefit from torch
        # capabilities.
        sizes = pointers_new[1:] - pointers_new[:-1]
        val_idx = torch.arange(pointers_new[-1], device=device)
        val_idx -= torch.arange(pointers_new[-1] + 1, device=device)[
            pointers_new[:-1]].repeat_interleave(sizes)
        val_idx += pointers[indices].repeat_interleave(sizes).to(device)

        return pointers_new, val_idx

    def __getitem__(
            self,
            idx: Union[int, List[int], torch.Tensor, np.ndarray]
    ) -> 'CSRData':
        """Indexing CSRData format. Supports Numpy and torch indexing
        mechanisms.

        Return a copy of self with updated pointers and values.
        """
        idx = tensor_idx(idx, device=self.device)

        # Check whether the indexing is actually effective. If not, all
        # indexing-related behavior can be skipped for simplicity
        no_indexing_required = idx is None or is_arange(idx, self.num_groups)
        if no_indexing_required:
            pointers = torch.zeros(1, dtype=torch.long, device=self.device)
            values = [v[[]] for v in self.values]
            out = self.__class__(
                pointers,
                *values,
                is_index_value=self.is_index_value)

        else:
            # Select the pointers and prepare the values indexing
            pointers, val_idx = self.__class__.index_select_pointers(
                self.pointers, idx)
            values = [v[val_idx] for v in self.values]
            out = self.__class__(
                pointers,
                *values,
                is_index_value=self.is_index_value)

        if src.is_debug_enabled():
            out.debug()

        return out

    def select(
            self,
            idx: Union[int, List[int], torch.Tensor, np.ndarray],
            **kwargs
    ) -> 'CSRData':
        """Returns a new CSRData which indexes `self` using entries
        in `idx`. Supports torch and numpy fancy indexing.

        :param idx: int or 1D torch.LongTensor or numpy.NDArray
            Cluster indices to select from 'self'. Must NOT contain
            duplicates
        """
        # Normal CSRData indexing, creates a new object in memory
        return self[idx]

    def __len__(self):
        return self.num_groups

    def __repr__(self):
        info = [
            f"{key}={int(getattr(self, key))}"
            for key in ['num_groups', 'num_items']]
        info.append(f"device={self.device}")
        return f"{self.__class__.__name__}({', '.join(info)})"

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            if src.is_debug_enabled():
                print(f'{self.__class__.__name__}.equal: classes differ')
            return False
        if not torch.equal(self.pointers, other.pointers):
            if src.is_debug_enabled():
                print(f'{self.__class__.__name__}.equal: pointers differ')
            return False
        if not torch.equal(self.is_index_value, other.is_index_value):
            if src.is_debug_enabled():
                print(f'{self.__class__.__name__}.equal: is_index_value differ')
            return False
        if self.num_values != other.num_values:
            if src.is_debug_enabled():
                print(f'{self.__class__.__name__}.equal: num_values differ')
            return False
        for v1, v2 in zip(self.values, other.values):
            # NB: this may be a bit strong a constraint for Cluster
            # where values are well-attributed to the proper clusters
            # but their order differ inside the cluster. In reality, we
            # want a set, the order does not matter. We could normalize
            # things by using a lexsort on cluster and point indices but
            # this be a bit costly...
            if not torch.equal(v1, v2):
                if src.is_debug_enabled():
                    print(f'{self.__class__.__name__}.equal: values differ')
                return False
        return True

    def __hash__(self) -> int:
        """Hashing for an CSRData.
        """
        return hash((
            self.__class__.__name__, self.pointers, *(v for v in self.values)))

    def save(
            self,
            f: Union[str, h5py.File, h5py.Group],
            fp_dtype: torch.dtype = torch.float):
        """Save CSRData to HDF5 file.

        :param f: h5 file path of h5py.File or h5py.Group
        :param fp_dtype: torch dtype
            Data type to which floating point tensors will be cast
            before saving
        :return:
        """
        if not isinstance(f, (h5py.File, h5py.Group)):
            with h5py.File(f, 'w') as file:
                self.save(file, fp_dtype=fp_dtype)
            return

        save_tensor(
            self.pointers,
            f,
            self._pointer_serialization_key,
            fp_dtype=fp_dtype)

        save_tensor(
            self.is_index_value,
            f,
            self._iiv_serialization_key,
            fp_dtype=fp_dtype)

        if self.values is None:
            return
        value_keys = [
            f"{self._value_serialization_prefix}{i}"
            for i in range(self.num_values)]
        for k, v in zip(value_keys, self.values):
            save_tensor(v, f, k, fp_dtype=fp_dtype)

    @classmethod
    def load(
            cls,
            f: Union[str, h5py.File, h5py.Group],
            idx: Union[int, List, np.ndarray, torch.Tensor] = None,
            non_fp_to_long: bool = False,
            verbose: bool = False,
            **kwargs
    ) -> 'CSRData':
        """Load CSRData from an HDF5 file. See `CSRData.save`
        for writing such file. Options allow reading only part of the
        clusters.

        :param f: h5 file path of h5py.File or h5py.Group
        :param idx: int, list, numpy.ndarray, torch.Tensor
            Used to select clusters when reading. Supports fancy
            indexing
        :param non_fp_to_long: bool
            By default `save_tensor()` cast all non-float tensors to the
            smallest integer dtype before saving. This allows saving
            memory and I/O bandwidth. Upon reading, `non_fp_to_long`
            rules whether these should be cast back to int64 or kept in
            this "compressed" dtype. One good reason for not doing so is
            to accelerate data loading and device transfer. To cast the
            tensors to int64 later on in the pipeline, use the `NAGCast`
            and `Cast` transforms
        :param verbose: bool
        """
        if not isinstance(f, (h5py.File, h5py.Group)):
            with h5py.File(f, 'r') as file:
                out = cls.load(
                    file,
                    idx=idx,
                    non_fp_to_long=non_fp_to_long,
                    verbose=verbose,
                    **kwargs)
            return out

        start = time()
        idx = tensor_idx(idx)
        if verbose:
            print(f'{cls.__name__}.load tensor_idx         : {time() - start:0.5f}s')

        # Check whether the indexing is actually effective. If not, all
        # indexing-related behavior can be skipped for simplicity
        num_groups = f[cls._pointer_serialization_key].shape[0] - 1
        no_indexing_required = idx is None or is_arange(idx, num_groups)

        # Check if the file actually corresponds to a batch object
        # rather than its corresponding base object. If so, load with
        # the appropriate class
        has_sizes = '__sizes__' in f.keys()
        is_not_batch = cls != cls.get_batch_class()
        if has_sizes and is_not_batch and no_indexing_required:
            return cls.get_batch_class().load(
                f,
                idx=None,
                non_fp_to_long=non_fp_to_long,
                verbose=verbose,
                **kwargs)

        # Check expected keys are in the file
        pointer_key = cls._pointer_serialization_key
        iiv_key = cls._iiv_serialization_key
        assert pointer_key in f.keys(),\
            f"Expected key '{pointer_key}' but could not find it."
        assert iiv_key in f.keys(),\
            f"Expected key '{iiv_key}' but could not find it."

        prefix = cls._value_serialization_prefix
        num_values, max_inc, all_inc_used = check_incremental_keys(
            f, prefix=prefix)
        value_keys = [f"{prefix}{i}" for i in range(num_values)]
        assert all_inc_used

        # If slicing is not needed
        if no_indexing_required:
            start = time()
            pointers = load_tensor(
                f[pointer_key], non_fp_to_long=non_fp_to_long)
            values = [
                load_tensor(f[k], non_fp_to_long=non_fp_to_long)
                for k in value_keys]
            if verbose:
                print(f'{cls.__name__}.load read all           : {time() - start:0.5f}s')
            start = time()
            out = cls(pointers, *values)
            out.is_index_value = load_tensor(f[iiv_key]).bool()
            if verbose:
                print(f'{cls.__name__}.load init               : {time() - start:0.5f}s')
            return out

        # Read only pointers start and end indices based on idx
        start = time()
        ptr_start = load_tensor(f[pointer_key], idx=idx, non_fp_to_long=True)
        ptr_end = load_tensor(f[pointer_key], idx=idx + 1, non_fp_to_long=True)
        if verbose:
            print(f'{cls.__name__}.load read ptr       : {time() - start:0.5f}s')

        # Create the new pointers
        start = time()
        pointers = torch.cat([
            torch.zeros(1, dtype=ptr_start.dtype),
            torch.cumsum(ptr_end - ptr_start, 0)])
        if verbose:
            print(f'{cls.__name__}.load new pointers   : {time() - start:0.5f}s')

        # Create the indexing tensor to select and order values.
        # Simply, we could have used a list of slices, but we want to
        # avoid for loops and list concatenations to benefit from torch
        # capabilities.
        start = time()
        sizes = pointers[1:] - pointers[:-1]
        val_idx = torch.arange(pointers[-1])
        val_idx -= torch.arange(pointers[-1] + 1)[
            pointers[:-1]].repeat_interleave(sizes)
        val_idx += ptr_start.repeat_interleave(sizes)
        if verbose:
            print(f'{cls.__name__}.load val_idx        : {time() - start:0.5f}s')

        # Read the values now we have computed the val_idx
        start = time()
        values = [
            load_tensor(
                f[k],
                idx=val_idx,
                non_fp_to_long=non_fp_to_long)
            for k in value_keys]
        if verbose:
            print(f'{cls.__name__}.load read values    : {time() - start:0.5f}s')

        # Build the CSRData object
        start = time()
        out = cls(pointers, *values)
        out.is_index_value = load_tensor(f[iiv_key]).bool()
        if verbose:
            print(f'{cls.__name__}.load init           : {time() - start:0.5f}s')
        return out


class CSRBatch(CSRData):
    """
    Wrapper class of CSRData to build a batch from a list of CSRData
    data and reconstruct it afterward.

    When defining a subclass A of CSRData, it is recommended to create
    an associated CSRBatch subclass by doing the following:
        - ABatch inherits from (A, CSRBatch)
        - A.get_base_class() returns A
        - A.get_batch_class() returns ABatch
    """
    def __init__(
            self,
            pointers: torch.Tensor,
            *args,
            dense: bool = False,
            is_index_value: List[bool] = None):
        """Basic constructor for a CSRBatch. Batches are rather
        intended to be built using the from_list() method.
        """
        super(CSRBatch, self).__init__(
            pointers, *args, dense=dense, is_index_value=is_index_value)
        self.__sizes__ = None

    def _items(self) -> Dict:
        """Return a dictionary containing all attributes."""
        items = super()._items()
        items['__sizes__'] = self.__sizes__
        return items

    @property
    def batch_pointers(self) -> torch.Tensor:
        return sizes_to_pointers(self.__sizes__) if self.__sizes__ is not None \
            else None

    @property
    def batch_items_sizes(self) -> torch.Tensor:
        return self.__sizes__ if self.__sizes__ is not None else None

    @property
    def num_batch_items(self):
        return len(self.__sizes__) if self.__sizes__ is not None else 0

    @classmethod
    def from_list(cls, csr_list: List['CSRData']) -> 'CSRBatch':
        assert isinstance(csr_list, list) and len(csr_list) > 0
        assert isinstance(csr_list[0], CSRData), \
            "All provided items must be CSRData objects."
        csr_cls = type(csr_list[0]).get_base_class()
        assert all([isinstance(csr, csr_cls) for csr in csr_list]), \
            "All provided items must have the same class."
        device = csr_list[0].device
        assert all([csr.device == device for csr in csr_list]), \
            "All provided items must be on the same device."
        num_values = csr_list[0].num_values
        assert all([csr.num_values == num_values for csr in csr_list]), \
            "All provided items must have the same number of values."
        is_index_value = csr_list[0].is_index_value
        if is_index_value is not None:
            assert all([
                torch.equal(csr.is_index_value, is_index_value)
                for csr in csr_list]), \
                "All provided items must have the same is_index_value."
        else:
            assert all([csr.is_index_value is None for csr in csr_list]), \
                "All provided items must have the same is_index_value."
        if src.is_debug_enabled():
            for csr in csr_list:
                csr.debug()

        # Offsets are used to stack pointer indices and values
        # identified as "index" value by `is_index_value` without
        # losing the indexing information they carry.
        offsets = torch.cumsum(
            torch.tensor(
                [0] + [csr.num_items for csr in csr_list[:-1]],
                dtype=torch.long,
                device=device),
            dim=0)

        # Stack pointers
        pointers = torch.cat((
            torch.tensor([0], dtype=torch.long, device=device),
            *[csr.pointers[1:].long() + offset
              for csr, offset in zip(csr_list, offsets)]), dim=0)

        # Stack values
        values = []
        for i in range(num_values):
            val_list = [csr.values[i] for csr in csr_list]
            if len(val_list) > 0 and isinstance(val_list[0], CSRData):
                val = val_list[0].from_list(val_list)
            elif is_index_value[i]:
                # "Index" values are stacked with updated indices.
                # For Clusters, this implies all point indices are
                # assumed to be present in the Cluster.points. There can
                # be no point with no cluster.
                # It may be that the value is currently stored in a
                # dtype that would overflow when incrementing values for
                # collation. To this end, we explicitly cast to int64
                # when sensitive
                offsets = torch.tensor(
                    [0] + [
                        v.max().long() + 1 if v.shape[0] > 0 else 0
                        for v in val_list[:-1]],
                    dtype=torch.long,
                    device=device)
                cum_offsets = torch.cumsum(offsets, dim=0)
                val = torch.cat([
                    v.long() + o for v, o in zip(val_list, cum_offsets)], dim=0)
            else:
                val = torch.cat(val_list, dim=0)
            values.append(val)

        # Create the Batch object, depending on the data type
        # Default of CSRData is CSRBatch, but subclasses of CSRData
        # may define their own batch class inheriting from CSRBatch.
        batch = csr_list[0].get_batch_class()(
            pointers, *values, dense=False, is_index_value=is_index_value)
        batch.__sizes__ = torch.tensor(
            [csr.num_groups for csr in csr_list],
            dtype=torch.long,
            device=device)

        return batch

    def to_list(self) -> List['CSRData']:
        if self.__sizes__ is None:
            raise RuntimeError(
                'Cannot reconstruct CSRData data list from batch because the '
                'CSRBatch was not created using `CSRBatch.from_list()`.')

        group_pointers = self.batch_pointers
        item_pointers = self.pointers[group_pointers]

        # Recover pointers and index offsets for each CSRData item
        pointers = [
            self.pointers[group_pointers[i]:group_pointers[i + 1] + 1]
            - item_pointers[i]
            for i in range(self.num_batch_items)]

        # Recover the values for each CSRData item
        values = []
        for i in range(self.num_values):
            batch_value = self.values[i]

            if isinstance(batch_value, CSRData):
                val = batch_value.to_list()

            elif self.is_index_value[i]:
                val = [
                    batch_value[item_pointers[j]:item_pointers[j + 1]]
                    - (batch_value[:item_pointers[j]].max() + 1 if j > 0 else 0)
                    for j in range(self.num_batch_items)]

                # Hacky fix for a pesky edge case. When a `CSRBatch` is
                # "manually" created and `__sizes__` is populated (this
                # happens in multiple places across the project),
                # `to_list()` may produce negative items for
                # `is_index_value` has not been constructed using the
                # expecting offsetting. To fix this, we enforce that the
                # returned is `is_index_value` indices start at 0. This
                # is a subjective choice and may very well break the
                # meaning of the corresponding indices. To reproduce
                # this issue, create an `InstanceData` with
                # dense `obj` attribute, manually convert it to a
                # `InstanceBatch` with `__sizes__`, then call
                # `to_list()`. The `obj` attributes of the second and
                # later items will contain negatives
                for j in range(self.num_batch_items):
                    if val[j].min() < 0:
                        val[j] -= val[j].min()

            else:
                val = [batch_value[item_pointers[j]:item_pointers[j + 1]]
                       for j in range(self.num_batch_items)]

            values.append(val)
        values = [list(x) for x in zip(*values)]

        csr_list = [
            self.get_base_class()(
                j, *v, dense=False, is_index_value=self.is_index_value)
            for j, v in zip(pointers, values)]

        return csr_list

    def __repr__(self):
        info = [f"{key}={getattr(self, key)}"
                for key in [
                    'num_batch_items', 'num_groups', 'num_items', 'device']]
        return f"{self.__class__.__name__}({', '.join(info)})"

    def __getitem__(
            self,
            idx: Union[int, List[int], torch.Tensor, np.ndarray]
    ) -> 'CSRData':
        """Indexing CSRBatch format. Supports Numpy and torch indexing
        mechanisms.

        Since indexing breaks batching, this will return a CSRData
        object with updated pointers and values.
        """
        # Default indexing will return a CSRBatch object
        out_batch = super().__getitem__(idx)

        # Convert to a base object, since batching mechanism is lost
        # after indexing. For this, we populate an object of the proper
        # class, initialized with fake data
        return out_batch.forget_batching()

    def save(
            self,
            f: Union[str, h5py.File, h5py.Group],
            fp_dtype: torch.dtype = torch.float):
        """Save CSRBatch to HDF5 file.

        :param f: h5 file path of h5py.File or h5py.Group
        :param fp_dtype: torch dtype
            Data type to which floating point tensors will be cast
            before saving
        :return:
        """
        if not isinstance(f, (h5py.File, h5py.Group)):
            with h5py.File(f, 'w') as file:
                self.save(file, fp_dtype=fp_dtype)
            return

        # CSRData.save stores pointers, values, and is_index_value
        super().save(f, fp_dtype=fp_dtype)

        # Need to additionally save __sizes__ to be able to maintain
        # the batching mechanism throughout serialization
        save_tensor(self.__sizes__, f, '__sizes__', fp_dtype=fp_dtype)

    @classmethod
    def load(
            cls,
            f: Union[str, h5py.File, h5py.Group],
            idx: Union[int, List, np.ndarray, torch.Tensor] = None,
            non_fp_to_long: bool = False,
            verbose: bool = False,
            **kwargs
    ) -> Union['CSRBatch', 'CSRData']:
        """Load CSRBatch from an HDF5 file. See `CSRData.save`
        for writing such file. Options allow reading only part of the
        clusters.

        :param f: h5 file path of h5py.File or h5py.Group
        :param idx: int, list, numpy.ndarray, torch.Tensor
            Used to select clusters when reading. Supports fancy
            indexing
        :param non_fp_to_long: bool
            By default `save_tensor()` cast all non-float tensors to the
            smallest integer dtype before saving. This allows saving
            memory and I/O bandwidth. Upon reading, `non_fp_to_long`
            rules whether these should be cast back to int64 or kept in
            this "compressed" dtype. One good reason for not doing so is
            to accelerate data loading and device transfer. To cast the
            tensors to int64 later on in the pipeline, use the `NAGCast`
            and `Cast` transforms
        :param verbose: bool
        """
        # Indexing breaks batching, so we return a base object if
        # indexing is required
        idx = tensor_idx(idx)
        if idx is not None:
            return cls.get_base_class().load(
                f,
                idx=idx,
                non_fp_to_long=non_fp_to_long,
                verbose=verbose,
                **kwargs)

        if not isinstance(f, (h5py.File, h5py.Group)):
            with h5py.File(f, 'r') as file:
                out = cls.load(
                    file,
                    non_fp_to_long=non_fp_to_long,
                    verbose=verbose,
                    **kwargs)
            return out

        # Check if the file actually corresponds to a batch object
        # rather than its corresponding base object
        if '__sizes__' not in f.keys():
            return cls.get_base_class().load(
                f,
                non_fp_to_long=non_fp_to_long,
                verbose=verbose,
                **kwargs)

        # Load all attributes like the parent class, and also load
        # attributes necessary for batching
        out = super().load(
            f,
            non_fp_to_long=non_fp_to_long,
            verbose=verbose,
            **kwargs)
        out.__sizes__ = load_tensor(
            f['__sizes__'],
            non_fp_to_long=non_fp_to_long)
        return out
