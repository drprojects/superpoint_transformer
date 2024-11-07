import copy
import h5py
import torch
import numpy as np
from time import time
from typing import List, Tuple, Union, Any

import src
from src.utils import tensor_idx, is_sorted, indices_to_pointers, \
    sizes_to_pointers, fast_repeat, save_tensor, load_tensor


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
class CSRData:
    """Implements the CSRData format and associated mechanisms in Torch.

    When defining a subclass A of CSRData, it is recommended to create
    an associated CSRBatch subclass by doing the following:
        - ABatch inherits from (A, CSRBatch)
        - A.get_base_class() returns A
        - A.get_batch_class() returns ABatch
    """

    __value_serialization_keys__ = None
    __pointer_serialization_key__ = 'pointers'
    __is_index_value_serialization_key__ = 'is_index_value'

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
            self.is_index_value = torch.zeros(self.num_values, dtype=torch.bool)
        else:
            self.is_index_value = torch.BoolTensor(is_index_value)
        if src.is_debug_enabled():
            self.debug()

    def debug(self):
        if self.pointer_key in self.value_keys:
            raise ValueError(
                f"Cannot serialize {self.__class__.__name__} object because"
                f"'{self.pointer_key}' is both in `self.pointer_key` and "
                f"`self.value_keys`.")

        if len(self.value_keys) != self.num_values:
            raise ValueError(
                f"Cannot serialize {self.__class__.__name__} object because"
                f"`self.value_keys` has length {len(self.value_keys)} but "
                f"`self.num_values` is {self.num_values}.")

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
            assert isinstance(self.is_index_value, torch.BoolTensor), \
                "is_index_value must be a torch.BoolTensor."
            assert self.is_index_value.dtype == torch.bool, \
                "is_index_value must be an tensor of booleans."
            assert self.is_index_value.ndim == 1, \
                "is_index_value must be a 1D tensor."
            assert self.is_index_value.shape[0] == self.num_values, \
                "is_index_value size must match the number of value tensors."

    def detach(self) -> 'CSRData':
        """Detach all tensors in the CSRData."""
        self.pointers = self.pointers.detach()
        for i in range(self.num_values):
            self.values[i] = self.values[i].detach()
        return self

    def to(self, device, **kwargs) -> 'CSRData':
        """Move the CSRData to the specified device."""
        self.pointers = self.pointers.to(device, **kwargs)
        for i in range(self.num_values):
            self.values[i] = self.values[i].to(device, **kwargs)
        return self

    def cpu(self, **kwargs) -> 'CSRData':
        """Move the CSRData to the CPU."""
        return self.to('cpu', **kwargs)

    def cuda(self, **kwargs) -> 'CSRData':
        """Move the CSRData to the first available GPU."""
        return self.to('cuda', **kwargs)

    @property
    def device(self) -> torch.device:
        return self.pointers.device

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

    def clone(self) -> 'CSRData':
        """Shallow copy of self. This may cause issues for certain types
        of downstream operations, but it saves time and memory. In
        practice, it shouldn't be problematic in this project.
        """
        out = copy.copy(self)
        out.pointers = copy.copy(self.pointers)
        out.values = copy.copy(self.values)
        return out

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
            torch.LongTensor([-1]).to(self.device),
            group_indices.to(self.device)])
        ends = torch.cat([
            group_indices.to(self.device),
            torch.LongTensor([num_groups]).to(self.device)])
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
        idx = tensor_idx(idx).to(self.device)

        # Shallow copy self and edit pointers and values. This
        # preserves the class for CSRData subclasses.
        out = self.clone()

        # If idx is empty, return an empty CSRData with empty values
        # of consistent type
        if idx.shape[0] == 0:
            out.pointers = torch.LongTensor([0])
            out.values = [v[[]] for v in self.values]

        else:
            # Select the pointers and prepare the values indexing
            pointers, val_idx = self.__class__.index_select_pointers(
                self.pointers, idx)
            out.pointers = pointers
            out.values = [v[val_idx] for v in self.values]

        if src.is_debug_enabled():
            out.debug()

        return out

    def select(
            self,
            idx: Union[int, List[int], torch.Tensor, np.ndarray],
            *args,
            **kwargs
    ) -> 'CSRData':
        """Returns a new CSRData which indexes `self` using entries
        in `idx`. Supports torch and numpy fancy indexing.

        :parameter
        idx: int or 1D torch.LongTensor or numpy.NDArray
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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            if src.is_debug_enabled():
                print(f'{self.__class__.__name__}.__eq__: classes differ')
            return False
        if not torch.equal(self.pointers, other.pointers):
            if src.is_debug_enabled():
                print(f'{self.__class__.__name__}.__eq__: pointers differ')
            return False
        if not torch.equal(self.is_index_value, other.is_index_value):
            if src.is_debug_enabled():
                print(f'{self.__class__.__name__}.__eq__: is_index_value differ')
            return False
        if self.num_values != other.num_values:
            if src.is_debug_enabled():
                print(f'{self.__class__.__name__}.__eq__: num_values differ')
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
                    print(f'{self.__class__.__name__}.__eq__: values differ')
                return False
        return True

    def __hash__(self) -> int:
        """Hashing for an CSRData.
        """
        return hash((
            self.__class__.__name__, self.pointers, *(v for v in self.values)))

    @property
    def pointer_key(self) -> str:
        """Key name for pointers. This will be used as labels for
        serialization.
        """
        return self.__pointer_serialization_key__
    
    @property
    def value_keys(self) -> List[str]:
        """List of names for each value. These will be used as labels
        for serialization.
        """
        if self.__value_serialization_keys__ is None:
            return [str(i) for i in range(self.num_values)]
        return self.__value_serialization_keys__

    @property
    def is_index_value_key(self) -> str:
        """Key name for is_index_value. This will be used as labels for
        serialization.
        """
        return self.__is_index_value_serialization_key__

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

        save_tensor(self.pointers, f, self.pointer_key, fp_dtype=fp_dtype)

        if self.is_index_value_key is not None:
            save_tensor(
                self.is_index_value, f, self.is_index_value_key,
                fp_dtype=fp_dtype)

        if self.values is None:
            return
        for k, v in zip(self.value_keys, self.values):
            save_tensor(v, f, k, fp_dtype=fp_dtype)

    @classmethod
    def load(
            cls,
            f: Union[str, h5py.File, h5py.Group],
            idx: Union[int, List, np.ndarray, torch.Tensor] = None,
            verbose: bool = False
    ) -> 'CSRData':
        """Load CSRData from an HDF5 file. See `CSRData.save`
        for writing such file. Options allow reading only part of the
        clusters.

        :param f: h5 file path of h5py.File or h5py.Group
        :param idx: int, list, numpy.ndarray, torch.Tensor
            Used to select clusters when reading. Supports fancy
            indexing
        :param verbose: bool
        """
        if not isinstance(f, (h5py.File, h5py.Group)):
            with h5py.File(f, 'r') as file:
                out = cls.load(file, idx=idx, verbose=verbose)
            return out

        start = time()
        idx = tensor_idx(idx)
        if verbose:
            print(f'{cls.__name__}.load tensor_idx         : {time() - start:0.5f}s')

        # Check if the file actually corresponds to a batch object
        # rather than its corresponding base object
        has_sizes = '__sizes__' in f.keys()
        is_not_batch = cls != cls.get_batch_class()
        has_no_indexing = idx is None or idx.shape[0] == 0
        if has_sizes and is_not_batch and has_no_indexing:
            return cls.get_batch_class().load(f, idx=idx, verbose=verbose)

        # Check expected keys are in the file
        pointer_key = cls.__pointer_serialization_key__
        value_keys = cls.__value_serialization_keys__
        value_keys = value_keys if value_keys is not None else []
        is_index_value_key = cls.__is_index_value_serialization_key__
        assert pointer_key in f.keys()
        assert all(k in f.keys() for k in value_keys)
        assert is_index_value_key is None or is_index_value_key in f.keys()

        # If no value keys are provided, CSRData.save() falls back to
        # using integers to index values. So, need to infer the number
        # of values from the consecutive integer keys in the file
        if len(value_keys) == 0:
            num_values = 0
            while str(num_values) in f.keys():
                num_values += 1
            value_keys = [str(i) for i in range(num_values)]

        if idx is None or idx.shape[0] == 0:
            start = time()
            pointers = load_tensor(f[pointer_key])
            values = [load_tensor(f[k]) for k in value_keys]
            if verbose:
                print(f'{cls.__name__}.load read all           : {time() - start:0.5f}s')
            start = time()
            out = cls(pointers, *values)
            if is_index_value_key is not None:
                out.is_index_value = load_tensor(f[is_index_value_key]).bool()
            if verbose:
                print(f'{cls.__name__}.load init               : {time() - start:0.5f}s')
            return out

        # Read only pointers start and end indices based on idx
        start = time()
        ptr_start = load_tensor(f[pointer_key], idx=idx)
        ptr_end = load_tensor(f[pointer_key], idx=idx + 1)
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
        values = [load_tensor(f[k], idx=val_idx) for k in value_keys]
        if verbose:
            print(f'{cls.__name__}.load read values    : {time() - start:0.5f}s')

        # Build the CSRData object
        start = time()
        out = cls(pointers, *values)
        if is_index_value_key is not None:
            out.is_index_value = load_tensor(f[is_index_value_key]).bool()
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

    def detach(self) -> 'CSRBatch':
        """Detach all tensors in the CSRBatch."""
        self = super().detach()
        self.__sizes__ = self.__sizes__.detach() if self.__sizes__ is not None \
            else None
        return self

    def to(self, device, **kwargs) -> 'CSRBatch':
        """Move the CSRBatch to the specified device."""
        out = super().to(device, **kwargs)
        out.__sizes__ = self.__sizes__.to(device, **kwargs) \
            if self.__sizes__ is not None else None
        return out

    @classmethod
    def from_list(cls, csr_list: List['CSRData']) -> 'CSRBatch':
        assert isinstance(csr_list, list) and len(csr_list) > 0
        assert isinstance(csr_list[0], CSRData), \
            "All provided items must be CSRData objects."
        csr_cls = type(csr_list[0])
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
                np.array_equal(csr.is_index_value, is_index_value)
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
        offsets = torch.cumsum(torch.LongTensor(
            [0] + [csr.num_items for csr in csr_list[:-1]]), dim=0).to(device)

        # Stack pointers
        pointers = torch.cat((
            torch.LongTensor([0]).to(device),
            *[csr.pointers[1:] + offset
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
                # be no point with no cluster
                offsets = torch.LongTensor(
                    [0] + [
                        v.max() + 1 if v.shape[0] > 0 else 0
                        for v in val_list[:-1]])
                cum_offsets = torch.cumsum(offsets, dim=0).to(device)
                val = torch.cat([
                    v + o for v, o in zip(val_list, cum_offsets)], dim=0)
            else:
                val = torch.cat(val_list, dim=0)
            values.append(val)

        # Create the Batch object, depending on the data type
        # Default of CSRData is CSRBatch, but subclasses of CSRData
        # may define their own batch class inheriting from CSRBatch.
        batch = csr_list[0].get_batch_class()(
            pointers, *values, dense=False, is_index_value=is_index_value)
        batch.__sizes__ = torch.LongTensor([csr.num_groups for csr in csr_list])

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

    def select(
            self,
            idx: Union[int, List[int], torch.Tensor, np.ndarray],
            *args,
            **kwargs
    ) -> 'CSRData':
        """Indexing CSRBatch format. Supports Numpy and torch indexing
        mechanisms.

        Since indexing breaks batching, this will return a CSRData
        object with updated pointers and values.
        """
        # Default indexing will return a CSRBatch object
        out_batch = super().select(idx, *args, **kwargs)

        # Convert to a base object, since batching mechanism is lost
        # after indexing. For this, we populate an object of the proper
        # class, initialized with fake data
        out = self.get_base_class()(
            torch.arange(1),
            *[torch.empty(0, dtype=v.dtype) for v in self.values])
        out.pointers = out_batch.pointers
        out.values = out_batch.values
        out.is_index_value = out_batch.is_index_value

        return out

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
            verbose: bool = False
    ) -> Union['CSRBatch', 'CSRData']:
        """Load CSRBatch from an HDF5 file. See `CSRData.save`
        for writing such file. Options allow reading only part of the
        clusters.

        :param f: h5 file path of h5py.File or h5py.Group
        :param idx: int, list, numpy.ndarray, torch.Tensor
            Used to select clusters when reading. Supports fancy
            indexing
        :param verbose: bool
        """
        # Indexing breaks batching, so we return a base object if
        # indexing is required
        idx = tensor_idx(idx)
        if idx is not None and idx.shape[0] != 0:
            return cls.get_base_class().load(f, idx=idx, verbose=verbose)

        if not isinstance(f, (h5py.File, h5py.Group)):
            with h5py.File(f, 'r') as file:
                out = cls.load(file, idx=idx, verbose=verbose)
            return out

        # Check if the file actually corresponds to a batch object
        # rather than its corresponding base object
        if '__sizes__' not in f.keys():
            return cls.get_base_class().load(f, idx=idx, verbose=verbose)

        # Load all attributes like the parent class, and also load
        # attributes necessary for batching
        out = super().load(f, idx=idx, verbose=verbose)
        out.__sizes__ = load_tensor(f['__sizes__'])
        return out
