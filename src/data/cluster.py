import h5py
import torch
from time import time
from src.data.csr import CSRData, CSRBatch
from src.utils import has_duplicates, tensor_idx, \
    save_tensor, load_tensor
from torch_geometric.nn.pool.consecutive import consecutive_cluster


__all__ = ['Cluster', 'ClusterBatch']


class Cluster(CSRData):
    """Child class of CSRData to simplify some common operations
    dedicated to cluster-point indexing.
    """

    def __init__(self, pointers, points, dense=False, **kwargs):
        super().__init__(
            pointers, points, dense=dense, is_index_value=[True])

    @staticmethod
    def get_batch_type():
        """Required by CSRBatch.from_list."""
        return ClusterBatch

    @property
    def points(self):
        return self.values[0]

    @points.setter
    def points(self, points):
        assert points.device == self.device, \
            f"Points is on {points.device} while self is on {self.device}"
        self.values[0] = points
        # if src.is_debug_enabled():
        #     self.debug()

    @property
    def num_clusters(self):
        return self.num_groups

    @property
    def num_points(self):
        return self.num_items

    def to_super_index(self):
        """Return a 1D tensor of indices converting the CSR-formatted
        clustering structure in 'self' into the 'super_index' format.
        """
        # TODO: this assumes 'self.point' is a permutation, shall we
        #  check this (although it requires sorting) ?
        device = self.device
        out = torch.empty((self.num_items,), dtype=torch.long, device=device)
        cluster_idx = torch.arange(self.num_groups, device=device)
        out[self.points] = cluster_idx.repeat_interleave(self.sizes)
        return out

    def select(self, idx, update_sub=True):
        """Returns a new Cluster with updated clusters and points, which
        indexes `self` using entries in `idx`. Supports torch and numpy
        fancy indexing. `idx` must NOT contain duplicate entries, as
        this would cause ambiguities in super- and sub- indices.

        NB: if `self` belongs to a NAG, calling this function in
        isolation may break compatibility with point and cluster indices
        in the other hierarchy levels. If consistency matters, prefer
        using NAG indexing instead.

        :parameter
        idx: int or 1D torch.LongTensor or numpy.NDArray
            Cluster indices to select from 'self'. Must NOT contain
            duplicates
        update_sub: bool
            If True, the point (i.e. subpoint) indices will also be
            updated to maintain dense indices. The output will then
            contain '(idx_sub, sub_super)' which can help apply these
            changes to maintain consistency with lower hierarchy levels
            of a NAG.

        :return: cluster, (idx_sub, sub_super)
            clusters: Cluster
                indexed cluster
            idx_sub: torch.LongTensor
                to be used with 'Data.select()' on the sub-level
            sub_super: torch.LongTensor
                to replace 'Data.super_index' on the sub-level
        """
        # Normal CSRData indexing, creates a new object in memory
        cluster = self[idx]

        if not update_sub:
            return cluster, (None, None)

        # Convert subpoint indices, in case some subpoints have
        # disappeared. 'idx_sub' is intended to be used with
        # Data.select() on the level below
        # TODO: IMPORTANT consecutive_cluster is a bottleneck for NAG
        #  and Data indexing, can we do better ?
        new_cluster_points, perm = consecutive_cluster(cluster.points)
        idx_sub = cluster.points[perm]
        cluster.points = new_cluster_points

        # Selecting the subpoints with 'idx_sub' will not be
        # enough to maintain consistency with the current points. We
        # also need to update the sub-level's 'Data.super_index', which
        # can be computed from 'cluster'
        sub_super = cluster.to_super_index()

        return cluster, (idx_sub, sub_super)

    def debug(self):
        super().debug()
        assert not has_duplicates(self.points)

    def __repr__(self):
        info = [
            f"{key}={getattr(self, key)}"
            for key in ['num_clusters', 'num_points', 'device']]
        return f"{self.__class__.__name__}({', '.join(info)})"

    def save(self, f, fp_dtype=torch.float):
        """Save Cluster to HDF5 file.

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

        save_tensor(self.pointers, f, 'pointers', fp_dtype=fp_dtype)
        save_tensor(self.points, f, 'points', fp_dtype=fp_dtype)

    @staticmethod
    def load(f, idx=None, update_sub=True, verbose=False):
        """Load Cluster from an HDF5 file. See `Cluster.save` for
        writing such file. Options allow reading only part of the
        clusters.

        This reproduces the behavior of Cluster.select but without
        reading the full pointer data from disk.

        :param f: h5 file path of h5py.File or h5py.Group
        :param idx: int, list, numpy.ndarray, torch.Tensor
            Used to select clusters when reading. Supports fancy
            indexing
        :param update_sub: bool
            If True, the point (i.e. subpoint) indices will also be
            updated to maintain dense indices. The output will then
            contain '(idx_sub, sub_super)' which can help apply these
            changes to maintain consistency with lower hierarchy levels
            of a NAG.
        :param verbose: bool

        :return: cluster, (idx_sub, sub_super)
        """
        KEYS = ['pointers', 'points']

        if not isinstance(f, (h5py.File, h5py.Group)):
            with h5py.File(f, 'r') as file:
                out = Cluster.load(
                    file, idx=idx, update_sub=update_sub, verbose=verbose)
            return out

        assert all(k in f.keys() for k in KEYS)

        start = time()
        idx = tensor_idx(idx)
        if verbose:
            print(f'Cluster.load tensor_idx         : {time() - start:0.5f}s')

        if idx is None or idx.shape[0] == 0:
            start = time()
            pointers = load_tensor(f['pointers'])
            points = load_tensor(f['points'])
            if verbose:
                print(f'Cluster.load read all           : {time() - start:0.5f}s')
            start = time()
            out = Cluster(pointers, points), (None, None)
            if verbose:
                print(f'Cluster.load init               : {time() - start:0.5f}s')
            return out

        # Read only pointers start and end indices based on idx
        start = time()
        ptr_start = load_tensor(f['pointers'], idx=idx)
        ptr_end = load_tensor(f['pointers'], idx=idx + 1)
        if verbose:
            print(f'Cluster.load read ptr       : {time() - start:0.5f}s')

        # Create the new pointers
        start = time()
        pointers = torch.cat([
            torch.zeros(1, dtype=ptr_start.dtype),
            torch.cumsum(ptr_end - ptr_start, 0)])
        if verbose:
            print(f'Cluster.load new pointers   : {time() - start:0.5f}s')

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
            print(f'Cluster.load val_idx        : {time() - start:0.5f}s')

        # Read the points, now we have computed the val_idx
        start = time()
        points = load_tensor(f['points'], idx=val_idx)
        if verbose:
            print(f'Cluster.load read points    : {time() - start:0.5f}s')

        # Build the Cluster object
        start = time()
        cluster = Cluster(pointers, points)
        if verbose:
            print(f'Cluster.load init           : {time() - start:0.5f}s')

        if not update_sub:
            return cluster, (None, None)

        # Convert subpoint indices, in case some subpoints have
        # disappeared. 'idx_sub' is intended to be used with
        # Data.select() on the level below
        # TODO: IMPORTANT consecutive_cluster is a bottleneck for NAG
        #  and Data indexing, can we do better ?
        start = time()
        new_cluster_points, perm = consecutive_cluster(cluster.points)
        idx_sub = cluster.points[perm]
        cluster.points = new_cluster_points
        if verbose:
            print(f'Cluster.load update_sub     : {time() - start:0.5f}s')

        # Selecting the subpoints with 'idx_sub' will not be
        # enough to maintain consistency with the current points. We
        # also need to update the sublevel's 'Data.super_index', which
        # can be computed from 'cluster'
        start = time()
        sub_super = cluster.to_super_index()
        if verbose:
            print(f'Cluster.load super_index    : {time() - start:0.5f}s')

        return cluster, (idx_sub, sub_super)


class ClusterBatch(Cluster, CSRBatch):
    """Wrapper for Cluster batching."""
    __csr_type__ = Cluster
