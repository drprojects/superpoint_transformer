import sys
import os.path as osp
import torch
import numpy as np
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.transforms import Transform
from src.data import Data, Cluster, NAG
from src.utils.cpu import available_cpu_count

dependencies_folder = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(dependencies_folder)
sys.path.append(osp.join(dependencies_folder, "dependencies/grid_graph/python/bin"))
sys.path.append(osp.join(dependencies_folder, "dependencies/parallel_cut_pursuit/python/wrappers"))

from grid_graph import edge_list_to_forward_star
from cp_d0_dist import cp_d0_dist

__all__ = ['CutPursuitPartition', 'GridPartition']


class CutPursuitPartition(Transform):
    """Partition Data using cut-pursuit.

    :param regularization: float or List(float)
    :param spatial_weight: float or List(float)
        Weight used to mitigate the impact of the point position in the
        partition. The larger, the less spatial coordinates matter. This
        can be loosely interpreted as the inverse of a maximum
        superpoint radius. If a list is passed, it must match the length
        of `regularization`
    :param cutoff: float or List(float)
        Minimum number of points in each cluster. If a list is passed,
        it must match the length of `regularization`
    :param parallel: bool
        Whether cut-pursuit should run in parallel
    :param iterations: int
        Maximum number of iterations for each partition
    :param k_adjacency: int
        When a node is isolated after a partition, we connect it to the
        nearest nodes. This rules the number of neighbors it should be
        connected to
    :param verbose: bool
    """

    _IN_TYPE = Data
    _OUT_TYPE = NAG
    _MAX_NUM_EGDES = 4294967295
    _NO_REPR = ['verbose', 'parallel']

    def __init__(
            self, regularization=5e-2, spatial_weight=1, cutoff=10,
            parallel=True, iterations=10, k_adjacency=5, verbose=False):
        self.regularization = regularization
        self.spatial_weight = spatial_weight
        self.cutoff = cutoff
        self.parallel = parallel
        self.iterations = iterations
        self.k_adjacency = k_adjacency
        self.verbose = verbose

    def _process(self, data):
        # Sanity checks
        assert data.has_edges, \
            "Cannot compute partition, no edges in Data"
        assert data.num_nodes < np.iinfo(np.uint32).max, \
            "Too many nodes for `uint32` indices"
        assert data.num_edges < np.iinfo(np.uint32).max, \
            "Too many edges for `uint32` indices"
        assert isinstance(self.regularization, (int, float, list)), \
            "Expected a scalar or a List"
        assert isinstance(self.cutoff, (int, list)), \
            "Expected an int or a List"
        assert isinstance(self.spatial_weight, (int, float, list)), \
            "Expected a scalar or a List"

        # Trim the graph
        data = data.to_trimmed()

        # Initialize the hierarchical partition parameters. In particular,
        # prepare the output as list of Data objects that will be stored in
        # a NAG structure
        num_threads = available_cpu_count() if self.parallel else 1
        data.node_size = torch.ones(
            data.num_nodes, device=data.device, dtype=torch.long)  # level-0 points all have the same importance
        data_list = [data]
        regularization = self.regularization
        if not isinstance(regularization, list):
            regularization = [regularization]
        cutoff = self.cutoff
        if isinstance(cutoff, int):
            cutoff = [cutoff] * len(regularization)
        spatial_weight = self.spatial_weight
        if isinstance(spatial_weight, (float, int)):
            spatial_weight = [spatial_weight] * len(regularization)
        assert len(regularization) == len(cutoff) == len(spatial_weight)
        n_dim = data.pos.shape[1]
        n_feat = data.x.shape[1] if data.x is not None else 0

        # Iteratively run the partition on the previous partition level
        for level, (reg, cut, sw) in enumerate(zip(
                regularization, cutoff, spatial_weight)):

            if self.verbose:
                print(
                    f'Launching partition level={level} reg={reg}, '
                    f'cutoff={cut}')

            # Recover the Data object on which we will run the partition
            d1 = data_list[level]

            # Exit if the graph contains only one node
            if d1.num_nodes < 2:
                break

            # User warning if the number of edges exceeds uint32 limits
            if d1.edge_index.shape[1] > self._MAX_NUM_EGDES and self.verbose:
                print(
                    f"WARNING: number of edges {d1.edge_index.shape[1]} "
                    f"exceeds the uint32 limit {self._MAX_NUM_EGDES}. Please"
                    f"update the cut-pursuit source code to accept a larger "
                    f"data type for `index_t`.")

            # Convert edges to forward-star (or CSR) representation
            source_csr, target, reindex = edge_list_to_forward_star(
                d1.num_nodes, d1.edge_index.T.contiguous().cpu().numpy())
            source_csr = source_csr.astype('uint32')
            target = target.astype('uint32')
            edge_weights = d1.edge_attr.cpu().numpy()[reindex] * reg \
                if d1.edge_attr is not None else reg

            # Recover attributes features from Data object
            pos_offset = d1.pos.mean(dim=0)
            if d1.x is not None:
                x = torch.cat((d1.pos - pos_offset, d1.x), dim=1)
            else:
                x = d1.pos - pos_offset
            x = np.asfortranarray(x.cpu().numpy().T)
            node_size = d1.node_size.float().cpu().numpy()
            coor_weights = np.ones(n_dim + n_feat, dtype=np.float32)
            coor_weights[:n_dim] *= sw

            # Partition computation
            super_index, x_c, cluster, edges, times = cp_d0_dist(
                n_dim + n_feat, x, source_csr, target,
                edge_weights=edge_weights, vert_weights=node_size,
                coor_weights=coor_weights, min_comp_weight=cut,
                cp_dif_tol=1e-2, cp_it_max=self.iterations,
                split_damp_ratio=0.7, verbose=self.verbose,
                max_num_threads=num_threads, balance_parallel_split=True,
                compute_Time=True, compute_List=True, compute_Graph=True)

            if self.verbose:
                delta_t = (times[1:] - times[:-1]).round(2)
                print(f'Level {level} iteration times: {delta_t}')
                print(f'partition {level} done')

            # Save the super_index for the i-level
            super_index = torch.from_numpy(super_index.astype('int64'))
            d1.super_index = super_index

            # Save cluster information in another Data object. Convert
            # cluster-to-point indices in a CSR format
            size = torch.LongTensor([c.shape[0] for c in cluster])
            pointer = torch.cat([torch.LongTensor([0]), size.cumsum(dim=0)])
            value = torch.cat([
                torch.from_numpy(x.astype('int64')) for x in cluster])
            pos = torch.from_numpy(x_c[:n_dim].T) + pos_offset.cpu()
            x = torch.from_numpy(x_c[n_dim:].T)
            s = torch.arange(edges[0].shape[0] - 1).repeat_interleave(
                torch.from_numpy((edges[0][1:] - edges[0][:-1]).astype("int64")))
            t = torch.from_numpy(edges[1].astype("int64"))
            edge_index = torch.vstack((s, t))
            edge_attr = torch.from_numpy(edges[2] / reg)
            node_size = torch.from_numpy(node_size)
            node_size_new = scatter_sum(
                node_size.cuda(), super_index.cuda(), dim=0).cpu().long()
            d2 = Data(
                pos=pos, x=x, edge_index=edge_index, edge_attr=edge_attr,
                sub=Cluster(pointer, value), node_size=node_size_new)

            # Trim the graph
            d2 = d2.to_trimmed()

            # If some nodes are isolated in the graph, connect them to
            # their nearest neighbors, so their absence of connectivity
            # does not "pollute" higher levels of partition
            if d2.num_nodes > 1:
                d2 = d2.connect_isolated(k=self.k_adjacency)

            # Aggregate some point attributes into the clusters. This
            # is not performed dynamically since not all attributes can
            # be aggregated (eg 'neighbor_index', 'neighbor_distance',
            # 'edge_index', 'edge_attr'...)
            if 'y' in d1.keys:
                assert d1.y.dim() == 2, \
                    "Expected Data.y to hold `(num_nodes, num_classes)` " \
                    "histograms, not single labels"
                d2.y = scatter_sum(
                    d1.y.cuda(), d1.super_index.cuda(), dim=0).cpu()
                torch.cuda.empty_cache()

            if 'pred' in d1.keys:
                assert d1.pred.dim() == 2, \
                    "Expected Data.pred to hold `(num_nodes, num_classes)` " \
                    "histograms, not single labels"
                d2.pred = scatter_sum(
                    d1.pred.cuda(), d1.super_index.cuda(), dim=0).cpu()
                torch.cuda.empty_cache()

            # Add the l+1-level Data object to data_list and update the
            # l-level after super_index has been changed
            data_list[level] = d1
            data_list.append(d2)

            if self.verbose:
                print('\n' + '-' * 64 + '\n')

        # Create the NAG object
        nag = NAG(data_list)

        return nag


class GridPartition(Transform):
    """XY-grid-based hierarchical partition of Data. The nodes are
    aggregated based on their coordinates in a grid of step `size`.

    :param size: int or List(int)
    """

    _IN_TYPE = Data
    _OUT_TYPE = NAG

    def __init__(self, size=2):
        self.size = size

    def _process(self, data):
        # Sanity checks
        assert data.num_nodes < np.iinfo(np.uint32).max, \
            "Too many nodes for `uint32` indices"
        assert data.num_edges < np.iinfo(np.uint32).max, \
            "Too many edges for `uint32` indices"
        assert isinstance(self.size, (int, float, list)), \
            "Expected a scalar or a List"

        # Initialize the partition data
        size = self.size
        if not isinstance(size, list):
            size = [size]
        data_list = [data]

        # XY-grid partitions
        for w in size:
            # Compute the (i, j) coordinates on the XY grid size
            d = data_list[-1]
            i = d.pos[:, 0].div(w, rounding_mode='trunc').long()
            j = d.pos[:, 1].div(w, rounding_mode='trunc').long()

            # Compute a "manual" partition based on the grid coordinates
            super_index = i * (max(i.max(), j.max()) + 1) + j
            super_index = consecutive_cluster(super_index)[0]
            pos = scatter_mean(d.pos, super_index, dim=0)
            cluster = Cluster(
                super_index, torch.arange(d.num_nodes), dense=True)

            # Update the super_index of the previous level and create
            # the Data object for the new level
            data_list[-1].super_index = super_index
            data_list.append(Data(pos=pos, sub=cluster))

        # Create the NAG object
        nag = NAG(data_list)

        return nag
