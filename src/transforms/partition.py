import torch
import numpy as np
from torch_scatter import scatter_sum, scatter_mean
from omegaconf import OmegaConf
from pycut_pursuit.cp_d0_dist import cp_d0_dist
from grid_graph import edge_list_to_forward_star

from src.transforms import Transform
from src.data import Data, NAG, Cluster, InstanceData
from src.utils import (
    xy_partition, 
    xyz_partition, 
    compute_edge_distances_batch,
    available_cpu_count)

from src.utils.components_merge import merge_components_by_contour_prior_on_data
from src.nn import CatFusion


__all__ = ['CutPursuitPartition', 'GridPartition', 'GreedyContourPriorPartition']

class CutPursuitPartition(Transform):
    """Partition a graph contained in a `Data` object using cut-pursuit.

    The input `Data` object is assumed to hold the following attributes:
      - `pos` carrying node spatial coordinates
      - `x` carrying node features
      - `edge_index` carrying the adjacency graph edges in Pytorch
         Geometric format (typically generated with `AdjacencyGraph`)
      - `edge_attr` carrying the scalar edge weights in Pytorch
         Geometric format (typically generated with `AdjacencyGraph`)

    The input `Data` object is assumed to be the atomic level. This means
    that the outputed `NAG` object `has_atoms `is set to `True`. This parameter
    is used in particular when the network is in “nano” mode.

    The quality of a partition may be assessed in terms of efficiency
    (how much it simplifies the input graph) and accuracy (how well it
    respects the semantic boundaries). We provide two tools for
    assessing these: `NAG.level_ratios` which computes the ratio of the
    number of elements between successive partition levels, and
    `Data.semantic_segmentation_oracle()` which computes the semantic
    segmentation metrics of a hypothetical oracle model capable of
    predicting the majority label for each superpoint. See our
    Superpoint Transformer tutorial
    `notebooks/superpoint_transformer_tutorial.ipynb` for more on this.

    :param regularization: float or List(float)
        Regularization strength used for each partition level. This is
        the primary parameter for adjusting cut-pursuit partitions. The
        larger the regularization, the coarser the partition, the fewer
        the superpoints, the bigger the superpoints, the lower their
        semantic purity (ie superpoints are more likely to bleed across
        semantic object boundaries). And vice versa. If a list is
        passed, the values are assumed to be increasing
    :param spatial_weight: float or List(float)
        Weight used to mitigate the impact of the point position in the
        partition. The smaller, the less spatial coordinates matter.
        This can be loosely interpreted as the inverse of a maximum
        superpoint radius. It typically affects the size of superpoints
        in geometrically/radiometrically homogeneous regions such as the
        ground, walls, or ceilings. Setting a large `spatial_weight`
        will have a "voronoi tessellation" effect on the superpoint
        partition, preventing too-large superpoints from being
        constructed in these otherwise-homogeneous regions. Inversely,
        setting a small `spatial_weight` will encourage cut-pursuit to
        create superpoints as large as possible, so long as the features
        of the points inside are homogeneous. In an extreme case: the
        entire floor would then be a single superpoint. If a list is
        passed, it must match the length of `regularization`
    :param cutoff: float or List(float)
        Minimum number of points in each superpoint. The output
        partition will not contain any superpoint smaller than `cutoff`.
        If a list is passed, it must match the length of
        `regularization`
    :param parallel: bool
        Whether cut-pursuit should run in parallel (ie on multiple CPU
        threads)
    :param iterations: int
        Maximum number of iterations for the cut-pursuit algorithm. The
        higher, the longer the processing. A value in $[10, 15]$ is
        usually sufficient
    :param k_adjacency: int
        When a node is isolated after a partition, we connect it to the
        nearest nodes. This rules the number of neighbors it should be
        connected to
    :param edge_reduce: str
        How to reduce duplicate edges when merging components.
        Options: 'add', 'mean', 'max', 'min', 'mul'. Default: 'add'.
        Especially useful for hierarchical partition, when trimming the 
        graph obtained after computing a partition.
    :param verbose: bool
    """

    _IN_TYPE = Data
    _OUT_TYPE = NAG
    _MAX_NUM_EDGES = 4294967295
    _NO_REPR = ['verbose', 'parallel']

    def __init__(
            self,
            regularization=5e-2,
            spatial_weight=1,
            cutoff=10,
            parallel=True,
            iterations=10,
            k_adjacency=5,
            edge_reduce='add',
            verbose=False):
        self.regularization = regularization
        self.spatial_weight = spatial_weight
        self.cutoff = cutoff
        self.parallel = parallel
        self.iterations = iterations
        self.k_adjacency = k_adjacency
        self.edge_reduce = edge_reduce
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

        # Keep track of the input device
        device = data.device

        # Trim the graph
        # TODO: calling this on the level-0 adjacency graph is a bit sluggish
        #  but still saves partition time overall. May be worth finding a
        #  quick way of removing self loops and redundant edges...
        data = data.to_trimmed(reduce=self.edge_reduce)

        # Initialize the hierarchical partition parameters. In particular,
        # prepare the output as list of Data objects that will be stored in
        # a NAG structure
        num_threads = available_cpu_count() if self.parallel else 1
        data.node_size = torch.ones(
            data.num_nodes,
            device=data.device,
            dtype=torch.long)  # level-0 points all have the same importance
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
            if d1.edge_index.shape[1] > self._MAX_NUM_EDGES and self.verbose:
                print(
                    f"WARNING: number of edges {d1.edge_index.shape[1]} "
                    f"exceeds the uint32 limit {self._MAX_NUM_EDGES}. Please"
                    f"update the cut-pursuit source code to accept a larger "
                    f"data type for `index_t`.")

            # Convert edges to forward-star (or CSR) representation
            source_csr, target, reindex = edge_list_to_forward_star(
                d1.num_nodes,
                d1.edge_index.T.contiguous().cpu().numpy())
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
                n_dim + n_feat,
                x,
                source_csr,
                target,
                edge_weights=edge_weights,
                vert_weights=node_size,
                coor_weights=coor_weights,
                min_comp_weight=cut,
                cp_dif_tol=1e-2,
                cp_it_max=self.iterations,
                split_damp_ratio=0.7,
                verbose=self.verbose,
                max_num_threads=num_threads,
                balance_parallel_split=True,
                compute_Time=True,
                compute_List=True,
                compute_Graph=True)

            if self.verbose:
                delta_t = (times[1:] - times[:-1]).round(2)
                print(f'Level {level} iteration times: {delta_t}')
                print(f'partition {level} done')

            # Save the super_index for the i-level
            super_index = torch.from_numpy(super_index.astype('int64'))
            d1.super_index = super_index

            # Save cluster information in another Data object. Convert
            # cluster-to-point indices in a CSR format
            size = torch.tensor([c.shape[0] for c in cluster], dtype=torch.long)
            pointer = torch.cat([
                torch.tensor([0], dtype=torch.long),
                size.cumsum(dim=0)])
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
                node_size.cuda(),
                super_index.cuda(), dim=0).cpu().long()
            d2 = Data(
                pos=pos,
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                sub=Cluster(pointer, value),
                node_size=node_size_new)

            # Merge the lower level's instance annotations, if any
            if d1.obj is not None and isinstance(d1.obj, InstanceData):
                d2.obj = d1.obj.merge(d1.super_index.to(d1.obj.device))

            # Trim the graph
            d2 = d2.to_trimmed(reduce=self.edge_reduce)

            # If some nodes are isolated in the graph, connect them to
            # their nearest neighbors, so their absence of connectivity
            # does not "pollute" higher levels of partition
            if d2.num_nodes > 1:
                d2 = d2.connect_isolated(k=self.k_adjacency)

            # Aggregate some point attributes into the clusters. This
            # is not performed dynamically since not all attributes can
            # be aggregated (e.g. 'neighbor_index', 'neighbor_distance',
            # 'edge_index', 'edge_attr'...)
            if 'y' in d1.keys:
                assert d1.y.dim() == 2, \
                    "Expected Data.y to hold `(num_nodes, num_classes)` " \
                    "histograms, not single labels"
                d2.y = scatter_sum(
                    d1.y.cuda(), d1.super_index.cuda(), dim=0).cpu()
                torch.cuda.empty_cache()

            if 'semantic_pred' in d1.keys:
                assert d1.semantic_pred.dim() == 2, \
                    "Expected Data.semantic_pred to hold `(num_nodes, num_classes)` " \
                    "histograms, not single labels"
                d2.semantic_pred = scatter_sum(
                    d1.semantic_pred.cuda(), d1.super_index.cuda(), dim=0).cpu()
                torch.cuda.empty_cache()

            # TODO: aggregate other attributes ?

            # TODO: if scatter operations are bottleneck, use scatter_csr

            # Add the l+1-level Data object to data_list and update the
            # l-level after super_index has been changed
            data_list[level] = d1
            data_list.append(d2)

            if self.verbose:
                print('\n' + '-' * 64 + '\n')

        # Create the NAG object
        nag = NAG(data_list, start_i_level = 0).to(device)

        return nag


class GridPartition(Transform):
    """XY(Z)-grid-based hierarchical partition of Data. The nodes are
    aggregated based on their coordinates in a grid of step `size`.

    The input `Data` object is assumed to be the atomic level. This
    means that the outputed `NAG` object `has_atoms `is set to `True`.
    This parameter is used in particular when the network is in “nano”
    mode.

    :param size: int or List(int)
    :param dim: int
        Dimension of the grid.
            2 for XY partitoin,
            3 for XYZ partition
    """

    _IN_TYPE = Data
    _OUT_TYPE = NAG

    def __init__(self, size=2, dim=2):
        self.size = size
        self.dim = dim

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

        # XY(Z)-grid partitions
        for w in size:
            # Compute a "manual" partition based on the grid coordinates
            d = data_list[-1]
            if self.dim == 2:
                super_index = xy_partition(d.pos, w, consecutive=True)
            elif self.dim == 3:
                super_index = xyz_partition(d.pos, w, consecutive=True)

            # Compute the superpoint centroids and Cluster object
            pos = scatter_mean(d.pos, super_index, dim=0)
            cluster = Cluster(
                super_index, torch.arange(d.num_nodes, device=d.device), dense=True)

            # TODO: support more Data attributes and more advanced
            #  grouping, probably by interfacing with
            #  src.transforms.sampling._group_data()

            # Update the super_index of the previous level and create
            # the Data object for the new level
            data_list[-1].super_index = super_index
            data_list.append(Data(pos=pos, sub=cluster))

        # Create the NAG object
        nag = NAG(data_list, start_i_level = 0)

        return nag


class GreedyContourPriorPartition(Transform):
    _IN_TYPE = Data
    _OUT_TYPE = NAG
    _NO_REPR = ['verbose', 'sharding']

    """Computes a hierarchical partition of 3D points cloud based.
    The objective function is an energy-based function with a contour prior.
    The algorithm uses a greedy approach to efficiently merge components based 
    on the energy function. It starts with all points being its own node, 
    and iteratively merges nodes based on the objective function and the min_size 
    parameter.
    
    The method assumes the Data object has the following attributes:
        - `pos` : the point coordinates (if k>0)
        - `x` : the point features
        - `edge_index` : the edge index
    
    `x` and a graph (defined by `edge_index`) are explicitly used 
    in the definition of the objective energy function.
    
    :param reg: float or List[float]
        Regularization strenght used for each partition level,
        ruling the importance of edges in the energy.
        Larger values result in more coarser partitions.
        Typical values tested on DALES, S3DIS and KITTI-360 :  2e-2.
        
    :param min_size: int or List[int]
        Minimum number of points (voxels) in each component for each level.
        This is the main parameter to control the partition granularity.
        
        Typical values : [5, 30, 90].
        This means that first level of the partition will be composed of 
        superpoints of at least 5 points, the second level of at least 30 points,
        and the third level of at least 90 points.
    
    :param spatial_weight: None or float
        If None, the position of the points is not concatenated to the point features.
        This is parametrization of EZ-SP.
        If a float is provided, the position of the points is weighted by the `spatial_weight`
        and concatenated to the point features.
        Briefly, x <- [x, spatial_weight * pos]
        
    :param edge_weight_mode: str
        Mode to compute the edge weights.
        See docstring of the `edge_weights` method.
    :param d_0: float
        Reference distance used to compute the edge weights.
        See docstring of the `edge_weights` method.
    :param edge_reduce: str
        How to reduce duplicate edges when merging components.
        Options: 'add', 'mean', 'max', 'min', 'mul'. Default: 'add'.
        Especially useful for hierarchical partition, when trimming the 
        graph obtained after computing a partition.
        
    :param k : int
        Number of neighbors to connect isolated nodes to.
    :param w_adjacency: float
        Scalar used to modulate the newly created edge weights when `k > 0`.
    
    :param max_iterations: int
        Maximum number of merging iterations.
        If `max_iterations <= 0`, the algorithm will run until the min_size 
        requirements are met.

    :param verbose: bool
        Whether to print verbose information.
    :param sharding: int, float
        Allows mitigating memory use. If `sharding > 1`,
        `edge_index` will be processed into chunks of `sharding` during 
        the bottleneck of the algorithm. If `0 < sharding < 1`, then 
        `edge_index` will be divided into parts of 
        `edge_index.shape[1] * sharding` or less
        
    """

    def __init__(self,
                 reg,
                 min_size,
                 spatial_weight=None,
                 edge_weight_mode='unit',
                 d_0=None,
                 edge_reduce='add',
                 k=0,
                 w_adjacency=0,
                 max_iterations=-1,

                 verbose=False,
                 sharding=None):

        # Basic parameters setting
        self.spatial_weight = spatial_weight
        self.edge_weight_mode = edge_weight_mode
        self.d_0 = d_0
        self.edge_reduce = edge_reduce
        self.k = k
        self.w_adjacency = w_adjacency
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.sharding = sharding

        self.feature_fusion = CatFusion()
        assert edge_weight_mode in ['unit', 'inverse_distance', 'exp_neg_distance', 'exp_neg_latent_distance', 'affinity_latent_distance', 'affinity_latent_distance_exp_neg'], \
            f"Invalid edge weight mode: {edge_weight_mode}.\n Valid options are: 'unit', 'inverse_distance', 'exp_neg_distance', 'exp_neg_latent_distance', 'affinity_latent_distance', 'affinity_latent_distance_exp_neg'"

        # Initialize the hierarchical partition parameters : reg and min_size
        if isinstance(min_size, list) or OmegaConf.is_list(min_size):
            num_levels = len(min_size)
        elif isinstance(reg, list) or OmegaConf.is_list(reg):
            num_levels = len(reg)
        else:
            num_levels = 1

        assert isinstance(reg, (int, float, list)) or OmegaConf.is_list(reg), \
            "`reg` parameter expected a scalar or a List"
        self.reg = reg if (isinstance(reg, list) or OmegaConf.is_list(reg)) else [reg]*num_levels

        assert isinstance(min_size, (int, list)) or OmegaConf.is_list(min_size), \
            "`min_size` parameter expected an int or a List"
        self.min_size = min_size if (isinstance(min_size, list) or OmegaConf.is_list(min_size)) else [min_size]*num_levels

        assert len(self.reg) == len(self.min_size), \
            "Expected the same number of `reg` and `min_size` parameters"

    def _process(self, data):

        data_list = [data]

        for level, (reg, min_size) in enumerate(zip(self.reg, self.min_size)):
            # Recover the Data object on which we will run the partition
            d1 = data_list[level]

            d1 = self.edge_weights(d1, d1.edge_index)
            d1 = self.concatenate_pos_to_x(d1)

            d1, (X_merged, S_merged, E_cp, W_cp, P_merged, sub) = \
                merge_components_by_contour_prior_on_data(
                    d1,
                    reg,
                    min_size,
                    False,
                    self.k,
                    self.w_adjacency,
                    self.max_iterations,
                    self.verbose,
                    self.sharding,
                    self.edge_reduce)

            # We give the Data object the computed super_index attribute
            data_list[level] = d1


            d2 = Data(
                pos=P_merged,
                node_size=S_merged,
                sub=sub,
                x=X_merged,
                edge_index=E_cp,
                edge_attr=W_cp)

            # No need to trim the graph, as the `merge_components_by_contour_prior_on_data`
            # function already does it.
            # d2 = d2.to_trimmed()

            # Isolated nodes are also managed in the `merge_components_by_contour_prior_on_data`
            # function. See argument `self.k` and `self.w_adjacency`.

            # Aggregate some point attributes into the clusters. This
            # is not performed dynamically since not all attributes can
            # be aggregated (e.g. 'neighbor_index', 'neighbor_distance',
            # 'edge_index', 'edge_attr'...)
            if 'y' in d1.keys:
                assert d1.y.dim() == 2, \
                    "Expected Data.y to hold `(num_nodes, num_classes)` " \
                    "histograms, not single labels"
                d2.y = scatter_sum(
                    d1.y, d1.super_index, dim=0)
                torch.cuda.empty_cache()

            if 'semantic_pred' in d1.keys:
                assert d1.semantic_pred.dim() == 2, \
                    "Expected Data.semantic_pred to hold `(num_nodes, num_classes)` " \
                    "histograms, not single labels"
                d2.semantic_pred = scatter_sum(
                    d1.semantic_pred, d1.super_index, dim=0)
                torch.cuda.empty_cache()

            data_list.append(d2)

        nag = NAG(data_list, start_i_level = 0)

        return nag

    def edge_weights(self, data: Data, edge_index: torch.Tensor) -> Data:
        """
        Compute the edge weights for the graph.

        self.edge_weight_mode:
         - `unit` : 1
         - `inverse_distance` : 1/(1+distance/d_0)
         - `exp_neg_distance` : exp(-distance/d_0)
         - `exp_neg_latent_distance` : exp(-latent_distance/d_0)
         - `affinity_latent_distance` : d_neg_exp/(1 - d_neg_exp + self.epsilon_edge_weight) with d_neg_exp = exp(-latent_distance/d_0)
        """

        data.edge_index = edge_index

        # First, compute the distance or latent distance of all edges
        if self.edge_weight_mode == 'unit':
            pass

        elif self.edge_weight_mode in ['inverse_distance', 'exp_neg_distance']:
            #TODO : the distance was already computed in the `KNN` transform.
            # To avoid recomputing the distance, we could update `AdjacencyGraph`
            # so that it can store the raw neighbor_distance (and not a transformation of the neighbor_distance)
            # and remove the `NAGRemoveKeys('edge_attr')` in the transforms
            distance = compute_edge_distances_batch(data.pos, edge_index, self.sharding)
            d_0 = distance.mean() if self.d_0 is None else self.d_0

        else :
            latent_distance = compute_edge_distances_batch(data.x, edge_index, self.sharding)
            d_0 = latent_distance.mean() if self.d_0 is None else self.d_0


        # Then, computes the edge weights
        if self.edge_weight_mode == 'unit':
            data.edge_attr = torch.ones_like(edge_index[0])

        elif self.edge_weight_mode == 'inverse_distance':
            data.edge_attr = 1/(1+distance/d_0)
        elif self.edge_weight_mode == 'exp_neg_distance':
            data.edge_attr = torch.exp(-distance/d_0)


        elif self.edge_weight_mode == 'exp_neg_latent_distance':
            data.edge_attr = torch.exp(-latent_distance/d_0)

        elif self.edge_weight_mode == 'affinity_latent_distance':
            print(f"epsilon_edge_weight: {self.epsilon_edge_weight}")
            d_neg_exp = torch.exp(-latent_distance/d_0)
            data.edge_attr = d_neg_exp/(1 - d_neg_exp + self.epsilon_edge_weight)


        else:
            raise ValueError(f"Invalid edge weight mode: {self.edge_weight_mode}.\n Valid options are: 'unit', 'inverse_distance', 'exp_neg_distance', 'exp_neg_latent_distance', 'affinity_latent_distance', 'affinity_latent_distance_exp_neg'")

        return data

    def concatenate_pos_to_x(self, data: Data) -> Data:
        """
        Concatenate the position of the points to the point features weighted by the `self.spatial_weight`.

        Briefly, x <- [x, self.spatial_weight * pos]

        """

        if (self.spatial_weight is None) or (self.spatial_weight == 0):
            return data
        else :
            data.x = self.feature_fusion(data.x, data.pos * self.spatial_weight)
            return data

