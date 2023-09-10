import torch
import numpy as np
import itertools
from scipy.spatial import Delaunay
from torch_scatter import scatter_mean, scatter_std
from torch_geometric.utils import add_self_loops
from src.transforms import Transform
import src
from src.data import NAG
from pgeof import pgeof
from src.utils import print_tensor_info, isolated_nodes, edge_to_superedge, \
    subedges, to_trimmed, cluster_radius_nn, is_trimmed, base_vectors_3d, \
    scatter_mean_orientation, POINT_FEATURES, SEGMENT_BASE_FEATURES, \
    SUBEDGE_FEATURES, ON_THE_FLY_HORIZONTAL_FEATURES, \
    ON_THE_FLY_VERTICAL_FEATURES, sanitize_keys

__all__ = [
    'AdjacencyGraph', 'SegmentFeatures', 'DelaunayHorizontalGraph',
    'RadiusHorizontalGraph', 'OnTheFlyHorizontalEdgeFeatures',
    'OnTheFlyVerticalEdgeFeatures', 'NAGAddSelfLoops', 'ConnectIsolated',
    'NodeSize']


class AdjacencyGraph(Transform):
    """Create the adjacency graph in `edge_index` and `edge_attr` based
    on the `Data.neighbor_index` and `Data.neighbor_distance`.

    NB: this graph is directed wrt Pytorch Geometric, but cut-pursuit
    happily takes this as an input.

    :param k: int
        Number of neighbors to consider for the adjacency graph
    :param w: float
        Scalar used to modulate the edge weight. If `w <= 0`, all edges
        will have a weight of 1. Otherwise, edges weights will follow:
        ```1 / (w + neighbor_distance / neighbor_distance.mean())```
    """

    def __init__(self, k=10, w=-1):
        self.k = k
        self.w = w

    def _process(self, data):
        assert data.has_neighbors, \
            "Data must have 'neighbor_index' attribute to allow adjacency " \
            "graph construction."
        assert getattr(data, 'neighbor_distance', None) is not None \
               or self.w <= 0, \
            "Data must have 'neighbor_distance' attribute to allow adjacency " \
            "graph construction."
        assert self.k <= data.neighbor_index.shape[1]

        # Compute source and target indices based on neighbors
        source = torch.arange(
            data.num_nodes, device=data.device).repeat_interleave(self.k)
        target = data.neighbor_index[:, :self.k].flatten()

        # Account for -1 neighbors and delete corresponding edges
        mask = target >= 0
        source = source[mask]
        target = target[mask]

        # Save edges and edge features in data
        data.edge_index = torch.stack((source, target))
        if self.w > 0:
            # Recover the neighbor distances and apply the masking
            distances = data.neighbor_distance[:, :self.k].flatten()[mask]
            data.edge_attr = 1 / (self.w + distances / distances.mean())
        else:
            data.edge_attr = torch.ones_like(source, dtype=torch.float)

        return data


class SegmentFeatures(Transform):
    """Compute segment features for all the NAG levels except its first
    (ie the 0-level). These are handcrafted node features that will be
    saved in the node attributes. To make use of those at training time,
    remember to move them to the `x` attribute using `AddKeysTo` and
    `NAGAddKeysTo`.

    The supported feature keys are the following:
      - linearity
      - planarity
      - scattering
      - verticality
      - curvature
      - log_length
      - log_surface
      - log_volume
      - normal
      - log_size

    :param n_max: int
        Maximum number of level-0 points to sample in each cluster to
        when building node features
    :param n_min: int
        Minimum number of level-0 points to sample in each cluster,
        unless it contains fewer points
    :param keys: List(str), str, or None
        Features to be computed segment-wise and saved under `<key>`.
        If None, all supported features will be computed
    :param mean_keys: List(str), str, or None
        Features to be computed from the points and the segment-wise
        mean aggregation will be saved under `mean_<key>`. If None, all
        supported features will be computed
    :param std_keys: List(str), str, or None
        Features to be computed from the points and the segment-wise
        std aggregation will be saved under `std_<key>`. If None, all
        supported features will be computed
    :param strict: bool
        If True, will raise an exception if an attribute from key is
        not within the input point Data keys
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _NO_REPR = ['strict']

    def __init__(
            self,
            n_max=32,
            n_min=5,
            keys=None,
            mean_keys=None,
            std_keys=None,
            strict=True):
        self.n_max = n_max
        self.n_min = n_min
        self.keys = sanitize_keys(keys, default=SEGMENT_BASE_FEATURES)
        self.mean_keys = sanitize_keys(mean_keys, default=POINT_FEATURES)
        self.std_keys = sanitize_keys(std_keys, default=POINT_FEATURES)
        self.strict = strict

    def _process(self, nag):
        for i_level in range(1, nag.num_levels):
            nag = _compute_cluster_features(
                i_level,
                nag,
                n_max=self.n_max,
                n_min=self.n_min,
                keys=self.keys,
                mean_keys=self.mean_keys,
                std_keys=self.std_keys,
                strict=self.strict)
        return nag


def _compute_cluster_features(
        i_level,
        nag,
        n_max=32,
        n_min=5,
        keys=None,
        mean_keys=None,
        std_keys=None,
        strict=True):
    assert isinstance(nag, NAG)
    assert i_level > 0, "Cannot compute cluster features on level-0"
    assert nag[0].num_nodes < np.iinfo(np.uint32).max, \
        "Too many nodes for `uint32` indices"

    keys = sanitize_keys(keys, default=SEGMENT_BASE_FEATURES)
    mean_keys = sanitize_keys(mean_keys, default=POINT_FEATURES)
    std_keys = sanitize_keys(std_keys, default=POINT_FEATURES)

    # Recover the i_level Data object we will be working on
    data = nag[i_level]
    num_nodes = data.num_nodes
    device = nag.device

    # Compute how many level-0 points each level cluster contains
    sub_size = nag.get_sub_size(i_level, low=0)

    # Sample points among the clusters. These will be used to compute
    # cluster geometric features
    idx_samples, ptr_samples = nag.get_sampling(
        high=i_level, low=0, n_max=n_max, n_min=n_min,
        return_pointers=True)

    # Compute cluster geometric features
    xyz = nag[0].pos[idx_samples].cpu().numpy()
    nn = np.arange(idx_samples.shape[0]).astype('uint32')
    nn_ptr = ptr_samples.cpu().numpy().astype('uint32')

    # Heuristic to avoid issues when a cluster sampling is such that
    # it produces singular covariance matrix (eg the sampling only
    # contains the same point repeated multiple times)
    xyz = xyz + torch.rand(xyz.shape).numpy() * 1e-5

    # C++ geometric features computation on CPU
    f = pgeof(xyz, nn, nn_ptr, k_min=5, k_step=-1, verbose=False)
    f = torch.from_numpy(f.astype('float32'))

    # Recover length, surface and volume
    if 'linearity' in keys:
        data.linearity = f[:, 0].to(device).view(-1, 1)

    if 'planarity' in keys:
        data.planarity = f[:, 1].to(device).view(-1, 1)

    if 'scattering' in keys:
        data.scattering = f[:, 2].to(device).view(-1, 1)

    if 'verticality' in keys:
        data.verticality = f[:, 3].to(device).view(-1, 1)

    if 'curvature' in keys:
        data.curvature = f[:, 10].to(device).view(-1, 1)

    if 'log_length' in keys:
        data.log_length = torch.log(f[:, 7] + 1).to(device).view(-1, 1)

    if 'log_surface' in keys:
        data.log_surface = torch.log(f[:, 8] + 1).to(device).view(-1, 1)

    if 'log_volume' in keys:
        data.log_volume = torch.log(f[:, 9] + 1).to(device).view(-1, 1)

    # As a way to "stabilize" the normals' orientation, we choose to
    # express them as oriented in the z+ half-space
    if 'normal' in keys:
        data.normal = f[:, 4:7].view(-1, 3).to(device)
        data.normal[data.normal[:, 2] < 0] *= -1

    if 'log_size' in keys:
        data.log_size = (torch.log(sub_size + 1).view(-1, 1) - np.log(2)) / 10

    # Get the cluster index each poitn belongs to
    super_index = nag.get_super_index(i_level)

    # Add the mean of point attributes, identified by their key
    for key in mean_keys:
        f = getattr(nag[0], key, None)
        if f is None and strict:
            raise ValueError(f"No point key `{key}` to build 'mean_{key} key'")
        if f is None:
            continue
        if key == 'normal':
            data[f'mean_{key}'] = scatter_mean_orientation(
                nag[0][key], super_index)
        else:
            data[f'mean_{key}'] = scatter_mean(nag[0][key], super_index, dim=0)

    # Add the std of point attributes, identified by their key
    for key in std_keys:
        f = getattr(nag[0], key, None)
        if f is None and strict:
            raise ValueError(f"No point key `{key}` to build 'std_{key} key'")
        if f is None:
            continue
        data[f'std_{key}'] = scatter_std(nag[0][key], super_index, dim=0)

    # To debug sampling
    if src.is_debug_enabled():
        data.super_super_index = super_index.to(device)
        data.node_idx_samples = idx_samples.to(device)
        data.node_xyz_samples = torch.from_numpy(xyz).to(device)
        data.node_nn_samples = torch.from_numpy(nn.astype('int64')).to(device)
        data.node_nn_ptr_samples = torch.from_numpy(
            nn_ptr.astype('int64')).to(device)

        end = ptr_samples[1:]
        start = ptr_samples[:-1]
        super_index_samples = torch.repeat_interleave(
            torch.arange(num_nodes), end - start)
        print('\n\n' + '*' * 50)
        print(f'        cluster graph for level={i_level}')
        print('*' * 50 + '\n')
        print(f'nag: {nag}')
        print(f'data: {data}')
        print('\n* Sampling for superpoint features')
        print_tensor_info(idx_samples, name='idx_samples')
        print_tensor_info(ptr_samples, name='ptr_samples')
        print(f'all clusters have a ptr:                   '
              f'{ptr_samples.shape[0] - 1 == num_nodes}')
        print(f'all clusters received n_min+ samples:      '
              f'{(end - start).ge(n_min).all()}')
        print(f'clusters which received no sample:         '
              f'{torch.where(end == start)[0].shape[0]}/{num_nodes}')
        print(f'all points belong to the correct clusters: '
              f'{torch.equal(super_index[idx_samples], super_index_samples)}')

    # Update the i_level Data in the NAG
    nag._list[i_level] = data

    return nag


class DelaunayHorizontalGraph(Transform):
    """Compute horizontal edges for all NAG levels except its first
    (ie the 0-level). These are the edges connecting the segments at
    each level, equipped with handcrafted edge features.

    This approach relies on the dual graph of the Delaunay triangulation
    of the point cloud. To reduce computation, each segment is susampled
    based on its size. This sampling still has downsides and the
    triangulation remains fairly long for large clouds, due to its O(N²)
    complexity. Besides, the horizontal graph induced by the
    triangulation is a visibility-based graph, meaning neighboring
    segments may not be connected if a large enough segment separates
    them. A faster alternative is `RadiusHorizontalGraph`.

    By default, a series of handcrafted edge attributes are computed and
    stored in the corresponding `Data.edge_attr`. However, if one only
    needs a subset of those at train time, one may make use of
    `SelectColumns` and `NAGSelectColumns`.

    The supported feature keys are the following:
      - mean_off
      - std_off
      - mean_dist

    :param n_max_edge: int
        Maximum number of level-0 points to sample in each cluster to
        when building edges and edge features from Delaunay
        triangulation and edge features
    :param n_min: int
        Minimum number of level-0 points to sample in each cluster,
        unless it contains fewer points
    :param max_dist: float or List(float)
        Maximum distance allowed for edges. If zero, this is ignored.
        Otherwise, edges whose distance is larger than max_dist. We pay
        particular attention here to avoid isolating nodes by distance
        filtering. If a node was isolated by max_dist filtering, we
        preserve its shortest edge to avoid it, even if it is larger
        than max_dist
    :param keys: List(str)
        Features to be computed. Attributes will be saved under `<key>`
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, n_max_edge=64, n_min=5, max_dist=-1, keys=None):
        self.n_max_edge = n_max_edge
        self.n_min = n_min
        self.max_dist = max_dist
        self.keys = sanitize_keys(keys, default=SUBEDGE_FEATURES)

    def _process(self, nag):
        assert isinstance(self.max_dist, (int, float, list)), \
            "Expected a scalar or a List"

        max_dist = self.max_dist
        if not isinstance(max_dist, list):
            max_dist = [max_dist] * (nag.num_levels - 1)

        for i_level, md in zip(range(1, nag.num_levels), max_dist):
            nag = _horizontal_graph_by_delaunay(
                i_level, nag,
                n_max_edge=self.n_max_edge,
                n_min=self.n_min,
                max_dist=md,
                keys=self.keys)

        return nag


def _horizontal_graph_by_delaunay(
        i_level,
        nag,
        n_max_edge=64,
        n_min=5,
        max_dist=-1,
        keys=None):
    assert isinstance(nag, NAG)
    assert i_level > 0, "Cannot compute cluster graph on level 0"
    assert nag[0].has_edges, \
        "Level-0 must have an adjacency structure in 'edge_index' to allow " \
        "guided sampling for superedges construction."
    assert nag[0].num_nodes < np.iinfo(np.uint32).max, \
        "Too many nodes for `uint32` indices"
    assert nag[0].num_edges < np.iinfo(np.uint32).max, \
        "Too many edges for `uint32` indices"

    keys = sanitize_keys(keys, default=SUBEDGE_FEATURES)

    # Recover the i_level Data object we will be working on
    data = nag[i_level]
    num_nodes = data.num_nodes
    device = nag.device

    # Exit in case the i_level graph contains only one node
    if num_nodes < 2:
        data.edge_index = None
        data.edge_attr = None
        nag._list[i_level] = data
        return nag

    # To guide the sampling for superedges, we want to sample among
    # points whose neighbors in the level-0 adjacency graph belong to
    # a different cluster in the i_level graph. To this end, we first
    # need to tell whose i_level cluster each level-0 point belongs to.
    # This step requires having access to the whole NAG, since we need
    # to convert level-0 point indices into their corresponding level-i
    # superpoint indices
    super_index = nag.get_super_index(i_level)

    # Once we know the i_level cluster each level-0 point belongs to,
    # we can search for level-0 edges between i_level clusters. These
    # in turn tell us which level-0 points to sample from
    edges_point_adj = super_index[nag[0].edge_index]
    inter_cluster = torch.where(edges_point_adj[0] != edges_point_adj[1])[0]
    edges_point_adj_inter = edges_point_adj[:, inter_cluster]
    idx_edge_point = nag[0].edge_index[:, inter_cluster].unique()

    # Some nodes may be isolated and not be connected to the other nodes
    # in the level-0 adjacency graph. For that reason, we need to look
    # for such isolated nodes and sample point inside them, since the
    # above approach will otherwise ignore them
    is_isolated = isolated_nodes(edges_point_adj_inter, num_nodes=num_nodes)
    is_isolated_point = is_isolated[super_index]

    # Combine the point indices into a point mask
    mask = is_isolated_point
    mask[idx_edge_point] = True
    mask = torch.where(mask)[0]

    # Sample points among the clusters. These will be used to compute
    # cluster adjacency graph and edge features. Note we sample more
    # generously here than for cluster features, because we need to
    # capture fine-grained adjacency
    idx_samples, ptr_samples = nag.get_sampling(
        high=i_level, low=0, n_max=n_max_edge, n_min=n_min, mask=mask,
        return_pointers=True)

    # To debug sampling
    if src.is_debug_enabled():
        data.edge_idx_samples = idx_samples

        end = ptr_samples[1:]
        start = ptr_samples[:-1]
        super_index_samples = torch.arange(
            num_nodes, device=device).repeat_interleave(end - start)

        print('\n* Sampling for superedge features')
        print_tensor_info(idx_samples, name='idx_samples')
        print_tensor_info(ptr_samples, name='ptr_samples')
        print(f'all clusters have a ptr:                   '
              f'{ptr_samples.shape[0] - 1 == num_nodes}')
        print(f'all clusters received n_min+ samples:      '
              f'{(end - start).ge(n_min).all()}')
        print(f'clusters which received no sample:         '
              f'{torch.where(end == start)[0].shape[0]}/{num_nodes}')
        print(f'all points belong to the correct clusters: '
              f'{torch.equal(super_index[idx_samples], super_index_samples)}')

    # Delaunay triangulation on the sampled points. The tetrahedra edges
    # are voronoi graph edges. This is the bottleneck of this function,
    # may be worth investigating alternatives if speedups are needed
    pos = nag[0].pos[idx_samples]
    tri = Delaunay(pos.cpu().numpy())

    # Concatenate all edges of the triangulation
    pairs = torch.tensor(
        list(itertools.combinations(range(4), 2)), device=device,
        dtype=torch.long)
    edges_point = torch.from_numpy(np.hstack([
        np.vstack((tri.simplices[:, i], tri.simplices[:, j]))
        for i, j in pairs])).long().to(device)
    edges_point = idx_samples[edges_point]

    # Remove duplicate edges. For now, (i,j) and (j,i) are considered
    # to be duplicates. We remove duplicate point-wise graph edges at
    # this point to mitigate memory use. The symmetric edges and edge
    # features will be created at the very end
    edges_point = to_trimmed(edges_point)

    # Now we are only interested in the edges connecting two different
    # clusters and not in the intra-cluster connections. Select only
    # inter-cluster edges and compute the corresponding source and
    # target point and cluster indices
    se, se_id, edges_point, _ = edge_to_superedge(edges_point, super_index)

    # Remove edges whose distance is too large. We pay articular
    # attention here to avoid isolating nodes by distance filtering. If
    # a node was isolated by max_dist filtering, we preserve its
    # shortest edge to avoid it, even if it is larger than max_dist
    if max_dist > 0:
        # Identify the edges that are too long
        dist = (nag[0].pos[edges_point[1]]
                - nag[0].pos[edges_point[0]]).norm(dim=1)
        too_far = dist > max_dist

        # Recover the corresponding cluster indices for each edge
        edges_super = super_index[edges_point]

        # Identify the clusters which would be isolated if all edges
        # beyond max_dist were removed
        potential_isolated = isolated_nodes(
            edges_super[:, ~too_far], num_nodes=num_nodes)

        # For those clusters, we will tolerate 1 edge larger than
        # max_dist and that connects to another cluster
        source_isolated = potential_isolated[edges_super[0]]
        target_isolated = potential_isolated[edges_super[1]]
        tricky_edge = too_far & (source_isolated | target_isolated) \
                      & (edges_super[0] != edges_super[1])

        # Sort tricky edges by distance in descending order and sort the
        # edge indices and cluster indices consequently. By populating a
        # 'shortest edge index' tensor for the clusters using the sorted
        # edge indices, we can ensure the last edge is the shortest.
        order = dist[tricky_edge].sort(descending=True).indices
        idx = edges_super[:, tricky_edge][:, order]
        val = torch.where(tricky_edge)[0][order]
        cluster_shortest_edge = -torch.ones(
            num_nodes, dtype=torch.long, device=device)
        cluster_shortest_edge[idx[0]] = val
        cluster_shortest_edge[idx[1]] = val
        idx_edge_to_keep = cluster_shortest_edge[potential_isolated]

        # Update the too-far mask so as to preserve at least one edge
        # for each cluster
        too_far[idx_edge_to_keep] = False
        edges_point = edges_point[:, ~too_far]

        # Since this filtering might have affected edges_point, we
        # recompute the super edges indices and ids
        se, se_id, edges_point, _ = edge_to_superedge(edges_point, super_index)

        del dist

    # Prepare data attributes before computing edge features
    data.edge_index = se
    data.is_artificial = is_isolated

    # Edge feature computation. NB: operates on trimmed graphs only.
    # Features for all undirected edges can be computed later using
    # `_on_the_fly_horizontal_edge_features()`
    data = _minimalistic_horizontal_edge_features(
        data, nag[0].pos, edges_point, se_id, keys=keys)

    # Restore the i_level Data object, if need be
    nag._list[i_level] = data

    return nag


class RadiusHorizontalGraph(Transform):
    """Compute horizontal edges for all NAG levels except its first
    (ie the 0-level). These are the edges connecting the segments at
    each level, equipped with handcrafted edge features.

    This approach relies on a fast heuristics to search neighboring
    segments as well as to identify level-0 points making up the
    'subedges' between the segments.

    By default, a series of handcrafted edge attributes are computed and
    stored in the corresponding `Data.edge_attr`.

    The supported feature keys are the following:
      - mean_off
      - std_off
      - mean_dist

    :param k_max: int, List(int)
        Maximum number of neighbors per segment
    :param gap: float, List(float)
        Two segments A and B are considered neighbors if there is a in A
        and b in B such that dist(a, b) < gap
    :param se_ratio: float
        Maximum ratio of a segment's points than can be used in a
        superedge's subedges
    :param se_min: int
        Minimum of subedges per superedge
    :param cycles: int
        Number of iterations for nearest neighbor search between
        segments
    :param margin: float
        Tolerance margin used for selecting subedges points and
        excluding segment points from potential subedge candidates
    :param chunk_size: int, float
        Allows mitigating memory use. If `chunk_size > 1`,
        `edge_index` will be processed into chunks of `chunk_size`. If
        `0 < chunk_size < 1`, then `edge_index` will be divided into
        parts of `edge_index.shape[1] * chunk_size` or less
    :param halfspace_filter: bool
        Whether the halfspace filtering should be applied
    :param bbox_filter: bool
        Whether the bounding box filtering should be applied
    :param target_pc_flip: bool
        Whether the subedge point pairs should be carefully ordered
    :param source_pc_sort: bool
        Whether the source and target subedge point pairs should be
        ordered along the same vector
    :param keys: List(str)
        Features to be computed. Attributes will be saved under `<key>`
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _NO_REPR = ['chunk_size']

    def __init__(
            self,
            k_min=1,
            k_max=100,
            gap=0,
            se_ratio=0.2,
            se_min=20,
            cycles=3,
            margin=0.2,
            chunk_size=100000,
            halfspace_filter=True,
            bbox_filter=True,
            target_pc_flip=True,
            source_pc_sort=False,
            keys=None):

        if isinstance(k_min, list):
            assert all([k > 0 for k in k_min]), \
                "k_min must be 1 or more, to avoid any unpleasant downstream " \
                "issues where nodes have no edge"
        else:
            assert k_min > 0, \
                "k_min must be 1 or more, to avoid any unpleasant downstream " \
                "issues where nodes have no edge"

        self.k_max = k_max
        self.k_min = k_min
        self.gap = gap
        self.se_ratio = se_ratio
        self.se_min = se_min
        self.cycles = cycles
        self.margin = margin
        self.chunk_size = chunk_size
        self.halfspace_filter = halfspace_filter
        self.bbox_filter = bbox_filter
        self.target_pc_flip = target_pc_flip
        self.source_pc_sort = source_pc_sort
        self.keys = sanitize_keys(keys, default=SUBEDGE_FEATURES)

    def _process(self, nag):
        # Convert parameters to list for each NAG level, if need be
        se_ratio = self.se_ratio if isinstance(self.se_ratio, list) \
            else [self.se_ratio] * (nag.num_levels - 1)
        se_min = self.se_min if isinstance(self.se_min, list) \
            else [self.se_min] * (nag.num_levels - 1)
        cycles = self.cycles if isinstance(self.cycles, list) \
            else [self.cycles] * (nag.num_levels - 1)
        margin = self.margin if isinstance(self.margin, list) \
            else [self.margin] * (nag.num_levels - 1)
        chunk_size = self.chunk_size if isinstance(self.chunk_size, list) \
            else [self.chunk_size] * (nag.num_levels - 1)

        # Compute the horizontal graph, without edge features
        nag = _horizontal_graph_by_radius(
            nag, k_min=self.k_min, k_max=self.k_max, gap=self.gap, trim=True,
            cycles=cycles, chunk_size=chunk_size)

        # Compute the edge features, level by level
        for i_level, ser, sem, cy, mg, cs in zip(
                range(1, nag.num_levels), se_ratio, se_min, cycles, margin,
                chunk_size):
            nag = self._process_edge_features_for_single_level(
                nag, i_level, ser, sem, cy, mg, cs)

        return nag

    def _process_edge_features_for_single_level(
            self, nag, i_level, se_ratio, se_min, cycles, margin, chunk_size):
        # Compute 'subedges', ie edges between level-0 points making up
        # the edges between the segments. These will be used for edge
        # features computation. NB: this operation simplifies the
        # edge_index graph into a trimmed graph. To restore
        # the bidirectional edges, we will need to reconstruct the j<i
        # edges later on (done in `_horizontal_edge_features`)
        edge_index, se_point_index, se_id = subedges(
            nag[0].pos,
            nag.get_super_index(i_level),
            nag[i_level].edge_index,
            ratio=se_ratio,
            k_min=se_min,
            cycles=cycles,
            pca_on_cpu=True,
            margin=margin,
            halfspace_filter=self.halfspace_filter,
            bbox_filter=self.bbox_filter,
            target_pc_flip=self.target_pc_flip,
            source_pc_sort=self.source_pc_sort,
            chunk_size=chunk_size)

        # Prepare for edge feature computation
        data = nag[i_level]
        data.edge_index = edge_index

        # Edge feature computation. NB: operates on trimmed graph only
        # to alleviate memory and compute. Features for all undirected
        # edges can be computed later using
        # `_on_the_fly_horizontal_edge_features()`
        data = _minimalistic_horizontal_edge_features(
            data, nag[0].pos, se_point_index, se_id, keys=self.keys)

        # Restore the i_level Data object
        nag._list[i_level] = data

        return nag


def _horizontal_graph_by_radius(
        nag,
        k_min=1,
        k_max=100,
        gap=0,
        trim=True,
        cycles=3,
        chunk_size=None):
    """Search neighboring segments with points distant from `gap`or
    less.

    :param nag: NAG
        Hierarchical structure
    :param k_min: int, List(int)
        Minimum number of neighbors per segment
    :param k_max: int, List(int)
        Maximum number of neighbors per segment
    :param gap: float, List(float)
        Two segments A and B are considered neighbors if there is a in A
        and b in B such that dist(a, b) < gap
    :param trim: bool
        Whether the returned horizontal graph should be trimmed. If
        True, `to_trimmed()` will be called and all edges will be
        expressed with source_index < target_index, self-loops and
        redundant edges will be removed. This may be necessary to
        alleviate memory consumption before computing edge features
    :param cycles int
        Number of iterations. Starting from a point X in set A, one
        cycle accounts for searching the nearest neighbor, in A, of the
        nearest neighbor of X in set B
    :param chunk_size: int, float
        Allows mitigating memory use when computing the subedges. If
        `chunk_size > 1`, `edge_index` will be processed into chunks of
        `chunk_size`. If `0 < chunk_size < 1`, then `edge_index` will be
        divided into parts of `edge_index.shape[1] * chunk_size` or less
    :return:
    """
    assert isinstance(nag, NAG)
    if not isinstance(k_max, list):
        k_max = [k_max] * (nag.num_levels - 1)
    if not isinstance(k_min, list):
        k_min = [k_min] * (nag.num_levels - 1)
    if not isinstance(gap, list):
        gap = [gap] * (nag.num_levels - 1)
    if not isinstance(cycles, list):
        cycles = [cycles] * (nag.num_levels - 1)
    if not isinstance(chunk_size, list):
        chunk_size = [chunk_size] * (nag.num_levels - 1)

    for i_level, k_lo, k_hi, g, cy, cs in zip(
            range(1, nag.num_levels), k_min, k_max, gap, cycles, chunk_size):
        nag = _horizontal_graph_by_radius_for_single_level(
            nag, i_level, k_min=k_lo, k_max=k_hi, gap=g, trim=trim,
            cycles=cy, chunk_size=cs)

    return nag


def _horizontal_graph_by_radius_for_single_level(
        nag,
        i_level,
        k_min=1,
        k_max=100,
        gap=0,
        trim=True,
        cycles=3,
        chunk_size=100000):
    """

    :param nag:
    :param i_level:
    :param k_min:
    :param k_max:
    :param gap:
    :param trim:
    :param cycles:
    :param chunk_size:
    :return:
    """
    assert isinstance(nag, NAG)
    assert i_level > 0, "Cannot compute cluster graph on level 0"
    assert nag[0].num_nodes < np.iinfo(np.uint32).max, \
        "Too many nodes for `uint32` indices"
    assert nag[0].num_edges < np.iinfo(np.uint32).max, \
        "Too many edges for `uint32` indices"

    # Recover the i_level Data object we will be working on
    data = nag[i_level]
    num_nodes = data.num_nodes

    # Remove any already-existing horizontal graph
    data.edge_index = None
    data.edge_attr = None

    # Exit in case the i_level graph contains only one node
    if num_nodes < 2:
        raise ValueError(
            f"Input NAG only has 1 node at level={i_level}. Cannot compute "
            f"radius-based horizontal graph.")

    # Compute the super_index for level-0 points wrt the target level
    super_index = nag.get_super_index(i_level)

    # Search neighboring clusters
    data.raise_if_edge_keys()
    edge_index, distances = cluster_radius_nn(
        nag[0].pos, super_index, k_max=k_max, gap=gap, trim=trim,
        cycles=cycles, chunk_size=chunk_size)

    # Save the graph in the Data object
    data.edge_index = edge_index
    data.edge_attr = None

    # Search for nodes which received no edges and connect them to their
    # k_min nearest neighbor
    data.connect_isolated(k=k_min)

    # Trim the graph. This is temporary, to alleviate edge features
    # computation
    if trim:
        data.to_trimmed(reduce='min')

    # Store the updated Data object in the NAG
    nag._list[i_level] = data

    return nag


def _minimalistic_horizontal_edge_features(
        data, points, se_point_index, se_id, keys=None):
    """Compute the features for horizontal edges, given the edge graph
    and the level-0 'subedges' making up each edge.

    The features computed here are partly based on:
    https://github.com/loicland/superpoint_graph

    :param data:
    :param points:
    :param se_point_index:
    :param se_id:
    :param keys:
    """

    keys = sanitize_keys(keys, default=SUBEDGE_FEATURES)

    # Recover the edges between the segments
    se = data.edge_index

    assert is_trimmed(se), \
        "Expects the graph to be trimmed, consider using " \
        "`src.utils.to_trimmed()` before computing the features"

    if not all(['mean_off' in keys, 'std_off' in keys, 'mean_dist' in keys]):
        raise NotImplementedError(
            "For now, 'mean_off', 'std_off' and 'mean_dist' must all be "
            "computed, since we must store them all into 'edge_attr'. Things"
            "will be different once we support custom 'edge_<key>' everywhere,"
            "but not for now.")

    # Direction are the pointwise source->target vectors, based on which
    # we will compute superedge descriptors
    offset = points[se_point_index[1]] - points[se_point_index[0]]

    # To stabilize the distance-based features' distribution, we use the
    # sqrt of the metric distance. This assumes coordinates are in meter
    # and that we are mostly interested in the range [1, 100]. Might
    # want to change this if your dataset is different
    dist = offset.norm(dim=1)

    # Compute mean subedge direction
    se_mean_off = scatter_mean(offset, se_id, dim=0)

    # Compute std of the offset, in a base built around the mean offset
    base = base_vectors_3d(se_mean_off)[se_id]
    u = (offset * base[:, 0]).sum(dim=1).view(-1, 1)
    v = (offset * base[:, 0]).sum(dim=1).view(-1, 1)
    w = (offset * base[:, 0]).sum(dim=1).view(-1, 1)
    se_std_off = scatter_std(torch.cat((u, v, w), dim=1), se_id, dim=0)
    se_std_off = se_std_off.clip(-2, 2)

    # Compute mean subedge distance
    se_mean_dist = scatter_mean(dist, se_id, dim=0).sqrt()

    # Save superedges and superedge features in the Data object
    f = []
    if 'mean_off' in keys:
        f.append(se_mean_off)
    if 'std_off' in keys:
        f.append(se_std_off)
    if 'mean_dist' in keys:
        f.append(se_mean_dist.view(-1, 1))
    data.edge_index = se
    data.edge_attr = torch.cat(f, dim=1)

    return data


class OnTheFlyHorizontalEdgeFeatures(Transform):
    """Compute edge features "on-the-fly" for all i->j and j->i
    horizontal edges of the NAG levels except its first (ie the
    0-level).

    Expects only trimmed edges as input, along with some edge-specific
    attributes that cannot be recovered from the corresponding source
    and target node attributes (see `src.utils.to_trimmed`).

    Accepts input edge_attr to be float16, to alleviate memory use and
    accelerate data loading and transforms. Output edge_<key> will,
    however, be in float32.

    Optionally adds some edge features that can be recovered from the
    source and target node attributes.

    Builds the j->i edges and corresponding features based on their i->j
    counterparts in the trimmed graph.

    Equips the output NAG with all i->j and j->i nodes and corresponding
    features.

    Note: this transform is intended to be called after all sampling
    transforms, to mitigate compute and memory impact of horizontal
    edges.

    The supported feature keys are the following:
      - mean_off: mean offset (subedges)
      - std_off: std offset (subedges)
      - mean_dist: mean offset (subedges) distance
      - angle_source: cosine of the angle between the mean offset
        (subedges) and the source normal
      - angle_target: cosine of the angle between the mean offset
        (subedges) and the target normal
      - centroid_dir: unit-normalized direction between the i and
        j centroids
      - centroid_dist: distance between the i and j centroids
      - normal_angle: cosine of the angle between the i and j
        normals
      - log_length: i/j log length ratio
      - log_surface: i/j log surface ratio
      - log_volume: i/j log volume ratio
      - log_size: i/j log size ratio

    :param keys: List(str)
        Features to be computed. Attributes will be saved under `<key>`
    :param use_mean_normal: bool
        Whether the 'normal' or the 'mean_normal' segment attribute
        should be used for computing normal-related edge features
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(
            self, keys=None, use_mean_normal=False):
        self.keys = sanitize_keys(keys, default=ON_THE_FLY_HORIZONTAL_FEATURES)
        self.use_mean_normal = use_mean_normal

    def _process(self, nag):
        for i_level in range(1, nag.num_levels):
            nag._list[i_level] = _on_the_fly_horizontal_edge_features(
                nag[i_level],
                keys=self.keys,
                use_mean_normal=self.use_mean_normal)
        return nag


def _on_the_fly_horizontal_edge_features(
        data, keys=None, use_mean_normal=False):
    """Compute all edges and edge features for a horizontal graph, given
    a trimmed graph and some precomputed edge attributes.
    """
    keys = sanitize_keys(keys, default=ON_THE_FLY_HORIZONTAL_FEATURES)

    # Recover the edges between the segments
    se = data.edge_index

    data.raise_if_edge_keys()

    normal_key = 'mean_normal' if use_mean_normal else 'normal'

    assert is_trimmed(se), \
        "Expects the graph to be trimmed, consider using " \
        "`src.utils.to_trimmed()` before computing the features"
    if 'mean_off' in keys:
        assert getattr(data, 'edge_attr', None) is not None, \
            "Expected input Data to have a 'edge_attr' attribute precomputed " \
            "using `_minimalistic_horizontal_edge_features`"
    if 'std_off' in keys:
        assert getattr(data, 'edge_attr', None) is not None, \
            "Expected input Data to have a 'edge_attr' attribute precomputed " \
            "using `_minimalistic_horizontal_edge_features`"
    if 'mean_dist' in keys:
        assert getattr(data, 'edge_attr', None) is not None, \
            "Expected input Data to have a 'edge_attr' attribute precomputed " \
            "using `_minimalistic_horizontal_edge_features`"
    if 'angle_source' in keys or 'angle_target' in keys:
        assert getattr(data, normal_key, None) is not None and \
               getattr(data, 'edge_attr', None) is not None, \
            f"Expected input Data to have a '{normal_key}' " \
            "attribute and an 'edge_attr' attribute precomputed using " \
            "`_minimalistic_horizontal_edge_features`"
    if 'normal_angle' in keys:
        assert getattr(data, normal_key, None) is not None, \
            f"Expected input Data to have a '{normal_key}'"
    if 'log_length' in keys:
        assert getattr(data, 'log_length', None) is not None, \
            "Expected input Data to have a 'log_length' attribute"
    if 'log_surface' in keys:
        assert getattr(data, 'log_surface', None) is not None, \
            "Expected input Data to have a 'log_surface' attribute"
    if 'log_volume' in keys:
        assert getattr(data, 'log_volume', None) is not None, \
            "Expected input Data to have a 'log_volume' attribute"
    if 'log_size' in keys:
        assert getattr(data, 'log_size', None) is not None, \
            "Expected input Data to have a 'log_size' attribute"

    f_list = []

    if 'std_off' in keys:
        # Precomputed edge features might be expressed in float16, so we
        # convert them to float32 here
        f = data.edge_attr[:, 3:6].float()
        f_list.append(torch.cat((f, f), dim=0))

    if 'mean_dist' in keys:
        # Precomputed edge features might be expressed in float16, so we
        # convert them to float32 here
        f = data.edge_attr[:, 6].float().view(-1, 1)
        f_list.append(torch.cat((f, f), dim=0))

    if 'mean_off' in keys or 'angle_source' in keys or 'angle_target' in keys:
        # Precomputed edge features might be expressed in float16, so we
        # convert them to float32 here
        se_mean_off = data.edge_attr[:, :3].float()

        # Compute the mean subedge (normalized) direction
        se_direction = se_mean_off / se_mean_off.norm(dim=1).view(-1, 1)

        # Sanity checks on normalized directions
        se_direction[se_direction.isnan()] = 0
        se_direction = se_direction.clip(-1, 1)

        if 'mean_off' in keys:
            # We place mean_off in the first 3 edge_attr columns, for
            # homogeneity with input edge_attr from
            # _minimalistic_horizontal_edge_features
            f_list = [torch.cat((se_mean_off, -se_mean_off), dim=0)] + f_list

        if 'angle_source' in keys:
            normal = getattr(data, normal_key, None)
            f = (se_direction * normal[se[0]]).sum(dim=1).abs()
            f_list.append(torch.cat((f, f), dim=0).view(-1, 1))

        if 'angle_target' in keys:
            normal = getattr(data, normal_key, None)
            f = (se_direction * normal[se[1]]).sum(dim=1).abs()
            f_list.append(torch.cat((f, f), dim=0).view(-1, 1))

    if 'normal_angle' in keys:
        normal = getattr(data, normal_key, None)
        f = (normal[se[0]] * normal[se[1]]).sum(dim=1).abs()
        f_list.append(torch.cat((f, f), dim=0).view(-1, 1))

    if 'log_length' in keys:
        f = data.log_length[se[0]] - data.log_length[se[1]]
        f_list.append(torch.cat((f, -f), dim=0).view(-1, 1))

    if 'log_surface' in keys:
        f = data.log_surface[se[0]] - data.log_surface[se[1]]
        f_list.append(torch.cat((f, -f), dim=0).view(-1, 1))

    if 'log_volume' in keys:
        f = data.log_volume[se[0]] - data.log_volume[se[1]]
        f_list.append(torch.cat((f, -f), dim=0).view(-1, 1))

    if 'log_size' in keys:
        f = data.log_size[se[0]] - data.log_size[se[1]]
        f_list.append(torch.cat((f, -f), dim=0).view(-1, 1))

    if 'centroid_dir' in keys or 'centroid_dist' in keys:
        # Compute the distance and direction between the segments'
        # centroids
        se_centroid_dir = data.pos[se[1]] - data.pos[se[0]]
        se_centroid_dist = se_centroid_dir.norm(dim=1).view(-1, 1)
        se_centroid_dir /= se_centroid_dist.view(-1, 1)
        se_centroid_dist = se_centroid_dist.sqrt()

        # Sanity checks on normalized directions
        se_centroid_dir[se_centroid_dir.isnan()] = 0
        se_centroid_dir = se_centroid_dir.clip(-1, 1)

        if 'centroid_dir' in keys:
            f_list.append(torch.cat((se_centroid_dir, -se_centroid_dir), dim=0))

        if 'centroid_dist' in keys:
            f_list.append(torch.cat((se_centroid_dist, se_centroid_dist), dim=0))

    # Update the edge_index with j->i edges
    data.edge_index = torch.cat((se, se.flip(0)), dim=1)

    # Update all edge features into edge_attr and remove all other
    # edge_<key> to save memory
    for k in ['edge_attr'] + data.edge_keys:
        data[k] = None
    if len(f_list) > 0:
        data.edge_attr = torch.cat(f_list, dim=1)

    return data


class OnTheFlyVerticalEdgeFeatures(Transform):
    """Compute edge features "on-the-fly" for all vertical edges of the
    NAG levels.

    Optionally build some edge features that can be recovered from the
    source and target node attributes.

    Note: this transform is intended to be called after all sampling
    transforms, to mitigate compute and memory impact of vertical
    edges.

    The supported feature keys are the following:
      - centroid_dir: unit-normalized direction between the child
        centroid and the parent centroid
      - centroid_dist: distance between the child and parent centroids
      - normal_angle: cosine of the angle between the child and parent
        normals
      - log_length: parent/child log length ratio
      - log_surface: parent/child log surface ratio
      - log_volume: parent/child log volume ratio
      - log_size: parent/child log size ratio

    :param keys: List(str)
        Features to be computed. Attributes will be saved under `<key>`
    :param use_mean_normal: bool
        Whether the 'normal' or the 'mean_normal' segment attribute
        should be used for computing normal-related edge features
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, keys=None, use_mean_normal=False):
        self.keys = sanitize_keys(keys, default=ON_THE_FLY_VERTICAL_FEATURES)
        self.use_mean_normal = use_mean_normal

    def _process(self, nag):
        for i_level in range(1, nag.num_levels):
            data_child = nag[i_level - 1]
            data_parent = nag[i_level]

            # For level-0 points, we artificially set 'mean_normal' from
            # 'normal', if need be
            if self.use_mean_normal and i_level == 1:
                if getattr(data_child, 'mean_normal', None):
                    data_child.mean_normal = getattr(data_child, 'normal', None)

            nag._list[i_level - 1] = _on_the_fly_vertical_edge_features(
                data_child,
                data_parent,
                keys=self.keys,
                use_mean_normal=self.use_mean_normal)
        return nag


def _on_the_fly_vertical_edge_features(
        data_child, data_parent, keys=None, use_mean_normal=False):
    """Compute edge features for a vertical graph, given child and
    parent nodes.
    """
    keys = sanitize_keys(keys, default=ON_THE_FLY_VERTICAL_FEATURES)

    if len(keys) == 0:
        return data_child

    normal_key = 'mean_normal' if use_mean_normal else 'normal'

    # Recover the parent index of each child node
    idx = data_child.super_index
    assert idx is not None, \
        "Expected input child Data to have a 'super_index' attribute"

    for d in [data_child, data_parent]:
        if 'normal_angle' in keys:
            assert getattr(d, normal_key, None) is not None, \
                f"Expected input Data to have a '{normal_key}' attribute"
        if 'log_length' in keys:
            assert getattr(d, 'log_length', None) is not None, \
                "Expected input Data to have a 'log_length' attribute"
        if 'log_surface' in keys:
            assert getattr(d, 'log_surface', None) is not None, \
                "Expected input Data to have a 'log_surface' attribute"
        if 'log_volume' in keys:
            assert getattr(d, 'log_volume', None) is not None, \
                "Expected input Data to have a 'log_volume' attribute"
        if 'log_size' in keys:
            assert getattr(d, 'log_size', None) is not None, \
                "Expected input Data to have a 'log_size' attribute"

    f_list = []

    if 'centroid_dir' in keys or 'centroid_dist' in keys:
        # Compute the distance and direction between the child and
        # parent segments' centroids
        ve_centroid_dir = data_parent.pos[idx] - data_child.pos
        ve_centroid_dist = ve_centroid_dir.norm(dim=1)
        ve_centroid_dir /= ve_centroid_dist.view(-1, 1)
        ve_centroid_dist = ve_centroid_dist.sqrt()

        # Sanity checks on normalized directions
        ve_centroid_dir[ve_centroid_dir.isnan()] = 0
        ve_centroid_dir = ve_centroid_dir.clip(-1, 1)

        if 'centroid_dir' in keys:
            f_list.append(ve_centroid_dir)

        if 'centroid_dist' in keys:
            f_list.append(ve_centroid_dist.view(-1, 1))

    if 'normal_angle' in keys:
        child_normal = getattr(data_child, normal_key, None)
        parent_normal = getattr(data_parent, normal_key, None)
        f = (child_normal * parent_normal[idx]).sum(dim=1).abs()
        f_list.append(f.view(-1, 1))

    if 'log_length' in keys:
        f = data_parent.log_length[idx] - data_child.log_length
        f_list.append(f.view(-1, 1))

    if 'log_surface' in keys:
        f = data_parent.log_surface[idx] - data_child.log_surface
        f_list.append(f.view(-1, 1))

    if 'log_volume' in keys:
        f = data_parent.log_volume[idx] - data_child.log_volume
        f_list.append(f.view(-1, 1))

    if 'log_size' in keys:
        f = data_parent.log_size[idx] - data_child.log_size
        f_list.append(f.view(-1, 1))

    # Stack all the vertical edge features into the child 'v_edge_attr'
    data_child.v_edge_attr = None
    if len(f_list) > 0:
        data_child.v_edge_attr = torch.cat(f_list, dim=1)

    return data_child


class NAGAddSelfLoops(Transform):
    """Add self-loops to all NAG levels having a horizontal graph. If
    the edges have attributes, the self-loops will receive 0-features.
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def _process(self, nag):
        for i_level in range(1, nag.num_levels):

            # Skip if the level has no horizontal graph
            if not nag[i_level].has_edges:
                continue

            # Recover edges and attributes
            num_nodes = nag[i_level].num_nodes
            edge_index = nag[i_level].edge_index
            edge_attr = nag[i_level].edge_attr

            nag[i_level].raise_if_edge_keys()

            # Add self-loops
            edge_index, edge_attr = add_self_loops(
                edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                fill_value=0)

            # Update the edges and attributes
            nag[i_level].edge_index = edge_index
            nag[i_level].edge_attr = edge_attr

        return nag


class ConnectIsolated(Transform):
    """Creates edges for isolated nodes. Each isolated node is connected
    to the `k` nearest nodes. If the Data graph contains edge features
    in `Data.edge_attr`, the new edges will receive features based on
    their length and a linear regression of the relation between
    existing edge features and their corresponding edge length.

    NB: this is an inplace operation that will modify the input data.

    :param k: int
        Number of neighbors the isolated nodes should be connected to
    """

    def __init__(self, k=1):
        self.k = k

    def _process(self, data):
        return data.connect_isolated(k=self.k)


class NodeSize(Transform):
    """Compute the number of `low`-level elements are contained in each
    segment, at each above-level. Results are save in the `node_size`
    attribute of the corresponding Data objects.

    Note: `low=-1` is accepted when level-0 has a `sub` attribute
    (ie level-0 points are themselves segments of `-1` level absent
    from the NAG object).

    :param low: int
        Level whose elements we want to count
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, low=0):
        assert isinstance(low, int) and low >= -1
        self.low = low

    def _process(self, nag):
        for i_level in range(self.low + 1, nag.num_levels):
            nag[i_level].node_size = nag.get_sub_size(i_level, low=self.low)
        return nag
