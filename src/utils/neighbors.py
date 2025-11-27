import torch
from time import time

import src
from src.dependencies.FRNN import frnn
from torch_scatter import scatter
from torch_geometric.utils import coalesce
from src.utils.scatter import scatter_nearest_neighbor
from src.utils.sparse import sizes_to_pointers


__all__ = [
    'knn_1',
    'knn_1_graph',
    'knn_2',
    'knn_brute_force',
    'inliers_split',
    'outliers_split',
    'inliers_outliers_splits',
    'cluster_radius_nn_graph',
    'neighbors_dense_to_csr']


def _frnn_grid_points(xyz_query, xyz_search, K=1, r=1):
    """A wrapper around frnn.frnn_grid_points to facilitate the handling
    ot K and r input formats.
    """
    device = xyz_query.device

    if K > 1000:
        print(
            f"Warning: FRNN is not optimal for for K>1000. "
            f"You may want to consider using brute force search with "
            f"`knn_brute_force()` if your query set is not too large.")

    # Make sure k and r are 1D tensors on the same device
    if isinstance(r, torch.Tensor):
        r = r.view(-1).to(device)
    else:
        r = torch.tensor([r], device=device)

    if isinstance(K, torch.Tensor):
        K = K.view(-1).to(device)
    else:
        K = torch.tensor([K], device=device)

    # Run the actual FRNN search
    return frnn.frnn_grid_points(xyz_query, xyz_search, K=K, r=r)


def knn_1(
        xyz,
        k,
        r_max=1,
        batch=None,
        oversample=False,
        self_is_neighbor=False,
        verbose=False):
    """Search k-NN for a 3D point cloud xyz. This search differs
    from `knn_2` in that it operates on a single cloud input (search and
    query are the same) and it allows oversampling the neighbors when
    less than `k` neighbors are found within `r_max`. Optionally,
    passing `batch` will ensure the neighbor search does not mix up
    batch items.
    """
    assert isinstance(xyz, torch.Tensor)
    assert k >= 1
    assert xyz.dim() == 2
    assert batch is None or batch.shape[0] == xyz.shape[0]

    # To take the batch into account, we add an offset to the Z
    # coordinates. The offset is designed so that any points from two
    # batch different batch items are separated by at least `r_max + 1`
    batch_offset = 0
    if batch is not None:
        z_offset = xyz[:, 2].max() - xyz[:, 2].min() + r_max + 1
        batch_offset = torch.zeros_like(xyz)
        batch_offset[:, 2] = batch * z_offset

    # Data initialization
    device = xyz.device
    xyz_query = (xyz + batch_offset).view(1, -1, 3)
    xyz_search = (xyz + batch_offset).view(1, -1, 3)
    if not xyz.is_cuda:
        xyz_query = xyz_query.cuda()
        xyz_search = xyz_search.cuda()

    # KNN on GPU. Actual neighbor search now
    k_search = k if self_is_neighbor else k + 1
    distances, neighbors, _, _ = _frnn_grid_points(
        xyz_query, xyz_search, K=k_search, r=r_max)

    # Remove each point from its own neighborhood
    neighbors = neighbors[0] if self_is_neighbor else neighbors[0][:, 1:]
    distances = distances[0] if self_is_neighbor else distances[0][:, 1:]

    # Oversample the neighborhoods where less than k points were found
    if oversample:
        neighbors, distances = oversample_partial_neighborhoods(
            neighbors, distances, k)

    # Restore the neighbors and distances to the input device
    if neighbors.device != device:
        neighbors = neighbors.to(device)
        distances = distances.to(device)

    if not verbose and not src.is_debug_enabled():
        return neighbors, distances

    # Warn the user of partial and empty neighborhoods
    num_nodes = neighbors.shape[0]
    n_missing = (neighbors < 0).sum(dim=1)
    n_partial = (n_missing > 0).sum()
    n_empty = (n_missing == k).sum()
    if n_partial == 0:
        return neighbors, distances

    print(
        f"\nWarning: {n_partial}/{num_nodes} points have partial "
        f"neighborhoods and {n_empty}/{num_nodes} have empty "
        f"neighborhoods (missing neighbors are indicated by -1 indices).")

    return neighbors, distances


def knn_1_graph(
        xyz,
        k,
        r_max=1,
        batch=None,
        oversample=False,
        self_is_neighbor=False,
        verbose=False,
        trim=True):
    """Search k-NN for a 3D point cloud xyz and convert the output into
    torch_geometric's `edge_index`, `edge_attr` format. This search
    differs from `knn_2` in that it operates on a single cloud input
    (search and query are the same) and it allows oversampling the
    neighbors when less than `k` neighbors are found within `r_max`.
    Optionally, passing `batch` will ensure the neighbor search does not
    mix up batch items.

    Importantly, the output graph will be coalesced: duplicate edges
    will be removed. Besides, if `trim=True`, the graph will be further
    reduced using `to_trimmed()` (see function documentation for more
    information).
    """
    # Nearest neighbor search
    neighbors, distances = knn_1(
        xyz,
        k,
        r_max=r_max,
        batch=batch,
        oversample=oversample,
        self_is_neighbor=self_is_neighbor,
        verbose=verbose)

    # Build the corresponding graph
    num_points = xyz.shape[0]
    source = torch.arange(num_points, device=xyz.device).repeat_interleave(k)
    target = neighbors.flatten()
    edge_index = torch.vstack((source, target))
    distances = distances.flatten()

    # Trim edges where points are missing (i.e. -1 neighbor indices)
    missing_point_edge = edge_index[1] == -1
    edge_index = edge_index[:, ~missing_point_edge]
    distances = distances[~missing_point_edge]

    # Trim the graph. This is required before computing the actual
    # nearest points between all cluster pairs. Since this operation is
    # so costly, we first built on a coarse neighborhood edge_index to
    # alleviate compute and memory cost
    if trim:
        from src.utils import to_trimmed
        edge_index, distances = to_trimmed(
            edge_index, edge_attr=distances, reduce='min')
    # Coalesce edges to remove duplicates
    else:
        edge_index, distances = coalesce(
            edge_index, edge_attr=distances, reduce='min')

    return edge_index, distances


def knn_2(
        x_search,
        x_query,
        k,
        r_max=1,
        batch_search=None,
        batch_query=None):
    """Search k-NN of x_query inside x_search, within radius `r_max`.
    Optionally, passing `batch_search` and `batch_query` will ensure the
    neighbor search does not mix up batch items.
    """
    assert isinstance(x_search, torch.Tensor)
    assert isinstance(x_query, torch.Tensor)
    assert k >= 1
    assert x_search.dim() == 2
    assert x_query.dim() == 2
    assert x_query.shape[1] == x_search.shape[1]
    assert (batch_search is None) == (batch_query is None)
    assert batch_search is None or batch_search.shape[0] == x_search.shape[0]
    assert batch_query is None or batch_query.shape[0] == x_query.shape[0]

    # To take the batch into account, we add an offset to the Z
    # coordinates. The offset is designed so that any points from two
    # batch different batch items are separated by at least `r_max + 1`
    batch_search_offset = 0
    batch_query_offset = 0
    if batch_search is not None:
        hi = max(x_search[:, 2].max(), x_query[:, 2].max())
        lo = min(x_search[:, 2].min(), x_query[:, 2].min())
        z_offset = hi - lo + r_max + 1
        batch_search_offset = torch.zeros_like(x_search)
        batch_search_offset[:, 2] = batch_search * z_offset
        batch_query_offset = torch.zeros_like(x_query)
        batch_query_offset[:, 2] = batch_query * z_offset

    # Data initialization
    xyz_query = (x_query + batch_query_offset).view(1, -1, 3)
    xyz_search = (x_search + batch_search_offset).view(1, -1, 3)

    # Move to GPU if need be
    if not xyz_query.is_cuda:
        xyz_query = xyz_query.cuda()
        xyz_search = xyz_search.cuda()

    # KNN on GPU. Actual neighbor search now
    distances, neighbors, _, _ = _frnn_grid_points(
        xyz_query, xyz_search, K=k, r=r_max)

    # Move back to original device
    device = x_search.device
    neighbors = neighbors[0].to(device)
    distances = distances[0].to(device)
    if k == 1:
        neighbors = neighbors[:, 0]
        distances = distances[:, 0]

    return neighbors, distances


def knn_brute_force(
        x_search,
        x_query,
        k,
        r_max=1,
        batch_search=None,
        batch_query=None):
    """Search k-NN of x_query inside x_search, within radius `r_max`.
    Optionally, passing `batch_search` and `batch_query` will ensure the
    neighbor search does not mix up batch items. This is the same as
    `knn_2()` but uses a brute force method, which does not scale.
    """
    assert isinstance(x_search, torch.Tensor)
    assert isinstance(x_query, torch.Tensor)
    assert k >= 1
    assert x_search.dim() == 2
    assert x_query.dim() == 2
    assert x_query.shape[1] == x_search.shape[1]
    assert (batch_search is None) == (batch_query is None)
    assert batch_search is None or batch_search.shape[0] == x_search.shape[0]
    assert batch_query is None or batch_query.shape[0] == x_query.shape[0]

    # To take the batch into account, we add an offset to the Z
    # coordinates. The offset is designed so that any points from two
    # batch different batch items are separated by at least `r_max + 1`
    batch_search_offset = 0
    batch_query_offset = 0
    if batch_search is not None:
        hi = max(x_search[:, 2].max(), x_query[:, 2].max())
        lo = min(x_search[:, 2].min(), x_query[:, 2].min())
        z_offset = hi - lo + r_max + 1
        batch_search_offset = torch.zeros_like(x_search)
        batch_search_offset[:, 2] = batch_search * z_offset
        batch_query_offset = torch.zeros_like(x_query)
        batch_query_offset[:, 2] = batch_query * z_offset

    # Data initialization
    xyz_query = (x_query + batch_query_offset)
    xyz_search = (x_search + batch_search_offset)

    # Brute force compute all distances and neighbors for all query
    # points
    distances = (xyz_search.unsqueeze(0) - xyz_query.unsqueeze(1)).norm(dim=2)
    distances, neighbors = distances.sort(dim=1)
    distances = distances[:, :k]
    neighbors = neighbors[:, :k]
    mask = distances > r_max
    distances[mask] = -1
    neighbors[mask] = -1

    return neighbors, distances


def inliers_split(
        xyz_query,
        xyz_search,
        k_min,
        r_max=1,
        recursive=False,
        q_in_s=False):
    """Optionally recursive inlier search. The `xyz_query` and
    `xyz_search`. Search for points with less than `k_min` neighbors
    within a radius of `r_max`.

    Since removing outliers may cause some points to become outliers
    themselves, this problem can be tackled with the `recursive` option.
    Note that this recursive search holds no guarantee of reasonable
    convergence as one could design a point cloud for given `k_min` and
    `r_max` whose points would all recursively end up as outliers.
    """
    return inliers_outliers_splits(
        xyz_query,
        xyz_search,
        k_min,
        r_max=r_max,
        recursive=recursive,
        q_in_s=q_in_s)[0]


def outliers_split(
        xyz_query,
        xyz_search,
        k_min,
        r_max=1,
        recursive=False,
        q_in_s=False):
    """Optionally recursive outlier search. The `xyz_query` and
    `xyz_search`. Search for points with less than `k_min` neighbors
    within a radius of `r_max`.

    Since removing outliers may cause some points to become outliers
    themselves, this problem can be tackled with the `recursive` option.
    Note that this recursive search holds no guarantee of reasonable
    convergence as one could design a point cloud for given `k_min` and
    `r_max` whose points would all recursively end up as outliers.
    """
    return inliers_outliers_splits(
        xyz_query,
        xyz_search,
        k_min,
        r_max=r_max,
        recursive=recursive,
        q_in_s=q_in_s)[1]


def inliers_outliers_splits(
        xyz_query,
        xyz_search,
        k_min,
        r_max=1,
        recursive=False,
        q_in_s=False):
    """Optionally recursive outlier search. The `xyz_query` and
    `xyz_search`. Search for points with less than `k_min` neighbors
    within a radius of `r_max`.

    Since removing outliers may cause some points to become outliers
    themselves, this problem can be tackled with the `recursive` option.
    Note that this recursive search holds no guarantee of reasonable
    convergence as one could design a point cloud for given `k_min` and
    `r_max` whose points would all recursively end up as outliers.
    """
    # Data initialization
    device = xyz_query.device
    xyz_query = xyz_query.view(1, -1, 3).cuda()
    xyz_search = xyz_search.view(1, -1, 3).cuda()

    # KNN on GPU. Actual neighbor search now
    neighbors = _frnn_grid_points(
        xyz_query, xyz_search, K=k_min + q_in_s, r=r_max)[1]

    # If the Query points are included in the Search points, remove each
    # point from its own neighborhood
    if q_in_s:
        neighbors = neighbors[0][:, 1:]

    # Get the number of found neighbors for each point. Indeed,
    # depending on the cloud properties and the chosen K and radius,
    # some points may receive "-1" neighbors
    n_found_nn = (neighbors != -1).sum(dim=1)

    # Identify points which have less than k_min neighbor. Those are
    # treated as outliers
    mask_outliers = n_found_nn < k_min
    idx_outliers = torch.where(mask_outliers)[0]
    idx_inliers = torch.where(~mask_outliers)[0]

    # Exit here if not recursively searching for outliers
    if not recursive:
        return idx_outliers.to(device), idx_inliers.to(device)

    # Identify the points affected by the removal of the outliers. Those
    # inliers are potential outliers
    idx_potential = torch.where(
        torch.isin(neighbors[idx_inliers], idx_outliers).any(dim=1))[0]

    # Exit here if there are no potential new outliers among the inliers
    if idx_potential.shape[0] == 0:
        return idx_outliers.to(device), idx_inliers.to(device)

    # Recursively search actual outliers among the potential
    xyz_query_sub = xyz_query[0, idx_inliers[idx_potential]]
    xyz_search_sub = xyz_search[0, idx_inliers]
    idx_outliers_sub, idx_inliers_sub = inliers_outliers_splits(
        xyz_query_sub, xyz_search_sub, k_min, r_max=r_max, recursive=True,
        q_in_s=True)

    # Update the outliers mask
    mask_outliers[idx_inliers[idx_potential][idx_outliers_sub]] = True
    idx_outliers = torch.where(mask_outliers)[0]
    idx_inliers = torch.where(~mask_outliers)[0]

    return idx_outliers.to(device), idx_inliers.to(device)


def oversample_partial_neighborhoods(neighbors, distances, k):
    """Oversample partial neighborhoods with less than k points. Missing
    neighbors are indicated by the "-1" index.

    Remarks
      - Neighbors and distances are assumed to be sorted in order of
      increasing distance
      - All neighbors are assumed to have at least one valid neighbor.
      See `search_outliers` to remove points with not enough neighbors
    """
    # Initialization
    assert neighbors.dim() == distances.dim() == 2
    device = neighbors.device

    # Get the number of found neighbors for each point. Indeed,
    # depending on the cloud properties and the chosen K and radius,
    # some points may receive `-1` neighbors
    n_found_nn = (neighbors != -1).sum(dim=1)

    # Identify points which have more than k_min and less than k
    # neighbors within R. For those, we oversample the neighbors to
    # reach k
    idx_partial = torch.where(n_found_nn < k)[0]
    neighbors_partial = neighbors[idx_partial]
    distances_partial = distances[idx_partial]

    # Since the neighbors are sorted by increasing distance, the missing
    # neighbors will always be the last ones. This helps finding their
    # number and position, for oversampling.

    # *******************************************************************
    # The above statement is actually INCORRECT because the outlier
    # removal may produce "-1" neighbors at unexpected positions. So
    # either we manage to treat this in a clean vectorized way, or we
    # fall back to the 2-searches solution...
    # Honestly, this feels like it is getting out of hand, let's keep
    # things simple, since we are not going to save so much computation
    # time with KNN wrt the partition.
    # *******************************************************************

    # For each missing neighbor, compute the size of the discrete set to
    # oversample from.
    n_valid = n_found_nn[idx_partial].repeat_interleave(
        k - n_found_nn[idx_partial])

    # Compute the oversampling row indices.
    idx_x_sampling = torch.arange(
        neighbors_partial.shape[0], device=device).repeat_interleave(
        k - n_found_nn[idx_partial])

    # Compute the oversampling column indices. The 0.9999 factor is a
    # security to handle the case where torch.rand is to close to 1.0,
    # which would yield incorrect sampling coordinates that would in
    # result in sampling '-1' indices (i.e. all we try to avoid here)
    idx_y_sampling = (n_valid * torch.rand(
        n_valid.shape[0], device=device) * 0.9999).floor().long()

    # Apply the oversampling
    idx_missing = torch.where(neighbors_partial == -1)
    neighbors_partial[idx_missing] = neighbors_partial[
        idx_x_sampling, idx_y_sampling]
    distances_partial[idx_missing] = distances_partial[
        idx_x_sampling, idx_y_sampling]

    # Restore the oversampled neighborhoods with the rest
    neighbors[idx_partial] = neighbors_partial
    distances[idx_partial] = distances_partial

    return neighbors, distances


def cluster_radius_nn_graph(
        x_points,
        idx,
        k_max=100,
        gap=0,
        batch=None,
        trim=True,
        cycles=3,
        chunk_size=100000,
        verbose=False):
    """Compute the radius neighbors of clusters. Two clusters are
    considered neighbors if 2 of their points are distant of `gap` of
    less.

    The underlying strategy searches the cluster centroids within a
    certain radius, based each cluster's radius and the chosen `gap`.
    This approach is a proxy to avoid the actual computation of all
    pointwise distances.

    :param x_points:
    :param idx:
    :param k_max:
    :param gap:
    :param batch:
        Passing `batch` will ensure the neighbor search does
        not mix up batch items. This batch tensor is a tensor of size
        `num_clusters=idx.max() + 1` indicating which batch item each
        cluster belongs to
    :param trim bool
        If True, the output `edge_index` will be trimmed using
        `to_trimmed`, to save compute and memory
    :param cycles int
        Number of iterations. Starting from a point X in set A, one
        cycle accounts for searching the nearest neighbor, in A, of the
        nearest neighbor of X in set B
    :param chunk_size: int, float
        Allows mitigating memory use when computing the neighbors. If
        `chunk_size > 1`, `edge_index` will be processed into chunks of
        `chunk_size`. If `0 < chunk_size < 1`, then `edge_index` will be
        divided into parts of `edge_index.shape[1] * chunk_size` or less
    :return:
    """
    assert batch is None or batch.shape[0] == idx.max() + 1

    device = x_points.device

    if verbose or src.is_debug_enabled():
        torch.cuda.synchronize()
        start = time()

    # Roughly estimate the diameter and center of each segment. Note we
    # do not use the centroid (center of mass) but rather the center of
    # the bounding box
    bbox_low = scatter(x_points, idx, dim=0, reduce='min')
    bbox_high = scatter(x_points, idx, dim=0, reduce='max')
    diam = (bbox_high - bbox_low).max(dim=1).values
    center = (bbox_high + bbox_low) / 2

    if verbose or src.is_debug_enabled():
        torch.cuda.synchronize()
        print(f"    cluster_radius_nn_graph | center: {time() - start:0.3f} sec")
        start = time()

    # Conservative heuristic for the global search radius: we search the
    # segments whose centroids are separated by the largest segment
    # diameter plus the input gap. This approximates the true operation
    # we would like to perform (but which is too costly): searching, for
    # each segment, the segments with at least one point within gap.
    # Obviously, the r_search may produce more neighbors than needed and
    # some subsequent pruning will be needed
    r_search = float(diam.max() + gap)
    neighbors, distances = knn_1(center, k_max, r_max=r_search, batch=batch)

    if verbose or src.is_debug_enabled():
        torch.cuda.synchronize()
        print(f"    cluster_radius_nn_graph | knn_1: {time() - start:0.3f} sec")
        start = time()

    # Build the corresponding edge_index
    num_clusters = idx.max() + 1
    source = torch.arange(num_clusters, device=device).repeat_interleave(k_max)
    target = neighbors.flatten()
    edge_index = torch.vstack((source, target))
    distances = distances.flatten()

    if verbose or src.is_debug_enabled():
        torch.cuda.synchronize()
        print(f"    cluster_radius_nn_graph | edge_index: {time() - start:0.3f} sec")
        start = time()

    # Trim edges based on the actual segment radii and not the
    # overly-conservative maximum radius used for the search. For this
    # step, we use a gap of `sqrt(3) * gap` to account for some edge
    # case where two 3D boxes touch each other by their corners. This
    # avoids removing neighbors too aggressively before the next step
    # TODO: for some reason, this trimming does not remove as many
    #  neighbors as I'd thought, so I need to call
    #  `scatter_nearest_neighbor` afterward. There must be something
    #  wrong in the `r_max_edge` or the `distances` here. Solving this
    #  is no priority but could maybe avoid the call to
    #  `scatter_nearest_neighbor`
    r_segment = diam / 2
    r_max_edge = r_segment[edge_index].sum(dim=0) + 1.732 * gap
    in_gap_range = distances <= r_max_edge
    edge_index = edge_index[:, in_gap_range]
    distances = distances[in_gap_range]

    if verbose or src.is_debug_enabled():
        torch.cuda.synchronize()
        print(f"    cluster_radius_nn_graph | in_gap: {time() - start:0.3f} sec")
        start = time()

    # Trim edges where points are missing (i.e. -1 neighbor indices)
    missing_point_edge = edge_index[1] == -1
    edge_index = edge_index[:, ~missing_point_edge]
    distances = distances[~missing_point_edge]

    if verbose or src.is_debug_enabled():
        torch.cuda.synchronize()
        print(f"    cluster_radius_nn_graph | missing_point : {time() - start:0.3f} sec")
        start = time()

    # Trim the graph. This is required before computing the actual
    # nearest points between all cluster pairs. Since this operation is
    # so costly, we first built on a coarse neighborhood edge_index to
    # alleviate compute and memory cost
    if trim:
        from src.utils import to_trimmed
        edge_index, distances = to_trimmed(
            edge_index,
            edge_attr=distances,
            reduce='min')
    # Coalesce edges to remove duplicates
    else:
        edge_index, distances = coalesce(
            edge_index,
            edge_attr=distances,
            reduce='min')

    if verbose or src.is_debug_enabled():
        torch.cuda.synchronize()
        print(f"    cluster_radius_nn_graph | coalesce: {time() - start:0.3f} sec")
        start = time()

    # For each cluster pair in edge_index, compute (approximately) the
    # two closest points (coined "anchors" here). The heuristic used
    # here to find those points runs in O(E) with E the number of
    # edges, which is O(N) with N the number of points. This is a
    # workaround for the actual anchor points search, which is O(N²)
    # TODO: scatter_nearest_neighbor is the bottleneck of cluster_nn_radius(),
    #  we could accelerate things by randomly sampling in the clusters
    anchors = scatter_nearest_neighbor(
        x_points,
        idx,
        edge_index,
        cycles=cycles,
        chunk_size=chunk_size)[1]
    d_nn = (x_points[anchors[0]] - x_points[anchors[1]]).norm(dim=1)

    if verbose or src.is_debug_enabled():
        torch.cuda.synchronize()
        print(f"    cluster_radius_nn_graph | scatter_nn: {time() - start:0.3f} sec")
        start = time()

    # Trim edges wrt the anchor points distance
    in_gap_range = d_nn <= gap
    edge_index = edge_index[:, in_gap_range]
    distances = d_nn[in_gap_range]

    if verbose or src.is_debug_enabled():
        torch.cuda.synchronize()
        print(f"    cluster_radius_nn_graph | final_filter: {time() - start:0.3f} sec")
        start = time()

    return edge_index, distances


def neighbors_dense_to_csr(nn):
    """Convert to CSR format a [N, K] tensor of dense neighbor indices
    containing -1 values to indicate missing/absent neighbors. This is
    typically useful for efficiently representing spherical
    neighborhoods with varying sizes.
    """
    k = nn.shape[1]

    # Convert neighbor indices from 2D tensor to CSR format
    # Check for missing neighbors (indicated by -1 indices)
    mask = nn < 0
    n_missing = mask.sum(dim=1)
    sizes = k - n_missing
    nn_val = nn[~mask]
    nn_ptr = sizes_to_pointers(sizes)

    return nn_ptr, nn_val, sizes
