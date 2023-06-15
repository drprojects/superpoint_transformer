import torch
import math
from torch_scatter import scatter_min, scatter_max, scatter_mean
from torch_geometric.utils import coalesce, remove_self_loops
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.utils.tensor import arange_interleave
from src.utils.geometry import base_vectors_3d
from src.utils.sparse import sizes_to_pointers, sparse_sort, \
    sparse_sort_along_direction
from src.utils.scatter import scatter_pca, scatter_nearest_neighbor, \
    idx_preserving_mask
from src.utils.edge import edge_wise_points

__all__ = [
    'is_pyg_edge_format', 'isolated_nodes', 'edge_to_superedge', 'subedges',
    'to_trimmed', 'is_trimmed']


def is_pyg_edge_format(edge_index):
    """Check whether edge_index follows pytorch geometric graph edge
    format: a [2, N] torch.LongTensor.
    """
    return \
        isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 \
        and edge_index.dtype == torch.long and edge_index.shape[0] == 2


def isolated_nodes(edge_index, num_nodes=None):
    """Return a boolean mask of size num_nodes indicating which node has
    no edge in edge_index.
    """
    assert is_pyg_edge_format(edge_index)
    num_nodes = edge_index.max() + 1 if num_nodes is None else num_nodes
    device = edge_index.device
    mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    mask[edge_index.unique()] = False
    return mask


def edge_to_superedge(edges, super_index, edge_attr=None):
    """Convert point-level edges into superedges between clusters, based
    on point-to-cluster indexing 'super_index'. Optionally 'edge_attr'
    can be passed to describe edge attributes that will be returned
    filtered and ordered to describe the superedges.

    NB: this function treats (i, j) and (j, i) superedges as identical.
    By default, the final edges are expressed with i <= j
    """
    # We are only interested in the edges connecting two different
    # clusters and not in the intra-cluster connections. So we first
    # identify the edges of interest. This step requires having access
    # to the 'super_index' to convert point indices into their
    # corresponding cluster indices
    se = super_index[edges]
    inter_cluster = torch.where(se[0] != se[1])[0]

    # Now only consider the edges of interest (ie inter-cluster edges)
    edges_inter = edges[:, inter_cluster]
    edge_attr = edge_attr[inter_cluster] if edge_attr is not None else None
    se = se[:, inter_cluster]

    # Search for undirected edges, ie edges with (i,j) and (j,i)
    # both present in edge_index. Flip (j,i) into (i,j) to make them
    # redundant. By default, the final edges are expressed with i <= j
    s_larger_t = se[0] > se[1]
    se[:, s_larger_t] = se[:, s_larger_t].flip(0)

    # So far we are manipulating inter-cluster edges, but there may be
    # multiple of those for a given source-target pair. If, we want to
    # aggregate those into 'superedges' and compute corresponding
    # features (designated with 'se_'), we will need unique and
    # compact inter-cluster edge identifiers for torch_scatter
    # operations. We use 'se' to designate 'superedge' (ie an edge
    # between two clusters)
    se_id = \
        se[0] * (max(se[0].max(), se[1].max()) + 1) + se[1]
    se_id, perm = consecutive_cluster(se_id)
    se = se[:, perm]

    return se, se_id, edges_inter, edge_attr


def subedges(
        points,
        index,
        edge_index,
        ratio=0.2,
        k_min=20,
        cycles=3,
        pca_on_cpu=True,
        margin=0.2,
        halfspace_filter=True,
        bbox_filter=True,
        target_pc_flip=True,
        source_pc_sort=False,
        chunk_size=None):
    """Compute the subedges making up each edge between segments. These
    are needed for superedge features computation. This approach relies
    on heuristics to avoid the Delaunay triangulation or any other O(N²)
    operation.

    NB: the input edges will be trimmed (see `to_trimmed`) in the first
    place and the returned edge_index will reflect this change. This is
    because subedge computation relies on costly operations. To save
    compute and memory, we only build subedges for the trimmed graph.

    :param points:
        Level-0 points
    :param index:
        Index of the segment each point belongs to
    :param edge_index:
        Edges of the graph between segments
    :param ratio:
        Maximum ratio of a segment's points than can be used in a
        superedge's subedges
    :param k_min:
        Minimum of subedges per superedge
    :param cycles:
        Number of iterations for nearest neighbor search between
        segments
    :param pca_on_cpu:
        Whether PCA should be computed on CPU if need be. Should be kept
        as True
    :param margin:
        Tolerance margin used for selecting subedges points and
        excluding segment points from potential subedge candidates
    :param halfspace_filter:
        Whether the halfspace filtering should be applied
    :param bbox_filter:
        Whether the bounding box filtering should be applied
    :param target_pc_flip:
        Whether the subedge point pairs should be carefully ordered
    :param source_pc_sort:
        Whether the source and target subedge point pairs should be
        ordered along the same vector
    :param chunk_size: int, float
        Allows mitigating memory use when computing the subedges. If
        `chunk_size > 1`, `edge_index` will be processed into chunks of
        `chunk_size`. If `0 < chunk_size < 1`, then `edge_index` will be
        divided into parts of `edge_index.shape[1] * chunk_size` or less
    :return:
    """
    # Trim the graph
    edge_index = to_trimmed(edge_index)

    # Number of segments
    num_segments = index.max() + 1

    # Recursive call in case chunk is specified. Chunk allows limiting
    # the number of edges processed at once. This might alleviate
    # memory use
    if chunk_size is not None and chunk_size > 0:

        # Recursive call on smaller edge_index chunks
        chunk_size = int(chunk_size) if chunk_size > 1 \
            else math.ceil(edge_index.shape[1] * chunk_size)
        num_chunks = math.ceil(edge_index.shape[1] / chunk_size)
        out_list = []
        for i_chunk in range(num_chunks):
            start = i_chunk * chunk_size
            end = (i_chunk + 1) * chunk_size
            out_list.append(subedges(
                points,
                index,
                edge_index[:, start:end],
                ratio=ratio,
                k_min=k_min,
                cycles=cycles,
                pca_on_cpu=pca_on_cpu,
                margin=margin,
                halfspace_filter=halfspace_filter,
                bbox_filter=bbox_filter,
                target_pc_flip=target_pc_flip,
                source_pc_sort=source_pc_sort,
                chunk_size=None))

        # Combine outputs
        device = points.device
        edge_index = torch.cat([elt[0] for elt in out_list], dim=1)
        ST_pairs = torch.cat([elt[1] for elt in out_list], dim=1)
        size = torch.tensor([o[0].shape[1] for o in out_list], device=device)
        offset = sizes_to_pointers(size[:-1])
        ST_uid = torch.cat([elt[2] + o for elt, o in zip(out_list, offset)])

        return edge_index, ST_pairs, ST_uid

    # Compute the nearest neighbors between superedge segments. This
    # pair of points will be crucial in finding the other level-0
    # points making up the superedge
    _, edge_anchor_idx = scatter_nearest_neighbor(
        points, index, edge_index, cycles=cycles)

    # Compute base vectors based on the anchor points source->target
    # direction
    s_anchor = points[edge_anchor_idx[0]]
    t_anchor = points[edge_anchor_idx[1]]
    anchor_base = base_vectors_3d(t_anchor - s_anchor)

    # Recover the number of points in source and target segments. 's_'
    # and 't_' indicate we are dealing with edge-wise values
    s_size, t_size = index.bincount(minlength=num_segments)[edge_index]

    # Expand the points to point-edge values. That is, the concatenation
    # of all the source --or target-- points for each edge. The
    # corresponding variables are prepended with 'S_' and 'T_' for
    # clarity
    (S_points, S_points_idx, S_uid), (T_points, T_points_idx, T_uid) = \
        edge_wise_points(points, index, edge_index)

    # Local helper function to convert absolute points coordinates to
    # their local edge coordinate system. This system is defined as
    # such: the origin is the source --target-- anchor point, the 1st
    # axis is given by the source->target direction of the anchor
    # points, and the 2nd and 3rd axes are constructed in the orthogonal
    # plane. NB: the base construction has a degree of freedom in
    # rotation around the 1st axis, but we do not care too much about it
    # here
    def to_anchor_base(source=True):
        if source:
            x_size, x_anchor, X_points = s_size, s_anchor, S_points
        else:
            x_size, x_anchor, X_points = t_size, t_anchor, T_points

        # Center the points wrt their anchor
        X_points = X_points - x_anchor.repeat_interleave(x_size, dim=0)

        # Project on the base vectors
        X_proj = []
        for i in range(3):
            v = anchor_base[:, i].repeat_interleave(x_size, dim=0)
            X_proj.append(torch.einsum('nd, nd -> n', X_points, v))

        return torch.vstack(X_proj).T

    # Project points in their local edge coordinate system
    S_points = to_anchor_base(source=True)
    T_points = to_anchor_base(source=False)
    del s_anchor, t_anchor, anchor_base

    # Select points that are in the half-space before their anchor.
    # Since subedge points (level-0 point pairs making up the superedge
    # between two segments) are searched along the nearest-neighbors
    # (ie anchor points) direction, this operation aims at dealing with
    # edges located in concave regions of the segment boundaries
    if halfspace_filter:
        in_S_halfspace = S_points[:, 0] <= margin
        in_S_halfspace = idx_preserving_mask(in_S_halfspace, S_uid)
        in_S_halfspace = torch.where(in_S_halfspace)[0]
        S_points = S_points[in_S_halfspace]
        S_points_idx = S_points_idx[in_S_halfspace]
        S_uid = S_uid[in_S_halfspace]
        del in_S_halfspace
        in_T_halfspace = T_points[:, 0] >= -margin
        in_T_halfspace = idx_preserving_mask(in_T_halfspace, T_uid)
        in_T_halfspace = torch.where(in_T_halfspace)[0]
        T_points = T_points[in_T_halfspace]
        T_points_idx = T_points_idx[in_T_halfspace]
        T_uid = T_uid[in_T_halfspace]
        del in_T_halfspace

    # Compute the bbox intersection in the 2nd and 3rd coordinates
    # plane. This is a proxy for computing the intersection of the
    # projection areas of the segments in the 2nd and 3rd coordinates
    # plane. This operation prevents subedge points from lying too far
    # from the source segment's projection on the target segment along
    # the anchor direction (and conversely)
    if bbox_filter:
        s_min, _ = scatter_min(S_points[:, 1:], S_uid, dim=0)
        s_max, _ = scatter_max(S_points[:, 1:], S_uid, dim=0)
        t_min, _ = scatter_min(T_points[:, 1:], T_uid, dim=0)
        t_max, _ = scatter_max(T_points[:, 1:], T_uid, dim=0)
        st_min = torch.max(s_min, t_min).clamp(max=-margin)
        st_max = torch.min(s_max, t_max).clamp(min=margin)
        del s_min, s_max, t_min, t_max

        # Local helper to select points inside the bbox intersection
        def select_in_bbox(source=True):
            if source:
                X_points, X_points_idx, X_uid = S_points, S_points_idx, S_uid
            else:
                X_points, X_points_idx, X_uid = T_points, T_points_idx, T_uid

            in_bbox = (X_points[:, 1:] >= st_min[X_uid]).all(dim=1) & \
                      (X_points[:, 1:] <= st_max[X_uid]).all(dim=1)
            in_bbox = idx_preserving_mask(in_bbox, X_uid)
            in_bbox = torch.where(in_bbox)[0]

            return X_points[in_bbox], X_points_idx[in_bbox], X_uid[in_bbox]

        # Select points inside the bbox intersection
        S_points, S_points_idx, S_uid = select_in_bbox(source=True)
        T_points, T_points_idx, T_uid = select_in_bbox(source=False)

    # Sort points along the edge direction, the first point being the
    # anchor point and subsequent points farther and farther away from
    # the anchor
    _, perm = sparse_sort(S_points[:, 0], S_uid, descending=True)
    S_points = S_points[perm]
    S_points_idx = S_points_idx[perm]
    S_uid = S_uid[perm]
    del perm
    _, perm = sparse_sort(T_points[:, 0], T_uid, descending=False)
    T_points = T_points[perm]
    T_points_idx = T_points_idx[perm]
    T_uid = T_uid[perm]
    del perm

    # Update the number of selected points in the source/target segments
    # and compute the number of points to keep for each edge. The
    # heuristic we use here is: the top ratio points, with a minimum
    # of k_min, within the limits of the cluster
    s_size = S_uid.bincount()
    t_size = T_uid.bincount()
    s_k = (s_size * ratio).long().clamp(min=k_min).clamp(max=s_size)
    t_k = (t_size * ratio).long().clamp(min=k_min).clamp(max=t_size)
    st_k = torch.min(s_k, t_k)
    del s_k, t_k

    # Select only the first k points for each edge
    S_k_idx = arange_interleave(st_k, start=sizes_to_pointers(s_size[:-1]))
    S_points = S_points[S_k_idx]
    S_points_idx = S_points_idx[S_k_idx]
    S_uid = S_uid[S_k_idx]
    del S_k_idx
    T_k_idx = arange_interleave(st_k, start=sizes_to_pointers(t_size[:-1]))
    T_points = T_points[T_k_idx]
    T_points_idx = T_points_idx[T_k_idx]
    T_uid = T_uid[T_k_idx]
    del T_k_idx

    # Local helper to compute, for each edge, the first eigen vector of
    # the selected subedge points for the source --target,
    # respectively-- segment
    def first_component(source=True):
        if source:
            X_points, X_uid = S_points, S_uid
        else:
            X_points, X_uid = T_points, T_uid
        return scatter_pca(X_points, X_uid, on_cpu=pca_on_cpu)[1][:, :, -1]

    # Compute the first component of the source and target subedge
    # points, to be used to sort the points and eventually build the
    # subedge pair
    s_v = first_component(source=True)
    t_v = first_component(source=False)

    # Flip the target first component direction when needed. This is to
    # limit subedge crossings. This is motivated by the desire to mimick
    # Delaunay's visibility-based edges
    if target_pc_flip and not source_pc_sort:
        T_proj = (T_points * t_v.repeat_interleave(st_k, dim=0)).sum(dim=1)
        s_mean = scatter_mean(S_points, S_uid, dim=0)
        t_min = T_points[scatter_min(T_proj, T_uid, dim=0)[1]]
        st_u = t_min - s_mean
        st_u /= st_u.norm(dim=1).view(-1, 1)
        to_flip = torch.where((s_v * t_v).sum(dim=1) <= (s_v * st_u).sum(dim=1))[0]
        t_v[to_flip] *= -1
    elif source_pc_sort:
        t_v = s_v

    # Local helper to sort points along their first component
    def sort_by_first_component(source=True):
        if source:
            X_points, X_points_idx, X_uid, x_v = \
                S_points, S_points_idx, S_uid, s_v
        else:
            X_points, X_points_idx, X_uid, x_v = \
                T_points, T_points_idx, T_uid, t_v

        # Sort points along the first component
        X_points, perm = sparse_sort_along_direction(X_points, X_uid, x_v)

        return X_points, X_points_idx[perm], X_uid[perm]

    # Sort the subedge points along their first component
    S_points, S_points_idx, S_uid = sort_by_first_component(source=True)
    T_points, T_points_idx, T_uid = sort_by_first_component(source=False)

    # Bring the subedge points together to make up the final pairs
    ST_pairs = torch.vstack((S_points_idx, T_points_idx))
    ST_uid = S_uid

    return edge_index, ST_pairs, ST_uid


def to_trimmed(edge_index, edge_attr=None, reduce='mean'):
    """Convert to 'trimmed' graph: same as coalescing with the
    additional constraint that (i, j) and (j, i) edges are duplicates.

    If edge attributes are passed, 'reduce' will indicate how to fuse
    duplicate edges' attributes.

    NB: returned edges are expressed with i<j by default.

    :param edge_index: 2xE LongTensor
        Edges in `torch_geometric` format
    :param edge_attr: ExC Tensor, optional
        Edge attributes
    :param reduce: str, optional
        Reduction modes supported by `torch_geometric.utils.coalesce`
    :return:
    """
    # Search for undirected edges, ie edges with (i,j) and (j,i)
    # both present in edge_index. Flip (j,i) into (i,j) to make them
    # redundant
    s_larger_t = edge_index[0] > edge_index[1]
    edge_index[:, s_larger_t] = edge_index[:, s_larger_t].flip(0)

    # Sort edges by row and remove duplicates
    if edge_attr is None:
        edge_index = coalesce(edge_index)
    else:
        edge_index, edge_attr = coalesce(
            edge_index, edge_attr=edge_attr, reduce=reduce)

    # Remove self loops
    edge_index, edge_attr = remove_self_loops(
        edge_index, edge_attr=edge_attr)

    if edge_attr is None:
        return edge_index
    return edge_index, edge_attr


def is_trimmed(edge_index, return_trimmed=False):
    """Check if the graph is 'trimmed': same as coalescing with the
    additional constraint that (i, j) and (j, i) edges are duplicates.

    :param edge_index: 2xE LongTensor
        Edges in `torch_geometric` format
    :param return_trimmed: bool
        If True, the trimmed graph will also be returned. Since checking
        if the graph is trimmed requires computing the actual trimmed
        graph, this may save some compute in certain situations
    :return:
    """
    edge_index_trimmed = to_trimmed(edge_index)
    trimmed = edge_index.shape == edge_index_trimmed.shape
    if return_trimmed:
        return trimmed, edge_index_trimmed
    return trimmed
