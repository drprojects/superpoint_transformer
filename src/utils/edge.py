from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.utils.sparse import indices_to_pointers
from src.utils.tensor import arange_interleave


__all__ = ['edge_index_to_uid', 'edge_wise_points']


def edge_index_to_uid(edge_index):
    """Compute consecutive unique identifiers for the edges. This may be
    needed for scatter operations.
    """
    assert edge_index.dim() == 2
    assert edge_index.shape[0] == 2
    source = edge_index[0]
    target = edge_index[1]
    edge_uid = source * (max(source.max(), target.max()) + 1) + target
    edge_uid = consecutive_cluster(edge_uid)[0]
    return edge_uid


def edge_wise_points(points, index, edge_index):
    """Given a graph of point segments, compute the concatenation of
    points belonging to either source or target segments for each edge
    of the segment graph. This operation arises when dealing with
    pairwise relationships between point segments.

    Warning: the output tensors might be memory-intensive

    :param points: (N, D) tensor
        Points
    :param index: (N) LongTensor
        Segment index, for each point
    :param edge_index: (2, E) LongTensor
        Edges of the segment graph
    """
    assert points.dim() == 2
    assert index.dim() == 1
    assert points.shape[0] == index.shape[0]
    assert edge_index.dim() == 2
    assert edge_index.shape[0] == 2
    assert edge_index.max() <= index.max()

    # We define the segments in the first row of edge_index as 'source'
    # segments, while the elements of the second row are 'target'
    # segments. The corresponding variables are prepended with 's_' and
    # 't_' for clarity
    s_idx = edge_index[0]
    t_idx = edge_index[1]

    # Compute consecutive unique identifiers for the edges
    uid = edge_index_to_uid(edge_index)

    # Compute the pointers and ordering to express the segments and the
    # points they hold in CSR format
    pointers, order = indices_to_pointers(index)

    # Compute the size of each segment
    segment_size = index.bincount()

    # Expand the edge variables to point-edge values. That is, the
    # concatenation of all the source -or target- points for each edge.
    # The corresponding variables are prepended with 'S_' and 'T_' for
    # clarity
    def expand(source=True):
        x_idx = s_idx if source else t_idx
        size = segment_size[x_idx]
        start = pointers[:-1][x_idx]
        X_points_idx = order[arange_interleave(size, start=start)]
        X_points = points[X_points_idx]
        X_uid = uid.repeat_interleave(size, dim=0)
        return X_points, X_points_idx, X_uid

    S_points, S_points_idx, S_uid = expand(source=True)
    T_points, T_points_idx, T_uid = expand(source=False)

    return (S_points, S_points_idx, S_uid), (T_points, T_points_idx, T_uid)
