from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.utils.point import is_xyz_tensor


__all__ = ['xy_partition']


def xy_partition(pos, grid, consecutive=True):
    """Partition a point cloud based on a regular XY grid. Returns, for
    each point, the index of the grid cell it falls into.

    :param pos: Tensor
        Point cloud
    :param grid: float
        Grid size
    :param consecutive: bool
        Whether the grid cell indices should be consecutive. That is to
        say all indices in [0, idx_max] are used. Note that this may
        prevent trivially mapping an index value back to the
        corresponding XY coordinates
    :return:
    """
    assert is_xyz_tensor(pos)

    # Compute the (i, j) coordinates on the XY grid size
    i = pos[:, 0].div(grid, rounding_mode='trunc').long()
    j = pos[:, 1].div(grid, rounding_mode='trunc').long()

    # Shift coordinates to positive integer to avoid negatives
    # clashing with our downstream indexing mechanism
    i -= i.min()
    j -= j.min()

    # Compute a "manual" partition based on the grid coordinates
    super_index = i * (max(i.max(), j.max()) + 1) + j

    # If required, update the used indices to be consecutive
    if consecutive:
        super_index = consecutive_cluster(super_index)[0]

    return super_index
