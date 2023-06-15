from src.transforms import Transform
from src.utils.neighbors import knn_1, inliers_split, \
    outliers_split


__all__ = ['KNN', 'Inliers', 'Outliers']


class KNN(Transform):
    """K-NN search for each point in Data.

    Neighbors and corresponding distances are stored in
    `Data.neighbor_index` and `Data.neighbor_distance`, respectively.

    To accelerate search, neighbors are searched within a maximum radius
    of each point. This may result in points having less-than-expected
    neighbors (missing neighbors are indicated by -1 indices). The
    `oversample` mechanism allows for oversampling the found neighbors
    so as to replace the missing ones.

    :param k: int
        Number of neighbors to search for
    :param r_max: float
        Radius within which neighbors are searched around each point
    :param oversample: bool
        Whether partial neighborhoods should be oversampled to reach
        the target `k` neighbors per point
    :param self_is_neighbor: bool
        Whether each point should be considered as its own nearest
        neighbor or should be excluded from the search
    :param verbose: bool
    """

    _NO_REPR = ['verbose']

    def __init__(
            self, k=50, r_max=1, oversample=False, self_is_neighbor=False,
            verbose=False):
        self.k = k
        self.r_max = r_max
        self.oversample = oversample
        self.self_is_neighbor = self_is_neighbor
        self.verbose = verbose

    def _process(self, data):
        neighbors, distances = knn_1(
            data.pos, self.k, r_max=self.r_max, oversample=self.oversample,
            self_is_neighbor=self.self_is_neighbor, verbose=self.verbose)
        data.neighbor_index = neighbors
        data.neighbor_distance = distances
        return data


class Inliers(Transform):
    """Search for points with `k_min` OR MORE neighbors within a
    radius of `r_max`.

    Since removing outliers may cause some points to become outliers
    themselves, this problem can be tackled with the `recursive` option.
    Note that this recursive search holds no guarantee of reasonable
    convergence as one could design a point cloud for given `k_min` and
    `r_max` whose points would all recursively end up as outliers.
    """

    def __init__(
            self, k_min, r_max=1, recursive=False, update_sub=False,
            update_super=False):
        self.k_min = k_min
        self.r_max = r_max
        self.recursive = recursive
        self.update_sub = update_sub
        self.update_super = update_super

    def _process(self, data):
        # Actual outlier search, optionally recursive
        idx = inliers_split(
            data.pos, data.pos, self.k_min, r_max=self.r_max,
            recursive=self.recursive, q_in_s=True)

        # Select the points of interest in Data
        return data.select(
            idx, update_sub=self.update_sub, update_super=self.update_super)


class Outliers(Transform):
    """Search for points with LESS THAN `k_min` neighbors within a
    radius of `r_max`.

    Since removing outliers may cause some points to become outliers
    themselves, this problem can be tackled with the `recursive` option.
    Note that this recursive search holds no guarantee of reasonable
    convergence as one could design a point cloud for given `k_min` and
    `r_max` whose points would all recursively end up as outliers.
    """

    def __init__(
            self, k_min, r_max=1, recursive=False, update_sub=False,
            update_super=False):
        self.k_min = k_min
        self.r_max = r_max
        self.recursive = recursive
        self.update_sub = update_sub
        self.update_super = update_super

    def _process(self, data):
        # Actual outlier search, optionally recursive
        idx = outliers_split(
            data.pos, data.pos, self.k_min, r_max=self.r_max,
            recursive=self.recursive, q_in_s=True)

        # Select the points of interest in Data
        return data.select(
            idx, update_sub=self.update_sub, update_super=self.update_super)
