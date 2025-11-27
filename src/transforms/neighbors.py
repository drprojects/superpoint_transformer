import torch
from src.transforms import Transform
from src.utils.neighbors import knn_1, inliers_split, \
    outliers_split
from src.utils import fill_list_with_string_indexing
from src.data import CSRData, NAG

__all__ = ['KNN', 'NAGKNN', 'Inliers', 'Outliers']


class KNN(Transform):
    """K-NN search for each point in Data.

    Neighbors and corresponding distances are stored in
    `Data.neighbor_index` and `Data.neighbor_distance`, respectively.

    To accelerate search, neighbors are searched within a maximum radius
    of each point. This may result in points having less-than-expected
    neighbors (missing neighbors are indicated by -1 indices). The
    `oversample` mechanism allows for oversampling the found neighbors
    to replace the missing ones.

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
    :param save_as_csr: bool
        Whether to save the neighbors as a CSR matrix.
        The attribute `neighbors` will be a CSRData object, where the first element 
        of `neighbors.values` is the neighbor indices and the second element is the neighbor distances.
        Useful to get rid of the missing neighbors.
    """

    _NO_REPR = ['verbose']

    def __init__(
            self, k=50, r_max=1, oversample=False, self_is_neighbor=False,
            verbose=False, save_as_csr=False):
        self.k = k
        self.r_max = r_max
        self.oversample = oversample
        self.self_is_neighbor = self_is_neighbor
        self.verbose = verbose
        self.save_as_csr = save_as_csr

    def _process(self, data):
        # Mechanism to skip the transform if needed
        if self.r_max <= 0 or self.k <= 0:
            return data
        
        neighbors, distances = knn_1(
            data.pos,
            self.k,
            r_max=self.r_max,
            batch=data.batch,
            oversample=self.oversample,
            self_is_neighbor=self.self_is_neighbor,
            verbose=self.verbose)
        
        if self.save_as_csr:
            num_points = neighbors.shape[0]
            
            mask_legal_neighbors = neighbors != -1
            
            pointers = torch.zeros(num_points + 1, dtype=torch.long, device=neighbors.device)
            pointers[1:] = mask_legal_neighbors.sum(dim=-1)
            pointers = pointers.cumsum(dim=0)
            
            
            data.neighbors = CSRData(pointers,
                                    neighbors[mask_legal_neighbors],
                                    distances[mask_legal_neighbors],
                                    is_index_value=[True, False])
            

        else:
            data.neighbor_index = neighbors
            data.neighbor_distance = distances
            
        return data

import time

class NAGKNN(Transform):
    """KNN search for the specified levels of the NAG.
    
    See more details in the `KNN` transform.
    
    WARNING : if we `NAG.select` the points, the neighbors index are not updated.
    
    :param level: int or str
        Level at which to search for neighbors. Can be an int or a str. If
        the latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    """
    
    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    
    def __init__(self, level='all', k=50, r_max=1, oversample=False, self_is_neighbor=False,
            verbose=False, save_as_csr=False,):
        
        super().__init__()
        
        self.level = level
        self.k = k
        self.r_max = r_max
        self.oversample = oversample
        self.self_is_neighbor = self_is_neighbor
        self.verbose = verbose
        self.save_as_csr = save_as_csr
        
    def _process(self, nag):
        k_per_level = fill_list_with_string_indexing(
            level=self.level,
            default=0, 
            value=self.k,
            output_length=nag.absolute_num_levels,
            start_index=nag.start_i_level)
        
        transforms = [KNN(k=k, r_max=self.r_max, oversample=self.oversample, self_is_neighbor=self.self_is_neighbor,
                          verbose=self.verbose, save_as_csr=self.save_as_csr) for k in k_per_level]
        
        nag.apply_data_transform(transforms)
        
        return nag

    

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
        return data.select(idx, update_sub=self.update_sub)


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
        return data.select(idx, update_sub=self.update_sub)
