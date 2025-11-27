import re
import math
import torch
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import k_hop_subgraph, to_undirected
from torch_cluster import grid_cluster
from torch_scatter import scatter_mean
from torch_geometric.nn.pool.consecutive import consecutive_cluster

from src.utils import (
    fast_randperm,
    sparse_sample,
    scatter_pca,
    sanitize_keys,
    knn_2,
    knn_brute_force,
    split_histogram,
    cast_to_optimal_integer_type,
    fill_list_with_string_indexing)
from src.transforms import Transform
from src.data import (
    Data,
    Batch,
    NAG,
    NAGBatch,
    CSRData,
    InstanceData,
    Cluster)
from src.utils.histogram import atomic_to_histogram

__all__ = [
    'Shuffle',
    'SaveNodeIndex',
    'NAGSaveNodeIndex',
    'GridSampling3D',
    'SampleXYTiling',
    'SampleRecursiveMainXYAxisTiling',
    'SampleSubNodes',
    'SampleKHopSubgraphs',
    'SampleRadiusSubgraphs',
    'SampleSegments',
    'SampleEdges',
    'RestrictSize',
    'NAGRestrictSize',
    'QuantizePointCoordinates']


class Shuffle(Transform):
    """Shuffle the order of points in a Data object."""

    def _process(self, data):
        idx = fast_randperm(data.num_points, device=data.device)
        return data.select(idx, update_sub=False)


class SaveNodeIndex(Transform):
    """Adds the index of the nodes to the Data object attributes. This
    allows tracking nodes from the output back to the input Data object.
    """

    DEFAULT_KEY = 'node_id'

    def __init__(self, key=None):
        self.key = key if key is not None else self.DEFAULT_KEY

    def _process(self, data):
        idx = torch.arange(0, data.pos.shape[0], device=data.device)
        setattr(data, self.key, idx)
        return data


class NAGSaveNodeIndex(SaveNodeIndex):
    """SaveNodeIndex, applied to each NAG level.
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def _process(self, nag):
        transform = SaveNodeIndex(key=self.key)
        for i_level in range(nag.num_levels):
            nag._list[i_level] = transform(nag._list[i_level])
        return nag


class GridSampling3D(Transform):
    """ Clusters 3D points into voxels with size :attr:`size`.

    By default, some special keys undergo dedicated grouping mechanisms.
    The `_VOTING_KEYS=['y', 'super_index', 'is_val']` keys are grouped
    by their majority label. The `_INSTANCE_KEYS=['obj', 'obj_pred']`
    keys are grouped into an `InstanceData`, which stores all
    instance/panoptic overlap data values in CSR format. The
    `_CLUSTER_KEYS=['point_id']` keys are grouped into a `Cluster`
    object, which stores indices of child elements for parent clusters
    in CSR format. The `_LAST_KEYS=['batch', SaveNodeIndex.DEFAULT_KEY]`
    keys are by default grouped following `mode='last'`.

    Besides, for keys where a more subtle histogram mechanism is needed,
    (e.g. for 'y'), the 'hist_key' and 'hist_size' arguments can be
    used.

    Modified from: https://github.com/torch-points3d/torch-points3d

    :param size: float
        Size of a voxel (in each dimension).
    :param quantize_coords: bool
        If True, it will convert the points into their associated sparse
        coordinates within the grid and store the value into a new
        `coords` attribute.
    :param allow_negative_coords: bool
        If True, some coordinates will be negative if some pos values are negative.
        It is needed to map coordinates back to approximate pos values.
        If False, the coordinates will be shifted to be positive.
        This is needed to avoid some overhead of torchsparse.
    :param mode: string:
        The mode can be either `last` or `mean`.
        If mode is `mean`, all the points and their features within a
        cell will be averaged. If mode is `last`, one random points per
        cell will be selected with its associated features.
    :param hist_key: str or List(str)
        Data attributes for which we would like to aggregate values into
        a histogram. This is typically needed when we want to aggregate
        points labels without losing the distribution, as opposed to
        majority voting.
    :param hist_size: str or List(str)
        Must be of same size as `hist_key`, indicates the number of
        bins for each key-histogram. This is typically needed when we
        want to aggregate points labels without losing the distribution,
        as opposed to majority voting.
    :param inplace: bool
        Whether the input Data object should be modified in-place
    :param chunk_size: int, float
        Allows mitigating memory use when the inputs are on GPU. If
        `chunk_size > 1`, the input point cloud will be processed into
        chunks of `chunk_size`. If `0 < chunk_size < 1`, then the point
        cloud will be divided into parts of `xyz.shape[1] * chunk_size`
        or smaller
    :param verbose: bool
        Verbosity
    """

    _NO_REPR = ['verbose', 'inplace']

    def __init__(
            self,
            size,
            quantize_coords=False,
            allow_negative_coords=False,
            mode="mean",
            hist_key=None,
            hist_size=None,
            inplace=False,
            chunk_size=10_000_000,
            verbose=False):

        hist_key = [] if hist_key is None else hist_key
        hist_size = [] if hist_size is None else hist_size
        hist_key = [hist_key] if isinstance(hist_key, str) else hist_key
        hist_size = [hist_size] if isinstance(hist_size, int) else hist_size

        assert isinstance(hist_key, list)
        assert isinstance(hist_size, list)
        assert len(hist_key) == len(hist_size)

        self.grid_size = size
        self.quantize_coords = quantize_coords
        self.allow_negative_coords = allow_negative_coords
        self.mode = mode
        self.bins = {k: v for k, v in zip(hist_key, hist_size)}
        self.inplace = inplace
        self.chunk_size = chunk_size

        if verbose:
            print(
                f"If you need to keep track of the position of your points, "
                f"use SaveNodeIndex transform before using "
                f"{self.__class__.__name__}.")

            if self.mode == "last":
                print(
                    "The tensors within data will be shuffled each time this "
                    "transform is applied. Be careful that if an attribute "
                    "doesn't have the size of num_nodes, it won't be shuffled")

    def _process(self, data_in):
        # In-place option will modify the input Data object directly
        data = data_in if self.inplace else data_in.clone()

        # If the aggregation mode is 'last', shuffle the points order.
        # Note that voxelization of point attributes will be stochastic
        if self.mode == 'last':
            data = Shuffle()(data)

        # Convert point coordinates to the voxel grid coordinates
        coords = torch.round((data.pos) / self.grid_size)

        # Match each point with a voxel identifier
        if 'batch' not in data:
            cluster = grid_cluster(coords, torch.ones(3, device=coords.device))
        else:
            cluster = voxel_grid(coords, 1, data.batch)

        # Reindex the clusters to make sure the indices used are
        # consecutive. Basically, we do not want cluster indices to span
        # [0, i_max] without all in-between indices to be used, because
        # this will affect the speed and output size of torch_scatter
        # operations
        cluster = cast_to_optimal_integer_type(cluster)
        cluster, unique_pos_indices = consecutive_cluster(cluster)

        # Perform voxel aggregation
        data = _group_data(
            data,
            cluster,
            unique_pos_indices,
            mode=self.mode,
            bins=self.bins,
            chunk_size=self.chunk_size)

        # Optionally convert quantize the coordinates. This is useful
        # for sparse convolution models
        if self.quantize_coords:
            data.coords = coords[unique_pos_indices].int()

            # Shift coordinates to positive integer to avoid negatives coordinates
            # (in order to avoid some overhead of torchsparse)
            if not self.allow_negative_coords:
                data.coords -= torch.min(data.coords, dim=0, keepdim=True).values

        # Save the grid size in the Data attributes
        data.grid_size = torch.tensor([self.grid_size])

        return data


def _group_data(
        data,
        cluster=None,
        unique_pos_indices=None,
        mode="mean",
        skip_keys=None,
        bins={},
        chunk_size=10_000_000):
    """Group data based on indices in cluster. The option ``mode``
    controls how data gets aggregated within each cluster.

    By default, some special keys undergo dedicated grouping mechanisms.
    The `_VOTING_KEYS=['y', 'super_index', 'is_val']` keys are grouped
    by their majority label. The `_INSTANCE_KEYS=['obj', 'obj_pred']`
    keys are grouped into an `InstanceData`, which stores all
    instance/panoptic overlap data values in CSR format. The
    `_CLUSTER_KEYS=['point_id']` keys are grouped into a `Cluster`
    object, which stores indices of child elements for parent clusters
    in CSR format. The `_LAST_KEYS=['batch', SaveNodeIndex.DEFAULT_KEY]`
    keys are by default grouped following `mode='last'`.

    Besides, for keys where a more subtle histogram mechanism is needed,
    (e.g. for 'y'), the 'bins' argument can be used.

    Warning: this function modifies the input Data object in-place.

    :param data : Data
    :param cluster : torch.Tensor
        torch.Tensor of the same size as the number of points in data.
        Each element is the cluster index of that point
    :param unique_pos_indices : torch.Tensor
        torch.Tensor containing one index per cluster, this index will
        be used to select features and labels
    :param mode : str
        Option to select how the features and labels for each voxel is
        computed. Can be ``last`` or ``mean``. ``last`` selects the last
        point falling in a voxel as the representative, ``mean`` takes
        the average
    :param skip_keys: list
        Keys of attributes to skip in the grouping
    :param bins: dict
        Dictionary holding ``{'key': n_bins}`` where ``key`` is a Data
        attribute for which we would like to aggregate values into an
        histogram and ``n_bins`` accounts for the corresponding number
        of bins. This is typically needed when we want to aggregate
        point labels without losing the distribution, as opposed to
        majority voting
    :param chunk_size: int, float
        Allows mitigating memory use when the inputs are on GPU. If
        `chunk_size > 1`, the input point cloud will be processed into
        chunks of `chunk_size`. If `0 < chunk_size < 1`, then the point
        cloud will be divided into parts of `xyz.shape[1] * chunk_size`
        or smaller
    """
    # Recursive call in case chunk is specified. Chunk allows limiting
    # the number of voxels processed at once. This might alleviate
    # memory use
    N = data.num_points
    if chunk_size is not None and chunk_size > 0:
        chunk_size = int(chunk_size) if chunk_size > 1 \
            else math.ceil(N * chunk_size)

        # Keep it simple if no chunking required
        if chunk_size >= N:
            return _group_data(
                data,
                cluster=cluster,
                unique_pos_indices=unique_pos_indices,
                mode=mode,
                skip_keys=skip_keys,
                bins=bins,
                chunk_size=None)

        data_list = []
        counts = cluster.bincount()
        splits = split_histogram(counts, chunk_size)

        for split in splits:
            # TODO: the reliance on Data.select() here forbids handling
            #  some attributes which are supported by select. For
            #  instance edge_index, edge_attr,  neighbor_index,
            #  neighbor_distance, ...
            mask = torch.logical_and(split[0] <= cluster, cluster <= split[-1])
            data_ = data.select(mask, update_sub=False, update_super=True)[0]
            cluster_, unique_pos_indices_ = consecutive_cluster(cluster[mask])
            data_list.append(_group_data(
                data_,
                cluster=cluster_,
                unique_pos_indices=unique_pos_indices_,
                mode=mode,
                skip_keys=skip_keys,
                bins=bins,
                chunk_size=None))

        # Dirty hack to allow batching together InstanceData objects
        # referring to the same object indices. In general, we assume
        # the InstanceData.obj indices are referring to different
        # objects when merging. But in our specific case here, this is
        # not the case
        has_obj = data_.obj is not None
        if has_obj:
            for data_ in data_list:
                data_.obj.is_index_value[0] = False
        batch = Batch.from_data_list(data_list)
        if has_obj:
            batch.obj.is_index_value[0] = True

        # TODO: this will not handle well objects which are not of size
        #  num_points. e.g. if pos_offset is a single scalar for the
        #  input Data, it will be duplicated for each chunk here. This
        #  is mostly a duplication problem, so for now we keep this as
        #  is, even if a bit dirty
        return batch.forget_batching()

    skip_keys = sanitize_keys(skip_keys, default=[])

    # Keys for which voxel aggregation will be based on majority voting
    _VOTING_KEYS = ['y', 'super_index', 'is_val']

    # Keys for which voxel aggregation will use an InstanceData object,
    # which store all input information in CSR format
    _INSTANCE_KEYS = ['obj', 'obj_pred']

    # Keys for which voxel aggregation will use a Cluster object, which 
    # store all input information in CSR format
    _CLUSTER_KEYS = ['sub']

    # Keys for which voxel aggregation will be based on majority voting
    _LAST_KEYS = ['batch', SaveNodeIndex.DEFAULT_KEY]

    # Keys to be treated as normal vectors, for which the unit-norm must
    # be preserved
    _NORMAL_KEYS = ['normal']

    # Supported mode for aggregation
    _MODES = ['mean', 'last']
    assert mode in _MODES
    if mode == "mean" and cluster is None:
        raise ValueError(
            "In mean mode the cluster argument needs to be specified")
    if mode == "last" and unique_pos_indices is None:
        raise ValueError(
            "In last mode the unique_pos_indices argument needs to be "
            "specified")

    # Save the number of nodes here because the subsequent in-place
    # modifications will affect it
    num_nodes = data.num_nodes

    # Aggregate Data attributes for same-cluster points
    for key, item in data:

        # `skip_keys` are not aggregated
        if key in skip_keys:
            continue

        # Edges cannot be aggregated
        if bool(re.search('edge', key)):
            raise NotImplementedError("Edges not supported. Wrong data type.")

        # For instance labels grouped into an InstanceData. Supports
        # input instance labels either as InstanceData or as a simple
        # index tensor
        if key in _INSTANCE_KEYS:
            if isinstance(item, InstanceData):
                data[key] = item.merge(cluster)
            else:
                count = torch.ones_like(item)
                y = data.y if getattr(data, 'y', None) is not None \
                    else torch.zeros_like(item)
                data[key] = InstanceData(cluster, item, count, y, dense=True)
            continue

        # For point indices to be grouped in Cluster. This allows
        # backtracking full-resolution point indices to the voxels
        if key in _CLUSTER_KEYS:
            if (isinstance(item, torch.Tensor) and item.dim() == 1
                    and not item.is_floating_point()):
                data[key] = Cluster(cluster, item, dense=True)
            else:
                raise NotImplementedError(
                    f"Cannot merge '{key}' with data type: {type(item)} into "
                    f"a Cluster object. Only supports 1D Tensor of integers.")
            continue

        # TODO: adapt to make use of CSRData batching ?
        if isinstance(item, CSRData):
            raise NotImplementedError(
                f"Cannot merge '{key}' with data type: {type(item)}")

        # Only torch.Tensor attributes of size Data.num_nodes are
        # considered for aggregation
        if not torch.is_tensor(item) or item.size(0) != num_nodes:
            continue

        # For 'last' mode, use unique_pos_indices to pick values
        # from a single point within each cluster. The same behavior
        # is expected for the _LAST_KEYS
        if mode == 'last' or key in _LAST_KEYS:
            data[key] = item[unique_pos_indices]
            continue

        # For 'mean' mode, the attributes will be aggregated
        # depending on their nature.

        # If the attribute is a boolean, temporarily convert to integer
        # to facilitate aggregation
        is_item_bool = item.dtype == torch.bool
        if is_item_bool:
            item = item.int()

        # For keys requiring a voting scheme or a histogram
        if key in _VOTING_KEYS or key in bins.keys():
            voting = key not in bins.keys()
            n_bins = item.max() + 1 if voting else bins[key]
            hist = atomic_to_histogram(item, cluster, n_bins=n_bins)
            data[key] = hist.argmax(dim=-1) if voting else hist

        # Standard behavior, where attributes are simply
        # averaged across the clusters
        else:
            data[key] = scatter_mean(item, cluster, dim=0)

        # For normals, make sure to re-normalize the mean-normal
        if key in _NORMAL_KEYS:
            data[key] = data[key] / data[key].norm(dim=1).view(-1, 1)

        # Convert back to boolean if need be
        if is_item_bool:
            data[key] = data[key].bool()

    return data


class SampleXYTiling(Transform):
    """Tile the input Data along the XY axes and select only a given
    tile. This is useful to reduce the size of very large clouds at
    preprocessing time.

    :param x: int
        x coordinate of the sample in the tiling grid
    :param y: int
        x coordinate of the sample in the tiling grid
    :param tiling: int or tuple(int, int)
        Number of tiles in the grid in each direction. If a tuple is
        passed, each direction can be tiled independently
    """

    def __init__(self, x=0, y=0, tiling=2):
        tiling = (tiling, tiling) if isinstance(tiling, int) else tiling
        assert 0 <= x < tiling[0]
        assert 0 <= y < tiling[1]
        self.tiling = torch.as_tensor(tiling)
        self.x = x
        self.y = y

    def _process(self, data):
        # Compute the xy coordinates in the tiling grid, for each point
        xy = data.pos[:, :2].clone().view(-1, 2)
        xy -= xy.min(dim=0).values.view(1, 2)
        xy /= xy.max(dim=0).values.view(1, 2)
        xy = xy.clip(min=0, max=1) * self.tiling.view(1, 2)
        xy = xy.long()

        # Select only the points in the desired tile
        idx = torch.where((xy[:, 0] == self.x) & (xy[:, 1] == self.y))[0]

        return data.select(idx)[0]


class QuantizePointCoordinates(Transform):

    """
    Quantization of point coordinates.

    Quantized coordinates cannot be computed during preprocessing,
    because of the augmentations.

    Even if using the same voxel size during preprocessing GridSampling3D,
    this transform may drop points.

    :param size: float
        if size <= 0, it will skip the transform
    :param allow_negative_coords: bool
        If True, some coordinates will be negative if some pos values are negative.
        It is needed to map coordinates back to approximate pos values.

        If False, the coordinates will be shifted to be positive.
        This is needed to avoid some overhead of torchsparse.
    """
    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    # Remark : GridSampling3D operates on Data objects, not NAG
    def __init__(self, size, allow_negative_coords=False):
        self.grid_size = size
        self.allow_negative_coords = allow_negative_coords


    def _process(self, nag_in):
        if self.grid_size <= 0:
            return nag_in

        data = nag_in[0]

        # Convert point coordinates to the voxel grid coordinates
        coords = torch.round((data.pos) / self.grid_size)

        # Match each point with a voxel identifier
        if 'batch' not in data:
            cluster = grid_cluster(coords, torch.ones(3, device=coords.device))
        else:
            cluster = voxel_grid(coords, 1, data.batch)

        # Reindex the clusters to make sure the indices used are
        # consecutive. Basically, we do not want cluster indices to span
        # [0, i_max] without all in-between indices to be used, because
        # this will affect the speed and output size of torch_scatter
        # operations
        cluster, unique_pos_indices = consecutive_cluster(cluster)

        nag_out = nag_in.select(0, unique_pos_indices)
        coords = coords[unique_pos_indices]

        # Shift coordinates to positive integer to avoid negatives coordinates
        # (in order to avoid some overhead of torchsparse)
        if not self.allow_negative_coords:
            coords -= torch.min(coords, dim=0, keepdim=True).values

        nag_out[0].coords = coords

        return nag_out


class SampleRecursiveMainXYAxisTiling(Transform):
    """Tile the input Data by recursively splitting the points along
    their principal XY direction and select only a given tile. This is
    useful to reduce the size of very large clouds at preprocessing
    time, when clouds are not XY-aligned or have non-trivial geometries.

    :param x: int
        x coordinate of the sample in the tiling structure. The tiles
        are "lexicographically" ordered, with the points lying below the
        median of each split considered before those above the median
    :param steps: int
        Number of splitting steps. By construction, the total number of
        tiles is 2**steps
    """

    def __init__(self, x=0, steps=2):
        assert 0 <= x < 2 ** steps
        self.steps = steps
        self.x = x

    def _process(self, data):
        # Nothing to do if less than 1 step required
        if self.steps <= 0:
            return data

        # Recursively split the data
        for p in self.binary_tree_path:
            data = self.split_by_main_xy_direction(data, left=not p, right=p)

        return data

    @property
    def binary_tree_path(self):
        # Converting x to a binary number gives the solution !
        path = bin(self.x)[2:]

        # Prepend with zeros to build path of length steps
        path = (self.steps - len(path)) * '0' + path

        # Convert string of 0 and 1 to list of booleans
        return [bool(int(i)) for i in path]

    @staticmethod
    def split_by_main_xy_direction(data, left=True, right=True):
        assert left or right, "At least one split must be returned"

        # Find the main XY direction and orient it along the x+ halfspace,
        # for repeatability
        v = SampleRecursiveMainXYAxisTiling.compute_main_xy_direction(data)
        if v[0] < 0:
            v *= -1

        # Project points along this direction and split around the median
        proj = (data.pos[:, :2] * v.view(1, -1)).sum(dim=1)
        mask = proj < proj.median()

        if left and not right:
            return data.select(mask)[0]
        if right and not left:
            return data.select(~mask)[0]
        return data.select(mask)[0], data.select(~mask)[0]

    @staticmethod
    def compute_main_xy_direction(data):
        # Work on local copy
        data = Data(pos=data.pos.clone())

        # Compute a voxel size to aggressively sample the data
        xy = data.pos[:, :2]
        xy -= xy.min(dim=0).values.view(1, -1)
        voxel = xy.max() / 100

        # Set Z to 0, we only want to compute the principal components in XY
        data.pos[:, 2] = 0

        # Voxelize
        data = GridSampling3D(size=voxel)(data)

        # Search first principal component
        idx = torch.zeros_like(data.pos[:, 0], dtype=torch.long)
        v = scatter_pca(data.pos, idx)[1][0, :2, -1]

        return v


class SampleSubNodes(Transform):
    """Sample elements at `low`-level, based on which segment they
    belong to at `high`-level.

    The sampling operation is run without replacement and each segment
    is sampled at least `n_min` and at most `n_max` times, within the
    limits allowed by its actual size.

    Optionally, a `mask` can be passed to filter out some `low`-level
    points.

    :param high: int
        Partition level of the segments we want to sample. By default,
        `high=1` to sample the level-1 segments
    :param low: int
        Partition level we will sample from, guided by the `high`
        segments. By default, `low=0` to sample the level-0 points.
        `low=-1` is accepted when level-0 has a `sub` attribute (i.e.
        level-0 points are themselves segments of `-1` level absent
        from the NAG object).
        Setting `low=high` will turn this transform into an Identity
    :param n_max: int
        Maximum number of `low`-level elements to sample in each
        `high`-level segment
    :param n_min: int
        Minimum number of `low`-level elements to sample in each
        `high`-level segment, within the limits of its size (i.e. no
        oversampling)
    :param mask: list, np.ndarray, torch.LongTensor, torch.BoolTensor
        Indicates a subset of `low`-level elements to consider. This
        allows ignoring
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, high=1, low=0, n_max=32, n_min=16, mask=None):
        assert isinstance(high, int)
        assert isinstance(low, int)
        assert isinstance(n_max, int)
        assert isinstance(n_min, int)
        self.high = high
        self.low = low
        self.n_max = n_max
        self.n_min = n_min
        self.mask = mask

    def _process(self, nag):
        # Skip if low and high levels are the same.
        # This is useful to turn this transform into an Identity
        if self.low == self.high:
            return nag

        idx = nag.get_sampling(
            high=self.high,
            low=self.low,
            n_max=self.n_max,
            n_min=self.n_min,
            return_pointers=False)
        return nag.select(self.low, idx)


class SampleSegments(Transform):
    """Remove randomly-picked nodes from each segment-level, that is
    from each available level 1+.
    This operation relies on `NAG.select()` to maintain index consistency
    across the NAG levels.

    Note: we do not directly prune atom-level points, see `SampleSubNodes`
    for that. For speed consideration, it is recommended to use
    `SampleSubNodes` first before `SampleSegments`, to minimize the
    number of level-0 points to manipulate.

    :param ratio: float or list(float)
        Portion of nodes to be dropped. A list may be passed to prune
        NAG 1+ levels with different probabilities
    :param by_size: bool
        If True, the segment size will affect the chances of being
        dropped out. The smaller the segment, the greater its chances
        to be dropped
    :param by_class: bool
        If True, the classes will affect the chances of being
        dropped out. The more frequent the segment class, the greater
        its chances to be dropped
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, ratio=0.2, by_size=False, by_class=False):
        assert isinstance(ratio, list) and all(0 <= r < 1 for r in ratio) \
               or (0 <= ratio < 1)
        self.ratio = ratio
        self.by_size = by_size
        self.by_class = by_class

    def _process(self, nag):
        if not isinstance(self.ratio, list):
            ratio = [self.ratio] * (nag.end_i_level - max(0, nag.start_i_level - 1))
        else:
            ratio = self.ratio

        # Drop some nodes from each NAG level. Note that we start
        # dropping from the highest to the lowest level, to accelerate
        # sampling
        device = nag.device
        for i_level in range(nag.end_i_level, max(0, nag.start_i_level-1), -1):

            # Negative max_ratios prevent dropout
            if ratio[i_level - 1] <= 0:
                continue

            # Prepare sampling
            num_nodes = nag[i_level].num_nodes
            num_keep = num_nodes - int(num_nodes * ratio[i_level - 1])

            # Initialize all segments with the same weights
            weights = torch.ones(num_nodes, device=device)

            # Compute per-segment weights solely based on the segment
            # size. This is biased towards preserving large segments in
            # the sampling
            if self.by_size:
                node_size = nag.get_sub_size(i_level, low=0)
                size_weights = node_size ** 0.333
                size_weights /= size_weights.sum()
                weights += size_weights

            # Compute per-class weights based on class frequencies in
            # the current NAG and give a weight to each segment
            # based on the rarest class it contains. This is biased
            # towards sampling rare classes
            if self.by_class and nag[i_level].y is not None:
                counts = nag[i_level].y.sum(dim=0).sqrt()
                scores = 1 / (counts + 1)
                scores /= scores.sum()
                mask = nag[i_level].y.gt(0)
                class_weights = (mask * scores.view(1, -1)).max(dim=1).values
                class_weights /= class_weights.sum()
                weights += class_weights.squeeze()

            # Normalize the weights again, in case size or class weights
            # were added
            weights /= weights.sum()

            # Generate sampling indices
            idx = torch.multinomial(weights, num_keep, replacement=False)

            # Select the nodes and update the NAG structure accordingly
            nag = nag.select(i_level, idx)

        return nag


class BaseSampleSubgraphs(Transform):
    """Base class for sampling subgraphs from a NAG. It randomly picks
    `k` seed nodes from `i_level`, from which `k` subgraphs can be
    grown. Child classes must implement `_sample_subgraphs()` to
    describe how these subgraphs are built. Optionally, the see sampling
    can be driven by their class, or their size, using `by_class` and
    `by_size`, respectively.

    This operation relies on `NAG.select()` to maintain index
    consistency across the NAG levels.

    :param i_level: int
        Partition level we want to pick from. By default, `i_level=-1`
        will sample the highest level of the input NAG
    :param k: int
        Number of sub-graphs/seeds to pick
    :param by_size: bool
        If True, the segment size will affect the chances of being
        selected as a seed. The larger the segment, the greater its
        chances to be picked
    :param by_class: bool
        If True, the classes will affect the chances of being
        selected as a seed. The scarcer the segment class, the greater
        its chances to be selected
    :param use_batch: bool
        If True, the 'Data.batch' attribute will be used to guide seed
        sampling across batches. More specifically, if the input NAG is
        a NAGBatch made up of multiple NAGs, the subgraphs will be
        sampled in a way that guarantees each NAG is sampled from.
        Obviously enough, if `k < batch.max() + 1`, not all NAGs will be
        sampled from
    :param disjoint: bool
        If True, subgraphs sampled from the same NAG will be separated
        as distinct NAGs themselves. Instead, when `disjoint=False`,
        subgraphs sampled in the same NAG will be long the same NAG.
        Hence, if two subgraphs share a node, they will be connected. It
        is worth noting that if `disjoint=True`, the output NAGBatch
        will consider each subgraph as a separate batch item. Meanwhile,
        if `disjoint=False` and the input is already a NAGBatch, the
        returned sampled nodes will still carry the 'batch' attribute of
        the input, allowing to distinguish between the different input
        graphs
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(
            self,
            i_level=1,
            k=1,
            by_size=False,
            by_class=False,
            use_batch=True,
            disjoint=True):
        self.i_level = i_level
        self.k = k
        self.by_size = by_size
        self.by_class = by_class
        self.use_batch = use_batch
        self.disjoint = disjoint

    def _process(self, nag):
        device = nag.device

        # Skip if i_level is None or k<=0. This may be useful to turn
        # this transform into an Identity, if need be
        if self.i_level is None or self.k <= 0:
            return nag

        # Initialization
        if self.i_level == -1:
            i_level = nag.end_i_level
        elif nag.start_i_level <= self.i_level < nag.absolute_num_levels :
            i_level = self.i_level
        else :
            raise ValueError(
                f"Invalid i_level: {self.i_level}. Must be in "
                f"range [{nag.start_i_level}, {nag.absolute_num_levels-1}],"
                f"\nor -1 for the highest level available.")

        k = self.k if self.k < nag[i_level].num_nodes \
            else 1

        # Initialize all segments with the same weights
        weights = torch.ones(nag[i_level].num_nodes, device=device)

        # Compute per-segment weights solely based on the segment
        # size. This is biased towards preserving large segments in
        # the sampling
        if self.by_size:
            node_size = nag.get_sub_size(i_level, low=0)
            size_weights = node_size ** 0.333
            size_weights /= size_weights.sum()
            weights += size_weights

        # Compute per-class weights based on class frequencies in
        # the current NAG and give a weight to each segment
        # based on the rarest class it contains. This is biased
        # towards sampling rare classes
        if self.by_class and nag[i_level].y is not None:
            counts = nag[i_level].y.sum(dim=0).sqrt()
            scores = 1 / (counts + 1)
            scores /= scores.sum()
            mask = nag[i_level].y.gt(0)
            class_weights = (mask * scores.view(1, -1)).max(dim=1).values
            class_weights /= class_weights.sum()
            weights += class_weights.squeeze()

        # Normalize the weights again, in case size or class weights
        # were added
        weights /= weights.sum()

        # Generate sampling indices. If the Data object has a 'batch'
        # attribute and 'self.use_batch', use it to guide the sampling
        # across the batches
        batch = getattr(nag[i_level], 'batch', None)
        if batch is not None and self.use_batch:
            idx_list = []

            # Shuffle the order of the batch items, to not always
            # prioritize the ones with the lowest batch index
            batch_indices = batch.unique()
            num_batch = batch_indices.numel()
            batch_indices = batch_indices[torch.randperm(num_batch)]

            # Estimate the maximum number of items to pick in each batch
            num_sampled = 0
            k_batch = torch.div(k, num_batch, rounding_mode='floor')
            k_batch = k_batch.maximum(torch.ones_like(k_batch))

            for i_step, i_batch in enumerate(batch_indices):

                # Try to sample all NAGs in the batch as evenly as
                # possible, within the constraints of k and
                # num_batch. Here, if we are sampling for the last batch
                # item, we fix k_batch to be all remaining "sampling
                # credit"
                if i_step >= num_batch - 1:
                    k_batch = k - num_sampled

                # Compute the sampling indices for the NAG at hand
                mask = torch.where(i_batch == batch)[0]
                idx_ = torch.multinomial(
                    weights[mask], k_batch, replacement=False)
                idx_list.append(mask[idx_])

                # Update number of sampled subgraphs
                num_sampled += k_batch
                if num_sampled >= k:
                    break

            # Aggregate sampling indices
            idx_seed = torch.cat(idx_list)
        else:
            idx_seed = torch.multinomial(weights, k, replacement=False)

        # Sample the NAG and allow subgraphs sharing the same nodes to
        # be connected if disjoint=False
        if self.disjoint:
            idx_subgraphs = [
                self._sample_subgraphs_from_seeds(nag, i_level, i.view(1))
                for i in idx_seed]

            # If one the returned subgraph samplings are None, this is
            # the signal we use to indicate not to sample at all and
            # return the input untouched. This is what will happen
            # downstream when setting `idx_subgraphs=None`, whereas
            # `idx_subgraphs=[None, ..., None]` would create a
            # `NAGBatch` of copies of the input
            if all(idx is None for idx in idx_subgraphs):
                idx_subgraphs = None
        else:
            idx_subgraphs = self._sample_subgraphs_from_seeds(
                nag, i_level, idx_seed)

        # Select the chosen subgraphs and update the NAG structure
        # accordingly
        if isinstance(idx_subgraphs, list):
            nag_subgraphs = NAGBatch.from_nag_list([
                nag.select(i_level, idx) for idx in idx_subgraphs])
        else:
            nag_subgraphs = nag.select(i_level, idx_subgraphs)

        return nag_subgraphs

    def _sample_subgraphs_from_seeds(self, nag, i_level, idx_seed):
        """Given a set of seed nodes `idx_seed`, sample subgraphs around
        them. The returned subgraphs are expressed as node indices.
        """
        raise NotImplementedError


class SampleKHopSubgraphs(BaseSampleSubgraphs):
    """Randomly pick segments from `i_level`, along with their `hops`
    neighbors. This can be thought as a spherical sampling in the graph
    of i_level.

    This operation relies on `NAG.select()` to maintain index
    consistency across the NAG levels.

    If the input is a `NAGBatch`, the corresponding subgraphs are
    assumed to be disjoint.

    Note: we do not directly sample level-0 points, see `SampleSubNodes`
    for that. For speed consideration, it is recommended to use
    `SampleSubNodes` first before `SampleKHopSubgraphs`, to minimize the
    number of level-0 points to manipulate.

    :param hops: int
        Number of hops ruling the neighborhood size selected around the
        seed nodes. If `hops` is `None` or `hops < 0`, this transform
        returns the input `NAG` without modification
    :param i_level: int
        Partition level we want to pick from. By default, `i_level=-1`
        will sample the highest level of the input NAG
    :param k: int
        Number of sub-graphs/seeds to pick
    :param by_size: bool
        If True, the segment size will affect the chances of being
        selected as a seed. The larger the segment, the greater its
        chances to be picked
    :param by_class: bool
        If True, the classes will affect the chances of being
        selected as a seed. The scarcer the segment class, the greater
        its chances to be selected
    :param use_batch: bool
        If True, the 'Data.batch' attribute will be used to guide seed
        sampling across batches. More specifically, if the input NAG is
        a NAGBatch made up of multiple NAGs, the subgraphs will be
        sampled in a way that guarantees each NAG is sampled from.
        Obviously enough, if `k < batch.max() + 1`, not all NAGs will be
        sampled from
    :param disjoint: bool
        If True, subgraphs sampled from the same NAG will be separated
        as distinct NAGs themselves. Instead, when `disjoint=False`,
        subgraphs sampled in the same NAG will be long the same NAG.
        Hence, if two subgraphs share a node, they will be connected
    """

    def __init__(
            self,
            hops=2,
            i_level=1,
            k=1,
            by_size=False,
            by_class=False,
            use_batch=True,
            disjoint=False):
        super().__init__(
            i_level=i_level,
            k=k,
            by_size=by_size,
            by_class=by_class,
            use_batch=use_batch,
            disjoint=disjoint)
        self.hops = hops

    def _sample_subgraphs_from_seeds(self, nag, i_level, idx_seed):
        """Given a set of seed nodes `idx_seed`, sample subgraphs around
        them. The returned subgraphs are expressed as node indices.
        """
        if self.hops is None or self.hops < 0:
            return None

        assert nag[i_level].has_edges, \
            "Expected Data object to have edges for k-hop subgraph sampling"

        # Convert the graph to undirected graph. This is needed because
        # it is likely that the graph has been trimmed (see
        # `src.utils.to_trimmed`), in which case the trimmed edge
        # direction would affect the k-hop search
        edge_index = to_undirected(nag[i_level].edge_index).long()

        # Search the k-hop neighbors of the sampled nodes
        idx_subgraph = k_hop_subgraph(
            idx_seed,
            self.hops,
            edge_index,
            num_nodes=nag[i_level].num_nodes)[0]

        return idx_subgraph


class SampleRadiusSubgraphs(BaseSampleSubgraphs):
    """Randomly pick segments from `i_level`, along with their
    spherical or cylindrical neighborhood of given radius.

    This operation relies on `NAG.select()` to maintain index
    consistency across the NAG levels.

    If the input is a `NAGBatch` or a `NAG` with `Data` objects holding,
    a 'batch' attribute, it will be used to guide the search. More
    specifically, nodes that are spatially close but not in the same
    batch item will not be considered as neighbors.

    Note: we do not directly sample level-0 points, see `SampleSubNodes`
    for that. For speed consideration, it is recommended to use
    `SampleSubNodes` first before `SampleRadiusSubgraphs`, to minimize
    the number of level-0 points to manipulate.

    :param r: float
        Radius used for spherical (or cylindrical) sampling. If `r` is
        `None` or `r <= 0`, this transform returns the input `NAG`
        without modification
    :param k_max: int
        Maximum number of neighbors to consider when sampling within
        radius `r`. This is needed to avoid memory explosion. If more
        than `k_max` neighbors are found within `r`, only the closest
        will be kept
    :param i_level: int
        Partition level we want to pick from. By default, `i_level=-1`
        will sample the highest level of the input NAG
    :param k: int
        Number of sub-graphs/seeds to pick
    :param by_size: bool
        If True, the segment size will affect the chances of being
        selected as a seed. The larger the segment, the greater its
        chances to be picked
    :param by_class: bool
        If True, the classes will affect the chances of being
        selected as a seed. The scarcer the segment class, the greater
        its chances to be selected
    :param use_batch: bool
        If True, the 'Data.batch' attribute will be used to guide seed
        sampling across batches. More specifically, if the input NAG is
        a NAGBatch made up of multiple NAGs, the subgraphs will be
        sampled in a way that guarantees each NAG is sampled from.
        Obviously enough, if `k < batch.max() + 1`, not all NAGs will be
        sampled from
    :param disjoint: bool
        If True, subgraphs sampled from the same NAG will be separated
        as distinct NAGs themselves. Instead, when `disjoint=False`,
        subgraphs sampled in the same NAG will be long the same NAG.
        Hence, if two subgraphs share a node, they will be connected
    :param cylindrical: bool
        If True the sampling will not be based on sphere of radius `r`
        but on a cylinder of axis Z. This is typically adapted for
        outdoor environments
    """

    def __init__(
            self,
            r=2,
            k_max=10000,
            i_level=1,
            k=1,
            by_size=False,
            by_class=False,
            use_batch=True,
            disjoint=False,
            cylindrical=False):
        super().__init__(
            i_level=i_level,
            k=k,
            by_size=by_size,
            by_class=by_class,
            use_batch=use_batch,
            disjoint=disjoint)
        self.r = r
        self.k_max = k_max
        self.cylindrical = cylindrical

    def _sample_subgraphs_from_seeds(self, nag, i_level, idx_seed):
        """Given a set of seed nodes `idx_seed`, sample subgraphs around
        them. The returned subgraphs are expressed as node indices.
        """
        # Skip if r<=0. This may be useful to turn this transform into
        # an Identity, if need be
        if self.r is None or self.r <= 0:
            return None

        # Neighbors are searched using the node coordinates. This is not
        # the optimal search for cluster-cluster distances, but is the
        # fastest for our needs here. If need be, one could make this
        # search more accurate using something like:
        # `src.utils.neighbors.cluster_radius_nn_graph`

        # Recover batch indices, if any. This is needed to prevent
        # points from different batch items to be associated with one
        # another
        batch = getattr(nag[i_level], 'batch', None)
        batch_query = batch[idx_seed] if batch is not None else None

        # Prepare the inputs for the neighborhood search
        mask = torch.tensor([[1, 1, not self.cylindrical]], device=nag.device)
        pos_search = nag[i_level].pos * mask
        pos_query = nag[i_level].pos[idx_seed] * mask

        # Neighbor search. This search scenario is a bit special
        # compared to other k-NN and radius-NN elsewhere in the project.
        # Here, we have a very small pos_query set, a large pos_search,
        # and a large k. FRNN works well for small k values, but as soon
        # as k becomes too large, brute force seems faster. However,
        # this will not scale if we have large k AND large query set.
        # But for now, for our use case here, we can assume the query
        # set to remain small while k is large.
        # NB: we need to specify k beforehand here, which is not super
        # convenient, as it forces the user to pass
        # SampleRadiusSubgraphs(k_max=...). But this is needed to avoid
        # potential memory explosion here...
        if self.k_max > 100:
            neighbors = knn_brute_force(
                pos_search,
                pos_query,
                self.k_max,
                r_max=self.r,
                batch_search=batch,
                batch_query=batch_query)[0]
        else:
            neighbors = knn_2(
                pos_search,
                pos_query,
                self.k_max,
                r_max=self.r,
                batch_search=batch,
                batch_query=batch_query)[0]

        # Recover the selected node indices
        idx_subgraph = neighbors[neighbors >= 0].unique()

        return idx_subgraph


class SampleEdges(Transform):
    """Sample edges based on which source node they belong to.

    The sampling operation is run without replacement and each source
    segment has at least `n_min` and at most `n_max` edges, within the
    limits allowed by its actual number of edges.

    :param level: int or str
        Level at which to sample edges. Can be an int or a str. If the
        latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param n_min: int or List(int)
        Minimum number of edges for each node, within the limits of its
        input number of edges
    :param n_max: int or List(int)
        Maximum number of edges for each node
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, level='1+', n_min=16, n_max=32):
        assert isinstance(level, (int, str))
        assert isinstance(n_min, (int, list))
        assert isinstance(n_max, (int, list))
        self.level = level
        self.n_min = n_min
        self.n_max = n_max

    def _process(self, nag):

        level_n_min = fill_list_with_string_indexing(
            level=self.level,
            default=-1,
            value=self.n_min,
            output_length=nag.absolute_num_levels,
            start_index=nag.start_i_level)

        level_n_max = fill_list_with_string_indexing(
            level=self.level,
            default=-1,
            value=self.n_max,
            output_length=nag.absolute_num_levels,
            start_index=nag.start_i_level)

        transforms = []
        for n_min, n_max in zip(level_n_min, level_n_max):
            if n_min < 0 or n_max < 0:
                transforms.append(None)
            else :
                transforms.append(lambda data : self._process_single_level(data, n_min, n_max))

        nag.apply_data_transform(transforms)

        return nag

    @staticmethod
    def _process_single_level(data, n_min, n_max):
        # Skip process if n_min or n_max is negative or if in put Data
        # has not edges
        if n_min < 0 or n_max < 0 or not data.has_edges:
            return data

        # Compute a sampling for the edges, based on the source node
        # they belong to
        idx = sparse_sample(
            data.edge_index[0],
            n_max=n_max,
            n_min=n_min,
            return_pointers=False)

        # Select edges and their attributes, if relevant
        data.edge_index = data.edge_index[:, idx]
        if data.has_edge_attr:
            data.edge_attr = data.edge_attr[idx]
        for key in data.edge_keys:
            data[key] = data[key][idx]

        return data


class RestrictSize(Transform):
    """Randomly sample nodes and edges to restrict their number within
    given limits. This is useful for stabilizing memory use of the
    model.

    :param num_nodes: int
        Maximum number of nodes. If the input has more, a subset of
        `num_nodes` nodes will be randomly sampled. No sampling if <=0
    :param num_edges: int
        Maximum number of edges. If the input has more, a subset of
        `num_edges` edges will be randomly sampled. No sampling if <=0
    """

    def __init__(self, num_nodes=0, num_edges=0):
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def _process(self, data):
        if data.num_nodes > self.num_nodes and self.num_nodes > 0:
            weights = torch.ones(data.num_nodes, device=data.device)
            idx = torch.multinomial(weights, self.num_nodes, replacement=False)
            data = data.select(idx)

        if data.num_edges > self.num_edges and self.num_edges > 0:
            weights = torch.ones(data.num_edges, device=data.device)
            idx = torch.multinomial(weights, self.num_edges, replacement=False)

            data.edge_index = data.edge_index[:, idx]
            if data.has_edge_attr:
                data.edge_attr = data.edge_attr[idx]
            for key in data.edge_keys:
                data[key] = data[key][idx]

        return data


class NAGRestrictSize(Transform):
    """Randomly sample nodes and edges to restrict their number within
    given limits. This is useful for stabilizing memory use of the
    model.

    :param num_nodes: int
        Maximum number of nodes. If the input has more, a subset of
        `num_nodes` nodes will be randomly sampled. No sampling if <=0
    :param num_edges: int
        Maximum number of edges. If the input has more, a subset of
        `num_edges` edges will be randomly sampled. No sampling if <=0
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, level='1+', num_nodes=0, num_edges=0):
        assert isinstance(level, (int, str))
        assert isinstance(num_nodes, (int, list))
        assert isinstance(num_edges, (int, list))
        self.level = level
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def _process(self, nag):

        # If 'level' is an int, we only need to process a single level
        if isinstance(self.level, int):
            return self._restrict_level(
                nag, self.level, self.num_nodes, self.num_edges)

        level_num_nodes = fill_list_with_string_indexing(
            level=self.level,
            default=-1,
            value=self.num_nodes,
            output_length=nag.absolute_num_levels,
            start_index=nag.start_i_level)

        level_num_edges = fill_list_with_string_indexing(
            level=self.level,
            default=-1,
            value=self.num_edges,
            output_length=nag.absolute_num_levels,
            start_index=nag.start_i_level)

        level_num_nodes = level_num_nodes[nag.start_i_level :]
        level_num_edges = level_num_edges[nag.start_i_level :]

        for i_level, num_nodes, num_edges in zip(
                nag.level_range, level_num_nodes, level_num_edges):
            nag = self._restrict_level(nag, i_level, num_nodes, num_edges)

        return nag

    @staticmethod
    def _restrict_level(nag, i_level, num_nodes, num_edges):

        if nag[i_level].num_nodes > num_nodes and num_nodes > 0:
            weights = torch.ones(nag[i_level].num_nodes, device=nag.device)
            idx = torch.multinomial(weights, num_nodes, replacement=False)
            nag = nag.select(i_level, idx)

        if nag[i_level].num_edges > num_edges and num_edges > 0:
            weights = torch.ones(nag[i_level].num_edges, device=nag.device)
            idx = torch.multinomial(weights, num_edges, replacement=False)

            nag[i_level].edge_index = nag[i_level].edge_index[:, idx]
            if nag[i_level].has_edge_attr:
                nag[i_level].edge_attr = nag[i_level].edge_attr[idx]
            for key in nag[i_level].edge_keys:
                nag[i_level][key] = nag[i_level][key][idx]

        return nag
