import torch
from src.data import NAG, InstanceData
from src.transforms import Transform
from src.utils import cluster_radius_nn_graph, knn_1_graph, to_trimmed
from torch_geometric.nn.pool.consecutive import consecutive_cluster


__all__ = ['NAGPropagatePointInstances', 'OnTheFlyInstanceGraph']


class NAGPropagatePointInstances(Transform):
    """Compute the instances contained in each superpoint of each level,
    provided that the first level has an 'obj' attribute holding an
    `InstanceData`.

    :param strict: bool
        If True, will raise an exception if the level Data does not have
        instance
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _NO_REPR = ['strict']

    def __init__(self, strict=False):
        self.strict = strict

    def _process(self, nag):
        # Read the instances from the first data
        obj_0 = nag[0].obj
        if obj_0 is None or not isinstance(obj_0, InstanceData):
            if not self.strict:
                return nag
            raise ValueError(f"Could not find any InstanceData in `nag[0].obj`")

        for i_level in range(1, nag.num_levels):
            super_index = nag.get_super_index(i_level)
            nag[i_level].obj = obj_0.merge(super_index)

        return nag


class OnTheFlyInstanceGraph(Transform):
    """Compute the non-oriented graph used for instance and panoptic
    segmentation.

    We choose the following assignment rule:
      - each superpoint is assigned to the instance it overlaps the most

    Importantly, one could think of other assignment rules, such as the
    maximum IoU, for instance. But the latter would not work for
    over-segmented scenes with small superpoints and very large 'stuff'
    instances. We favor the overlap-size rule due to this scenario and
    for its simplicity.

    It is recommended to call this transform only AFTER all geometric
    transformations and sampling have been applied to the batch. This
    is typically important when using samplings of subgraphs, where the
    position of the entire clusters are maintained, even after being
    cropped. Such behavior may damage the instance centroid estimation
    step.

    :param level: int
        Partition level at which to compute the instance graph. Setting
        `level=-1` or `level=None` will skip the present Transform (can
        be useful for integrating this Transform in a pipeline and
        optionally skip it)
    :param num_classes: int
        Number of classes in the dataset. Specifying `num_classes`
        allows identifying 'void' labels. By convention, we assume
        `y ∈ [0, self.num_classes-1]` ARE ALL VALID LABELS (i.e. not
        'ignored', 'void', 'unknown', etc), while `y < 0` AND
        `y >= self.num_classes` ARE VOID LABELS. Void data is dealt
        with following https://arxiv.org/abs/1801.00868 and
        https://arxiv.org/abs/1905.01220
    :param adjacency_mode: str
        Method used to compute search for adjacent nodes. If 'available'
        the already-existing graph in the input's 'edge_index' will be
        used. If 'radius', the `radius` parameter will be used to search
        for all neighboring clusters with points within `radius` of each
        other. If 'radius-centroid', the `radius` parameter will be used
        to search for all neighboring clusters solely based on their
        centroid position. This is likely faster but less accurate than
        'radius'
    :param k_max: int
        Maximum number of neighbors per cluster if `adjacency_mode`
        calls for it
    :param radius: float
        Radius used for neighbor search if `adjacency_mode` calls for it
    :param use_batch: bool
        If True, the 'NAG[level].batch' attribute will be used to
        guide neighbor search if `adjacency_mode` calls for it. More
        specifically, if the input NAG is a NAGBatch made up of multiple
        NAGs, the neighbor search will ensure that clusters from
        different batch items cannot be neighbors. It is recommended to
        keep the default `use_batch=True`
    :param centroid_mode: str
        Method used to estimate the centroids. 'iou' will weigh down
        the centroids of the clusters overlapping each instance by
        their IoU. 'ratio-product' will use the product of the size
        ratios of the overlap wrt the cluster and wrt the instance
    :param centroid_level: int
        Partition level to use to estimate the centroids. The purer the
        partition, the better the estimation. But the larger the
        partition, the slower the computation
    :param smooth_affinity: bool
        If True, the affinity score computed for each edge will
        follow the 'smooth' formulation:
        `(overlap_i_obj_j / size_i + overlap_j_obj_i / size_j) / 2`
        for the edge `(i, j)`, where `obj_i` designates the target
        instance of `i`. If False, the affinity will be computed
        with the simpler formulation: `obj_i == obj_j`
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _ADJACENCY_MODES = ['available', 'radius', 'radius-centroid']
    _CENTROID_MODES = ['iou', 'ratio-product']

    def __init__(
            self,
            level=1,
            num_classes=None,
            adjacency_mode='radius',
            k_max=30,
            radius=1,
            use_batch=True,
            centroid_mode='iou',
            centroid_level=1,
            smooth_affinity=True):
        assert adjacency_mode.lower() in self._ADJACENCY_MODES, \
            f"Expected 'mode' to be one of {self._ADJACENCY_MODES}"
        assert centroid_mode.lower() in self._CENTROID_MODES, \
            f"Expected 'mode' to be one of {self._CENTROID_MODES}"
        self.level = level
        self.num_classes = num_classes
        self.adjacency_mode = adjacency_mode.lower()
        self.k_max = k_max
        self.radius = radius
        self.use_batch = use_batch
        self.centroid_mode = centroid_mode.lower()
        self.centroid_level = centroid_level
        self.smooth_affinity = smooth_affinity

    def _process(self, nag):
        # Skip the transform. This mechanism can be useful for skipping
        # this Transform in a pipeline
        if self.level is None or self.level < 0:
            return nag

        data = nag[self.level]

        # Build the edges on which the graph optimization will be run
        # for instance or panoptic segmentation.
        # Use the already-existing graph in `edge_index`
        if self.adjacency_mode == 'available':
            obj_edge_index = data.edge_index

        # Compute the neighbors based on the distances between the
        # points they hold
        elif self.adjacency_mode == 'radius':
            # TODO: accelerate with subsampling ?
            super_index = nag.get_super_index(self.level, low=0)
            obj_edge_index, _ = cluster_radius_nn_graph(
                nag[0].pos,
                super_index,
                k_max=self.k_max,
                gap=self.radius,
                batch=nag[self.level].batch if self.use_batch else None)

        # Compute the neighbors solely based on the clusters' centroids
        elif self.adjacency_mode == 'radius-centroid':
            obj_edge_index, _ = knn_1_graph(
                nag[self.level].pos,
                self.k_max,
                r_max=self.radius,
                batch=nag[self.level].batch if self.use_batch else None)

        else:
            raise NotImplementedError

        # If, for some reason, the graph is None, we convert it to an
        # empty torch_geometric-friendly `edge_index`-like format
        if obj_edge_index is None:
            obj_edge_index = torch.empty(
                2, 0, dtype=torch.long, device=data.device)

        # If the Data does not contain any InstanceData with instance
        # annotations, set the target object positions and edge
        # affinities to None and save the trimmed instance graph
        if data.obj is None:
            data.obj_edge_index = to_trimmed(obj_edge_index)
            data.obj_edge_affinity = None
            data.obj_pos = None
            nag._list[self.level] = data
            return nag

        # Compute the trimmed graph and the edge affinity scores
        data.obj_edge_index, data.obj_edge_affinity = data.obj.instance_graph(
            obj_edge_index,
            num_classes=self.num_classes,
            smooth_affinity=self.smooth_affinity)

        # Compute the superpoint target instance centroid position
        # NB: this is a proxy method assuming nag[0] is pure-enough
        i_level = min(self.centroid_level, nag.num_levels - 1)
        obj_pos, obj_idx = nag[i_level].estimate_instance_centroid(
            mode=self.centroid_mode)

        # Find the target instance for each superpoint: the instance it
        # has the biggest overlap with
        sp_obj_idx = data.obj.major(num_classes=self.num_classes)[0]

        # Recover, for each superpoint, the instance position. Since the
        # `estimate_instance_centroid()` output is sorted by increasing
        # obj indices, `consecutive_cluster()` allows us to convert
        # `obj_idx` into proper indices to gather object positions from
        # `obj_pos`
        joint_obj_idx = torch.cat((sp_obj_idx, obj_idx))
        joint_obj_idx_consec = consecutive_cluster(joint_obj_idx)[0]
        sp_obj_idx_consec = joint_obj_idx_consec[:sp_obj_idx.numel()]
        data.obj_pos = obj_pos[sp_obj_idx_consec]

        # Save in the data in the NAG structure
        nag._list[self.level] = data

        return nag
