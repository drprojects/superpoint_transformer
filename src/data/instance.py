import torch
import numpy as np
from torch.nn.functional import one_hot
from typing import List, Tuple, Union
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter_max, scatter_sum

from src.data.csr import CSRData, CSRBatch
from src.utils import tensor_idx, is_dense, has_duplicates, to_trimmed


__all__ = ['InstanceData', 'InstanceBatch']


class InstanceData(CSRData):
    """Child class of CSRData to simplify some common operations
    dedicated to instance labels clustering. In particular, this data
    structure stores the cluster-object overlaps: for each cluster (i.e.
    segment, superpoint, node in the superpoint graph, etc), we store
    all the object instances with which it overlaps. Concretely, for
    each cluster-object pair, we store:
      - `obj`: the object's index
      - `count`: the number of points in the cluster-object overlap
      - `y`: the object's semantic label

    Importantly, each object in the InstanceData is expected to be
    described by a unique index in `obj', regardless of its actual
    semantic class. It is not required for the object instances to be
    contiguous in `[0, obj_max]`, although enforcing it may have
    beneficial downstream effects on memory and I/O times. Finally,
    when two InstanceData are batched in an InstanceBatch, the `obj'
    indices will be updated to avoid collision between the batch items.

    :param pointers: torch.LongTensor
        Pointers to address the data in the associated value tensors.
        `values[Pointers[i]:Pointers[i+1]]` hold the values for the ith
        cluster. If `dense=True`, the `pointers` are actually the dense
        indices to be converted to pointer format.
    :param obj: torch.LongTensor
        Object index for each cluster-object pair. Assumes there are
        NO DUPLICATE CLUSTER-OBJECT pairs in the input data, unless
        'dense=True'.
    :param count: torch.LongTensor
        Number of points in the overlap for each cluster-object pair.
    :param y: torch.LongTensor
        Semantic label the object for each cluster-object pair. By
        definition, we assume the objects to be SEMANTICALLY PURE. For
        that reason, we only store a single semantic label for objects,
        as opposed to superpoints, for which we want to maintain a
        histogram of labels.
    :param dense: bool
        If `dense=True`, the `pointers` are actually the dense indices
        to be converted to pointer format. Besides, any duplicate
        cluster-obj pairs will be merged and the corresponding `count`
        will be updated.
    :param kwargs:
        Other kwargs will be ignored.
    """

    __value_keys__ = ['obj', 'count', 'y']
    __is_index_value_serialization_key__ = None

    def __init__(
            self,
            pointers: torch.Tensor,
            obj: torch.Tensor,
            count: torch.Tensor,
            y: torch.Tensor,
            dense: bool = False,
            **kwargs):
        # If the input data is passed in 'dense' format, we merge the
        # potential duplicate cluster-obj pairs before anything else.
        # NB: if dense=True, 'pointers' are not actual pointers but
        # dense cluster indices instead
        if dense:
            # Build indices to uniquely identify each cluster-obj pair
            cluster_obj_idx = pointers * (obj.max() + 1) + obj

            # Make the indices contiguous in [0, max] to alleviate
            # downstream scatter operations. Compute the cluster and obj
            # for each unique cluster_obj_idx index. These will be
            # helpful in building the cluster_idx and obj of the new
            # merged data
            cluster_obj_idx, perm = consecutive_cluster(cluster_obj_idx)
            pointers = pointers[perm]
            obj = obj[perm]
            y = y[perm]

            # Compute the actual count for each cluster-obj pair in the
            # input data
            count = scatter_sum(count, cluster_obj_idx)

        super().__init__(
            pointers, obj, count, y, dense=dense,
            is_index_value=[True, False, False])

    @classmethod
    def get_base_class(cls) -> type:
        """Helps `self.from_list()` and `self.to_list()` identify which
        classes to use for batch collation and un-collation.
        """
        return InstanceData

    @classmethod
    def get_batch_class(cls) -> type:
        """Helps `self.from_list()` and `self.to_list()` identify which
        classes to use for batch collation and un-collation.
        """
        return InstanceBatch

    @property
    def obj(self) -> torch.Tensor:
        return self.values[0]

    @obj.setter
    def obj(self, obj: torch.Tensor):
        assert obj.device == self.device, \
            f"obj is on {obj.device} while self is on {self.device}"
        self.values[0] = obj
        # if src.is_debug_enabled():
        #     self.debug()

    @property
    def count(self) -> torch.Tensor:
        return self.values[1]

    @count.setter
    def count(self, count: torch.Tensor):
        assert count.device == self.device, \
            f"count is on {count.device} while self is on {self.device}"
        self.values[1] = count
        # if src.is_debug_enabled():
        #     self.debug()

    @property
    def y(self) -> torch.Tensor:
        return self.values[2]

    @y.setter
    def y(self, y: torch.Tensor):
        assert y.device == self.device, \
            f"y is on {y.device} while self is on {self.device}"
        self.values[2] = y
        # if src.is_debug_enabled():
        #     self.debug()

    @property
    def num_clusters(self):
        return self.num_groups

    @property
    def num_overlaps(self):
        return self.num_items

    @property
    def num_obj(self):
        return self.obj.unique().numel()

    def major(
            self,
            num_classes: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the obj, count, and y of the majority instance in each
        cluster (i.e. the object with which it has the highest overlap).

        :param num_classes: int
            Number of classes in the dataset. Specifying `num_classes`
            allows identifying 'void' labels. By convention, we assume
            `y ∈ [0, self.num_classes-1]` ARE ALL VALID LABELS (i.e. not
            'ignored', 'void', 'unknown', etc), while `y < 0` AND
            `y >= self.num_classes` ARE VOID LABELS. Void data is dealt
            with following https://arxiv.org/abs/1801.00868 and
            https://arxiv.org/abs/1905.01220
        """

        # If `num_classes` was not passed, we set it to `y_max + 1`
        # (i.e. there are no 'void' objects)
        num_classes = num_classes if num_classes else self.y.max() + 1

        # Compute the cluster index for each overlap (i.e. each row in
        # self.values)
        cluster_idx = self.indices

        # Search the overlaps with void objects
        pair_is_void = (self.y < 0) | (self.y >= num_classes)

        # Search for the obj with the largest overlap, for each cluster
        x = torch.stack((self.count, self.count * ~pair_is_void)).T
        res = scatter_max(x, cluster_idx, dim=0)
        count = res[0][:, 0]
        argmax = res[1][:, 0]
        obj = self.obj[argmax]
        y = self.y[argmax]

        # If no cluster mainly overlaps with a void object, exit here
        is_major_void = (y < 0) | (y >= num_classes)
        if (~is_major_void).all():
            return obj, count, y

        # Otherwise, we need to find those clusters which overlap with
        # void, but with less than 50%. These clusters will not be
        # assigned to their main void cluster, but to their second-best
        # overlap. This way, only clusters with +50% void overlap will
        # be excluded from metrics computation, as defined in:
        # https://arxiv.org/abs/1801.00868

        # Search if any of the clusters assigned to a void object have
        # less than 50% void points
        total_count = scatter_sum(self.count, cluster_idx, dim=0)
        major_50_plus = (count / total_count) > 0.5
        if major_50_plus[is_major_void].all():
            return obj, count, y

        # Assign the clusters with less than 50% void overlap to their
        # second-best overlap
        count_no_void = res[0][:, 1]
        argmax_no_void = res[1][:, 1]
        count[is_major_void] = count_no_void[is_major_void]
        obj[is_major_void] = self.obj[argmax_no_void][is_major_void]
        y[is_major_void] = self.y[argmax_no_void][is_major_void]

        return obj, count, y

    def merge(
            self,
            idx: Union[int, List[int], torch.Tensor, np.ndarray]
    ) -> 'InstanceData':
        """Merge clusters based on `idx` and return the result in a new
        InstanceData object.

        :param idx: 1D torch.LongTensor or numpy.NDArray
            Indices of the parent cluster each cluster should be merged
            into. Must have the same size as `self.num_clusters` and
            indices must start at 0 and be contiguous.
        """
        # Make sure each cluster has a merge index and that the merge
        # indices are dense
        idx = tensor_idx(idx)
        assert idx.shape == torch.Size([self.num_clusters]), \
            f"Expected indices of shape {torch.Size([self.num_clusters])}, " \
            f"but received shape {idx.shape} instead"
        assert is_dense(idx), f"Expected contiguous indices in [0, max]"

        # Compute the merged cluster index for each cluster-obj pair
        merged_idx = idx[self.indices].long()

        # Return a new object holding the merged data.
        # NB: specifying 'dense=True' will do all the merging for us
        return self.__class__(
            merged_idx, self.obj, self.count, self.y, dense=True)

    def iou_and_size(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the Intersection over Union (IoU) and the individual
        size for each cluster-object pair in the data. This is typically
        needed for computing the Average Precision.
        """
        # Prepare the indices for sets A (i.e. predictions) and B (i.e.
        # targets). In particular, we want the indices to be contiguous
        # in [0, idx_max], to alleviate scatter operations' computation.
        # Since `self.obj` contains potentially-large and non-contiguous
        # global object indices, we update these indices locally
        a_idx = self.indices
        b_idx = consecutive_cluster(self.obj)[0]

        # Compute the size of each set and redistribute to each a-b pair
        a_size = scatter_sum(self.count, a_idx)[a_idx]
        b_size = scatter_sum(self.count, b_idx)[b_idx]

        # If self was created using `self.remove_void()`, use the
        # `self.pair_cropped_count` attribute to account for cropped
        # parts of b
        # TODO: `self.pair_cropped_count` is not accounted for in the
        #  `self.values`. InstanceBatch mechanisms will discard this
        #  value. i.e. 'pair_cropped_count' will disappear when calling
        #  `InstanceBatch.from_list` or `InstanceBatch.to_list`
        if getattr(self, 'pair_cropped_count', None) is not None:
            b_size += self.pair_cropped_count

        # Compute the IoU
        iou = self.count / (a_size + b_size - self.count)

        return iou, a_size, b_size

    def estimate_centroid(
            self,
            cluster_pos: torch.Tensor,
            mode: str = 'iou'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate the centroid position of each object, based on the
        position of the clusters.

        Based on the hypothesis that clusters are relatively
        instance-pure, we can approximate the centroid of each object by
        taking the barycenter of the centroids of the clusters
        overlapping with each object, weighed down by their respective
        IoUs.

        NB: This is a proxy and one could design failure cases, when
        clusters are not pure enough.

        :param cluster_pos: Tensor of size [num_clusters, D]
            Centroid position of each cluster
        :param mode: str
            Method used to estimate the centroids. 'iou' will weigh down
            the centroids of the clusters overlapping each instance by
            their IoU. 'ratio-product' will use the product of the size
            ratios of the overlap wrt the cluster and wrt the instance.
            'overlap' will use the size of the overlap between the
            cluster and the instance.

        :return obj_pos, obj_idx
            obj_pos: Tensor
                Estimated position for each object
            obj_idx: Tensor
                Corresponding object indices
        """
        # Prepare the indices for sets A (i.e. clusters) and B (i.e.
        # objects). In particular, we want the indices to be contiguous
        # in [0, idx_max], to alleviate scatter operations' computation.
        # Since `self.obj` contains potentially-large and non-contiguous
        # global object indices, we update these indices locally
        a_idx = self.indices
        b_idx, perm = consecutive_cluster(self.obj)
        obj_idx = self.obj[perm]

        # Expand per-cluster positions to each overlap
        a_pos = cluster_pos[a_idx]

        # Compute the weight for each overlap
        mode = mode.lower()
        if mode == 'iou':
            iou, _, _ = self.iou_and_size()
            w = iou
        elif mode == 'product-iou':
            _, a_size, b_size = self.iou_and_size()
            w = self.count**2 / (a_size * b_size)
        elif mode == 'overlap':
            w = self.count
        else:
            raise NotImplementedError
        w = w.view(-1, 1)

        # To avoid running 2 scatter operations, we concatenate the data
        # we want to sum before
        a_wpos = torch.cat((a_pos * w, w), dim=1)
        res = scatter_sum(a_wpos, b_idx, dim=0)
        obj_pos = res[:, :-1] / res[:, -1].view(-1, 1)

        return obj_pos, obj_idx

    def instance_graph(
            self,
            edge_index: torch.Tensor,
            num_classes: int = None,
            smooth_affinity: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute instance graph and per-edge affinity scores.

        :param edge_index: Tensor of size [2, num_edges]
            Edges connecting the clusters in of the instance graph. The
            output instance graph will be a trimmed version of this
            graph, where only (i, j) edges with (i < j) are preserved.
        :param num_classes: int
            Number of classes in the dataset. Specifying `num_classes`
            allows identifying 'void' labels. By convention, we assume
            `y ∈ [0, self.num_classes-1]` ARE ALL VALID LABELS (i.e. not
            'ignored', 'void', 'unknown', etc), while `y < 0` AND
            `y >= self.num_classes` ARE VOID LABELS. Void data is dealt
            with following https://arxiv.org/abs/1801.00868 and
            https://arxiv.org/abs/1905.01220
        :param smooth_affinity: bool
            If True, the affinity score computed for each edge will
            follow the 'smooth' formulation:
            `(overlap_i_obj_j / size_i + overlap_j_obj_i / size_j) / 2`
            for the edge `(i, j)`, where `obj_i` designates the target
            instance of `i`. If False, the affinity will be computed
            with the simpler formulation: `obj_i == obj_j`

        :return obj_edge_index, obj_edge_affinity
            obj_edge_index: Tensor of size [2, num_trimmed_edges]
                Edges of the trimmed instance graph
            obj_edge_affinity: Tensor
                Affinity for each edge
        """
        # In order to save compute and memory, and because the
        # cut-pursuit partition algorithm considers edges to be
        # non-oriented, we do not need to express both (i, j) and (j, i)
        # edges in the instance graph. So we start by trimming the input
        # edges to only have unique (i, j) edges with i < j.
        # Importantly, this operation also removes self-loops, which is
        # what we want here
        obj_edge_index = to_trimmed(edge_index.to(self.device))

        # Return here if the graph is empty
        if obj_edge_index.numel() == 0:
            return obj_edge_index, torch.zeros(0, device=self.device)

        # Find the target instance for each cluster: the instance it has
        # the biggest overlap with
        sp_obj_idx = self.major(num_classes=num_classes)[0]

        # Propagate the instance object to the edges' source and target
        # clusters
        i_obj_idx = sp_obj_idx[obj_edge_index[0]]
        j_obj_idx = sp_obj_idx[obj_edge_index[1]]

        # In case smooth affinity computation is not required, the
        # affinity is directly calculated by `obj_i == obj_j`
        if not smooth_affinity:
            return obj_edge_index, (i_obj_idx == j_obj_idx).float()

        # In order to efficiently compute the overlaps `overlap_i_obj_j`
        # and `overlap_j_obj_i`, we will need to recover from self the
        # overlaps that exist (those are non-zero) and set the other
        # ones to zero. By definition, since we assume the data
        # contained in self accounts for two partitions of the scene, if
        # an overlap is not present in self, then the overlap is empty.
        # To properly align edge-wise overlaps and cluster-object
        # overlaps, we will build a shared indexing to uniquely identify
        # each cluster-object pair (including the pairs not in self). We
        # will build this indexing in such a way that it is compact, to
        # avoid ever constructing any brutal [num_clusters, num_objects]
        # matrix. We will compute the corresponding index for each
        # cluster-object pair in self (A), each `overlap_i_obj_j` (B),
        # and each `overlap_j_obj_i` (C)
        base = self.obj.max() + 1
        A = self.indices * base + self.obj
        B = obj_edge_index[0] * base + j_obj_idx
        C = obj_edge_index[1] * base + i_obj_idx

        # Make the index contiguous
        all_uid_raw = torch.cat((A, B, C))
        uid, perm = consecutive_cluster(all_uid_raw)
        uid_raw = all_uid_raw[perm]
        num_uid = uid.max() + 1
        A_uid = uid[:A.shape[0]]
        B_uid = uid[A.shape[0]:A.shape[0] + B.shape[0]]
        C_uid = uid[-C.shape[0]:]

        # To compute the overlaps, we will initialize them all to 0.
        # Then, we will populate the non-zero overlaps using self.count.
        # Finally, we wil distribute the overlap to each relevant edge
        # for smooth affinity computation
        overlaps = torch.zeros(num_uid, device=self.device)
        overlaps[A_uid] = self.count.float()
        overlap_i_obj_j = overlaps[B_uid]
        overlap_j_obj_i = overlaps[C_uid]

        # Compute the size of each cluster and propagate it to each edge
        sp_size = scatter_sum(self.count, self.indices)
        size_i = sp_size[obj_edge_index[0]].float()
        size_j = sp_size[obj_edge_index[1]].float()

        # We can now compute the smooth affinity for each edge
        affinity = (overlap_i_obj_j / size_i + overlap_j_obj_i / size_j) / 2

        return obj_edge_index, affinity

    def search_void(
            self,
            num_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Search for clusters and objects with 'void' semantic labels.

        IMPORTANT:
        By convention, we assume `y ∈ [0, num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'void', 'ignored', 'unknown', etc),
        while `y < 0` AND `y >= num_classes` ARE VOID LABELS.
        This applies to both `Data.y` and `Data.obj.y`.

        Points with 'void' labels are handled following the procedure
        proposed in:
          - https://arxiv.org/abs/1801.00868
          - https://arxiv.org/abs/1905.01220

        More precisely, we remove from IoU and metrics computation:
          - predictions (i.e. clusters here) containing more than 50% of
            'void' points
          - targets (i.e. objects here) containing more than 50% of
            'void' points. In our case, we assume targets to be
            SEMANTICALLY PURE, so we remove a target even if it contains
            a single 'void' point

        To this end, the present function returns:
          - `cluster_mask`: boolean mask of the clusters containing more
            than 50% points with `void` labels
          - `pair_mask`: boolean mask of the cluster-object pairs whose
            object (i.e. target) has an `void` label
          - `pair_cropped_count`: tensor of cropped target size, for
            each pair. Indeed, blindly removing the predictions with 50%
            or more void points will affect downstream IoU computation.
            To account for this, this, `pair_cropped_count` is intended
            to be used at IoU computation time, when assessing the
            prediction and target sizes

        NB: by construction, removing pairs in `pair_mask` from the
            InstanceData will also remove all target objects containing
            'void' points. Importantly, this assumes, however, that the
            raw instance annotations in the datasets are semantically
            pure: all annotated instances contain points of the same
            class. Said otherwise: IF AN INSTANCE CONTAINS A SINGLE
            'VOID' POINT, THEN ALL OF ITS POINTS ARE 'VOID'.
        """
        # Identify the pairs whose object (i.e. target instance) is void.
        # For simplicity, we note 'a' for clusters/predictions and 'b'
        # for objects/targets/ground truths
        is_pair_b_void = (self.y < 0) | (self.y >= num_classes)

        # Get the cluster indices, for each cluster-object pair
        pair_a_idx = self.indices

        # Compute the size of each set and redistribute to each a-b pair
        a_size = scatter_sum(self.count, pair_a_idx)

        # Identify the indices of the clusters included in a void pair
        void_a_idx = pair_a_idx[is_pair_b_void].unique()

        # For those clusters specifically, identify those whose total
        # size encompasses more than 50% void points
        void_a_total_size = a_size[void_a_idx]
        void_a_void_size = scatter_sum(
            self.count[is_pair_b_void], pair_a_idx[is_pair_b_void])[void_a_idx]
        void_a_50_plus = (void_a_void_size / void_a_total_size.float()) > 0.5
        void_a_50_plus_idx = void_a_idx[void_a_50_plus]

        # Convert the indices to a boolean mask spanning the clusters
        is_a_void = torch.zeros(
            self.num_clusters, dtype=torch.bool, device=self.device)
        is_a_void[void_a_50_plus_idx] = True

        # Blindly removing the predictions with 50% or more void points
        # will affect downstream IoU computation. To account for this,
        # we search the affected target indices and compute the size of
        # the corresponding crop induced by void prediction removal.
        # Finally, we expand this as a pair-wise tensor, indicating the
        # missing crop size for each pair
        b_idx = consecutive_cluster(self.obj)[0]
        pair_cropped_count = scatter_sum(
            self.count * is_a_void[pair_a_idx], b_idx)[b_idx]

        # Update the pair-wise void mask, to account for the removal of
        # +50%-void predictions
        is_pair_void = is_pair_b_void | is_a_void[pair_a_idx]

        return is_a_void, is_pair_void, pair_cropped_count

    def remove_void(
            self,
            num_classes: int
    ) -> Tuple['InstanceData', torch.Tensor]:
        """Return a new InstanceData with void clusters, objects and
        pairs removed.

        IMPORTANT:
        By convention, we assume `y ∈ [0, num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'void', 'ignored', 'unknown', etc),
        while `y < 0` AND `y >= num_classes` ARE VOID LABELS.
        This applies to both `Data.y` and `Data.obj.y`.

        Points with 'void' labels are handled following the procedure
        proposed in:
          - https://arxiv.org/abs/1801.00868
          - https://arxiv.org/abs/1905.01220

        More precisely:
          - predictions (i.e. clusters here) containing more than 50% of
            'void' points are removed from the metrics computation
          - targets (i.e. objects here) containing more than 50% of
            'void' points are removed from the metrics computation
          - the remaining 'void' points are ignored when computing the
            prediction-target (i.e. cluster-object here) IoUs

        To this end, the present function returns:
          - `instance_data`: a new InstanceData object with all void
            clusters, objects, and pairs removed
          - `non_void_mask`: boolean mask spanning the clusters,
            indicating the clusters that were preserved in the
            `instance_data`. This mask can be used outside of this
            function to subsample cluster-wise information after
            void-removal

        NB: by construction, removing pairs in `pair_mask` from the
            InstanceData will also remove all target objects containing
            'void' points. Importantly, this assumes, however, that the
            raw instance annotations in the datasets are semantically
            pure: all annotated instances contain points of the same
            class. Said otherwise: IF AN INSTANCE CONTAINS A SINGLE
            'VOID' POINT, THEN ALL OF ITS POINTS ARE 'VOID'.
        """
        # Get the masks for indexing void clusters and pairs
        is_cluster_void, is_pair_void, pair_cropped_count = \
            self.search_void(num_classes)

        # Create a new InstanceData without void data
        idx = self.indices
        idx = idx[~is_pair_void]
        idx = consecutive_cluster(idx)[0]
        obj = self.obj[~is_pair_void]
        count = self.count[~is_pair_void]
        y = self.y[~is_pair_void]
        pair_cropped_count = pair_cropped_count[~is_pair_void]
        instance_data = self.__class__(idx, obj, count, y, dense=True)

        # Save the pair_cropped_count in the new InstanceData. This will
        # be used by `self.iou_and_size()` to cleanly account for the
        # removal of +50%-void predictions
        instance_data.pair_cropped_count = pair_cropped_count

        return instance_data, ~is_cluster_void

    def debug(self):
        super().debug()

        # Make sure there are no duplicate cluster-obj pairs
        cluster_obj_idx = self.indices * (self.obj.max() + 1) + self.obj
        assert not has_duplicates(cluster_obj_idx)

    def __repr__(self):
        info = [
            f"{key}={getattr(self, key)}"
            for key in ['num_clusters', 'num_overlaps', 'num_obj', 'device']]
        return f"{self.__class__.__name__}({', '.join(info)})"

    def target_label_histogram(self, num_classes: int) -> torch.Tensor:
        """Compute the target histogram for semantic segmentation. That
        is, for each cluster, the histogram of pointwise labels of its
        overlaps. When joined with cluster-wise semantic predictions,
        this histogram can be passed to a ConfusionMatrix metric.

        :param num_classes: int
            Number of valid classes. By convention, we assume
            `y ∈ [0, num_classes-1]` are VALID LABELS, while
            `y < 0` AND `y >= num_classes` ARE VOID LABELS

        :return: Tensor of shape [num_clusters, num_classes + 1]
        """
        # Set all void labels to `num_classes`, if any
        y = self.y.clone()
        y[(y < 0) | (y > num_classes)] = num_classes

        # Accumulate all pair labels into pre-cluster label histograms
        y_hist = one_hot(y, num_classes=num_classes + 1) * self.count.view(-1, 1)
        return scatter_sum(y_hist, self.indices, dim=0)

    def semantic_segmentation_oracle(
            self,
            num_classes: int,
            *metric_args,
            **metric_kwargs
    ) -> 'SemanticMetricResults':
        """Compute the oracle performance for semantic segmentation,
        when all clusters predict the dominant label among their points.
        This corresponds to the highest achievable performance with the
        partition at hand.

        :param num_classes: int
            Number of valid classes. By convention, we assume
            `y ∈ [0, num_classes-1]` are VALID LABELS, while
            `y < 0` AND `y >= num_classes` ARE VOID LABELS
        :param metric_args:
            Args for the metrics computation
        :param metric_kwargs:
            Kwargs for the metrics computation

        :return: SemanticMetricResults
        """
        # Compute the label histogram for each cluster
        y_hist = self.target_label_histogram(num_classes)

        # We expect the network to predict the most frequent label. For
        # clusters where the dominant label is 'void', we expect the
        # network to predict the second most frequent label. In the
        # event where the cluster is 100% 'void', the metric will ignore
        # the prediction, regardless its value
        pred = y_hist[:, :num_classes].argmax(dim=1)
        target = y_hist

        # Performance evaluation
        from src.metrics import ConfusionMatrix
        cm = ConfusionMatrix(num_classes, *metric_args, **metric_kwargs)
        cm(pred.cpu(), target.cpu())
        metrics = cm.all_metrics()

        return metrics

    def oracle(
            self,
            num_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the oracle predictions for instance and panoptic
        segmentation. This is a proxy for the highest achievable
        performance with the cluster partition at hand. The output data
        can be passed to the relevant metrics in `src.metrics` for
        performance computation.

        More precisely, for the oracle prediction:
          - each cluster is assigned to the instance it shares the most
            points with
          - clusters assigned to the same instance are merged into a
            single prediction
          - each predicted instance has a score equal to its IoU with
            the assigned target instance

        :param num_classes: int
            Number of valid classes. By convention, we assume
            `y ∈ [0, num_classes-1]` are VALID LABELS, while
            `y < 0` AND `y >= num_classes` ARE VOID LABELS
        :return: oracle_scores, oracle_y, oracle_instance_data
        """
        # For each cluster, identify the dominant object (i.e. the object
        # with which the cluster has the most overlap)
        obj, count, y = self.major(num_classes=num_classes)
        idx, perm = consecutive_cluster(obj)

        # Group together clusters with the same target object index.
        # This amounts to constructing the oracle predictions instances
        oracle = self.merge(idx)

        # Compute the oracle predicted semantic label for each grouped
        # cluster instance
        oracle_y = y[perm]

        # Compute the oracle predicted scores. Here, we choose to score
        # clusters by their IoU with their optimal target
        # NB: this choice is relatively arbitrary, one could design
        # another approach for scoring the predictions
        # (e.g. IoU * semantic purity). There is no guarantee that this
        # specific scoring function maximizes mAP, but it is a
        # reasonable proxy
        iou = oracle.iou_and_size()[0]
        argmax = scatter_max(oracle.count, oracle.indices)[1]
        oracle_scores = iou[argmax]

        return oracle_scores, oracle_y, oracle

    def instance_segmentation_oracle(
            self,
            num_classes: int,
            **metric_kwargs
    ) -> 'InstanceMetricResults':
        """Compute the oracle performance for instance segmentation.
        This is a proxy for the highest achievable performance with the
        cluster partition at hand.

        More precisely, for the oracle prediction:
          - each cluster is assigned to the instance it shares the most
            points with
          - clusters assigned to the same instance are merged into a
            single prediction
          - each predicted instance has a score equal to its IoU with
            the assigned target instance

        :param num_classes: int
            Number of valid classes. By convention, we assume
            `y ∈ [0, num_classes-1]` are VALID LABELS, while
            `y < 0` AND `y >= num_classes` ARE VOID LABELS
        :param metric_kwargs:
            Kwargs for the metrics computation

        :return: InstanceMetricResults
        """
        # Compute oracle predictions
        oracle_scores, oracle_y, oracle = self.oracle(num_classes)

        # Performance evaluation
        from src.metrics import MeanAveragePrecision3D
        metric = MeanAveragePrecision3D(num_classes, **metric_kwargs)
        metric.update(oracle_scores, oracle_y, oracle)
        results = metric.compute()

        return results

    def panoptic_segmentation_oracle(
            self,
            num_classes: int,
            **metric_kwargs
    ) -> 'PanopticMetricResults':
        """Compute the oracle performance for panoptic segmentation.
        This is a proxy for the highest achievable performance with the
        cluster partition at hand.

        More precisely, for the oracle prediction:
          - each cluster is assigned to the instance it shares the most
            points with
          - clusters assigned to the same instance are merged into a
            single prediction

        :param num_classes: int
            Number of valid classes. By convention, we assume
            `y ∈ [0, num_classes-1]` are VALID LABELS, while
            `y < 0` AND `y >= num_classes` ARE VOID LABELS
        :param metric_kwargs:
            Kwargs for the metrics computation

        :return: PanopticMetricResults
        """
        # Compute oracle predictions
        oracle_scores, oracle_y, oracle = self.oracle(num_classes)

        # Performance evaluation
        from src.metrics import PanopticQuality3D
        metric = PanopticQuality3D(num_classes, **metric_kwargs)
        metric.update(oracle_y, oracle)
        results = metric.compute()

        return results


class InstanceBatch(InstanceData, CSRBatch):
    """Wrapper for InstanceData batching. Importantly, although
    instance labels in 'obj' will be updated to avoid collisions between
    the different batch items.
    """
