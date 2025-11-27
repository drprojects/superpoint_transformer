import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor, LongTensor
from typing import Any, Dict, List, Optional, Sequence, Tuple
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.utilities.imports import _TORCHVISION_GREATER_EQUAL_0_8
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.data import InstanceData, InstanceBatch
from src.utils import arange_interleave, sizes_to_pointers


log = logging.getLogger(__name__)


__all__ = [
    'MeanAveragePrecision3D',
    'MAPMetricResults',
    'MARMetricResults',
    'InstanceMetricResults']


class BaseMetricResults(dict):
    """Base metric class, that allows fields for pre-defined metrics.
    """

    def __getattr__(self, key: str) -> Tensor:
        # Using this you get the correct error message, an
        # AttributeError instead of a KeyError
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key: str, value: Tensor) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        if key in self:
            del self[key]
        raise AttributeError(f"No such attribute: {key}")


class MAPMetricResults(BaseMetricResults):
    """Class to wrap the final mAP results.
    """
    __slots__ = (
        "map",
        "map_25",
        "map_50",
        "map_75",
        "map_small",
        "map_medium",
        "map_large")


class MARMetricResults(BaseMetricResults):
    """Class to wrap the final mAR results.
    """
    __slots__ = ("mar", "mar_small", "mar_medium", "mar_large")


class InstanceMetricResults(BaseMetricResults):
    """Class to wrap the final COCO metric results including various
    mAP/mAR values.
    """
    __slots__ = (
        "map",
        "map_25",
        "map_50",
        "map_75",
        "map_small",
        "map_medium",
        "map_large",
        "mar",
        "mar_small",
        "mar_medium",
        "mar_large",
        "map_per_class",
        "mar_per_class")


class MeanAveragePrecision3D(MeanAveragePrecision):
    """Computes the `Mean-Average-Precision (mAP) and
    Mean-Average-Recall (mAR)`_ for 3D instance segmentation.
    Optionally, the mAP and mAR values can be calculated per class.

    Importantly, this implementation expects predictions and targets to
    be passed as InstanceData, which assumes predictions and targets
    form two PARTITIONS of the scene: all points belong to one and only
    one prediction and one and only one target ('stuff' included).

    Predicted instances and targets have to be passed to
    :meth:``forward`` or :meth:``update`` within a custom format. See
    the :meth:`update` method for more information about the input
    format to this metric.

    As output of ``forward`` and ``compute`` the metric returns the
    following output:

    - ``map_dict``: A dictionary containing the following key-values:

        - map: (:class:`~torch.Tensor`)
        - map_25: (:class:`~torch.Tensor`)         (NaN if 0.25 not in the list of iou thresholds)
        - map_50: (:class:`~torch.Tensor`)         (NaN if 0.5 not in the list of iou thresholds)
        - map_75: (:class:`~torch.Tensor`)         (NaN if 0.75 not in the list of iou thresholds)
        - map_small: (:class:`~torch.Tensor`)      (NaN if `medium_size` and `large_size` are not specified)
        - map_medium:(:class:`~torch.Tensor`)      (NaN if `medium_size` and `large_size` are not specified)
        - map_large: (:class:`~torch.Tensor`)      (NaN if `medium_size` and `large_size` are not specified)
        - mar: (:class:`~torch.Tensor`)
        - mar_small: (:class:`~torch.Tensor`)      (NaN if `medium_size` and `large_size` are not specified)
        - mar_medium: (:class:`~torch.Tensor`)     (NaN if `medium_size` and `large_size` are not specified)
        - mar_large: (:class:`~torch.Tensor`)      (NaN if `medium_size` and `large_size` are not specified)
        - map_per_class: (:class:`~torch.Tensor`)  (NaN if class metrics are disabled)
        - mar_per_class: (:class:`~torch.Tensor`)  (NaN if class metrics are disabled)

    .. note::
        ``map`` score is calculated with @[ IoU=self.iou_thresholds | area=all ].
        Caution: If the initialization parameters are changed, dictionary keys for mAR can change as well.
        The default properties are also accessible via fields and will raise an ``AttributeError`` if not available.

    .. note::
        This metric is following the mAP implementation of
        `pycocotools <https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools>`_,
        a standard implementation for the mAP metric for object detection.

    .. note::
        This metric requires you to have `torchvision` version 0.8.0 or newer installed
        (with corresponding version 1.7.0 of torch or newer). Please install with ``pip install torchvision`` or
        ``pip install torchmetrics[detection]``.

    :param num_classes: int
        Number of valid semantic classes (union of 'thing' and 'stuff')
        in the dataset. These are all semantic classes whose
        points are not to be considered as 'void' or 'ignored' when
        computing the metrics. By convention, we assume
        `y ∈ [0, self.num_classes-1]` ARE ALL VALID LABELS (i.e. not
        'ignored', 'void', 'unknown', etc), while `y < 0` AND
        `y >= self.num_classes` ARE VOID LABELS. Void data is dealt
        with following https://arxiv.org/abs/1801.00868 and
        https://arxiv.org/abs/1905.01220
    :param iou_thresholds: List or Tensor
        List of IoU on which to evaluate the performance. If None,
        the mAP and mAR will be computed on thresholds
        [0.5, 0.55, ..., 0.95], and AP25, AP50, and AP75 will be
        computed.
    :param rec_thresholds: List or Tensor
        The recall steps to use when integrating the AUC.
    :param class_metrics: bool
        If True, the per-class AP and AR metrics will also be
        computed.
    :param stuff_classes: List or Tensor
        List of 'stuff' class labels to ignore in the metrics
        computation. NB: as opposed to 'void' classes, 'stuff' classes
        are not entirely ignored per se, as they are accounted for when
        computing the IoUs.
    :param min_size: int
        Minimum target instance size to consider when computing the
        metrics. If a target is smaller, it will be ignored, as well
        as its matched prediction, if any.
    :param medium_size: float
        Marks the frontier between small-sized and medium-sized
        objects. If both `medium_size` and `large_size` are
        provided, a breakdown of the metrics will be computed
        between 'small', 'medium', and 'large' objects. Setting
        either `medium_size` or `large_size` to None will skip this
        behavior.
    :param large_size: float
        Marks the frontier between medium-sized and large-sized
        objects. If both `medium_size` and `large_size` are
        provided, a breakdown of the metrics will be computed
        between 'small', 'medium', and 'large' objects. Setting
        either `medium_size` or `large_size` to None will skip this
        behavior.
    :param compute_on_cpu: bool
        If True, the accumulated prediction and target data will be
        stored on CPU, and the metrics computation will be performed
        on CPU. This can be necessary for particularly large
        datasets.
    :param remove_void: bool
        If True, points with 'void' labels will be removed following
        the procedure proposed in:
          - https://arxiv.org/abs/1801.00868
          - https://arxiv.org/abs/1905.01220
    :param plot: bool
        If True, the `mAP = f(IoU)` and `mAR = f(IoU)` curves be plotted
    :param kwargs:
        Additional keyword arguments, see :ref:`Metric kwargs` for
        more info.
    """
    prediction_score: List[Tensor]
    prediction_semantic: List[LongTensor]
    instance_data: List[InstanceData]

    def __init__(
            self,
            num_classes: int,
            iou_thresholds: Optional[List[float]] = None,
            rec_thresholds: Optional[List[float]] = None,
            class_metrics: bool = True,
            stuff_classes: Optional[List[int]] = None,
            min_size: int = 0,
            medium_size: Optional[float] = None,
            large_size: Optional[float] = None,
            compute_on_cpu: bool = True,
            remove_void: bool = True,
            plot: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(compute_on_cpu=compute_on_cpu, **kwargs)

        if not _TORCHVISION_GREATER_EQUAL_0_8:
            raise ModuleNotFoundError(
                f"`{self.__class__.__name__}` metric requires that "
                f"`torchvision` version 0.8.0 or newer is installed. "
                f"Please install with `pip install torchvision>=0.8` or "
                f"`pip install torchmetrics[detection]`.")

        # Store the number of valid semantic classes in the dataset
        self.num_classes = num_classes

        # The IoU thresholds are used for computing various mAP. The
        # standard mAP is the mean of the AP for IoU thresholds of
        # [0.5, 0.55, ..., 0.95]. It is also often common practice to
        # communicate AP25, AP50 and AP75, respectively associated with
        # IoU thresholds of 25, 50, and 75.
        base_ious = torch.cat((torch.tensor([0.25]), torch.arange(0.5, 1, 0.05)))
        iou_thresholds = torch.asarray(iou_thresholds).clamp(min=0) \
            if iou_thresholds else base_ious
        self.iou_thresholds = iou_thresholds.tolist()

        # The recall thresholds control the number of bins we use when
        # integrating the area under the precision-recall curve
        base_rec = torch.arange(0, 1.01, 0.01)
        rec_thresholds = torch.asarray(rec_thresholds) \
            if rec_thresholds else base_rec
        self.rec_thresholds = rec_thresholds.tolist()

        # A minimum size can be provided to ignore target instances
        # below this threshold. All ground truth objects below this
        # threshold will be ignored in the metrics computation. Besides,
        # a prediction with no match (after assigning predictions to
        # targets) and below this threshold will also be ignored in the
        # metrics
        self.min_size = min_size

        # Area ranges are used to compute the metrics for several
        # families of objects, based on their ground truth size
        if medium_size is not None and large_size is not None:
            min_size = float(min_size)
            medium_size = max(float(medium_size), min_size)
            large_size = max(float(large_size), min_size)
            self.size_ranges = dict(
                all=(min_size, float("inf")),
                small=(min_size, medium_size),
                medium=(medium_size, large_size),
                large=(large_size, float("inf")))
        else:
            self.size_ranges = dict(all=(float(min_size), float("inf")))

        # If class_metrics=True the per-class metrics will also be
        # returned, although this may impact overall speed
        if not isinstance(class_metrics, bool):
            raise ValueError("Expected argument `class_metrics` to be a bool")
        self.class_metrics = class_metrics

        # Stuff classes may be specified, to be properly accounted for
        # in metrics computation
        self.stuff_classes = stuff_classes or []

        # Whether points with 'void' labels should be removed following
        # the procedure proposed in:
        #   - https://arxiv.org/abs/1801.00868
        #   - https://arxiv.org/abs/1905.01220
        self.remove_void = remove_void

        # Whether the mAP and mAR curves should be plotted
        self.plot = plot

        # All torchmetric's Metrics have internal states they use to
        # store predictions and ground truths. Those are updated when
        # `self.forward()` or `self.update()` are called, and used for
        # computing the actual metrics when `self.compute()` is called.
        # Every time we want to restart our metric measurements, we
        # need to reset these internal states to their initial values.
        # This happens when calling `self.reset()`. For `self.reset()`
        # to know what to reset and to which value, these states to be
        # declared with `self.add_state()`
        self.add_state("prediction_score", default=[], dist_reduce_fx=None)
        self.add_state("prediction_semantic", default=[], dist_reduce_fx=None)
        self.add_state("instance_data", default=[], dist_reduce_fx=None)

    def update(
            self,
            prediction_score: Tensor,
            prediction_semantic: LongTensor,
            instance_data: InstanceData,
    ) -> None:
        """Update the internal state of the metric.

        :param prediction_score: Tensor
             1D tensor of size N_pred holding the confidence score of
             the predicted instances.
        :param prediction_semantic: LongTensor
             1D tensor of size N_pred holding the semantic label of the
             predicted instances.
        :param instance_data: InstanceData
             InstanceData object holding all information required for
             computing the iou between predicted and ground truth
             instances, as well as the target semantic label.
             Importantly, ALL PREDICTION AND TARGET INSTANCES ARE
             ASSUMED TO BE REPRESENTED in THIS InstanceData, even
             'stuff' classes and 'too small' instances, which will be
             accounted for in this metric. Besides the InstanceData
             assumes the predictions and targets form two PARTITIONS of
             the scene: all points belong to one and only one prediction
             and one and only one target object ('stuff' included).
        :return:
        """
        # Sanity checks
        self._input_validator(
            prediction_score, prediction_semantic, instance_data)

        # Store in the internal states
        self.prediction_score.append(prediction_score)
        self.prediction_semantic.append(prediction_semantic)
        self.instance_data.append(instance_data)

    @staticmethod
    def _input_validator(
            prediction_score: Tensor,
            prediction_semantic: LongTensor,
            instance_data: InstanceData):
        """Sanity checks executed on the input of `self.update()`.
        """
        if not isinstance(prediction_score, Tensor):
            raise ValueError(
                "Expected argument `prediction_score` to be of type Tensor")
        if not isinstance(prediction_semantic, Tensor):
            raise ValueError(
                "Expected argument `prediction_semantic` to be of type Tensor")
        if not prediction_semantic.dtype == torch.long:
            raise ValueError(
                "Expected argument `prediction_semantic` to have dtype=long")
        if not isinstance(instance_data, InstanceData):
            raise ValueError(
                "Expected argument `instance_data` to be of type InstanceData")

        if prediction_score.dim() != 1:
            raise ValueError(
                "Expected argument `prediction_score` to have dim=1")
        if prediction_semantic.dim() != 1:
            raise ValueError(
                "Expected argument `prediction_semantic` to have dim=1")
        if prediction_score.numel() != prediction_semantic.numel():
            raise ValueError(
                "Expected argument `prediction_score` and `prediction_semantic`"
                " to have the same size")
        if prediction_score.numel() != instance_data.num_clusters:
            raise ValueError(
                "Expected argument `prediction_score` and `instance_data` to "
                "have the same number of size")

    def compute(self) -> dict:
        """Metrics computation.
        """
        # Batch together the values stored in the internal states.
        # Importantly, the InstanceBatch mechanism ensures there is no
        # collision between object labels of the stored scenes
        pred_score = torch.cat(self.prediction_score)
        pred_semantic = torch.cat(self.prediction_semantic)
        pair_data = InstanceBatch.from_list(self.instance_data)
        device = pred_score.device

        # Remove some prediction, targets, and pairs to properly account
        # for points with 'void' labels in the data. For more details,
        # `InstanceData.remove_void` documentation
        if self.remove_void:
            pair_data, is_pred_valid = pair_data.remove_void(self.num_classes)
            pred_score = pred_score[is_pred_valid]
            pred_semantic = pred_semantic[is_pred_valid]
            del is_pred_valid

        # Recover the target index, IoU, sizes, and target label for
        # each pair. Importantly, the way the pair_pred_idx is built
        # guarantees the prediction indices are contiguous in
        # [0, pred_id_max]. Besides it is IMPORTANT that all points are
        # accounted for in the InstanceData, regardless of their class.
        # This is because the InstanceData will infer the total size
        # of each segment, as well as the IoUs from these values.
        pair_pred_idx = pair_data.indices
        pair_gt_idx = pair_data.obj
        pair_gt_semantic = pair_data.y
        pair_iou, pair_pred_size, pair_gt_size = pair_data.iou_and_size()

        # To alleviate memory and compute, we would rather store
        # ground-truth-specific attributes in tensors of size
        # num_gt rather than for all prediction-ground truth
        # pairs. We will keep track of the prediction and ground truth
        # indices for each pair, to be able to recover relevant
        # information when need be. To this end, since there is no
        # guarantee the ground truth indices are contiguous in
        # [0, gt_idx_max], we contract those and gather associated
        # pre-ground-truth attributes
        pair_gt_idx, gt_idx = consecutive_cluster(pair_gt_idx)
        gt_semantic = pair_gt_semantic[gt_idx]
        gt_size = pair_gt_size[gt_idx]
        del gt_idx, pair_gt_semantic, pair_gt_size

        # Similarly, we do not need to store all prediction attributes
        # at a pair-wise level. So we compute the corresponding
        # prediction-wise attributes
        pred_size = torch.zeros_like(pred_semantic)
        pred_size[pair_pred_idx] = pair_pred_size
        del pair_pred_size

        # To prepare for downstream matching assignments, we sort all
        # pairs by decreasing prediction score. We do this now because
        # we want to leverage the convenient order ensured by the
        # InstanceData structure. Indeed, we want to compute the sorting
        # on the prediction-wise scores and apply the resulting
        # reordering to the prediction-wise as well as the pair-wise
        # data
        order = pred_score.argsort(descending=True)
        pred_score = pred_score[order]
        pred_semantic = pred_semantic[order]
        pred_size = pred_size[order]
        width = pair_data.sizes
        start = sizes_to_pointers(width[:-1])
        pair_order = arange_interleave(width[order], start=start[order])
        pair_pred_idx = torch.arange(
            width.shape[0], device=device).repeat_interleave(width[order])
        pair_gt_idx = pair_gt_idx[pair_order]
        pair_iou = pair_iou[pair_order]
        del pair_data, order, width, start, pair_order

        # Recover the classes of interest for this metric. These are the
        # classes whose labels appear at least once in the predicted
        # semantic label or in the ground truth semantic labels.
        # By default, all labels in `[0, self.num_classes-1]` are
        # considered valid and all labels outside this range will be
        # considered 'void' and ignored.
        # Besides, if `stuff_classes` was provided, the corresponding
        # labels are ignored
        all_semantic = torch.cat((gt_semantic, pred_semantic))
        class_ids = all_semantic.unique()
        class_ids = class_ids[(class_ids >= 0) & (class_ids < self.num_classes)]
        class_ids = list(set(class_ids.tolist()) - set(self.stuff_classes))

        # For each class, each size range (and each IoU threshold),
        # compute the prediction-ground truth matches. Importantly, the
        # output of this step is formatted so as to be passed to
        # `MeanAveragePrecision.__calculate_recall_precision_scores`
        evaluations = [
            self._evaluate(
                class_id,
                size_range,
                pred_score,
                pair_iou,
                pair_pred_idx,
                pair_gt_idx,
                pred_semantic,
                gt_semantic,
                pred_size,
                gt_size)
            for class_id in class_ids
            for size_range in self.size_ranges.values()]

        num_iou = len(self.iou_thresholds)
        num_rec = len(self.rec_thresholds)
        num_classes = len(class_ids)
        num_sizes = len(self.size_ranges)
        precision = -torch.ones((num_iou, num_rec, num_classes, num_sizes, 1), device=device)
        recall = -torch.ones((num_iou, num_classes, num_sizes, 1), device=device)
        scores = -torch.ones((num_iou, num_rec, num_classes, num_sizes, 1), device=device)

        # Compute the recall, precision, and score for each IoU
        # threshold, class, and size range
        for idx_cls, _ in enumerate(class_ids):
            for i_size, _ in enumerate(self.size_ranges):
                recall, precision, scores = self.__calculate(
                    recall,
                    precision,
                    scores,
                    idx_cls,
                    i_size,
                    evaluations)

        # Compute all AP and AR metrics
        map_val, mar_val = self._summarize_results(
            precision, recall, plot=self.plot)

        # If class_metrics is enabled, also evaluate all metrics for
        # each class of interest
        map_per_class_val = torch.tensor([torch.nan])
        mar_per_class_val = torch.tensor([torch.nan])
        if self.class_metrics:
            map_per_class_list = []
            mar_per_class_list = []

            for i_class in range(len(class_ids)):
                cls_precisions = precision[:, :, i_class].unsqueeze(dim=2)
                cls_recalls = recall[:, i_class].unsqueeze(dim=1)
                cls_map, cls_mar = self._summarize_results(
                    cls_precisions, cls_recalls)
                map_per_class_list.append(cls_map.map)
                mar_per_class_list.append(cls_mar.mar)

            map_per_class_val = torch.as_tensor(map_per_class_list)
            mar_per_class_val = torch.as_tensor(mar_per_class_list)

        # Prepare the final metrics output
        metrics = InstanceMetricResults()
        metrics.update(map_val)
        metrics.update(mar_val)
        metrics.map_per_class = map_per_class_val
        metrics.mar_per_class = mar_per_class_val

        return metrics

    def _evaluate(
            self,
            class_id: int,
            size_range: Tuple[int, int],
            pred_score: Tensor,
            pair_iou: Tensor,
            pair_pred_idx: Tensor,
            pair_gt_idx: Tensor,
            pred_semantic: Tensor,
            gt_semantic: Tensor,
            pred_size: Tensor,
            gt_size: Tensor
    ) -> Optional[dict]:
        """Perform evaluation for single class and a single size range.
        The output evaluations cover all required IoU thresholds.
        Concretely, these "evaluations" are the prediction-target
        assignments, with respect to constraints on the semantic class
        of interest, the IoU threshold (for AP computation), the target
        size, etc.

        The following rules apply:
          - at most 1 prediction per target
          - predictions are assigned by order of decreasing score and to
            the not-already-matched target with highest IoU (within IoU
            threshold)

        NB: the input prediction-wise and pair-wise data is assumed to
        be ALREADY SORTED by descending prediction scores.

        The output is formatted so as to be passed to torchmetrics'
        `MeanAveragePrecision.__calculate_recall_precision_scores`.

        :param class_id: int
            Index of the class on which to compute the evaluations.
        :param size_range: List
            Upper and lower bounds for the size range of interest.
            Target objects outside those bounds will be ignored. As well
            as non-matched predictions outside those bounds.
        :param pred_score: Tensor of shape [N_pred]
        :param pair_iou: Tensor of shape [N_pred_gt_overlaps]
        :param pair_pred_idx: Tensor of shape [N_pred_gt_overlaps]
        :param pair_gt_idx: Tensor of shape [N_pred_gt_overlaps]
        :param pred_semantic: Tensor of shape [N_pred]
        :param gt_semantic: Tensor of shape [N_gt]
        :param pred_size: Tensor of shape [N_pred]
        :param gt_size: Tensor of shape [N_gt]
        :return:
        """
        device = gt_semantic.device

        # Compute masks on the prediction-target pairs, based on their
        # semantic label, as well as their size
        is_gt_class = gt_semantic == class_id
        is_pred_class = pred_semantic == class_id
        is_gt_in_size_range = (size_range[0] <= gt_size.float()) \
                              & (gt_size.float() <= size_range[1])
        is_pred_in_size_range = (size_range[0] <= pred_size.float()) \
                                & (pred_size.float() <= size_range[1])

        # Count the number of ground truths and predictions with the
        # class at hand
        num_gt = is_gt_class.count_nonzero().item()
        num_pred = is_pred_class.count_nonzero().item()

        # Get the number of IoU thresholds
        num_iou = len(self.iou_thresholds)

        # If no ground truth and no detection carry the class of
        # instance, return None
        if num_gt == 0 and num_pred == 0:
            return

        # Some targets have the class at hand but no prediction does
        if num_pred == 0:
            return {
                "dtMatches": torch.zeros(
                    num_iou, 0, dtype=torch.bool, device=device),
                "gtMatches": torch.zeros(
                    num_iou, num_gt, dtype=torch.bool, device=device),
                "dtScores": torch.zeros(0, device=device),
                "gtIgnore": ~is_gt_in_size_range[is_gt_class],
                "dtIgnore": torch.zeros(
                    num_iou, 0, dtype=torch.bool, device=device)}

        # Some predictions have the class at hand but no target does
        if num_gt == 0:
            pred_ignore = ~is_pred_in_size_range[is_pred_class]
            return {
                "dtMatches": torch.zeros(
                    num_iou, num_pred, dtype=torch.bool, device=device),
                "gtMatches": torch.zeros(
                    num_iou, 0, dtype=torch.bool, device=device),
                "dtScores": pred_score[is_pred_class],
                "gtIgnore": torch.zeros(
                    0, dtype=torch.bool, device=device),
                "dtIgnore": pred_ignore.view(1, -1).repeat(num_iou, 1)}

        # Compute the global indices of the prediction and ground truth
        # for the class at hand. These will be used to search for
        # relevant pairs
        gt_idx = torch.where(is_gt_class)[0]
        pred_idx = torch.where(is_pred_class)[0]
        is_pair_gt_class = torch.isin(pair_gt_idx, gt_idx)
        pair_idx = torch.where(is_pair_gt_class)[0]

        # Build the tensors used to track which ground truth and which
        # prediction has found a match, for each IoU threshold. This is
        # the data structure expected by torchmetrics'
        # `MeanAveragePrecision.__calculate_recall_precision_scores()`
        gt_matches = torch.zeros(
            num_iou, num_gt, dtype=torch.bool, device=device)
        det_matches = torch.zeros(
            num_iou, num_pred, dtype=torch.bool, device=device)
        gt_ignore = ~is_gt_in_size_range[is_gt_class]
        det_ignore = torch.zeros(
            num_iou, num_pred, dtype=torch.bool, device=device)

        # Each pair is associated with a prediction index and a ground
        # truth index. Except these indices are global, spanning across
        # all objects in the data, regardless of their class. Here, we
        # are also going to need to link a pair with the local ground
        # truth index (tracking objects with the current class of
        # interest), to be able to index and update the above-created
        # gt_matches and gt_ignore. To this end, we will compute a
        # simple mapping. NB: we do not need to build such mapping for
        # prediction indices, because we will simply enumerate
        # pred_idx, which provides both local and global indices
        gt_idx_to_i_gt = torch.full_like(gt_semantic, -1)
        gt_idx_to_i_gt[gt_idx] = torch.arange(num_gt, device=device)

        # For each prediction with the class at hand, we will need to
        # find the pairs involving this prediction and a target object
        # with the class. To avoid excessive computations, we trim the
        # predictions absent from the pairs and the pairs involving
        # predictions we are not interested in
        pair_pred_idx = pair_pred_idx[pair_idx]
        pred_isin_pairs = torch.isin(pred_idx, pair_pred_idx)
        pair_isin_preds = torch.isin(pair_pred_idx, pred_idx)
        pred_idx = pred_idx[pred_isin_pairs]
        pair_idx = pair_idx[pair_isin_preds]
        pair_pred_idx = pair_pred_idx[pair_isin_preds]

        # Then, to avoid searching through all the pairs for every new
        # prediction index, we reorder the pairs by increasing
        # prediction index. This allows the construction of pointers to
        # easily address consecutive pairs involving each prediction.
        # Since the tensor of prediction indices is sorted by
        # construction, we can then easily get the pair indices
        # involving each prediction
        order = pair_pred_idx.argsort()
        pair_idx = pair_idx[order]
        pair_pred_idx = pair_pred_idx[order]
        pred_ptr = sizes_to_pointers(pair_pred_idx.bincount()[pred_idx])
        del order

        # Match each prediction to a ground truth
        # NB: the assumed pre-ordering of predictions by decreasing
        # score ensures we are assigning high-confidence predictions
        # first
        for i, (j, pred_idx) in enumerate(zip(torch.where(pred_isin_pairs)[0], pred_idx)):

            # Get the indices of pairs which involve the prediction at
            # hand and whose ground truth has the class at hand
            _pair_idx = pair_idx[pred_ptr[i]:pred_ptr[i + 1]]

            # Gather the pairs' ground truth information for candidate
            # ground truth matches
            _pair_gt_idx = pair_gt_idx[_pair_idx]
            _pair_i_gt = gt_idx_to_i_gt[_pair_gt_idx]
            _pair_iou = pair_iou[_pair_idx]

            # Sort the candidates by decreasing gt size. In case the
            # prediction has multiple candidate ground truth matches
            # with equal IoU, we will select the one with the largest
            # size in priority
            if _pair_gt_idx.numel() > 1:
                order = gt_size[_pair_gt_idx].argsort(descending=True)
                _pair_idx = _pair_idx[order]
                _pair_gt_idx = _pair_gt_idx[order]
                _pair_i_gt = _pair_i_gt[order]
                _pair_iou = _pair_iou[order]
                del order

            # Among potential ground truth matches, remove those which
            # are already matched with a prediction.
            # NB: we do not remove the 'ignored' ground truth yet: if a
            # ground truth is 'ignored', we still want to match it to a
            # good prediction, if any. This way, the prediction in
            # question will also be marked as to be 'ignored', else it
            # would be unfairly penalized as a False Positive
            _iou_pair_gt_matched = gt_matches[:, _pair_i_gt]

            # For each IoU and each candidate ground truth, search the
            # available candidates with large-enough IoU.
            # NB: clamping the thresholds to 0 will facilitate searching
            # for the best match for each prediction while identifying
            # False Positives
            iou_thresholds = torch.tensor(self.iou_thresholds, device=device)

            # Get the best possible matching pair for each IoU threshold
            _iou_match, _iou_match_idx = \
                (~_iou_pair_gt_matched * _pair_iou.view(1, -1)).max(dim=1)

            # Check if the match found for each IoU threshold is valid.
            # A match is valid if:
            #   - the match's IoU is above the IoU threshold
            #   - the ground truth is not already matched
            # A match may be valid but still be ignored if:
            #   - the ground truth is marked as ignored
            _iou_match_ok = _iou_match > iou_thresholds

            # For each IoU threshold, get the corresponding ground truth
            # index. From there, we can update the det_ignore,
            # det_matches and gt_matches.
            _iou_match_i_gt = _pair_i_gt[_iou_match_idx]
            det_ignore[:, j] = gt_ignore[_iou_match_i_gt] * _iou_match_ok
            det_matches[:, j] = _iou_match_ok

            #  Special attention must be paid to gt_matches in case the
            #  prediction tried to match an already-assigned gt. In
            #  which case the prediction will not match (i.e.
            #  _iou_match_ok is False). To avoid re-setting the
            #  corresponding gt_matches to False, we need to make sure
            #  gt_matches was not already matched
            _iou_match_gt_ok = \
                gt_matches.gather(1, _iou_match_i_gt.view(-1, 1)).squeeze()
            _iou_match_gt_ok = _iou_match_gt_ok | _iou_match_ok
            gt_matches.scatter_(
                1, _iou_match_i_gt.view(-1, 1), _iou_match_gt_ok.view(-1, 1))

        # The above procedure may leave some predictions without match.
        # Those should count as False Positives, unless their size is
        # outside the size_range of interest, in which case it should be
        # ignored from metrics computation
        det_ignore = det_ignore | ~det_matches \
                     & ~is_pred_in_size_range[is_pred_class].view(1, -1)

        return {
            "dtMatches": det_matches,
            "gtMatches": gt_matches,
            "dtScores": pred_score[is_pred_class],
            "gtIgnore": gt_ignore,
            "dtIgnore": det_ignore}

    def __calculate(
            self,
            recall: Tensor,
            precision: Tensor,
            scores: Tensor,
            idx_cls: int,
            idx_size_range: int,
            evaluations: list,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return MeanAveragePrecision._MeanAveragePrecision__calculate_recall_precision_scores(
            recall,
            precision,
            scores,
            idx_cls,
            idx_size_range,
            0,
            evaluations,
            torch.tensor(self.rec_thresholds, device=recall.device),
            torch.iinfo(torch.int64).max,
            1,
            len(self.size_ranges))

    def _move_list_states_to_cpu(self) -> None:
        """Move list states to cpu to save GPU memory."""
        for key in self._defaults.keys():
            current_val = getattr(self, key)
            if isinstance(current_val, Sequence):
                setattr(self, key, [cur_v.to("cpu") for cur_v in current_val])

    def to(self, *args, **kwargs):
        """Overwrite torch.nn.Module.to() to handle the InstanceData
        stored in the internal states.
        """
        instance_data = getattr(self, 'instance_data', None)
        if instance_data is not None:
            self.instance_data = torch.zeros(1, device=self.device)
        out = super().to(*args, **kwargs)
        if instance_data is not None:
            self.instance_data = instance_data
            out.instance_data = [x.to(*args, **kwargs) for x in instance_data]
        return out

    def _summarize(
            self,
            results: Dict,
            avg_prec: bool = True,
            iou_threshold: Optional[float] = None,
            size_range: str = "all",
    ) -> Tensor:
        """Summarize Precision or Recall results, for a given IoU
        threshold.

        :param results: Dict
            Dictionary including precision, recall and scores for all
            combinations.
        :param avg_prec: bool
            Whether the average precision or the average recall should
            be returned.
        :param iou_threshold: float
            IoU threshold. If set to ``None`` it all values are used.
            Else results are filtered. NB: the passed IoU threshold MUST
            BE in `self.iou_thresholds`
        :param size_range: str
            String indicating the size range. The corresponding key MUST
            BE in `self.size_ranges`.
        """
        i_size = [
            i for i, k in enumerate(self.size_ranges.keys()) if k == size_range]
        i_iou = self.iou_thresholds.index(iou_threshold) \
            if iou_threshold else Ellipsis

        if avg_prec:
            # dimension of precision: [TxRxKxAx1]
            prec = results["precision"]
            prec = prec[i_iou, :, :, i_size, 0]
        else:
            # dimension of recall: [TxKxAx1]
            prec = results["recall"]
            prec = prec[i_iou, :, i_size, 0]

        mean_prec = torch.tensor([torch.nan]) if len(prec[prec > -1]) == 0 \
            else torch.mean(prec[prec > -1])

        return mean_prec

    def _summarize_results(
            self,
            precisions: Tensor,
            recalls: Tensor,
            plot: bool = False
    ) -> Tuple[MAPMetricResults, MARMetricResults]:
        """Summarizes the precision and recall values to calculate
        mAP/mAR.

        Args:
            precisions:
                Precision values for different thresholds
            recalls:
                Recall values for different thresholds
            plot:
                If True, the `mAP = f(IoU)` and `mAR = f(IoU)` curves
                 be plotted
        """
        # Compute all metrics for IoU > 0.5. This little trick is needed
        # to avoid including IoU=0.25 in the mAP computation, which, by
        # convention, is computed on IoUs=[0.5, 0.55, ..., 0.95]
        device = precisions.device
        idx = torch.tensor(self.iou_thresholds, device=device) >= 0.5
        res_above_50 = dict(precision=precisions[idx], recall=recalls[idx])

        map_metrics = MAPMetricResults()
        map_metrics.map = self._summarize(res_above_50, True)
        map_metrics.map_small = self._summarize(res_above_50, True, size_range="small")
        map_metrics.map_medium = self._summarize(res_above_50, True, size_range="medium")
        map_metrics.map_large = self._summarize(res_above_50, True, size_range="large")

        mar_metrics = MARMetricResults()
        mar_metrics.mar = self._summarize(res_above_50, False)
        mar_metrics.mar_small = self._summarize(res_above_50, False, size_range="small")
        mar_metrics.mar_medium = self._summarize(res_above_50, False, size_range="medium")
        mar_metrics.mar_large = self._summarize(res_above_50, False, size_range="large")

        # Finally, compute the results for specific IoU levels
        res = dict(precision=precisions, recall=recalls)
        map_metrics.map_25 = self._summarize(res, True, iou_threshold=0.25) \
            if 0.25 in self.iou_thresholds else torch.tensor([torch.nan])
        map_metrics.map_50 = self._summarize(res, True, iou_threshold=0.5) \
            if 0.5 in self.iou_thresholds else torch.tensor([torch.nan])
        map_metrics.map_75 = self._summarize(res, True, iou_threshold=0.75) \
            if 0.75 in self.iou_thresholds else torch.tensor([torch.nan])

        if plot:
            map_per_iou = []
            mar_per_iou = []
            for iou in sorted(self.iou_thresholds):
                map_per_iou.append(self._summarize(res, True, iou_threshold=iou).cpu())
                mar_per_iou.append(self._summarize(res, False, iou_threshold=iou).cpu())
            plt.plot(sorted(self.iou_thresholds), np.array(map_per_iou))
            plt.show()

        return map_metrics, mar_metrics
