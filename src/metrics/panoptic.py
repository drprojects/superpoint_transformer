import torch
import logging
from torch_scatter import scatter_sum
from torch import Tensor, LongTensor
from typing import Any, List, Optional, Sequence
from torchmetrics.metric import Metric
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.data import InstanceData, InstanceBatch
from src.metrics.mean_average_precision import BaseMetricResults


log = logging.getLogger(__name__)


__all__ = ['PanopticQuality3D']


class PanopticMetricResults(BaseMetricResults):
    """Class to wrap the final metric results for Panoptic Segmentation.
    """
    __slots__ = (
        'pq',
        'sq',
        'rq',
        'pq_modified',
        'pq_thing',
        'sq_thing',
        'rq_thing',
        'pq_stuff',
        'sq_stuff',
        'rq_stuff',
        'pq_per_class',
        'sq_per_class',
        'rq_per_class',
        'precision_per_class',
        'recall_per_class',
        'tp_per_class',
        'fp_per_class',
        'fn_per_class',
        'pq_modified_per_class',
        'mean_precision',
        'mean_recall')


class PanopticQuality3D(Metric):
    """Computes the `Panoptic Quality (PQ) and associated metrics`_ for
    3D panoptic segmentation. Optionally, the metrics can be calculated
    per class.

    Importantly, this implementation expects predictions and targets to
    be passed as InstanceData, which assumes predictions and targets
    form two PARTITIONS of the scene: all points belong to one and only
    one prediction and one and only one target ('stuff' included).

    By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL VALID
    LABELS (i.e. not 'ignored', 'void', 'unknown', etc), while `y < 0`
    AND `y >= self.num_classes` ARE VOID LABELS. Void data is dealt
    with following:
      - https://arxiv.org/abs/1801.00868
      - https://arxiv.org/abs/1905.01220

    Predicted instances and targets have to be passed to
    :meth:``forward`` or :meth:``update`` within a custom format. See
    the :meth:`update` method for more information about the input
    format to this metric.

    As output of ``forward`` and ``compute`` the metric returns the
    following output:

    - ``pq_dict``: A dictionary containing the following key-values:

        - pq: (:class:`~torch.Tensor`)
        - sq: (:class:`~torch.Tensor`)
        - rq: (:class:`~torch.Tensor`)
        - pq_modified: (:class:`~torch.Tensor`)
        - pq_thing: (:class:`~torch.Tensor`)
        - sq_thing: (:class:`~torch.Tensor`)
        - rq_thing: (:class:`~torch.Tensor`)
        - pq_stuff: (:class:`~torch.Tensor`)
        - sq_stuff: (:class:`~torch.Tensor`)
        - rq_stuff: (:class:`~torch.Tensor`)
        - pq_per_class: (:class:`~torch.Tensor`)
        - sq_per_class: (:class:`~torch.Tensor`)
        - rq_per_class: (:class:`~torch.Tensor`)
        - pq_modified_per_class: (:class:`~torch.Tensor`)
        - mean_precision: (:class:`~torch.Tensor`)
        - mean_recall: (:class:`~torch.Tensor`)

    :param num_classes: int
        Number of valid classes in the dataset. By convention, we assume
        `y ∈ [0, self.num_classes-1]` ARE ALL VALID LABELS (i.e. not
        'ignored', 'void', 'unknown', etc), while `y < 0` AND
        `y >= self.num_classes` ARE VOID LABELS. Void data is dealt
        with following https://arxiv.org/abs/1801.00868 and
        https://arxiv.org/abs/1905.01220
    :param ignore_unseen_classes: bool
        If True, the mean metrics will only be computed on seen classes.
        Otherwise, metrics for the unseen classes will be set to ZERO by
        default and those will affect the average metrics over all
        classes.
    :param stuff_classes: List or Tensor
        List of 'stuff' class labels, to distinguish between 'thing' and
        'stuff' classes.
    :param compute_on_cpu: bool
        If True, the accumulated prediction and target data will be
        stored on CPU, and the metrics computation will be performed
        on CPU. This can be necessary for particularly large
        datasets.
    :param kwargs:
        Additional keyword arguments, see :ref:`Metric kwargs` for
        more info.
    """
    prediction_semantic: List[LongTensor]
    instance_data: List[InstanceData]
    full_state_update: bool = False

    def __init__(
            self,
            num_classes: int,
            ignore_unseen_classes: bool = True,
            stuff_classes: Optional[List[int]] = None,
            compute_on_cpu: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(compute_on_cpu=compute_on_cpu, **kwargs)

        # Store the number of valid classes in the dataset
        self.num_classes = num_classes

        # Whether classes without any prediction or target in the data
        # should be ignored in the metrics computation
        self.ignore_unseen_classes = ignore_unseen_classes

        # Stuff classes may be specified, to be properly accounted for
        # in metrics computation
        self.stuff_classes = stuff_classes or []

        # All torchmetric's Metrics have internal states they use to
        # store predictions and ground truths. Those are updated when
        # `self.forward()` or `self.update()` are called, and used for
        # computing the actual metrics when `self.compute()` is called.
        # Every time we want to restart our metric measurements, we
        # need to reset these internal states to their initial values.
        # This happens when calling `self.reset()`. For `self.reset()`
        # to know what to reset and to which value, these states to be
        # declared with `self.add_state()`
        self.add_state("prediction_semantic", default=[], dist_reduce_fx=None)
        self.add_state("instance_data", default=[], dist_reduce_fx=None)

    def update(
            self,
            prediction_semantic: LongTensor,
            instance_data: InstanceData,
    ) -> None:
        """Update the internal state of the metric.

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
             Besides, for each 'stuff' class, AT MOST ONE prediction and
             AT MOST ONE target are allowed per scene/cloud/image.
        :return:
        """
        # Sanity checks
        self._input_validator(prediction_semantic, instance_data)

        # Store in the internal states
        self.prediction_semantic.append(prediction_semantic)
        self.instance_data.append(instance_data)

    @staticmethod
    def _input_validator(
            prediction_semantic: LongTensor,
            instance_data: InstanceData):
        """Sanity checks executed on the input of `self.update()`.
        """
        if not isinstance(prediction_semantic, Tensor):
            raise ValueError(
                "Expected argument `prediction_semantic` to be of type Tensor")
        if not prediction_semantic.dtype == torch.long:
            raise ValueError(
                "Expected argument `prediction_semantic` to have dtype=long")
        if not isinstance(instance_data, InstanceData):
            raise ValueError(
                "Expected argument `instance_data` to be of type InstanceData")

        if prediction_semantic.dim() != 1:
            raise ValueError(
                "Expected argument `prediction_semantic` to have dim=1")
        if prediction_semantic.numel() != instance_data.num_clusters:
            raise ValueError(
                "Expected argument `prediction_semantic` and `instance_data` to "
                "have the same number of size")

    def compute(self) -> dict:
        """Compute the metrics from the data stored in the internal
        states.

        NB: this implementation assumes the prediction and targets
        stored in the internal states represent two PARTITIONS of the
        data points. Said otherwise, all points belong to one and only
        one prediction and one and only one target.
        """
        # Batch together the values stored in the internal states.
        # Importantly, the InstanceBatch mechanism ensures there is no
        # collision between object labels of the stored scenes
        pred_semantic = torch.cat(self.prediction_semantic)
        pair_data = InstanceBatch.from_list(self.instance_data)
        device = pred_semantic.device

        # Remove some prediction, targets, and pairs to properly account
        # for points with 'void' labels in the data. For more details,
        # `InstanceData.remove_void` documentation
        pair_data, is_pred_valid = pair_data.remove_void(self.num_classes)
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
        pair_iou = pair_data.iou_and_size()[0]

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
        del gt_idx, pair_gt_semantic

        # Recover the classes present in the data.
        # NB: this step assumes `InstanceData.remove_void()` was called
        # beforehand. Otherwise, some data labels may be outside
        # `[0, self.num_classes-1]`
        # all_semantic = torch.cat((gt_semantic, pred_semantic))
        # num_classes = all_semantic.max().item() + 1
        num_classes = self.num_classes
        classes = range(num_classes)

        # Class-wise mask for stuff/thing
        is_stuff = torch.tensor(
            [i in self.stuff_classes for i in classes], device=device)
        has_stuff = is_stuff.count_nonzero() > 0

        # TP + FN
        gt_class_counts = torch.bincount(gt_semantic, minlength=num_classes)
        gt_class_counts = gt_class_counts[:num_classes]

        # TP + FP
        pred_class_counts = torch.bincount(pred_semantic, minlength=num_classes)
        pred_class_counts = pred_class_counts[:num_classes]

        # Preparing for TP search among candidate pairs
        pair_gt_semantic = gt_semantic[pair_gt_idx]
        pair_pred_semantic = pred_semantic[pair_pred_idx]
        pair_is_stuff = is_stuff[pair_gt_semantic]
        pair_agrees = pair_gt_semantic == pair_pred_semantic
        pair_iou_gt_50 = pair_iou > 0.5

        # TP
        idx_pair_tp = torch.where(pair_agrees & pair_iou_gt_50)[0]
        tp = torch.bincount(
            pair_gt_semantic[idx_pair_tp], minlength=num_classes)
        iou_sum = scatter_sum(
            pair_iou[idx_pair_tp],
            pair_gt_semantic[idx_pair_tp],
            dim_size=num_classes)

        # Precision & Recall
        precision = tp / pred_class_counts
        precision[precision.isnan()] = 0
        recall = tp / gt_class_counts
        fp = pred_class_counts - tp
        fn = gt_class_counts - tp

        # SQ - Segmentation Quality
        sq = iou_sum / tp
        sq[sq.isnan()] = 0

        # RQ - Recognition Quality
        rq = 2 * precision * recall / (precision + recall)
        rq[rq.isnan()] = 0

        # PQ - Panoptic Quality
        pq = sq * rq

        # PQ modified - more permissive for stuff classes, following:
        # https://arxiv.org/abs/1905.01220
        if has_stuff:
            idx_pair_tp_mod = torch.where(
                pair_agrees & (pair_iou_gt_50 | pair_is_stuff))[0]
            # tp_mod = torch.bincount(
            #     pair_gt_semantic[idx_pair_tp_mod], minlength=num_classes)
            iou_mod_sum = scatter_sum(
                pair_iou[idx_pair_tp_mod],
                pair_gt_semantic[idx_pair_tp_mod],
                dim_size=num_classes)
            denominator = (gt_class_counts + pred_class_counts).float() / 2
            denominator[is_stuff] = gt_class_counts[is_stuff].float()
            pq_mod = iou_mod_sum / denominator
        else:
            pq_mod = pq

        # Recover the classes present in the data. In particular, we
        # identify the expected classes which are absent from both
        # predictions and targets.
        # Unseen classes are classes whose label is in
        # `[0, self.num_classes-1]` (considered 'void' otherwise), and
        # appears at least once in the predictions or in the ground
        # truth data.
        # Those classes are not accounted for in the mean metrics
        # computation, unless `self.ignore_unseen_classes=False`
        all_semantic = torch.cat((gt_semantic, pred_semantic))
        class_ids = all_semantic.unique()
        class_ids = class_ids[(class_ids >= 0) & (class_ids < num_classes)]
        is_seen = torch.zeros(num_classes, dtype=torch.bool, device=device)
        is_seen[class_ids] = True
        default = torch.nan if self.ignore_unseen_classes else 0

        pq[~is_seen] = default
        sq[~is_seen] = default
        rq[~is_seen] = default
        pq_mod[~is_seen] = default
        precision[~is_seen] = default
        recall[~is_seen] = default

        if not self.ignore_unseen_classes:
            assert not pq.isnan().any()
            assert not sq.isnan().any()
            assert not rq.isnan().any()
            assert not pq_mod.isnan().any()
            assert not precision.isnan().any()

        # Compute the final metrics
        metrics = PanopticMetricResults()
        metrics.pq = pq.nanmean()
        metrics.sq = sq.nanmean()
        metrics.rq = rq.nanmean()
        metrics.pq_modified = pq_mod.nanmean()
        metrics.pq_thing = pq[~is_stuff].nanmean()
        metrics.sq_thing = sq[~is_stuff].nanmean()
        metrics.rq_thing = rq[~is_stuff].nanmean()
        metrics.pq_stuff = pq[is_stuff].nanmean() if has_stuff else torch.nan
        metrics.sq_stuff = sq[is_stuff].nanmean() if has_stuff else torch.nan
        metrics.rq_stuff = rq[is_stuff].nanmean() if has_stuff else torch.nan
        metrics.pq_per_class = pq
        metrics.sq_per_class = sq
        metrics.rq_per_class = rq
        metrics.precision_per_class = precision
        metrics.recall_per_class = recall
        metrics.tp_per_class = tp
        metrics.fp_per_class = fp
        metrics.fn_per_class = fn
        metrics.pq_modified_per_class = pq_mod
        metrics.mean_precision = precision.nanmean()
        metrics.mean_recall = recall.nanmean()

        return metrics

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
