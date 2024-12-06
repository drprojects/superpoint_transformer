import torch
import logging
from typing import Any, List, Tuple, Dict
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torch_geometric.nn.pool.consecutive import consecutive_cluster

from src.utils import init_weights, PanopticSegmentationOutput, \
    PartitionParameterSearchStorage
from src.metrics import MeanAveragePrecision3D, PanopticQuality3D, \
    ConfusionMatrix
from src.models.semantic import SemanticSegmentationModule
from src.loss import BCEWithLogitsLoss
from src.data import NAG


log = logging.getLogger(__name__)


__all__ = ['PanopticSegmentationModule']


class PanopticSegmentationModule(SemanticSegmentationModule):
    """A LightningModule for panoptic segmentation of point clouds.

    :param net: torch.nn.Module
        Backbone model. This can typically be an `SPT` object
    :param edge_affinity_head: torch.nn.Module
        Edge affinity prediction head for instance/panoptic graph
        clustering. This is typically an MLP
    :param partitioner: src.nn.instance.InstancePartitioner
        Instance partition head, expects a fully-fledged
        `InstancePartitioner` module as input. This module is only
        called when the actual instance/panoptic segmentation is
        required. At train time, it is not essential, since we do not
        propagate gradient to its parameters. However, we may still tune
        its parameters to maximize instance/panoptic metrics on the
        train set. This tuning involves a simple grid-search on a small
        range of parameters and needs to be called at least once at the
        very end of training
    :param criterion: torch.nn._Loss
        Loss
    :param optimizer: torch.optim.Optimizer
        Optimizer
    :param scheduler: torch.optim.lr_scheduler.LRScheduler
        Learning rate scheduler
    :param num_classes: int
        Number of classes in the dataset
    :param stuff_classes: List[int]
        Indices of the classes to be treated as 'stuff', as opposed to
        'thing'
    :param class_names: List[str]
        Name for each class
    :param sampling_loss:  bool
        If True, the target labels will be obtained from labels of
        the points sampled in the batch at hand. This affects
        training supervision where sampling augmentations may be
        used for dropping some points or superpoints. If False, the
        target labels will be based on exact superpoint-wise
        histograms of labels computed at preprocessing time,
        disregarding potential level-0 point down-sampling
    :param loss_type: str
        Type of loss applied.
        'ce': cross-entropy (if `multi_stage_loss_lambdas` is used,
        all 1+ levels will be supervised with cross-entropy).
        'kl': Kullback-Leibler divergence (if `multi_stage_loss_lambdas`
        is used, all 1+ levels will be supervised with cross-entropy).
        'ce_kl': cross-entropy on level 1 and Kullback-Leibler for
        all levels above
        'wce': not documented for now
        'wce_kl': not documented for now
    :param weighted_loss: bool
        If True, the loss will be weighted based on the class
        frequencies computed on the train dataset. See
        `BaseDataset.get_class_weight()` for more
    :param init_linear: str
        Initialization method for all linear layers. Supports
        'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
        'kaiming_normal', 'trunc_normal'
    :param init_rpe: str
        Initialization method for all linear layers producing
        relative positional encodings. Supports 'xavier_uniform',
        'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
        'trunc_normal'
    :param transformer_lr_scale: float
        Scaling parameter applied to the learning rate for the
        `TransformerBlock` in each `Stage` and for the pooling block
        in `DownNFuseStage` modules. Setting this to a value lower
        than 1 mitigates exploding gradients in attentive blocks
        during training
    :param multi_stage_loss_lambdas: List[float]
        List of weights for combining losses computed on the output
        of each partition level. If not specified, the loss will
        be computed on the level 1 outputs only
    :param edge_affinity_criterion: torch.nn._Loss
        Loss on the edges of the superpoint level 1 for affinity
        prediction
    :param edge_affinity_loss_weights: List[float]
        Weights for insisting on certain cases in the edge affinity
        loss:
         - 0: same-class same-object edges
         - 1: same-class different-object edges
         - 2: different-class same-object edges
         - 3: different-class different-object edges
    :param edge_affinity_loss_lambda: float
        Weight for combining the semantic segmentation loss with the
        node offset and edge affinity losses. The final loss will be:
        `L_node_classif + edge_affinity_loss_lambda * L_edge_affinity
        + node_offset_loss_lambda * L_node_offset`
    :param node_offset_criterion: torch.nn._Loss
        Loss on the nodes of the superpoint level 1 for node offset
        prediction
    :param node_offset_loss_lambda: float
        Weight for combining the semantic segmentation loss with the
        node offset and edge affinity losses. The final loss will be:
        `L_node_classif + edge_affinity_loss_lambda * L_edge_affinity
        + node_offset_loss_lambda * L_node_offset`
    :param gc_every_n_steps: int
        Explicitly call the garbage collector after a certain number
        of steps. May involve a computation overhead. Mostly hear
        for debugging purposes when observing suspicious GPU memory
        increase during training
    :param track_val_every_n_epoch: int
        If specified, the output for a validation batch of interest
        specified with `track_val_idx` will be stored to disk every
        `track_val_every_n_epoch` epochs. Must be a multiple of
        `check_val_every_n_epoch`. See `track_batch()` for more
    :param track_val_idx: int
        If specified, the output for the `track_val_idx`th
        validation batch will be saved to disk periodically based on
        `track_val_every_n_epoch`. Importantly, this index is expected
        to match the `Dataloader`'s index wrt the current epoch
        and NOT an index wrt the `Dataset`. Said otherwise, if the
        `Dataloader(shuffle=True)` then, the stored batch will not be
        the same at each epoch. For this reason, if tracking the same
        object across training is needed, the `Dataloader` and the
        transforms should be free from any stochasticity
    :param track_test_idx:
        If specified, the output for the `track_test_idx`th
        test batch will be saved to disk. If `track_test_idx=-1`,
        predictions for the entire test set will be saved to disk
    :param min_instance_size: int
        Minimum target instance size to consider when computing the
        metrics. If a target is smaller, it will be ignored, as well
        as its matched prediction, if any. See `MeanAveragePrecision3D`
    :param partition_every_n_epoch: int
        Since we do not need to compute the actual panoptic/instance
        segmentation to train the model, we can simply do so once in a
        while to track the training and validation metrics. This
        parameter rules the frequency at which the panoptic/instance
        partition and metrics are computed during training
    :param no_instance_metrics: bool
        Whether instance segmentation metrics should be computed. These
        may incur an overhead. Besides, the SuperCluster formulation is
        mainly targeted for panoptic segmentation, as the model is not
        specifically trained to maximize instance metrics, which, among
        other things, involve predicting an instance confidence score
    :param no_instance_metrics_on_train_set: bool
        Same as `no_instance_metrics` but specifically for the train
        set. This is in case we still want the instance metrics every
        partition_every_n_epoch` on the validation set, but want to
        avoid the compute overhead of computing the instance partition
        and metrics at every single training epoch
    :param kwargs: Dict
        Kwargs will be passed to `_load_from_checkpoint()`
    """

    _IGNORED_HYPERPARAMETERS = [
        'net',
        'edge_affinity_head',
        'partitioner',
        'criterion',
        'edge_affinity_criterion',
        'node_offset_criterion']

    def __init__(
            self,
            net: torch.nn.Module,
            edge_affinity_head: torch.nn.Module,
            partitioner: 'InstancePartitioner',
            criterion: 'torch.nn._Loss',
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            num_classes: int,
            stuff_classes: List[int],
            class_names: List[str] = None,
            sampling_loss: bool = False,
            loss_type: str = 'ce_kl',
            weighted_loss: bool = True,
            init_linear: str = None,
            init_rpe: str = None,
            transformer_lr_scale: float = 1,
            multi_stage_loss_lambdas: List[float] = None,
            edge_affinity_criterion: 'torch.nn._Loss' = None,
            edge_affinity_loss_weights: List[float] = None,
            edge_affinity_loss_lambda: float = 1,
            node_offset_criterion: 'torch.nn._Loss' = None,
            node_offset_loss_lambda: float = 1,
            gc_every_n_steps: int = 0,
            track_val_every_n_epoch: int = 1,
            track_val_idx: int = None,
            track_test_idx: int = None,
            min_instance_size: int = 100,
            partition_every_n_epoch: int = 50,
            no_instance_metrics: bool = True,
            no_instance_metrics_on_train_set: bool = True,
            **kwargs):
        super().__init__(
            net,
            criterion,
            optimizer,
            scheduler,
            num_classes,
            class_names=class_names,
            sampling_loss=sampling_loss,
            loss_type=loss_type,
            weighted_loss=weighted_loss,
            init_linear=init_linear,
            init_rpe=init_rpe,
            transformer_lr_scale=transformer_lr_scale,
            multi_stage_loss_lambdas=multi_stage_loss_lambdas,
            gc_every_n_steps=gc_every_n_steps,
            track_val_every_n_epoch=track_val_every_n_epoch,
            track_val_idx=track_val_idx,
            track_test_idx=track_test_idx,
            **kwargs)

        # Instance partition head, expects a fully-fledged
        # InstancePartitioner module as input.
        # This module is only called when the actual instance/panoptic
        # segmentation is required. At train time, it is not essential,
        # since we do not propagate gradient to its parameters. However,
        # we still tune its parameters to maximize instance/panoptic
        # metrics on the train set. This tuning involves a simple
        # grid-search on a small range of parameters and needs to be
        # called at least once at the very end of training
        self.partition_every_n_epoch = partition_every_n_epoch
        self.no_instance_metrics = no_instance_metrics
        self.no_instance_metrics_on_train_set = no_instance_metrics_on_train_set
        self.partitioner = partitioner

        # Store the stuff class indices
        self.stuff_classes = stuff_classes

        # Loss functions for edge affinity and node offset predictions.
        # NB: the semantic loss is already accounted for in the
        # SemanticSegmentationModule constructor
        self.edge_affinity_criterion = BCEWithLogitsLoss() \
            if edge_affinity_criterion is None else edge_affinity_criterion
        # self.node_offset_criterion = WeightedL2Loss() \
        #     if node_offset_criterion is None else node_offset_criterion

        # Model heads for edge affinity and node offset predictions
        # Initialize the model segmentation head (or heads)
        # out_dim = self.net.out_dim[0] if self.multi_stage_loss \
        #     else self.net.out_dim
        # self.edge_affinity_head = FFN(out_dim * 2, hidden_dim=32, out_dim=1)
        self.edge_affinity_head = edge_affinity_head
        # self.node_offset_head = FFN(out_dim, hidden_dim=32, out_dim=3)

        # Custom weight initialization. In particular, this applies
        # Xavier / Glorot initialization on Linear and RPE layers by
        # default, but can be tuned
        init = lambda m: init_weights(m, linear=init_linear, rpe=init_rpe)
        self.edge_affinity_head.apply(init)
        # self.node_offset_head.apply(init)

        # Metric objects for calculating panoptic segmentation scores on
        # each dataset split
        self.train_panoptic = PanopticQuality3D(
            self.num_classes,
            ignore_unseen_classes=True,
            stuff_classes=self.stuff_classes,
            compute_on_cpu=True,
            **kwargs)
        self.val_panoptic = PanopticQuality3D(
            self.num_classes,
            ignore_unseen_classes=True,
            stuff_classes=self.stuff_classes,
            compute_on_cpu=True,
            **kwargs)
        self.test_panoptic = PanopticQuality3D(
            self.num_classes,
            ignore_unseen_classes=True,
            stuff_classes=self.stuff_classes,
            compute_on_cpu=True,
            **kwargs)

        # Metric objects for calculating semantic segmentation scores on
        # predicted instances on each dataset split
        self.train_semantic = ConfusionMatrix(self.num_classes)
        self.val_semantic = ConfusionMatrix(self.num_classes)
        self.test_semantic = ConfusionMatrix(self.num_classes)

        # Metric objects for calculating instance segmentation scores on
        # each dataset split
        self.train_instance = MeanAveragePrecision3D(
            self.num_classes,
            stuff_classes=self.stuff_classes,
            min_size=min_instance_size,
            compute_on_cpu=True,
            remove_void=True,
            **kwargs)
        self.val_instance = MeanAveragePrecision3D(
            self.num_classes,
            stuff_classes=self.stuff_classes,
            min_size=min_instance_size,
            compute_on_cpu=True,
            remove_void=True,
            **kwargs)
        self.test_instance = MeanAveragePrecision3D(
            self.num_classes,
            stuff_classes=self.stuff_classes,
            min_size=min_instance_size,
            compute_on_cpu=True,
            remove_void=True,
            **kwargs)

        # Storage to accumulate multiple batch partition predictions, to
        # be used when searching for the best partition setting
        self.train_multi_partition_storage = []

        # Metric objects for calculating node offset prediction scores
        # on each dataset split
        # self.train_offset_wl2 = WeightedL2Error()
        # self.train_offset_wl1 = WeightedL1Error()
        # self.train_offset_l2 = L2Error()
        # self.train_offset_l1 = L1Error()
        # self.val_offset_wl2 = WeightedL2Error()
        # self.val_offset_wl1 = WeightedL1Error()
        # self.val_offset_l2 = L2Error()
        # self.val_offset_l1 = L1Error()
        # self.test_offset_wl2 = WeightedL2Error()
        # self.test_offset_wl1 = WeightedL1Error()
        # self.test_offset_l2 = L2Error()
        # self.test_offset_l1 = L1Error()

        # Metric objects for calculating edge affinity prediction scores
        # on each dataset split
        self.train_affinity_oa = BinaryAccuracy()
        self.train_affinity_f1 = BinaryF1Score()
        self.val_affinity_oa = BinaryAccuracy()
        self.val_affinity_f1 = BinaryF1Score()
        self.test_affinity_oa = BinaryAccuracy()
        self.test_affinity_f1 = BinaryF1Score()

        # For averaging losses across batches
        self.train_semantic_loss = MeanMetric()
        self.train_edge_affinity_loss = MeanMetric()
        # self.train_node_offset_loss = MeanMetric()
        self.val_semantic_loss = MeanMetric()
        self.val_edge_affinity_loss = MeanMetric()
        # self.val_node_offset_loss = MeanMetric()
        self.test_semantic_loss = MeanMetric()
        self.test_edge_affinity_loss = MeanMetric()
        # self.test_node_offset_loss = MeanMetric()

        # For tracking best-so-far validation metrics
        self.val_map_best = MaxMetric()
        self.val_pq_best = MaxMetric()
        self.val_pqmod_best = MaxMetric()
        self.val_mprec_best = MaxMetric()
        self.val_mrec_best = MaxMetric()
        self.val_instance_miou_best = MaxMetric()
        self.val_instance_oa_best = MaxMetric()
        self.val_instance_macc_best = MaxMetric()
        # self.val_offset_wl2_best = MinMetric()
        # self.val_offset_wl1_best = MinMetric()
        # self.val_offset_l2_best = MinMetric()
        # self.val_offset_l1_best = MinMetric()
        self.val_affinity_oa_best = MaxMetric()
        self.val_affinity_f1_best = MaxMetric()

    @property
    def needs_partition(self) -> bool:
        """Whether the `self.partitioner` should be called to compute
        the actual panoptic segmentation. During training, the actual
        partition is not really needed, as we do not learn to partition,
        but learn to predict inputs for the partition step instead. For
        this reason, we save compute and time during training by only
        computing the partition once in a while with
        `self.partition_every_n_epoch`.
        """
        # Get the current epoch. For the validation set, we alter the
        # epoch number so that `partition_every_n_epoch` can align
        # with `check_val_every_n_epoch`. Indeed, it seems the epoch
        # number during the validation step is always one increment
        # ahead
        epoch = self.current_epoch + 1

        # If no Trainer attached to the model, run the partition
        if self._trainer is None:
            return True

        # Come useful checks to decide whether the partition should be
        # triggered
        k = self.partition_every_n_epoch
        last_epoch = epoch == self.trainer.max_epochs
        first_epoch = epoch == 1
        kth_epoch = epoch % k == 0 if k > 0 else False

        # For training, the partition is computed based on
        # `partition_every_n_epoch`, or if we reached the last epoch.
        # The first epoch will be skipped, because trained weights are
        # unlikely to produce interesting inputs for the partition
        if self.trainer.training:
            return (kth_epoch and not first_epoch) or last_epoch

        # For validation, we have the same behavior as training, with
        # the difference that if `check_val_every_n_epoch` is larger
        # than `partition_every_n_epoch`, we automatically trigger the
        # partition
        if self.trainer.validating:
            k_val = self.trainer.check_val_every_n_epoch
            nearest_multiple = epoch % k < k_val if k > 0 else False
            if 0 < k <= k_val:
                return not first_epoch or last_epoch
            else:
                return (nearest_multiple and not first_epoch) or last_epoch

        # For all other Trainer stages, we run the partition by default
        return True

    @property
    def needs_instance(self) -> bool:
        """Returns True if the instance segmentation metrics need to be
        computed. In particular, since computing instance metrics can be
        computationally costly, we may want to skip it during training
        by setting `no_instance_metrics_on_train_set=True`, or all the
        time by setting `no_instance_metrics=True`.
        """
        if self.no_instance_metrics:
            return False

        if self._trainer is None:
            return self.needs_partition

        if self.trainer.training and self.no_instance_metrics_on_train_set:
            return False

        return self.needs_partition

    def forward(
            self,
            nag: NAG,
            grid: Any = None
    ) -> PanopticSegmentationOutput:
        # Extract features
        x = self.net(nag)

        # Compute level-1 or multi-level semantic predictions
        semantic_pred = [head(x_) for head, x_ in zip(self.head, x)] \
            if self.multi_stage_loss else self.head(x)

        # Recover level-1 features only
        x = x[0] if self.multi_stage_loss else x

        # Compute node offset predictions
        # node_offset_pred = self.node_offset_head(x)

        # # Forcefully set 0-offset for nodes with stuff predictions
        # node_logits = semantic_pred[0] if self.multi_stage_loss \
        #     else semantic_pred
        # is_stuff = get_stuff_mask(node_logits, self.stuff_classes)
        # node_offset_pred[is_stuff] = 0

        # TODO: OPTIONALLY REMOVE OFFSET
        # node_offset_pred = node_offset_pred * 0

        # TODO: offset soft-assigned to 0 based on the predicted
        #  stuff/thing probas. A stuff/thing classification loss could
        #  provide additional supervision

        # Compute edge affinity predictions
        # NB: we make edge features symmetric, since we want to compute
        # edge affinity, which is not directed
        x_edge = x[nag[1].obj_edge_index]
        x_edge = torch.cat(
            ((x_edge[0] - x_edge[1]).abs(), (x_edge[0] + x_edge[1]) / 2), dim=1)
        norm_index = torch.zeros(
            x_edge.shape[0], device=x_edge.device, dtype=torch.long)
        edge_affinity_logits = self.edge_affinity_head(
            x_edge, batch=norm_index).squeeze()

        # Gather results in an output object
        output = PanopticSegmentationOutput(
            semantic_pred,
            self.stuff_classes,
            edge_affinity_logits,
            # node_offset_pred,
            nag.get_sub_size(1))

        # Compute the panoptic partition
        output = self._forward_partition(nag, output, grid=grid)

        return output

    def _forward_partition(
            self,
            nag: NAG,
            output: PanopticSegmentationOutput,
            grid: Any = None,
            force: bool = False
    ) -> PanopticSegmentationOutput:
        """Compute the panoptic partition based on the predicted node
        offsets, node semantic logits, and edge affinity logits.

        The partition will only be computed if required. In general,
        during training, the actual partition is not needed for the
        model to be supervised. We only run it once in a while to
        evaluate the panoptic/instance segmentation metrics or tune
        the partition hyperparameters on the train set.

        :param nag: NAG object
        :param output: PanopticSegmentationOutput
        :param grid: Dict
            A dictionary containing settings for grid-searching optimal
            partition parameters
        :param force: bool
            Whether to forcefully compute the partition, regardless of
            `self.needs_partition`. This mechanism is typically needed
            during training when we want to store or log predictions for
            a batch of interest at an epoch when `self.needs_partition`
            is False

        :return: output
        """
        if not self.needs_partition and not force:
            return output

        # Recover some useful information from the NAG and
        # PanopticSegmentationOutput objects
        batch = nag[1].batch
        # node_x = nag[1].pos + output.node_offset_pred
        node_x = nag[1].pos
        node_size = nag.get_sub_size(1)
        node_logits = output.logits[0] if output.multi_stage else output.logits
        edge_index = nag[1].obj_edge_index
        edge_affinity_logits = output.edge_affinity_logits

        # Compute the instance partition
        # NB: we detach the tensors here: this operation runs on CPU and
        # is non-differentiable
        obj_index = self.partitioner(
            batch,
            node_x.detach(),
            node_logits.detach(),
            self.stuff_classes,
            node_size,
            edge_index,
            edge_affinity_logits.detach(),
            grid=grid)

        # Store the results in the output object
        output.obj_index_pred = obj_index

        return output

    def on_fit_start(self) -> None:
        super().on_fit_start()

        # Get the LightningDataModule stuff classes and make sure it
        # matches self.stuff_classes. We could also forcefully update
        # the LightningModule with this new information, but it could
        # easily become tedious to track all places where stuff_classes
        # affects the LightningModule object.
        stuff_classes = self.trainer.datamodule.train_dataset.stuff_classes
        assert sorted(stuff_classes) == sorted(self.stuff_classes), \
            f'LightningModule has the following stuff classes ' \
            f'{self.stuff_classes} while the LightningDataModule has ' \
            f'{stuff_classes}.'

    def on_train_start(self) -> None:
        # By default, lightning executes validation step sanity checks
        # before training starts, so we need to make sure `*_best`
        # metrics do not store anything from these checks
        super().on_train_start()
        self.val_panoptic.reset()
        self.val_semantic.reset()
        self.val_instance.reset()
        # self.val_offset_wl2.reset()
        # self.val_offset_wl1.reset()
        # self.val_offset_l2.reset()
        # self.val_offset_l1.reset()
        self.val_affinity_oa.reset()
        self.val_affinity_f1.reset()
        self.val_map_best.reset()
        self.val_pq_best.reset()
        self.val_pqmod_best.reset()
        self.val_mprec_best.reset()
        self.val_mrec_best.reset()
        self.val_instance_miou_best.reset()
        self.val_instance_oa_best.reset()
        self.val_instance_macc_best.reset()
        # self.val_offset_wl2_best.reset()
        # self.val_offset_wl1_best.reset()
        # self.val_offset_l2_best.reset()
        # self.val_offset_l1_best.reset()
        self.val_affinity_oa_best.reset()
        self.val_affinity_f1_best.reset()
        self.train_multi_partition_storage = []

    def _create_empty_output(self, nag: NAG) -> PanopticSegmentationOutput:
        """Local helper method to initialize an empty output for
        multi-run prediction.
        """
        # Prepare empty output for semantic segmentation
        output_semseg = super()._create_empty_output(nag)

        # Prepare empty edge affinity and node offset outputs
        num_edges = nag[1].obj_edge_index.shape[1]
        edge_affinity_logits = torch.zeros(num_edges, device=nag.device)
        # node_offset_pred = torch.zeros_like(nag[1].pos)
        node_size = nag.get_sub_size(1)

        return PanopticSegmentationOutput(
            output_semseg.logits,
            self.stuff_classes,
            edge_affinity_logits,
            # node_offset_pred,
            node_size)

    @staticmethod
    def _update_output_multi(
            output_multi: PanopticSegmentationOutput,
            nag: NAG, output: PanopticSegmentationOutput,
            nag_transformed: NAG,
            key: str
    ) -> PanopticSegmentationOutput:
        """Local helper method to accumulate multiple predictions on
        the same--or part of the same--point cloud.
        """
        raise NotImplementedError(
            "The current implementation does not properly support multi-run "
            "for instance/panoptic segmentation")

        # Update semantic segmentation logits only
        output_multi = super()._update_output_multi(
            output_multi, nag, output, nag_transformed, key)

        # Update node-wise predictions
        # TODO: this is INCORRECT accumulation of node offsets. Need to
        #  define the mean, not the mean of the successive predictions
        node_id = nag_transformed[1][key]
        output_multi.node_offset_pred[node_id] = \
            (output_multi.node_offset_pred[node_id]
             + output.node_offset_pred) / 2

        # Update edge-wise predictions
        edge_index_1 = nag[1].obj_edge_index
        edge_index_2 = node_id[nag_transformed[1].obj_edge_index]
        base = nag[1].num_points + 1
        edge_id_1 = edge_index_1[0] * base + edge_index_1[1]
        edge_id_2 = edge_index_2[0] * base + edge_index_2[1]
        edge_id_cat = consecutive_cluster(torch.cat((edge_id_1, edge_id_2)))[0]
        edge_id_1 = edge_id_cat[:edge_id_1.numel()]
        edge_id_2 = edge_id_cat[edge_id_1.numel():]
        pivot = torch.zeros(base ** 2, device=output.edge_affinity_logits)
        pivot[edge_id_1] = output_multi.edge_affinity_logits
        # TODO: this is INCORRECT accumulation of node offsets. Need to
        #  define the mean, not the mean of the successive predictions
        pivot[edge_id_2] = (pivot[edge_id_2] + output.edge_affinity_logits) / 2
        output_multi.edge_affinity_logits = pivot[edge_id_1]

        return output_multi

    @staticmethod
    def _propagate_output_to_unseen_neighbors(
            output: PanopticSegmentationOutput,
            nag: NAG, seen: torch.Tensor,
            neighbors: torch.Tensor
    ) -> PanopticSegmentationOutput:
        """Local helper method to propagate predictions to unseen
        neighbors.
        """
        # Propagate semantic segmentation to neighbors
        output = super()._propagate_output_to_unseen_neighbors(
            output, nag, seen, neighbors)

        # Heuristic for unseen node offsets: unseen nodes take the same
        # offset as their nearest neighbor
        seen_idx = torch.where(seen)[0]
        unseen_idx = torch.where(~seen)[0]
        output.node_offset_pred[unseen_idx] = \
            output.node_offset_pred[seen_idx][neighbors]

        # Heuristic for unseen edge affinity predictions: we set the
        # edge affinity to 0.5
        seen_edge = nag[1].obj_edge_index[seen]
        unseen_edge_idx = torch.where(~seen_edge)[0]
        output.edge_affinity_logits[unseen_edge_idx] = 0.5

        return output

    def get_target(
            self,
            nag: NAG,
            output: PanopticSegmentationOutput
    ) -> PanopticSegmentationOutput:
        """Recover the target data for semantic and panoptic
        segmentation and store it in the `output` object.

        More specifically:
          - label histogram(s) for semantic segmentation will be saved
            in `output.y_hist`
          - instance graph data `obj_edge_index` and `obj_edge_affinity`
            will be saved in `output.obj_edge_index` and
            `output.obj_edge_affinity`, respectively
          - node positions `pos` and `obj_pos` will be saved in
            `output.pos` and `output.obj_pos`, respectively. Besides,
            the `output.obj_offset` will carry the target offset,
            computed from those
        """
        # Recover targets for semantic segmentation
        output = super().get_target(nag, output)

        # Recover targets for instance/panoptic segmentation
        output.obj_edge_index = getattr(nag[1], 'obj_edge_index', None)
        output.obj_edge_affinity = getattr(nag[1], 'obj_edge_affinity', None)
        output.pos = nag[1].pos
        output.obj_pos = getattr(nag[1], 'obj_pos', None)
        output.obj = nag[1].obj

        return output

    def _edge_affinity_weights(
            self,
            is_same_class: torch.Tensor,
            is_same_obj: torch.Tensor
    ) -> torch.Tensor:
        """Helper function to compute edge weights to be used by the
        edge affinity loss. Each edge may have a different weight, based
        on whether its source and target nodes have the same class or
        belong to the same object. The weight given to each case
        (same-class and same-object, same-class and different object,
        etc..) is specified in `edge_affinity_loss_weights`.

        :param is_same_class: BoolTensor
            Mask indicating edges between nodes of the same semantic
            class
        :param is_same_obj: BoolTensor
            Mask indicating edges between nodes of the same object
        """
        # Recover the weights given to each case
        w = self.hparams.edge_affinity_loss_weights

        # If edge_affinity_loss_weights was not specified, no weighting
        # scheme will be applied to the edges
        if w is None or not len(w) == 4:
            return None

        # Compute the weight for each edge
        edge_weight = torch.ones_like(is_same_class).float()
        edge_weight[is_same_class * is_same_obj] = w[0]
        edge_weight[is_same_class * ~is_same_obj] = w[1]
        edge_weight[~is_same_class * is_same_obj] = w[2]
        edge_weight[~is_same_class * ~is_same_obj] = w[3]
        return edge_weight

    def model_step(
            self,
            batch: NAG
    ) -> Tuple[torch.Tensor, PanopticSegmentationOutput]:
        # Loss and predictions for semantic segmentation
        semantic_loss, output = super().model_step(batch)

        # Cannot compute losses if some target data are missing
        if not output.has_target:
            return None, output

        # Compute the node offset loss, weighted by the node size
        # node_offset_loss = self.node_offset_criterion(
        #     *output.sanitized_node_offsets)

        # Compute the edge affinity loss
        edge_affinity_pred, edge_affinity_target, is_same_class, is_same_obj = \
            output.sanitized_edge_affinities()
        edge_weight = self._edge_affinity_weights(is_same_class, is_same_obj)
        edge_affinity_loss = self.edge_affinity_criterion(
            edge_affinity_pred, edge_affinity_target, edge_weight)

        # Combine the losses together
        # TODO: remove node offset cleanly
        # loss = semantic_loss \
        #        + self.hparams.edge_affinity_loss_lambda * edge_affinity_loss \
        #        + self.hparams.node_offset_loss_lambda * node_offset_loss
        loss = semantic_loss \
               + self.hparams.edge_affinity_loss_lambda * edge_affinity_loss

        # Save individual losses in the output object
        output.semantic_loss = semantic_loss
        # TODO: remove node offset cleanly
        # output.node_offset_loss = 0
        output.edge_affinity_loss = edge_affinity_loss

        return loss, output

    def train_step_update_metrics(
            self,
            loss: torch.Tensor,
            output: PanopticSegmentationOutput
    ) -> None:
        """Update train metrics with the content of the output object.
        """
        # Update semantic segmentation metrics
        super().train_step_update_metrics(loss, output)

        # Update instance and panoptic metrics
        if self.needs_partition and not output.has_multi_instance_pred:
            obj_score, obj_y, instance_data = output.panoptic_pred()
            obj_score = obj_score.detach().cpu()
            obj_y = obj_y.detach()
            obj_hist = instance_data.target_label_histogram(self.num_classes)
            self.train_panoptic.update(obj_y.cpu(), instance_data.cpu())
            self.train_semantic(obj_y, obj_hist)
            if self.needs_instance:
                self.train_instance.update(obj_score, obj_y, instance_data.cpu())
        elif self.needs_partition:
            logits = output.logits[0] if output.multi_stage else output.logits
            storage = PartitionParameterSearchStorage(
                    logits.detach().cpu(),
                    self.stuff_classes,
                    output.node_size.detach().cpu(),
                    output.edge_affinity_logits.detach().cpu(),
                    output.obj.cpu(),
                    [(v[0], v[1].detach().cpu()) for v in output.obj_index_pred])
            self.train_multi_partition_storage.append(storage)

        # Update tracked losses
        self.train_semantic_loss(output.semantic_loss.detach())
        # self.train_node_offset_loss(output.node_offset_loss.detach())
        self.train_edge_affinity_loss(output.edge_affinity_loss.detach())

        # Update node offset metrics
        # node_offset_pred, node_offset, node_size = output.sanitized_node_offsets
        # node_offset_pred = node_offset_pred.detach()
        # node_offset = node_offset.detach()
        # node_size = node_size.detach()
        # self.train_offset_wl2(node_offset_pred, node_offset, node_size)
        # self.train_offset_wl1(node_offset_pred, node_offset, node_size)
        # self.train_offset_l2(node_offset_pred, node_offset)
        # self.train_offset_l1(node_offset_pred, node_offset)

        # Update edge affinity metrics
        ea_pred, ea_target, is_same_class, is_same_obj = \
            output.sanitized_edge_affinities()
        ea_pred = ea_pred.detach()
        ea_target_binary = (ea_target.detach() > 0.5).long()
        self.train_affinity_oa(ea_pred, ea_target_binary)
        self.train_affinity_f1(ea_pred, ea_target_binary)

    def train_step_log_metrics(self) -> None:
        """Log train metrics after a single step with the content of the
        output object.
        """
        super().train_step_log_metrics()
        self.log(
            "train/semantic_loss", self.train_semantic_loss, on_step=False,
            on_epoch=True, prog_bar=True)
        # self.log(
        #     "train/node_offset_loss", self.train_node_offset_loss, on_step=False,
        #     on_epoch=True, prog_bar=True)
        self.log(
            "train/edge_affinity_loss", self.train_edge_affinity_loss, on_step=False,
            on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        # Log semantic segmentation metrics and reset confusion matrix
        super().on_train_epoch_end()

        # TODO: support logging panoptic metrics for DDP
        if self.trainer.num_devices > 1:
            log.warning(
                "Panoptic and instance segmentation metrics are not guaranteed "
                "to be well-behaved on DDP yet.")

        if self.needs_partition:
            # If multiple partitions settings were tested during the
            # epoch, this will search for the best one, update the
            # internal states of train metrics with related predictions,
            # and update the partitioner's settings
            setting = self._compute_best_partition_settings()[0]

            # Compute the instance and panoptic metrics
            panoptic_results = self.train_panoptic.compute()
            if self.needs_instance:
                instance_results = self.train_instance.compute()

            # Gather tracked metrics
            pq = panoptic_results.pq
            sq = panoptic_results.sq
            rq = panoptic_results.rq
            pq_thing = panoptic_results.pq_thing
            pq_stuff = panoptic_results.pq_stuff
            pqmod = panoptic_results.pq_modified
            mprec = panoptic_results.mean_precision
            mrec = panoptic_results.mean_recall
            pq_per_class = panoptic_results.pq_per_class
            if self.needs_instance:
                map = instance_results.map
                map_50 = instance_results.map_50
                map_75 = instance_results.map_75
                map_per_class = instance_results.map_per_class

            # Log metrics
            self.log("train/pq", 100 * pq, prog_bar=True)
            self.log("train/sq", 100 * sq, prog_bar=True)
            self.log("train/rq", 100 * rq, prog_bar=True)
            self.log("train/pq_thing", 100 * pq_thing, prog_bar=True)
            self.log("train/pq_stuff", 100 * pq_stuff, prog_bar=True)
            self.log("train/pqmod", 100 * pqmod, prog_bar=True)
            self.log("train/mprec", 100 * mprec, prog_bar=True)
            self.log("train/mrec", 100 * mrec, prog_bar=True)
            self.log("train/instance_miou", self.train_semantic.miou(), prog_bar=True)
            self.log("train/instance_oa", self.train_semantic.oa(), prog_bar=True)
            self.log("train/instance_macc", self.train_semantic.macc(), prog_bar=True)
            for iou, seen, name in zip(*self.train_semantic.iou(), self.class_names):
                if seen:
                    self.log(f"train/instance_iou_{name}", iou, prog_bar=True)
            if self.needs_instance:
                self.log("train/map", 100 * map, prog_bar=True)
                self.log("train/map_50", 100 * map_50, prog_bar=True)
                self.log("train/map_75", 100 * map_75, prog_bar=True)
            for pq_c, name in zip(pq_per_class, self.class_names):
                self.log(f"train/pq_{name}", 100 * pq_c, prog_bar=True)
            if self.needs_instance:
                for map_c, name in zip(map_per_class, self.class_names):
                    self.log(f"train/map_{name}", 100 * map_c, prog_bar=True)
            if setting is not None:
                for k, v in setting.items():
                    self.log(f"partition_settings/{k}", v, prog_bar=True)

        # Log metrics
        # self.log("train/offset_wl2", self.train_offset_wl2.compute(), prog_bar=True)
        # self.log("train/offset_wl1", self.train_offset_wl1.compute(), prog_bar=True)
        # self.log("train/offset_l2", self.train_offset_l2.compute(), prog_bar=True)
        # self.log("train/offset_l1", self.train_offset_l1.compute(), prog_bar=True)
        self.log("train/affinity_oa", 100 * self.train_affinity_oa.compute(), prog_bar=True)
        self.log("train/affinity_f1", 100 * self.train_affinity_f1.compute(), prog_bar=True)

        # Reset metrics accumulated over the last epoch
        # self.train_offset_wl2.reset()
        # self.train_offset_wl1.reset()
        # self.train_offset_l2.reset()
        # self.train_offset_l1.reset()
        self.train_affinity_oa.reset()
        self.train_affinity_f1.reset()
        self.train_panoptic.reset()
        self.train_semantic.reset()
        self.train_instance.reset()

    def _compute_best_partition_settings(
            self,
            monitor: str = 'pq',
            maximize: bool = True
    ) -> Tuple[Dict, float]:
        """Compute the best partition settings from
        `self.train_multi_partition_storage`. This will have the
        following internal effects:
          - `self.partitioner` will be updated with the settings which
            produced the best metrics on the epoch
          - `self.train_panoptic` will be updated with the batch
            predictions with the best settings
          - `self.train_instance` will be updated with the batch
            predictions with the best settings, if required

        :param monitor: str
            The metric based on which we will select the best settings
        :param maximize: bool
            Whether the monitored metric should be maximized or
            minimized
        :return:
        """
        # Nothing happens if multi-partition was not activated during
        # the epoch
        if len(self.train_multi_partition_storage) == 0:
            return None, None

        # Reset the instance and panoptic metrics, these will be used to
        # compute metric performance
        self.train_panoptic.reset()
        self.train_instance.reset()

        # Check whether the metric to monitor is for the semantic or
        # panoptic segmentation task
        if monitor in self.train_panoptic.__slots__:
            task = 'panoptic'
            meter = self.train_panoptic
        elif monitor in self.train_instance.__slots__:
            task = 'instance'
            meter = self.train_instance
        else:
            raise ValueError(f"Unknown metric, cannot monitor '{monitor}'.")
        if task == 'instance' and not self.needs_instance:
            raise ValueError(
                'Cannot compute the best partition settings on the train set '
                'based on instance metrics if `self.needs_instance` is False')

        # Recover from the first PartitionParameterSearchStorage, which
        # settings were explored
        settings = self.train_multi_partition_storage[0].settings

        # Compute the metric for each partition setting while tracking
        # the best setting
        best_metric = -torch.inf if maximize else torch.inf
        best_setting = None
        for s in settings:

            # Accumulate batch predictions in the meter
            for storage in self.train_multi_partition_storage:
                obj_score, obj_y, instance_data = \
                    storage.panoptic_pred(s)
                if task == 'panoptic':
                    meter.update(obj_y, instance_data)
                else:
                    meter.update(obj_score, obj_y, instance_data)

            # Compute the monitored metric on the whole epoch
            metric = getattr(meter.compute(), monitor)

            # Update the best metric and settings
            condition = (metric > best_metric) if maximize \
                else (metric < best_setting)
            if condition:
                best_metric = metric
                best_setting = s

            # Reset the meter to avoid mixing predictions of different
            # settings
            meter.reset()

        # Update the partitioner with the best metrics
        for k, v in best_setting.items():
            setattr(self.partitioner, k, v)

        # Update the train meters with the data for computation of
        # logged metrics with the accumulated data from the best
        # setting, thus mimicking a normal epoch with a single partition
        # prediction per batch
        for storage in self.train_multi_partition_storage:
            obj_score, obj_y, instance_data = \
                storage.panoptic_pred(best_setting)
            obj_hist = instance_data.target_label_histogram(self.num_classes)
            self.train_panoptic.update(obj_y, instance_data)
            self.train_semantic(
                obj_y.to(self.train_semantic.device),
                obj_hist.to(self.train_semantic.device))
            if self.needs_instance:
                self.train_instance.update(obj_score, obj_y, instance_data)

        return best_setting, best_metric

    def validation_step_update_metrics(
            self,
            loss: torch.Tensor,
            output: PanopticSegmentationOutput
    ) -> None:
        """Update validation metrics with the content of the output
        object.
        """
        # Update semantic segmentation metrics
        super().validation_step_update_metrics(loss, output)

        # Update instance and panoptic metrics
        if self.needs_partition:
            obj_score, obj_y, instance_data = output.panoptic_pred()
            obj_score = obj_score.detach().cpu()
            obj_y = obj_y.detach()
            obj_hist = instance_data.target_label_histogram(self.num_classes)
            self.val_panoptic.update(obj_y.cpu(), instance_data.cpu())
            self.val_semantic(obj_y, obj_hist)
            if self.needs_instance:
                self.val_instance.update(obj_score, obj_y, instance_data.cpu())

        # Update tracked losses
        self.val_semantic_loss(output.semantic_loss.detach())
        # self.val_node_offset_loss(output.node_offset_loss.detach())
        self.val_edge_affinity_loss(output.edge_affinity_loss.detach())

        # Update node offset metrics
        # node_offset_pred, node_offset, node_size = output.sanitized_node_offsets
        # node_offset_pred = node_offset_pred.detach()
        # node_offset = node_offset.detach()
        # node_size = node_size.detach()
        # self.val_offset_wl2(node_offset_pred, node_offset, node_size)
        # self.val_offset_wl1(node_offset_pred, node_offset, node_size)
        # self.val_offset_l2(node_offset_pred, node_offset)
        # self.val_offset_l1(node_offset_pred, node_offset)

        # Update edge affinity metrics
        ea_pred, ea_target, is_same_class, is_same_obj = \
            output.sanitized_edge_affinities()
        ea_pred = ea_pred.detach()
        ea_target_binary = (ea_target.detach() > 0.5).long()
        self.val_affinity_oa(ea_pred, ea_target_binary)
        self.val_affinity_f1(ea_pred, ea_target_binary)

    def validation_step_log_metrics(self) -> None:
        """Log validation metrics after a single step with the content
        of the output object.
        """
        super().validation_step_log_metrics()
        self.log(
            "val/semantic_loss", self.val_semantic_loss, on_step=False,
            on_epoch=True, prog_bar=True)
        # self.log(
        #     "val/node_offset_loss", self.val_node_offset_loss, on_step=False,
        #     on_epoch=True, prog_bar=True)
        self.log(
            "val/edge_affinity_loss", self.val_edge_affinity_loss, on_step=False,
            on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        # Log semantic segmentation metrics and reset confusion matrix
        super().on_validation_epoch_end()

        # TODO: support logging panoptic metrics for DDP
        if self.trainer.num_devices > 1:
            log.warning(
                "Panoptic and instance segmentation metrics are not guaranteed "
                "to be well-behaved on DDP yet.")

        if self.needs_partition:
            # Compute the instance and panoptic metrics
            panoptic_results = self.val_panoptic.compute()
            if self.needs_instance:
                instance_results = self.val_instance.compute()

            # Gather tracked metrics
            pq = panoptic_results.pq
            sq = panoptic_results.sq
            rq = panoptic_results.rq
            pq_thing = panoptic_results.pq_thing
            pq_stuff = panoptic_results.pq_stuff
            pqmod = panoptic_results.pq_modified
            mprec = panoptic_results.mean_precision
            mrec = panoptic_results.mean_recall
            pq_per_class = panoptic_results.pq_per_class
            if self.needs_instance:
                map = instance_results.map
                map_50 = instance_results.map_50
                map_75 = instance_results.map_75
                map_per_class = instance_results.map_per_class

            # Log metrics
            self.log("val/pq", 100 * pq, prog_bar=True)
            self.log("val/sq", 100 * sq, prog_bar=True)
            self.log("val/rq", 100 * rq, prog_bar=True)
            self.log("val/pq_thing", 100 * pq_thing, prog_bar=True)
            self.log("val/pq_stuff", 100 * pq_stuff, prog_bar=True)
            self.log("val/pqmod", 100 * pqmod, prog_bar=True)
            self.log("val/mprec", 100 * mprec, prog_bar=True)
            self.log("val/mrec", 100 * mrec, prog_bar=True)
            instance_miou = self.val_semantic.miou()
            instance_oa = self.val_semantic.oa()
            instance_macc = self.val_semantic.macc()
            self.log("val/instance_miou", instance_miou, prog_bar=True)
            self.log("val/instance_oa", instance_oa, prog_bar=True)
            self.log("val/instance_macc", instance_macc, prog_bar=True)
            for iou, seen, name in zip(*self.val_semantic.iou(), self.class_names):
                if seen:
                    self.log(f"val/instance_iou_{name}", iou, prog_bar=True)
            if self.needs_instance:
                self.log("val/map", 100 * map, prog_bar=True)
                self.log("val/map_50", 100 * map_50, prog_bar=True)
                self.log("val/map_75", 100 * map_75, prog_bar=True)
            for pq_c, name in zip(pq_per_class, self.class_names):
                self.log(f"val/pq_{name}", 100 * pq_c, prog_bar=True)
            if self.needs_instance:
                for map_c, name in zip(map_per_class, self.class_names):
                    self.log(f"val/map_{name}", 100 * map_c, prog_bar=True)

            # Update best-so-far metrics
            self.val_pq_best(pq)
            self.val_pqmod_best(pqmod)
            self.val_mprec_best(mprec)
            self.val_mrec_best(mrec)
            if self.needs_instance:
                self.val_map_best(map)
            self.val_instance_miou_best(instance_miou)
            self.val_instance_oa_best(instance_oa)
            self.val_instance_macc_best(instance_macc)

            # Log best-so-far metrics, using `.compute()` instead of passing
            # the whole torchmetrics object, because otherwise metric would
            # be reset by lightning after each epoch
            self.log("val/pq_best", 100 * self.val_pq_best.compute(), prog_bar=True)
            self.log("val/pqmod_best", 100 * self.val_pqmod_best.compute(), prog_bar=True)
            self.log("val/mprec_best", 100 * self.val_mprec_best.compute(), prog_bar=True)
            self.log("val/mrec_best", 100 * self.val_mrec_best.compute(), prog_bar=True)
            if self.needs_instance:
                self.log("val/map_best", 100 * self.val_map_best.compute(), prog_bar=True)
            self.log("val/instance_miou_best", self.val_instance_miou_best.compute(), prog_bar=True)
            self.log("val/instance_oa_best", self.val_instance_oa_best.compute(), prog_bar=True)
            self.log("val/instance_macc_best", self.val_instance_macc_best.compute(), prog_bar=True)

        # Compute the metrics tracked for model selection on validation
        # offset_wl2 = self.val_offset_wl2.compute()
        # offset_wl1 = self.val_offset_wl1.compute()
        # offset_l2 = self.val_offset_l2.compute()
        # offset_l1 = self.val_offset_l1.compute()
        affinity_oa = self.val_affinity_oa.compute()
        affinity_f1 = self.val_affinity_f1.compute()

        # Log metrics
        # self.log("val/offset_wl2", offset_wl2, prog_bar=True)
        # self.log("val/offset_wl1", offset_wl1, prog_bar=True)
        # self.log("val/offset_l2", offset_l2, prog_bar=True)
        # self.log("val/offset_l1", offset_l1, prog_bar=True)
        self.log("val/affinity_oa", 100 * affinity_oa, prog_bar=True)
        self.log("val/affinity_f1", 100 * affinity_f1, prog_bar=True)

        # Update best-so-far metrics
        # self.val_offset_wl2_best(offset_wl2)
        # self.val_offset_wl1_best(offset_wl1)
        # self.val_offset_l2_best(offset_l2)
        # self.val_offset_l1_best(offset_l1)
        self.val_affinity_oa_best(affinity_oa)
        self.val_affinity_f1_best(affinity_f1)

        # Log best-so-far metrics, using `.compute()` instead of passing
        # the whole torchmetrics object, because otherwise metric would
        # be reset by lightning after each epoch
        # self.log("val/offset_wl2_best", self.val_offset_wl2_best.compute(), prog_bar=True)
        # self.log("val/offset_wl1_best", self.val_offset_wl1_best.compute(), prog_bar=True)
        # self.log("val/offset_l2_best", self.val_offset_l2_best.compute(), prog_bar=True)
        # self.log("val/offset_l1_best", self.val_offset_l1_best.compute(), prog_bar=True)
        self.log("val/affinity_oa_best", 100 * self.val_affinity_oa_best.compute(), prog_bar=True)
        self.log("val/affinity_f1_best", 100 * self.val_affinity_f1_best.compute(), prog_bar=True)

        # Reset metrics accumulated over the last epoch
        # self.val_offset_wl2.reset()
        # self.val_offset_wl1.reset()
        # self.val_offset_l2.reset()
        # self.val_offset_l1.reset()
        self.val_affinity_oa.reset()
        self.val_affinity_f1.reset()
        self.val_panoptic.reset()
        self.val_semantic.reset()
        self.val_instance.reset()

    def test_step_update_metrics(
            self,
            loss: torch.Tensor,
            output: PanopticSegmentationOutput
    ) -> None:
        """Update test metrics with the content of the output object.
        """
        # Update semantic segmentation metrics
        super().test_step_update_metrics(loss, output)

        # If the test set misses targets, we keep track of it, to skip
        # metrics computation on the test set
        if not self.test_has_target:
            return

        # Update instance and panoptic metrics
        if self.needs_partition:
            obj_score, obj_y, instance_data = output.panoptic_pred()
            obj_score = obj_score.detach().cpu()
            obj_y = obj_y.detach()
            obj_hist = instance_data.target_label_histogram(self.num_classes)
            self.test_panoptic.update(obj_y.cpu(), instance_data.cpu())
            self.test_semantic(obj_y, obj_hist)
            if self.needs_instance:
                self.test_instance.update(obj_score, obj_y, instance_data.cpu())

        # Update tracked losses
        self.test_semantic_loss(output.semantic_loss.detach())
        # self.test_node_offset_loss(output.node_offset_loss.detach())
        self.test_edge_affinity_loss(output.edge_affinity_loss.detach())

        # Update node offset metrics
        # node_offset_pred, node_offset, node_size = output.sanitized_node_offsets
        # node_offset_pred = node_offset_pred.detach()
        # node_offset = node_offset.detach()
        # node_size = node_size.detach()
        # self.test_offset_wl2(node_offset_pred, node_offset, node_size)
        # self.test_offset_wl1(node_offset_pred, node_offset, node_size)
        # self.test_offset_l2(node_offset_pred, node_offset)
        # self.test_offset_l1(node_offset_pred, node_offset)

        # Update edge affinity metrics
        ea_pred, ea_target, is_same_class, is_same_obj = \
            output.sanitized_edge_affinities()
        ea_pred = ea_pred.detach()
        ea_target_binary = (ea_target.detach() > 0.5).long()
        self.test_affinity_oa(ea_pred, ea_target_binary)
        self.test_affinity_f1(ea_pred, ea_target_binary)

    def test_step_log_metrics(self) -> None:
        """Log test metrics after a single step with the content of the
        output object.
        """
        super().test_step_log_metrics()

        # If the test set misses targets, we keep track of it, to skip
        # metrics computation on the test set
        if not self.test_has_target:
            return

        self.log(
            "test/semantic_loss", self.test_semantic_loss, on_step=False,
            on_epoch=True, prog_bar=True)
        # self.log(
        #     "test/node_offset_loss", self.test_node_offset_loss, on_step=False,
        #     on_epoch=True, prog_bar=True)
        self.log(
            "test/edge_affinity_loss", self.test_edge_affinity_loss, on_step=False,
            on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        # Log semantic segmentation metrics and reset confusion matrix
        super().on_test_epoch_end()

        # If test set misses target data, reset metrics and skip logging
        if not self.test_has_target:
            # self.test_offset_wl2.reset()
            # self.test_offset_wl1.reset()
            # self.test_offset_l2.reset()
            # self.test_offset_l1.reset()
            self.test_affinity_oa.reset()
            self.test_affinity_f1.reset()
            self.test_panoptic.reset()
            self.test_semantic.reset()
            self.test_instance.reset()
            return

        # TODO: support logging panoptic metrics for DDP
        if self.trainer.num_devices > 1:
            log.warning(
                "Panoptic and instance segmentation metrics are not guaranteed "
                "to be well-behaved on DDP yet.")

        if self.needs_partition:
            # Compute the instance and panoptic metrics
            panoptic_results = self.test_panoptic.compute()
            if self.needs_instance:
                instance_results = self.test_instance.compute()

            # Gather tracked metrics
            pq = panoptic_results.pq
            sq = panoptic_results.sq
            rq = panoptic_results.rq
            pq_thing = panoptic_results.pq_thing
            pq_stuff = panoptic_results.pq_stuff
            pqmod = panoptic_results.pq_modified
            mprec = panoptic_results.mean_precision
            mrec = panoptic_results.mean_recall
            pq_per_class = panoptic_results.pq_per_class
            if self.needs_instance:
                map = instance_results.map
                map_50 = instance_results.map_50
                map_75 = instance_results.map_75
                map_per_class = instance_results.map_per_class

            # Log metrics
            self.log("test/pq", 100 * pq, prog_bar=True)
            self.log("test/sq", 100 * sq, prog_bar=True)
            self.log("test/rq", 100 * rq, prog_bar=True)
            self.log("test/pq_thing", 100 * pq_thing, prog_bar=True)
            self.log("test/pq_stuff", 100 * pq_stuff, prog_bar=True)
            self.log("test/pqmod", 100 * pqmod, prog_bar=True)
            self.log("test/mprec", 100 * mprec, prog_bar=True)
            self.log("test/mrec", 100 * mrec, prog_bar=True)
            self.log("test/instance_miou", self.test_semantic.miou(), prog_bar=True)
            self.log("test/instance_oa", self.test_semantic.oa(), prog_bar=True)
            self.log("test/instance_macc", self.test_semantic.macc(), prog_bar=True)
            for iou, seen, name in zip(*self.test_semantic.iou(), self.class_names):
                if seen:
                    self.log(f"test/instance_iou_{name}", iou, prog_bar=True)
            if self.needs_instance:
                self.log("test/map", 100 * map, prog_bar=True)
                self.log("test/map_50", 100 * map_50, prog_bar=True)
                self.log("test/map_75", 100 * map_75, prog_bar=True)
            for pq_c, name in zip(pq_per_class, self.class_names):
                self.log(f"test/pq_{name}", 100 * pq_c, prog_bar=True)
            if self.needs_instance:
                for map_c, name in zip(map_per_class, self.class_names):
                    self.log(f"test/map_{name}", 100 * map_c, prog_bar=True)

        # Log metrics
        # self.log("test/offset_wl2", self.test_offset_wl2.compute(), prog_bar=True)
        # self.log("test/offset_wl1", self.test_offset_wl1.compute(), prog_bar=True)
        # self.log("test/offset_l2", self.test_offset_l2.compute(), prog_bar=True)
        # self.log("test/offset_l1", self.test_offset_l1.compute(), prog_bar=True)
        self.log("test/affinity_oa", 100 * self.test_affinity_oa.compute(), prog_bar=True)
        self.log("test/affinity_f1", 100 * self.test_affinity_f1.compute(), prog_bar=True)

        # Reset metrics accumulated over the last epoch
        # self.test_offset_wl2.reset()
        # self.test_offset_wl1.reset()
        # self.test_offset_l2.reset()
        # self.test_offset_l1.reset()
        self.test_affinity_oa.reset()
        self.test_affinity_f1.reset()
        self.test_panoptic.reset()
        self.test_semantic.reset()
        self.test_instance.reset()

    def track_batch(
            self,
            batch: NAG,
            batch_idx: int,
            output: PanopticSegmentationOutput,
            folder: str = None
    ) -> None:
        """Store a batch prediction to disk. The corresponding `NAG`
        object will be populated with panoptic segmentation predictions
        for:
        - levels 1+ if `multi_stage` output (i.e. loss supervision on
          levels 1 and above)
        - only level 1 otherwise

        Besides, we also pre-compute the level-0 predictions as this is
        frequently required for downstream tasks. However, we choose not
        to compute the full-resolution predictions for the sake of disk
        memory.

        If a `folder` is provided, the NAG will be saved there under:
          <folder>/predictions/<stage>/<epoch>/batch_<batch_idx>.h5
        If not, the folder will be the logger's directory, if any.
        If not, the current working directory will be used.

        :param batch: NAG
            Object that will be stored to disk. Before that, the
            model predictions will be added to the attributes of each
            level, to facilitate downstream use of the stored `NAG`
        :param batch_idx: int
            Index of the batch to be stored
        :param output: PanopticSegmentationOutput
             Output of `self.model_step()`
        :param folder: str
            Path where to save the tracked batch. If not provided, the
            logger's saving directory will be used as fallback. If not
            logger is found, the current working directory will be used
        :return:
        """
        # Sanity check in case using multi-run inference
        if not isinstance(batch, NAG):
            raise NotImplementedError(
                f"Expected as NAG, but received a {type(batch)}. Are you "
                f"perhaps running multi-run inference ? If so, this is not "
                f"compatible with batch_saving, please deactivate either one.")

        # Compute the panoptic partition if not already done
        if output.obj_index_pred is None:
            output = self._forward_partition(batch, output, force=True)

        # Store the output predictions in conveniently-accessible
        # attributes in the NAG, for easy downstream use of the saved
        # object
        sp_y_pred, sp_obj_index_pred, sp_obj_pred = (
            output.superpoint_panoptic_pred())
        vox_y_pred, vox_obj_index_pred, vox_obj_pred = (
            output.voxel_panoptic_pred(super_index=batch[0].super_index))
        batch[1].obj_y_pred = sp_y_pred
        batch[1].obj_index_pred = sp_obj_index_pred
        batch[1].obj_pred = sp_obj_pred
        batch[0].obj_y_pred = vox_y_pred
        batch[0].obj_index_pred = vox_obj_index_pred
        batch[0].obj_pred = vox_obj_pred
        batch[1].edge_affinity_logits = output.edge_affinity_logits

        # Parent behavior for saving semantic segmentation prediction
        super().track_batch(batch, batch_idx, output, folder=folder)

    def load_state_dict(self, state_dict: Dict, strict: bool = True) -> None:
        """Basic `load_state_dict` from `torch.nn.Module` with a bit of
        acrobatics due to `criterion.weight`.

        This attribute, when present in the `state_dict`, causes
        `load_state_dict` to crash. More precisely, `criterion.weight`
        is holding the per-class weights for classification losses.
        """
        # Special treatment for BCEWithLogitsLoss
        if self.edge_affinity_criterion.pos_weight is not None:
            pos_weight_bckp = self.edge_affinity_criterion.pos_weight
            self.edge_affinity_criterion.pos_weight = None

        if 'edge_affinity_criterion.pos_weight' in state_dict.keys():
            pos_weight = state_dict.pop('edge_affinity_criterion.pos_weight')
        else:
            pos_weight = None

        # Load the state_dict
        super().load_state_dict(state_dict, strict=strict)

        # If need be, assign the class weights to the criterion
        if self.edge_affinity_criterion.pos_weight is not None:
            self.edge_affinity_criterion.pos_weight = pos_weight \
                if pos_weight is not None else pos_weight_bckp

    def _load_from_checkpoint(
            self,
            checkpoint_path: str,
            **kwargs
    ) -> 'PanopticSegmentationModule':
        """Simpler version of `LightningModule.load_from_checkpoint()`
        for easier use: no need to explicitly pass `model.net`,
        `model.criterion`, etc.
        """
        return self.__class__.load_from_checkpoint(
            checkpoint_path,
            net=self.net,
            edge_affinity_head=self.edge_affinity_head,
            partitioner=self.partitioner,
            criterion=self.criterion,
            **kwargs)


# TODO: gridsearch instance partition parameters

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = str(pyrootutils.setup_root(__file__, pythonpath=True))
    cfg = omegaconf.OmegaConf.load(root + "/configs/model/panoptic/spt-2.yaml")
    _ = hydra.utils.instantiate(cfg)
