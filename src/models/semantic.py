import torch
import os
import os.path as osp
from torch.nn import ModuleList
import logging
from copy import deepcopy
from typing import Any, List, Tuple, Dict
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from pytorch_lightning.loggers.wandb import WandbLogger

from src.metrics import ConfusionMatrix
from src.utils import loss_with_target_histogram, atomic_to_histogram, \
    init_weights, wandb_confusion_matrix, knn_2, garbage_collection_cuda, \
    SemanticSegmentationOutput
from src.nn import Classifier
from src.loss import MultiLoss
from src.optim.lr_scheduler import ON_PLATEAU_SCHEDULERS
from src.data import NAG
from src.transforms import Transform, NAGSaveNodeIndex

log = logging.getLogger(__name__)


__all__ = ['SemanticSegmentationModule']


class SemanticSegmentationModule(LightningModule):
    """A LightningModule for semantic segmentation of point clouds.

    :param net: torch.nn.Module
        Backbone model. This can typically be an `SPT` object
    :param criterion: torch.nn._Loss
        Loss
    :param optimizer: torch.optim.Optimizer
        Optimizer
    :param scheduler: torch.optim.lr_scheduler.LRScheduler
        Learning rate scheduler
    :param num_classes: int
        Number of classes in the dataset
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
        `track_val_every_n_epoch`. If `track_test_idx=-1`, predictions
        for the entire test set will be saved to disk.
        Importantly, this index is expected to match the `Dataloader`'s
        index wrt the current epoch and NOT an index wrt the `Dataset`.
        Said otherwise, if the `Dataloader(shuffle=True)` then, the
        stored batch will not be the same at each epoch. For this
        reason, if tracking the same object across training is needed,
        the `Dataloader` and the transforms should be free from any
        stochasticity
    :param track_test_idx:
        If specified, the output for the `track_test_idx`th
        test batch will be saved to disk. If `track_test_idx=-1`,
        predictions for the entire test set will be saved to disk
    :param kwargs: Dict
        Kwargs will be passed to `_load_from_checkpoint()`
    """

    _IGNORED_HYPERPARAMETERS = ['net', 'criterion']

    def __init__(
            self,
            net: torch.nn.Module,
            criterion: 'torch.nn._Loss',
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            num_classes: int,
            class_names: List[str] = None,
            sampling_loss: bool = False,
            loss_type: str = 'ce_kl',
            weighted_loss: bool = True,
            init_linear: str = None,
            init_rpe: str = None,
            transformer_lr_scale: float = 1,
            multi_stage_loss_lambdas: List[float] = None,
            gc_every_n_steps: int = 0,
            track_val_every_n_epoch: int = 1,
            track_val_idx: int = None,
            track_test_idx: int = None,
            **kwargs):
        super().__init__()

        # Allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=self._IGNORED_HYPERPARAMETERS)

        # Store the number of classes and the class names
        self.num_classes = num_classes
        self.class_names = class_names if class_names is not None \
            else [f'class-{i}' for i in range(num_classes)]

        # Loss function. If `multi_stage_loss_lambdas`, a MultiLoss is
        # built based on the input criterion
        if isinstance(criterion, MultiLoss):
            self.criterion = criterion
        elif multi_stage_loss_lambdas is not None:
            criteria = [
                deepcopy(criterion)
                for _ in range(len(multi_stage_loss_lambdas))]
            self.criterion = MultiLoss(criteria, multi_stage_loss_lambdas)
        else:
            self.criterion = criterion

        # Ignore the `num_classes` labels, which, by construction, are
        # where we send all 'ignored'/'void' annotations
        if isinstance(self.criterion, MultiLoss):
            for i in range(len(self.criterion.criteria)):
                self.criterion.criteria[i].ignore_index = num_classes
        else:
            self.criterion.ignore_index = num_classes

        # Network that will do the actual computation. NB, we make sure
        # the net returns the output from all up stages, if a multi-stage
        # loss is expected
        self.net = net
        if self.multi_stage_loss:
            self.net.output_stage_wise = True
            assert len(self.net.out_dim) == len(self.criterion), \
                f"The number of items in the multi-stage loss must match the " \
                f"number of stages in the net. Found " \
                f"{len(self.net.out_dim)} stages, but {len(self.criterion)} " \
                f"criteria in the loss."

        # Initialize the model segmentation head (or heads)
        if self.multi_stage_loss:
            self.head = ModuleList([
                Classifier(dim, num_classes) for dim in self.net.out_dim])
        else:
            self.head = Classifier(self.net.out_dim, num_classes)

        # Custom weight initialization. In particular, this applies
        # Xavier / Glorot initialization on Linear and RPE layers by
        # default, but can be tuned
        init = lambda m: init_weights(m, linear=init_linear, rpe=init_rpe)
        self.net.apply(init)
        self.head.apply(init)

        # Metric objects for calculating scores on each dataset split.
        # We add `ignore_index=num_classes` to account for
        # void/unclassified/ignored points, which are given
        # `num_classes` labels
        self.train_cm = ConfusionMatrix(num_classes)
        self.val_cm = ConfusionMatrix(num_classes)
        self.test_cm = ConfusionMatrix(num_classes)

        # For averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # For tracking best-so-far validation metrics
        self.val_miou_best = MaxMetric()
        self.val_oa_best = MaxMetric()
        self.val_macc_best = MaxMetric()

        # For tracking whether the test set has target labels. By
        # default, we assume the test set to have labels. But if a
        # single test batch misses labels, this will be set to False and
        # all test metrics computation will be skipped
        self.test_has_target = True

        # Explicitly call the garbage collector after a certain number
        # of steps
        self.gc_every_n_steps = int(gc_every_n_steps)

    def forward(self, nag: NAG) -> SemanticSegmentationOutput:
        x = self.net(nag)
        logits = [head(x_) for head, x_ in zip(self.head, x)] \
            if self.multi_stage_loss else self.head(x)
        return SemanticSegmentationOutput(logits)

    @property
    def multi_stage_loss(self) -> bool:
        return isinstance(self.criterion, MultiLoss)

    def on_fit_start(self) -> None:
        # This is a bit of a late initialization for the LightningModule
        # At this point, we can access some LightningDataModule-related
        # parameters that were not available beforehand. So we take this
        # opportunity to catch the number of classes or class weights
        # from the LightningDataModule

        # Get the LightningDataModule number of classes and make sure it
        # matches self.num_classes. We could also forcefully update the
        # LightningModule with this new information, but it could easily
        # become tedious to track all places where num_classes affects
        # the LightningModule object.
        num_classes = self.trainer.datamodule.train_dataset.num_classes
        assert num_classes == self.num_classes, \
            f'LightningModule has {self.num_classes} classes while the ' \
            f'LightningDataModule has {num_classes} classes.'

        self.class_names = self.trainer.datamodule.train_dataset.class_names

        if not self.hparams.weighted_loss:
            return

        if not hasattr(self.criterion, 'weight'):
            log.warning(
                f"{self.criterion} does not have a 'weight' attribute. "
                f"Class weights will be ignored...")
            return

        # Set class weights for the criterion
        weight = self.trainer.datamodule.train_dataset.get_class_weight()
        self.criterion.weight = weight.to(self.device)

        # Check that the period of track_val_every_n_epoch` is a
        # multiple of check_val_every_n_epoch
        if self.trainer.check_val_every_n_epoch is not None:
            assert (self.hparams.track_val_every_n_epoch
                    % self.trainer.check_val_every_n_epoch == 0), \
                (f"Expected 'track_val_every_n_epoch' to be a multiple of "
                 f"'check_val_every_n_epoch', but received "
                 f"{self.hparams.track_val_every_n_epoch} and "
                 f"{self.trainer.check_val_every_n_epoch} instead.")

    def on_train_start(self) -> None:
        # By default, lightning executes validation step sanity checks
        # before training starts, so we need to make sure `*_best`
        # metrics do not store anything from these checks
        self.val_cm.reset()
        self.val_miou_best.reset()
        self.val_oa_best.reset()
        self.val_macc_best.reset()

    def gc_collect(self) -> None:
        num_steps = self.trainer.fit_loop.epoch_loop._batches_that_stepped + 1
        period = self.gc_every_n_steps
        if period is None or period < 1:
            return
        if num_steps % period == 0:
            garbage_collection_cuda()

    def on_train_batch_start(self, *args) -> None:
        self.gc_collect()

    def on_validation_batch_start(self, *args) -> None:
        self.gc_collect()

    def on_test_batch_start(self, *args) -> None:
        self.gc_collect()

    def model_step(
            self,
            batch: NAG
    ) -> Tuple[torch.Tensor, SemanticSegmentationOutput]:
        # Forward step on the input batch. If a (NAG, Transform, int)
        # tuple is passed, the multi-run inference will be triggered
        output = self.step_single_run_inference(batch) \
            if isinstance(batch, NAG) \
            else self.step_multi_run_inference(*batch)

        # If the input batch does not have labels (e.g. test set with
        # held-out labels), y_hist will be None and the loss will not be
        # computed
        if not output.has_target:
            return None, output

        # Compute the loss either in a point-wise or segment-wise
        # fashion. Cross-Entropy with pointwise_loss is equivalent to
        # KL-divergence
        if self.multi_stage_loss:
            if self.hparams.loss_type == 'ce':
                loss = self.criterion(
                    output.logits, [y.argmax(dim=1) for y in output.y_hist])
            elif self.hparams.loss_type == 'wce':
                y_hist_dominant = []
                for y in output.y_hist:
                    y_dominant = y.argmax(dim=1)
                    y_hist_dominant_ = torch.zeros_like(y)
                    y_hist_dominant_[:, y_dominant] = y.sum(dim=1)
                    y_hist_dominant.append(y_hist_dominant_)
                loss = 0
                enum = zip(
                    self.criterion.lambdas,
                    self.criterion.criteria,
                    output.logits,
                    y_hist_dominant)
                for lamb, criterion, a, b in enum:
                    loss = loss + lamb * loss_with_target_histogram(
                        criterion, a, b)
            elif self.hparams.loss_type == 'ce_kl':
                loss = 0
                enum = zip(
                    self.criterion.lambdas,
                    self.criterion.criteria,
                    output.logits,
                    output.y_hist)
                for i, (lamb, criterion, a, b) in enumerate(enum):
                    if i == 0:
                        loss = loss + criterion(a, b.argmax(dim=1))
                        continue
                    loss = loss + lamb * loss_with_target_histogram(
                        criterion, a, b)
            elif self.hparams.loss_type == 'wce_kl':
                loss = 0
                enum = zip(
                    self.criterion.lambdas,
                    self.criterion.criteria,
                    output.logits,
                    output.y_hist)
                for i, (lamb, criterion, a, b) in enumerate(enum):
                    if i == 0:
                        y_dominant = b.argmax(dim=1)
                        y_hist_dominant = torch.zeros_like(b)
                        y_hist_dominant[:, y_dominant] = b.sum(dim=1)
                        loss = loss + loss_with_target_histogram(
                            criterion, a, y_hist_dominant)
                        continue
                    loss = loss + lamb * loss_with_target_histogram(
                        criterion, a, b)
            elif self.hparams.loss_type == 'kl':
                loss = 0
                enum = zip(
                    self.criterion.lambdas,
                    self.criterion.criteria,
                    output.logits,
                    output.y_hist)
                for lamb, criterion, a, b in enum:
                    loss = loss + lamb * loss_with_target_histogram(
                        criterion, a, b)
            else:
                raise ValueError(
                    f"Unknown multi-stage loss '{self.hparams.loss_type}'")
        else:
            if self.hparams.loss_type == 'ce':
                loss = self.criterion(output.logits, output.y_hist.argmax(dim=1))
            elif self.hparams.loss_type == 'wce':
                y_dominant = output.y_hist.argmax(dim=1)
                y_hist_dominant = torch.zeros_like(output.y_hist)
                y_hist_dominant[:, y_dominant] = output.y_hist.sum(dim=1)
                loss = loss_with_target_histogram(
                    self.criterion, output.logits, y_hist_dominant)
            elif self.hparams.loss_type == 'kl':
                loss = loss_with_target_histogram(
                    self.criterion, output.logits, output.y_hist)
            else:
                raise ValueError(
                    f"Unknown single-stage loss '{self.hparams.loss_type}'")

        return loss, output

    def step_single_run_inference(self, nag: NAG) -> SemanticSegmentationOutput:
        """Single-run inference
        """
        output = self.forward(nag)
        output = self.get_target(nag, output)
        return output

    def step_multi_run_inference(
            self,
            nag: NAG,
            transform: Transform,
            num_runs: int,
            key: str = 'tta_node_id'
    ) -> SemanticSegmentationOutput:
        """Multi-run inference, typically with test-time augmentation.
        See `BaseDataModule.on_after_batch_transfer`
        """
        # Since the transform may change the sampling of the nodes, we
        # save their input id here before anything. This will allow us
        # to fuse the multiple predictions for each node
        transform.transforms = [NAGSaveNodeIndex(key=key)] \
                               + transform.transforms

        # Create empty output predictions, to be iteratively populated
        # with the multiple predictions
        output_multi = self._create_empty_output(nag)

        # Recover the target labels from the reference NAG
        output_multi = self.get_target(nag, output_multi)

        # Build the global logits, in which the multi-run
        # logits will be accumulated, before computing their final
        seen = torch.zeros(nag.num_points[1], dtype=torch.bool)

        for i_run in range(num_runs):

            # Apply transform
            nag_ = transform(nag.clone())

            # Forward pass
            output = self.forward(nag_)

            # Update the output results
            output_multi = self._update_output_multi(
                output_multi, nag, output, nag_, key)

            # Maintain the seen/unseen mask for level-1 nodes only
            node_id = nag_[1][key]
            seen[node_id] = True

        # Restore the original transform inplace modification
        transform.transforms = transform.transforms[1:]

        # If some nodes were not seen across any of the multi-runs,
        # search their nearest seen neighbor
        unseen_idx = torch.where(~seen)[0]
        batch = nag[1].batch
        if unseen_idx.shape[0] > 0:
            seen_idx = torch.where(seen)[0]
            x_search = nag[1].pos[seen_idx]
            x_query = nag[1].pos[unseen_idx]
            neighbors = knn_2(
                x_search,
                x_query,
                1,
                r_max=2,
                batch_search=batch[seen_idx] if batch is not None else None,
                batch_query=batch[unseen_idx] if batch is not None else None)[0]
            num_unseen = unseen_idx.shape[0]
            num_seen = seen_idx.shape[0]
            num_left_out = (neighbors == -1).sum().long()
            if num_left_out > 0:
                log.warning(
                    f"Could not find a neighbor for all unseen nodes: num_seen="
                    f"{num_seen}, num_unseen={num_unseen}, num_left_out="
                    f"{num_left_out}. These left out nodes will default to "
                    f"label-0 class prediction. Consider sampling less nodes "
                    f"in the augmentations, or increase the search radius")

            # Propagate the output to unseen neighbors
            output_multi = self._propagate_output_to_unseen_neighbors(
                output_multi, nag, seen, neighbors)

        return output_multi

    def _create_empty_output(self, nag: NAG) -> SemanticSegmentationOutput:
        """Local helper method to initialize an empty output for
        multi-run prediction.
        """
        device = nag.device
        num_classes = self.num_classes
        if self.multi_stage_loss:
            logits = [
                torch.zeros(num_points, num_classes, device=device)
                for num_points in nag.num_points[1:]]
        else:
            logits = torch.zeros(nag.num_points[1], num_classes, device=device)
        return SemanticSegmentationOutput(logits)

    @staticmethod
    def _update_output_multi(
            output_multi: SemanticSegmentationOutput,
            nag: NAG,
            output: SemanticSegmentationOutput,
            nag_transformed: NAG,
            key: str
    ) -> SemanticSegmentationOutput:
        """Local helper method to accumulate multiple predictions on
        the same--or part of the same--point cloud.
        """
        # Recover the node identifier that should have been
        # implanted by `NAGSaveNodeIndex` and forward on the
        # augmented data and update the global logits of the node
        if output.multi_stage:
            for i in range(len(output.logits)):
                node_id = nag_transformed[i + 1][key]
                output_multi.logits[i][node_id] += output.logits[i]
        else:
            node_id = nag_transformed[1][key]
            output_multi.logits[node_id] += output.logits
        return output_multi

    @staticmethod
    def _propagate_output_to_unseen_neighbors(
            output: SemanticSegmentationOutput,
            nag: NAG,
            seen: torch.Tensor,
            neighbors: torch.Tensor
    ) -> SemanticSegmentationOutput:
        """Local helper method to propagate predictions to unseen
        neighbors.
        """
        seen_idx = torch.where(seen)[0]
        unseen_idx = torch.where(~seen)[0]
        if output.multi_stage:
            output.logits[0][unseen_idx] = output.logits[0][seen_idx][neighbors]
        else:
            output.logits[unseen_idx] = output.logits[seen_idx][neighbors]
        return output

    def get_target(
            self,
            nag: NAG,
            output: SemanticSegmentationOutput
    ) -> SemanticSegmentationOutput:
        """Recover the target histogram of labels from the NAG object.
        The labels will be saved in `output.y_hist`.

        If the `multi_stage_loss=True`, a list of label histograms
        will be recovered (one for each prediction level).

        If `sampling_loss=True`, the histogram(s) will be updated based
        on the actual level-0 point sampling. That is, superpoints will
        be supervised by the labels of the sampled points at train time,
        rather than the true full-resolution label histogram.

        If no labels are found in the NAG, `output.y_hist` will be None.
        """
        # Return if the required labels cannot be found in the NAG
        if self.hparams.sampling_loss and nag[0].y is None:
            output.y_hist = None
            return output
        elif self.multi_stage_loss:
            for i in range(1, nag.num_levels):
                if nag[i].y is None:
                    output.y_hist = None
                    return output
        elif nag[1].y is None:
            output.y_hist = None
            return output

        # Recover level-1 label histograms, either from the level-0
        # sampled points (i.e. sampling will affect the loss and metrics)
        # or directly from the precomputed level-1 label histograms (i.e.
        # true annotations)
        if self.hparams.sampling_loss and self.multi_stage_loss:
            y_hist = [
                atomic_to_histogram(
                    nag[0].y,
                    nag.get_super_index(i_level), n_bins=self.num_classes + 1)
                for i_level in range(1, nag.num_levels)]

        elif self.hparams.sampling_loss:
            idx = nag[0].super_index
            y = nag[0].y

            # Convert level-0 labels to segment-level histograms, while
            # accounting for the extra class for unlabeled/ignored points
            y_hist = atomic_to_histogram(y, idx, n_bins=self.num_classes + 1)

        elif self.multi_stage_loss:
            y_hist = [nag[i_level].y for i_level in range(1, nag.num_levels)]

        else:
            y_hist = nag[1].y

        # Store the label histogram in the output object
        output.y_hist = y_hist

        return output

    def training_step(
            self,
            batch: NAG,
            batch_idx: int
    ) -> torch.Tensor:
        loss, output = self.model_step(batch)

        # Update and log metrics
        self.train_step_update_metrics(loss, output)
        self.train_step_log_metrics()

        # Explicitly delete the output, for memory release
        del output

        # return loss or backpropagation will fail
        return loss

    def train_step_update_metrics(
            self,
            loss: torch.Tensor,
            output: SemanticSegmentationOutput
    ) -> None:
        """Update train metrics after a single step, with the content of
        the output object.
        """
        self.train_loss(loss.detach())
        self.train_cm(output.semantic_pred().detach(), output.semantic_target.detach())

    def train_step_log_metrics(self) -> None:
        """Log train metrics after a single step with the content of the
        output object.
        """
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True,
            prog_bar=True)

    def on_train_epoch_end(self) -> None:
        # Log metrics
        self.log("train/miou", self.train_cm.miou(), prog_bar=True)
        self.log("train/oa", self.train_cm.oa(), prog_bar=True)
        self.log("train/macc", self.train_cm.macc(), prog_bar=True)
        for iou, seen, name in zip(*self.train_cm.iou(), self.class_names):
            if seen:
                self.log(f"train/iou_{name}", iou, prog_bar=True)

        # Reset metrics accumulated over the last epoch
        self.train_cm.reset()

    def validation_step(
            self,
            batch: NAG,
            batch_idx: int
    ) -> None:
        loss, output = self.model_step(batch)

        # Update and log metrics
        self.validation_step_update_metrics(loss, output)
        self.validation_step_log_metrics()

        # Get the current epoch. For the validation set, we alter the
        # epoch number so that `track_val_every_n_epoch` can align
        # with `check_val_every_n_epoch`. Indeed, it seems the epoch
        # number during the validation step is always one increment
        # ahead
        epoch = self.current_epoch + 1

        # Store features and predictions for a batch of interest
        # NB: the `batch_idx` produced by torch lightning here
        # corresponds to the `Dataloader`'s index wrt the current epoch
        # and NOT an index wrt the `Dataset`. Said otherwise, if the
        # `Dataloader(shuffle=True)` then, the stored batch will not be
        # the same at each epoch. For this reason, if tracking the same
        # object across training is needed, the `Dataloader` and the
        # transforms should be free from any stochasticity
        track_epoch = epoch % self.hparams.track_val_every_n_epoch == 0
        track_batch = batch_idx == self.hparams.track_val_idx
        track_all_batches = self.hparams.track_val_idx == -1
        if track_epoch and (track_batch or track_all_batches):
            self.track_batch(batch, batch_idx, output)

        # Explicitly delete the output, for memory release
        del output

    def validation_step_update_metrics(
            self,
            loss: torch.Tensor,
            output: SemanticSegmentationOutput
    ) -> None:
        """Update validation metrics with the content of the output
        object.
        """
        self.val_loss(loss.detach())
        self.val_cm(output.semantic_pred().detach(), output.semantic_target.detach())

    def validation_step_log_metrics(self) -> None:
        """Log validation metrics after a single step with the content
        of the output object.
        """
        self.log(
            "val/loss", self.val_loss, on_step=False, on_epoch=True,
            prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        miou = self.val_cm.miou()
        oa = self.val_cm.oa()
        macc = self.val_cm.macc()

        # Log metrics
        self.log("val/miou", miou, prog_bar=True)
        self.log("val/oa", oa, prog_bar=True)
        self.log("val/macc", macc, prog_bar=True)
        for iou, seen, name in zip(*self.val_cm.iou(), self.class_names):
            if seen:
                self.log(f"val/iou_{name}", iou, prog_bar=True)

        # Update best-so-far metrics
        self.val_miou_best(miou)
        self.val_oa_best(oa)
        self.val_macc_best(macc)

        # Log best-so-far metrics, using `.compute()` instead of passing
        # the whole torchmetrics object, because otherwise metric would
        # be reset by lightning after each epoch
        self.log("val/miou_best", self.val_miou_best.compute(), prog_bar=True)
        self.log("val/oa_best", self.val_oa_best.compute(), prog_bar=True)
        self.log("val/macc_best", self.val_macc_best.compute(), prog_bar=True)

        # Reset metrics accumulated over the last epoch
        self.val_cm.reset()

    def on_test_start(self) -> None:
        # Initialize the submission directory based on the time of the
        # beginning of test. This way, the test steps can all have
        # access to the same directory, regardless of their execution
        # time
        self.submission_dir = self.trainer.datamodule.test_dataset.submission_dir
        self.on_fit_start()

    def test_step(self, batch: NAG, batch_idx: int) -> None:
        loss, output = self.model_step(batch)

        # If the input batch does not have any labels (e.g. test set
        # with held-out labels), y_hist will be None and the loss will
        # not be computed. In this case, we arbitrarily set the loss to
        # 0 and do not update the confusion matrix
        loss = 0 if loss is None else loss

        # If the test set misses targets, we keep track of it, to skip
        # metrics computation on the test set
        if not output.has_target:
            self.test_has_target = False

        # Update and log metrics
        self.test_step_update_metrics(loss, output)
        self.test_step_log_metrics()

        # Prepare submission for held-out test sets
        if self.trainer.datamodule.hparams.submit:
            nag = batch if isinstance(batch, NAG) else batch[0]
            l0_pos = nag[0].pos.detach().cpu()
            l0_pred = output.semantic_pred()[nag[0].super_index].detach().cpu()
            self.trainer.datamodule.test_dataset.make_submission(
                batch_idx, l0_pred, l0_pos, submission_dir=self.submission_dir)

        # Store features and predictions for a batch of interest
        # NB: the `batch_idx` produced by torch lightning here
        # corresponds to the `Dataloader`'s index wrt the current epoch
        # and NOT an index wrt the `Dataset`. Said otherwise, if the
        # `Dataloader(shuffle=True)` then, the stored batch will not be
        # the same at each epoch. For this reason, if tracking the same
        # object across training is needed, the `Dataloader` and the
        # transforms should be free from any stochasticity
        track_batch = batch_idx == self.hparams.track_test_idx
        track_all_batches = self.hparams.track_test_idx == -1
        if track_batch or track_all_batches:
            self.track_batch(batch, batch_idx, output)

        # Explicitly delete the output, for memory release
        del output

    def test_step_update_metrics(
            self,
            loss: torch.Tensor,
            output: SemanticSegmentationOutput
    ) -> None:
        """Update test metrics with the content of the output object.
        """
        # If the test set misses targets, we keep track of it, to skip
        # metrics computation on the test set
        if not self.test_has_target:
            return

        self.test_loss(loss.detach())
        self.test_cm(output.semantic_pred().detach(), output.semantic_target.detach())

    def test_step_log_metrics(self) -> None:
        """Log test metrics after a single step with the content of the
        output object.
        """
        # If the test set misses targets, we keep track of it, to skip
        # metrics computation on the test set
        if not self.test_has_target:
            return

        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True,
            prog_bar=True)

    def on_test_epoch_end(self) -> None:
        # Finalize the submission
        if self.trainer.datamodule.hparams.submit:
            self.trainer.datamodule.test_dataset.finalize_submission(
                self.submission_dir)

        # If test set misses target data, reset metrics and skip logging
        if not self.test_has_target:
            self.test_cm.reset()
            return

        # Log metrics
        self.log("test/miou", self.test_cm.miou(), prog_bar=True)
        self.log("test/oa", self.test_cm.oa(), prog_bar=True)
        self.log("test/macc", self.test_cm.macc(), prog_bar=True)
        for iou, seen, name in zip(*self.test_cm.iou(), self.class_names):
            if seen:
                self.log(f"test/iou_{name}", iou, prog_bar=True)

        # Log confusion matrix to wandb
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({
                "test/cm": wandb_confusion_matrix(
                    self.test_cm.confmat, class_names=self.class_names)})

        # Reset metrics accumulated over the last epoch
        self.test_cm.reset()

    def predict_step(
            self,
            batch: NAG,
            batch_idx: int
    ) -> Tuple[NAG, SemanticSegmentationOutput]:
        _, output = self.model_step(batch)
        return batch, output

    def track_batch(
            self,
            batch: NAG,
            batch_idx: int,
            output: SemanticSegmentationOutput,
            folder: str = None
    ) -> None:
        """Store a batch prediction to disk. The corresponding `NAG`
        object will be populated with semantic segmentation predictions
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
        :param output: SemanticSegmentationOutput
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

        # Store the output predictions in conveniently-accessible
        # attributes in the NAG, for easy downstream use of the saved
        # object
        if not output.multi_stage:
            logits = output.logits
            pred = torch.argmax(logits, dim=1)

            # Store level-1 predictions and logits
            batch[1].semantic_pred = pred
            batch[1].logits = logits

            # Store level-0 (voxel-wise) predictions and logits
            batch[0].semantic_pred = pred[batch[0].super_index]
            batch[0].logits = logits[batch[0].super_index]

        else:
            for i, _logits in enumerate(output.logits):
                logits = _logits
                pred = torch.argmax(logits, dim=1)

                # Store level-1 predictions and logits
                batch[i + 1].semantic_pred = pred
                batch[i + 1].logits = logits

                # Store level-0 (voxel-wise) predictions and logits
                if i > 0:
                    continue
                batch[0].semantic_pred = pred[batch[0].super_index]
                batch[0].logits = logits[batch[0].super_index]

        # Detach the batch object and move it to CPU before saving
        batch = batch.detach().cpu()

        # Prepare the folder
        if self.trainer is None:
            stage = 'unknown_stage'
        elif self.trainer.training:
            stage = 'train'
        elif self.trainer.validating:
            stage = 'val'
        elif self.trainer.testing:
            stage = 'test'
        elif self.trainer.predicting:
            stage = 'predict'
        else:
            stage = 'unknown_stage'
        if folder is None:
            if self.logger and self.logger.save_dir:
                folder = self.logger.save_dir
            else:
                folder = ''
        folder = osp.join(folder, 'predictions', stage, str(self.current_epoch))
        if not osp.isdir(folder):
            os.makedirs(folder, exist_ok=True)

        # Save to disk
        path = osp.join(folder, f"batch_{batch_idx}.h5")
        batch.save(path)
        log.info(f'Stored predictions at: "{path}"')

        # TODO: log plotly plot to wandb
        if isinstance(self.logger, WandbLogger):
            pass

    def configure_optimizers(self) -> Dict:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # Differential learning rate for transformer blocks
        t_names = ['transformer_blocks', 'down_pool_block']
        lr = self.hparams.optimizer.keywords['lr']
        t_lr = lr * self.hparams.transformer_lr_scale
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if all([t not in n for t in t_names]) and p.requires_grad]},
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any([t in n for t in t_names]) and p.requires_grad],
                "lr": t_lr}]
        optimizer = self.hparams.optimizer(params=param_dicts)

        # Return the optimizer if no scheduler in the config
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        # Build the scheduler, with special attention for plateau-like
        # schedulers, which
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        reduce_on_plateau = isinstance(scheduler, ON_PLATEAU_SCHEDULERS)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": reduce_on_plateau}}

    def load_state_dict(
            self,
            state_dict: Dict,
            strict: bool = True
    ) -> None:
        """Basic `load_state_dict` from `torch.nn.Module` with a bit of
        acrobatics due to `criterion.weight`.

        This attribute, when present in the `state_dict`, causes
        `load_state_dict` to crash. More precisely, `criterion.weight`
        is holding the per-class weights for classification losses.
        """
        # Special treatment `criterion.weight`
        class_weight_bckp = self.criterion.weight
        self.criterion.weight = None

        # Recover the class weights from any `criterion.weight' or
        # 'criterion.*.weight' key and remove those keys from the
        # state_dict
        keys = []
        for key in state_dict.keys():
            if key.startswith('criterion.') and key.endswith('.weight'):
                keys.append(key)
        class_weight = state_dict[keys[0]] if len(keys) > 0 else None
        for key in keys:
            state_dict.pop(key)

        # Load the state_dict
        super().load_state_dict(state_dict, strict=strict)

        # If need be, assign the class weights to the criterion
        self.criterion.weight = class_weight if class_weight is not None \
            else class_weight_bckp

    def _load_from_checkpoint(
            self,
            checkpoint_path: str,
            **kwargs
    ) -> 'SemanticSegmentationModule':
        """Simpler version of `LightningModule.load_from_checkpoint()`
        for easier use: no need to explicitly pass `model.net`,
        `model.criterion`, etc.
        """
        return self.__class__.load_from_checkpoint(
            checkpoint_path, net=self.net, criterion=self.criterion, **kwargs)

    @staticmethod
    def sanitize_step_output(out_dict: Dict) -> Dict:
        """Helper to be used for cleaning up the `_step` functions.
        Lightning expects those to return the loss (on GPU, with the
        computation graph intact for the backward step. Any other
        element passed in this dict will be detached and moved to CPU
        here. This avoids memory leak.
        """
        return {
            k: v if ((k == "loss") or (not isinstance(v, torch.Tensor)))
            else v.detach().cpu()
            for k, v in out_dict.items()}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = str(pyrootutils.setup_root(__file__, pythonpath=True))
    cfg = omegaconf.OmegaConf.load(root + "/configs/model/semantic/spt-2.yaml")
    _ = hydra.utils.instantiate(cfg)
