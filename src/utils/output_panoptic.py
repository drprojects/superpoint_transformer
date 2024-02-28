import torch
import logging
from torch_scatter import scatter_mean
from src.utils.scatter import scatter_mean_weighted
from src.utils.output_semantic import SemanticSegmentationOutput


log = logging.getLogger(__name__)


__all__ = ['PanopticSegmentationOutput', 'PartitionParameterSearchStorage']


class PanopticSegmentationOutput(SemanticSegmentationOutput):
    """A simple holder for panoptic segmentation model output, with a
    few helper methods for manipulating the predictions and targets
    (if any).
    """

    def __init__(
            self,
            logits,
            stuff_classes,
            edge_affinity_logits,
            node_size,
            y_hist=None,
            obj=None,
            obj_edge_index=None,
            obj_edge_affinity=None,
            pos=None,
            obj_pos=None,
            obj_index_pred=None,
            semantic_loss=None,
            edge_affinity_loss=None):
        # We set the child class attributes before calling the parent
        # class constructor, because the parent constructor calls
        # `self.debug()`, which needs all attributes to be initialized
        device = edge_affinity_logits.device
        self.stuff_classes = torch.tensor(stuff_classes, device=device).long() \
            if stuff_classes is not None \
            else torch.empty(0, device=device).long()
        self.edge_affinity_logits = edge_affinity_logits
        self.node_size = node_size
        self.obj = obj
        self.obj_edge_index = obj_edge_index
        self.obj_edge_affinity = obj_edge_affinity
        self.pos = pos
        self.obj_pos = obj_pos
        self.obj_index_pred = obj_index_pred
        self.semantic_loss = semantic_loss
        self.edge_affinity_loss = edge_affinity_loss
        super().__init__(logits, y_hist=y_hist)

    def debug(self):
        # Parent class debugger
        super().debug()

        # Instance predictions
        assert self.edge_affinity_logits.dim() == 1

        # Node properties
        assert self.node_size.dim() == 1
        assert self.node_size.shape[0] == self.num_nodes

        if self.has_instance_pred:
            if not self.has_multi_instance_pred:
                assert self.obj_index_pred.dim() == 1
                assert self.obj_index_pred.shape[0] == self.num_nodes
            else:
                assert isinstance(self.obj_index_pred, list)
                item = self.obj_index_pred[0]
                assert isinstance(item[0], dict)
                assert isinstance(item[1], torch.Tensor)
                assert item[1].dim() == 1
                assert item[1].shape[0] == self.num_nodes

        # Instance target
        items = [
            self.obj_edge_index, self.obj_edge_affinity, self.pos, self.obj_pos]
        without_instance_target = all(x is None for x in items)
        with_instance_target = all(x is not None for x in items)
        assert without_instance_target or with_instance_target

        if without_instance_target:
            return

        # Local import to avoid import loop errors
        from src.data import InstanceData

        assert isinstance(self.obj, InstanceData)
        assert self.obj.num_clusters == self.num_nodes
        assert self.obj_edge_index.dim() == 2
        assert self.obj_edge_index.shape[0] == 2
        assert self.obj_edge_index.shape[1] == self.num_edges
        assert self.obj_edge_affinity.dim() == 1
        assert self.obj_edge_affinity.shape[0] == self.num_edges

    @property
    def has_target(self):
        """Check whether `self` contains target data for panoptic
        segmentation.
        """
        items = [
            self.obj,
            self.obj_edge_index,
            self.obj_edge_affinity,
            self.pos,
            self.obj_pos]
        return super().has_target and all(x is not None for x in items)

    @property
    def has_instance_pred(self):
        """Check whether `self` contains predicted data for panoptic
        segmentation `obj_index_pred`.
        """
        return self.obj_index_pred is not None

    @property
    def has_multi_instance_pred(self):
        """Check whether `self` contains predicted data for panoptic
        segmentation `obj_index_pred` as a list of results for
        performance comparison of partition settings.
        """
        return self.has_instance_pred \
               and not isinstance(self.obj_index_pred, torch.Tensor)

    @property
    def num_edges(self):
        """Number for edges in the instance graph.
        """
        return self.edge_affinity_logits.shape[1]

    @property
    def edge_affinity_pred(self):
        """Simply applies a sigmoid on `edge_affinity_logits` to produce
        the actual affinity predictions to be used for superpoint
        graph clustering.
        """
        return self.edge_affinity_logits.sigmoid()

    @property
    def void_edge_mask(self):
        """Returns a mask on the edges indicating those connecting two
        void nodes.
        """
        if not self.has_target:
            return

        mask = self.void_mask[self.obj_edge_index]
        return mask[0] & mask[1]

    @property
    def sanitized_edge_affinities(self):
        """Return the predicted and target edge affinities, along with
        masks indicating same-class and same-object edges. The output is
        sanitized for edge affinity loss and metrics computation.

        We return the edge affinity logits to the criterion and not
        the actual sigmoid-normalized predictions used for graph
        clustering. The reason for this is that we expect the edge
        affinity loss to be computed using `BCEWithLogitsLoss`.

        We choose to exclude edges connecting nodes/superpoints with
        more than 50% 'void' points from edge affinity loss and metrics
        computation. This is what the sanitization step consists in.

        To this end, the present function does the following:
          - remove predicted and target edges connecting two 'void'
            nodes (see `self.void_edge_mask`)
        """
        # Identify the sanitized edges
        idx = torch.where(~self.void_edge_mask)[0]

        # Compute the boolean masks indicating same-class and
        # same-object edges. These can be useful for losses with more
        # weights on hard edges
        obj, count, y = self.obj.major(num_classes=self.num_classes)
        is_same_class = y[self.obj_edge_index[0]] == y[self.obj_edge_index[1]]
        is_same_obj = obj[self.obj_edge_index[0]] == obj[self.obj_edge_index[1]]

        # Return sanitized predicted and target affinities, as well as
        # edge masks
        return self.edge_affinity_logits[idx], self.obj_edge_affinity[idx], \
               is_same_class[idx], is_same_obj[idx]

    @property
    def weighted_instance_semantic_pred(self):
        """Compute the predicted semantic label, score and logits for
        each predicted instance. This involves computing, for each
        predicted instance, the weighted average of the logits of the
        superpoints it contains.
        """
        if not self.has_instance_pred:
            return None, None, None

        # Compute the mean logits for each predicted object, weighted by
        # the node sizes
        node_logits = self.logits[0] if self.multi_stage else self.logits
        obj_logits = scatter_mean_weighted(
            node_logits, self.obj_index_pred, self.node_size)

        # Compute the predicted semantic label and proba for each node
        obj_semantic_score, obj_y = obj_logits.softmax(dim=1).max(dim=1)

        return obj_y, obj_semantic_score, obj_logits

    @property
    def panoptic_pred(self):
        """Panoptic predictions on the level-1 superpoints.

        Return the predicted semantic score and label for each predicted
        instance, along with the InstanceData object summarizing
        predictions.
        """
        if not self.has_instance_pred:
            return None, None, None

        # Merge the InstanceData based on the predicted instances and
        # target instances
        instance_data = self.obj.merge(self.obj_index_pred) if self.has_target \
            else None

        # Compute the semantic prediction for each predicted object,
        # weighted by the node sizes
        obj_y, obj_semantic_score, obj_logits = \
            self.weighted_instance_semantic_pred

        # TODO: should we take object size into account in the scoring ?

        # Compute, for each predicted object, the mean inter-object and
        # intra-object predicted edge affinity
        ie = self.obj_index_pred[self.obj_edge_index]
        intra = ie[0] == ie[1]
        idx = ie.flatten()
        intra = intra.repeat(2)
        a = self.edge_affinity_pred.repeat(2)
        n = self.obj_index_pred.max() + 1
        obj_mean_intra = scatter_mean(a[intra], idx[intra], dim_size=n)
        obj_mean_inter = scatter_mean(a[~intra], idx[~intra], dim_size=n)

        # Compute the inter-object and intra-object scores
        obj_intra_score = obj_mean_intra
        obj_inter_score = 1 / (1 + obj_mean_inter)

        # Final prediction score is the product of individual scores
        obj_score = obj_semantic_score

        return obj_score, obj_y, instance_data

    def voxel_panoptic_pred(self, super_index=None, sub=None):
        """Panoptic predictions on the level-0 voxels. Returns the
        predicted semantic label and instance index for each voxel,
        along with the voxel-wise InstanceData summarizing predictions.

        Final panoptic segmentation predictions are computed with
        respect to predicted instances, after level-1 superpoint-graph
        clustering.

        The predicted instance semantic labels are computed from the
        average of logits of level-1 superpoints they include, weighted
        by the superpoint sizes. These instance-aggregated semantic
        predictions may (slightly) differ from the per-superpoint
        semantic segmentation prediction obtained from
        `self.voxel_semantic_pred()`.

        This function then distributes semantic and instance index
        predictions to each level-0 point (ie voxel in our framework).

        :param super_index: LongTensor
            Tensor holding, for each level-0 point (ie voxel), the index
            of the level-1 superpoint it belongs to
        :param sub: Cluster
            Cluster object indicating, for each level-1 superpoint,
            the indices of the level-0 points (ie voxels) it contains
        """
        assert super_index is not None or sub is not None, \
            "Must provide either `super_index` or `sub`"

        # If super_index is not provided, build it from sub
        if super_index is None:
            super_index = sub.to_super_index()

        # Compute the semantic prediction for each predicted object,
        # weighted by the node sizes
        obj_y, _, _ = self.weighted_instance_semantic_pred

        # Distribute the per-instance predictions to level-1 superpoints
        sp_y = obj_y[self.obj_index_pred]

        # Distribute the level-1 superpoint semantic predictions and
        # instance indices to the voxels
        vox_y = sp_y[super_index]
        vox_index = self.obj_index_pred[super_index]

        # Local import to avoid import loop errors
        from src.data import InstanceData

        # Compute the voxel-wise InstanceData carrying voxel predictions
        # NB: we make an approximation here: each voxel is given a count
        # of 1 point, neglecting the actual number of points in each
        # voxel. This may slightly affect the metrics, compared to
        # the true full-resolution predictions
        num_voxels = super_index.shape[0]
        vox_obj_pred = InstanceData(
            torch.arange(num_voxels, device=self.device),
            vox_index,
            torch.ones(num_voxels, device=self.device, dtype=torch.long),
            vox_y,
            dense=True)

        return vox_y, vox_index, vox_obj_pred

    def full_res_panoptic_pred(
            self,
            super_index_level0_to_level1=None,
            super_index_raw_to_level0=None,
            sub_level1_to_level0=None,
            sub_level0_to_raw=None):
        """Panoptic predictions on the full-resolution input point
        cloud. Returns the predicted semantic label and instance index
        for each point, along with the point-wise InstanceData
        summarizing predictions.

        Final panoptic segmentation predictions are computed with
        respect to predicted instances, after level-1 superpoint-graph
        clustering.

        The predicted instance semantic labels are computed from the
        average of logits of level-1 superpoints they include, weighted
        by the superpoint sizes. These instance-aggregated semantic
        predictions may (slightly) differ from the per-superpoint
        semantic segmentation prediction obtained from
        `self.full_res_semantic_pred()`.

        This function then distributes these predictions to each raw
        point (ie full-resolution point cloud before voxelization in our
        framework).

        :param super_index_level0_to_level1: LongTensor
            Tensor holding, for each level-0 point (ie voxel), the index
            of the level-1 superpoint it belongs to
        :param super_index_raw_to_level0: LongTensor
            Tensor holding, for each raw full-resolution point, the
            index of the level-0 point (ie voxel) it belongs to
        :param sub_level1_to_level0: Cluster
            Cluster object indicating, for each level-1 superpoint,
            the indices of the level-0 points (ie voxels) it contains
        :param sub_level0_to_raw: Cluster
            Cluster object indicating, for each level-0 point (ie
            voxel), the indices of the raw full-resolution points it
            contains
        """
        assert super_index_level0_to_level1 is not None or sub_level1_to_level0 is not None, \
            "Must provide either `super_index_level0_to_level1` or `sub_level1_to_level0`"

        assert super_index_raw_to_level0 is not None or sub_level0_to_raw is not None, \
            "Must provide either `super_index_raw_to_level0` or `sub_level0_to_raw`"

        # If super_index are not provided, build them from sub
        if super_index_level0_to_level1 is None:
            super_index_level0_to_level1 = sub_level1_to_level0.to_super_index()
        if super_index_raw_to_level0 is None:
            super_index_raw_to_level0 = sub_level0_to_raw.to_super_index()

        # Distribute the level-1 superpoint semantic predictions and
        # instance indices to the voxels
        vox_y, vox_index, vox_obj_pred = self.voxel_panoptic_pred(
            super_index=super_index_level0_to_level1)

        # Distribute the level-1 superpoint predictions to the
        # full-resolution points
        raw_y = vox_y[super_index_raw_to_level0]
        raw_index = vox_index[super_index_raw_to_level0]

        # Local import to avoid import loop errors
        from src.data import InstanceData

        # Compute the voxel-wise InstanceData carrying voxel predictions
        # NB: we make an approximation here: each voxel is given a count
        # of 1 point, neglecting the actual number of points in each
        # voxel. This may slightly affect the metrics, compared to
        # the true full-resolution predictions
        num_points = super_index_raw_to_level0.shape[0]
        raw_obj_pred = InstanceData(
            torch.arange(num_points, device=self.device),
            raw_index,
            torch.ones(num_points, device=self.device, dtype=torch.long),
            raw_y,
            dense=True)

        return raw_y, raw_index, raw_obj_pred


class PartitionParameterSearchStorage:
    """A class to hold the output results of multiple partitions, when
    searching for the optimal partition parameter settings. Since
    metrics are only computed at the end of an epoch, we cannot compute
    the optimal parameter settings at each batch. On the other hand, we
    cannot store the whole content of the `PanopticSegmentationOutput`
    of each batch. This holder is used to store the strict necessary
    from the `PanopticSegmentationOutput` of each batch, to be able to
    call `PanopticSegmentationOutput.panoptic_pred` at
    the end of an epoch and pass its output to an instance or panoptic
    segmentation metric object.

    NB: make sure the input is detached and on CPU, you do not want to
    blow up your GPU memory. Still, for very large datasets, this
    approach will be RAM-hungry. If this causes CPU memory errors, you
    will need to save your predicted data in temp files on disk.
    """
    def __init__(
            self,
            logits,
            stuff_classes,
            node_size,
            edge_affinity_logits,
            obj,
            obj_index_pred):
        self.stuff_classes = stuff_classes
        self.logits = logits
        self.node_size = node_size
        self.edge_affinity_logits = edge_affinity_logits
        self.obj = obj
        self.obj_index_pred = obj_index_pred

    @property
    def settings(self):
        """This assumes all items in `self.obj_index_pred` follow the
        output format of `InstancePartitioner._grid_forward()`.
        """
        return [v[0] for v in self.obj_index_pred]

    @property
    def num_settings(self):
        """This assumes all items in `self.obj_index_pred` follow the
        output format of `InstancePartitioner._grid_forward()`.
        """
        return len(self.settings)

    def panoptic_pred(self, setting):
        """Return the predicted InstanceData, and the predicted instance
        semantic label and score, for a given batch item and a given
        partition setting.
        """
        # Recover the index of the setting in the stored results
        i_setting = self.settings.index(setting) \
            if not isinstance(setting, int) else setting

        # Recover the batch's partition results
        output = PanopticSegmentationOutput(
            self.logits,
            self.stuff_classes,
            self.edge_affinity_logits,
            self.node_size,
            obj=self.obj,
            obj_index_pred=self.obj_index_pred[i_setting][1])

        # Compute inputs for an instance or panoptic segmentation metric
        return output.panoptic_pred
