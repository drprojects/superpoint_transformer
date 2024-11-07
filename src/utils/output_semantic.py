import torch
import logging
import src

log = logging.getLogger(__name__)


__all__ = ['SemanticSegmentationOutput']


class SemanticSegmentationOutput:
    """A simple holder for semantic segmentation model output, with a
    few helper methods for manipulating the predictions and targets
    (if any).
    """

    def __init__(self, logits, y_hist=None):
        self.logits = logits
        self.y_hist = y_hist
        if src.is_debug_enabled():
            self.debug()

    def debug(self):
        """Runs a series of sanity checks on the attributes of self.
        """
        assert isinstance(self.logits, torch.Tensor) \
               or all(isinstance(l, torch.Tensor) for l in self.logits)
        if self.has_target:
            if self.multi_stage:
                assert len(self.y_hist) == len(self.logits)
                assert all(
                    y.shape[0] == l.shape[0]
                    for y, l in zip(self.y_hist, self.logits))
            else:
                assert self.y_hist.shape[0] == self.logits.shape[0]

    @property
    def device(self):
        """Returns the device on which the logits are stored, assuming
        all other output variables held by the object are also on the
        same device.
        """
        logits = self.logits[0] if self.multi_stage else self.logits
        return logits.device

    @property
    def has_target(self):
        """Check whether `self` contains target data for semantic
        segmentation.
        """
        return self.y_hist is not None

    @property
    def multi_stage(self):
        """If the semantic segmentation `logits` are stored in an
        enumerable, then the model output is multi-stage.
        """
        return not isinstance(self.logits, torch.Tensor)

    @property
    def num_classes(self):
        """Number for semantic classes in the output predictions.
        """
        logits = self.logits[0] if self.multi_stage else self.logits
        return logits.shape[1]

    @property
    def num_nodes(self):
        """Number for nodes/superpoints in the output predictions. By
        default, for a hierarchical partition, this means counting the
        number of level-1 nodes/superpoints.
        """
        logits = self.logits[0] if self.multi_stage else self.logits
        return logits.shape[0]

    def semantic_pred(self):
        """Semantic predictions on the level-1 superpoint.

        Final semantic segmentation predictions are the argmax of the
        first-level partition logits.
        """
        logits = self.logits[0] if self.multi_stage else self.logits
        return torch.argmax(logits, dim=1)

    @property
    def semantic_target(self):
        """Semantic target on the level-1 superpoint.

        Final semantic segmentation target are the label histogram
        of the first-level partition logits.
        """
        return self.y_hist[0] if self.multi_stage else self.y_hist

    @property
    def void_mask(self):
        """Returns a mask on the level-1 nodes indicating which is void.
        By convention, nodes/superpoints are void if they contain
        more than 50% void points. By convention in this project, void
        points have the label `num_classes`. In label histograms, void
        points are counted in the last column.
        """
        if not self.has_target:
            return

        # For simplicity, we only return the mask for the level-1
        y_hist = self.semantic_target
        total_count = y_hist.sum(dim=1)
        void_count = y_hist[:, -1]
        return void_count / total_count > 0.5

    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def voxel_semantic_pred(self, super_index=None, sub=None):
        """Semantic predictions on the level-0 voxels.

        Final semantic segmentation predictions are the argmax of the
        first-level partition logits. This function then distributes 
        these predictions to each level-0 point (ie voxel in our 
        framework).
        
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
        
        # Distribute the level-1 superpoint predictions to the voxels
        return self.semantic_pred()[super_index]

    def full_res_semantic_pred(
            self, 
            super_index_level0_to_level1=None, 
            super_index_raw_to_level0=None, 
            sub_level1_to_level0=None, 
            sub_level0_to_raw=None):
        """Semantic predictions on the full-resolution input point
        cloud.

        Final semantic segmentation predictions are the argmax of the
        first-level partition logits. This function then distributes 
        these predictions to each raw point (ie full-resolution point 
        cloud before voxelization in our framework).
        
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
        
        # Distribute the level-1 superpoint predictions to the 
        # full-resolution points
        return self.semantic_pred()[super_index_level0_to_level1][super_index_raw_to_level0]
