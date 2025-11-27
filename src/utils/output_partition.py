import torch
import logging

log = logging.getLogger(__name__)


__all__ = ['PartitionOutput']


class PartitionOutput:
    """A simple holder for the output of the model 
    when training the partition.
    
    NB: In evaluation (or with the flag `partition_during_training` set to True),
    the `PartitionAndSemanticModule` also stores the computed partition (based on
    the features `x`) in the `partition` attribute of the PartitionOutput object.
    
    In that case, it allows to get have access to the `y_superpoint` attribute 
    of the PartitionOutput object.

    :param y: torch.Tensor (n)
        Labels of the points
    :param x: torch.Tensor (n, D)
        Features of the points
    :param edge_index: torch.Tensor (2, E)
        Edge index of the graph
    """
    
    def __init__(self, 
                 y: torch.Tensor,
                 x: torch.Tensor = None,
                 edge_index: torch.Tensor = None,
                 **kwargs):
        
        self.y = y
        self.x = x
        self.edge_index = edge_index
        
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def y_superpoint(self):
        return self.partition.y
    
    def superpoint_size_histogram(self):
        """
        Get the histogram of the size of the superpoints.
        """
        superpoint_size = self.partition.sub.sizes
        histogram = torch.bincount(superpoint_size)
        
        return histogram

    @property
    def has_target(self):
        return self.y is not None