import torch
from torch import nn
from typing import Tuple

from src.utils import PartitionOutput, compute_edge_distances_batch
from src.loss.focal import BinaryFocalLoss

import logging
log = logging.getLogger(__name__)

__all__ = []

class PartitionCriterion(nn.Module):
    """
    A criterion for the partition learning stage.
    It aims to learn embeddings that remain homogeneous within objects while 
    being sharply contrasted across semantic boundaries.


    :param loss_function: torch.nn._Loss
        The loss function to compute between the predicted and target edge affinities.
        Example: 
        - Focal loss, see src.loss.BinaryFocalLoss
        - Entropy loss, see torch.nn.CrossEntropyLoss
        
    :param affinity_temperature: float
        If we note X and Y the features of the two nodes of an edge,
        the affinity is computed as `exp(-|| X-Y || / affinity_temperature)`.
        
    :param adaptive_sampling_ratio: float in [0, 1]
        Adaptive sampling by randomly dropping intra-edges until
        they constitute at most `1-adaptive_sampling_ratio` ratio of the sampled edges.
        In other words, until the minority class (intra-edges) constitutes at least 
        `adaptive_sampling_ratio` ratio of the sampled edges.
        
    :param num_classes: int
        The number of classes.
        
    """
    INTER_EDGE_LABEL = 0
    INTRA_EDGE_LABEL = 1

    def __init__(self, 
                 loss_function = BinaryFocalLoss(),
                 affinity_temperature: float = 1,
                 adaptive_sampling_ratio = None,
            
                 num_classes: int = None,
                 sharding: int = None,
                 ):
        
        super().__init__()
        
        
        self.loss_function = loss_function
        self.affinity_temperature = affinity_temperature
        self.adaptive_sampling_ratio = adaptive_sampling_ratio  
        
        self.sharding = sharding     
        self.num_classes = num_classes

    def forward(self, 
                partition_output: PartitionOutput
                ) -> Tuple[torch.Tensor, PartitionOutput]:
        
        loss, edge_classification_output, n_inter_edge \
            = self.edge_classification_loss(partition_output)
        
        # Store the number of inter-edges for logging
        partition_output.n_inter_edge = n_inter_edge

        return loss, partition_output

    
    def edge_classification_loss(self, partition_output: PartitionOutput):
        """
        Compute the edge classification loss. Contrastive loss between nodes from different class 
        linked by an edge.
        
        :param partition_output: PartitionOutput
            The partition output.
            
        :return: tuple
            A tuple containing:
            - loss: The edge classification loss for backpropagation
            - (predicted_class, groundtruth_class): Tuple of predicted and target edge classes 
              (inter=1, intra=0). Note that the 0.5 threshold used to define the predicted_class 
              from the predicted_affinity is arbitrary, so these metrics are not logged. 
              For meaningful partition evaluation, see the partition purity metrics logged in 
              `PartitionAndSemanticModule.on_validation_epoch_end`.
        """
        # A) Get the target edge affinity
        # y is an histogram of the classes of shape (n, C+1), where C is the
        # number of classes and the last column is the void class.
        y = partition_output.y 
        x = partition_output.x
        edge_index = partition_output.edge_index
        
        if edge_index.numel() == 0:
            raise ValueError("No edges found in batch.")
        
        # A.1) Keep only relevant edges
        
        # Turn histogram into unique label (by taking the mode per voxel)
        majority_class_count, y = y[:,:self.num_classes].max(dim=1)
        
        # Remove self-loops (which are useless for the loss)
        mask_self_loops = edge_index[0] == edge_index[1]
        edge_index = edge_index[:, ~mask_self_loops]
        
        # Discard edges containing a pure void voxel
        mask_void_voxels = majority_class_count == 0 # all points in the voxel were voids
        mask_void_edges = mask_void_voxels[edge_index[0]] | mask_void_voxels[edge_index[1]]
        edge_index = edge_index[:, ~mask_void_edges]
        
        if edge_index.numel() == 0:
            log.warning("No edges with two non-void nodes found in current batch.\n")
            return self.fake_edge_classification_loss(y.device)
        
        # Compute the target edge affinities 
        # (inter-edges are 0, intra-edges are 1 - consistent with class 
        # attributes `INTER_EDGE_LABEL` and `INTRA_EDGE_LABEL`)
        target_affinity = (y[edge_index[0]] == y[edge_index[1]]).int()
        n_inter_edge = (target_affinity == self.INTER_EDGE_LABEL).sum().item()
        
        if n_inter_edge == 0:
            log.warning("No inter-edges found in current batch.\n")
            return self.fake_edge_classification_loss(y.device)

        # A.2) Adaptive sampling on edges to balance between inter- and 
        # intra-edges (ONLY during training)
        if self.training and self.adaptive_sampling_ratio is not None:
            sampled_indices = self.binary_adaptive_sampling(target_affinity, 
                                                            minority_class=self.INTER_EDGE_LABEL)
            
            assert sampled_indices.numel() > 0, "No edges left after adaptive sampling."
            
            edge_index = edge_index[:, sampled_indices]
            target_affinity = target_affinity[sampled_indices]

        # B) Predict the edge affinities
        predicted_affinity = self.features_to_edge_affinity(x, edge_index)
        loss = self.loss_function(predicted_affinity, target_affinity.bool())

        return loss, ((predicted_affinity>=0.5).int(), target_affinity), n_inter_edge
    
    def fake_edge_classification_loss(self, device):
        """
        If for any reason, there are no edges or no inter-edges in the batch to 
        train on, we return a fake loss and other metrics to avoid errors.
        
        This is useful if it happens rarely during one training epoch (e.g., 
        due to small random crops or specific batch compositions).
        
        If during a WHOLE training epoch, no inter-edges are found, then 
        `PartitionAndSemanticModule.on_train_epoch_end` will raise an error.
        """
        # Create a zero loss that requires grad for backward compatibility
        fake_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        fake_pred = torch.tensor([0], device=device)
        fake_target = torch.tensor([0], device=device)
        n_inter_edge = 0
        return fake_loss, (fake_pred, fake_target), n_inter_edge
    
    def binary_adaptive_sampling(self, y, minority_class=1):
        """
        Adaptive sampling of the edges for the edge affinity loss.

        The goal is that the minority class is sampled as `self.adaptive_sampling_ratio`% of
        the samples.
        
        :param minority_class: int
            The class to consider as the minority class.
        
        :param y: torch.Tensor
            Labels in {0, 1}
        
        :return: torch.Tensor
            Tensor of shape (n_sample_per_class) containing random indices for each class
        """
        
        count_small = (y==minority_class).int().sum()
        count_large = y.shape[0] - count_small
        
        # We take all the samples from the minority class
        sample_minority = torch.where(y == minority_class)[0]

        # We sample n_sample_majority samples from the majority class
        n_sample_majority = ((1/self.adaptive_sampling_ratio - 1)*count_small).int()
        
        assert n_sample_majority <= count_large, "Not enough samples for the majority class"
        sample_majority = torch.where(y != minority_class)[0]
        perm_majority = torch.randperm(count_large, device=y.device)[:n_sample_majority]
        sample_majority = sample_majority[perm_majority]
        
        sampled_indices = torch.cat((sample_minority, sample_majority))

        return sampled_indices
        
   
    def adaptive_sampling(self, y, num_classes=2):
        """
        Adaptive sampling of the edges for the edge affinity loss.
        
        :param y: torch.Tensor
            Labels in [0, num_classes-1] (label `num_classes` is void)
            
        :return: torch.Tensor
            Tensor of shape (C, n_sample_per_class) containing random indices for each class
        """

        counts = torch.bincount(y, minlength=num_classes)[:num_classes]
        
        n_sample_per_class = counts.min()
        
        # Handle case where some classes have 0 samples
        if n_sample_per_class == 0:
            return torch.empty((num_classes, 0), dtype=torch.long, device=y.device).flatten()
        
        # Create tensor to store random indices for each class
        sampled_indices = torch.zeros((num_classes, n_sample_per_class), dtype=torch.long, device=y.device)
        
        # For each class, sample n_sample_per_class indices without replacement
        for class_idx in range(num_classes):
            
            if counts[class_idx] == n_sample_per_class:
                # All samples are used
                sampled_indices[class_idx] = torch.where(y == class_idx)[0]
                
            else:
                class_indices = torch.where(y == class_idx)[0]

                perm = torch.randperm(counts[class_idx], device=y.device)[:n_sample_per_class]
                sampled_indices[class_idx] = class_indices[perm]

        return sampled_indices.flatten()
    
    
    def features_to_edge_affinity(self, x: torch.Tensor, edge_index: torch.Tensor):
        
        distances = compute_edge_distances_batch(x, edge_index, self.sharding)
    
        affinity = torch.exp(-distances / self.affinity_temperature)
        
        return affinity