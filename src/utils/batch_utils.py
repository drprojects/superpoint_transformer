import math
import torch
from typing import Optional

__all__ = ['compute_edge_distances_batch', 'compute_pairwise_distances_batch',]

def compute_edge_distances_batch(x: torch.Tensor, 
                                edge_index: torch.Tensor, 
                                sharding: Optional[int] = None) -> torch.Tensor:
    """
    Computes the Euclidean distance of all pairs of nodes defined by `edge_index`.
    
    :param x: torch.Tensor of shape (num_nodes, feature_dim)
        Node features to compute the distances from.
    
    :param edge_index: torch.Tensor of shape (2, num_edges)
        Edge indices indicating the pairs of nodes for which to compute the distances.
        
    :param sharding: Optional[int], default=None
        Controls memory usage by processing edges in chunks.
        - If None or <= 0: processes all edges at once
        - If > 1: processes edges in chunks of size `sharding`
        - If 0 < sharding < 1: processes edges in chunks of size 
          `int(edge_index.shape[1] * sharding)`
        
    :return: torch.Tensor of shape (num_edges,)
        Euclidean distances
    """
    if sharding is None or sharding <= 0:
        # Original computation without batching
        d = torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=-1, p=2)
        return d
    
    else: # Recursive call in case of sharding.
        # Sharding allows limiting the number of edges processed at once.
        # This might alleviate memory use.
        
        sharding = int(sharding) if sharding > 1 \
            else math.ceil(edge_index.shape[1] * sharding)
        
        num_shards = math.ceil(edge_index.shape[1] / sharding)
        
        distances = []
        
        for i_shard in range(num_shards):
            
            start = i_shard * sharding
            end = min(start + sharding, edge_index.shape[1])
            
            edge_batch = edge_index[:, start:end]
            
            distances.append(compute_edge_distances_batch(x, edge_batch, sharding=None))
        
        # Concatenate all results
        return torch.cat(distances)



def compute_pairwise_distances_batch(x: torch.Tensor, 
                                   y: torch.Tensor, 
                                   sharding: Optional[int] = None) -> torch.Tensor:
    """
    Compute Euclidean distances between two sets of points,
    dividing the computation into batches to reduce memory usage.
    
    Args:
        x: First set of points of shape (num_points_x, feature_dim)
        y: Second set of points of shape (num_points_y, feature_dim)
        num_batch: Number of batches to divide the computation. If None, no batching.
    
    Returns:
        Distance matrix of shape (num_points_x, num_points_y)
    """
    if sharding is None or sharding <= 0:
        # Original computation without batching
        # Use broadcasting to avoid redundant computations
        x_expanded = x.unsqueeze(1)  # (num_points_x, 1, feature_dim)
        y_expanded = y.unsqueeze(0)  # (1, num_points_y, feature_dim)
        distances = torch.norm(x_expanded - y_expanded, dim=-1, p=2)
        return distances
    
    else:
        # Sharding allows limiting the number of points processed at once.
        # This might alleviate memory use.
        
        sharding = int(sharding) if sharding > 1 \
            else math.ceil(x.shape[0] * sharding)
            
        num_shards = math.ceil(x.shape[0] / sharding)
        
        num_points_x = x.shape[0]
        num_shards = math.ceil(num_points_x / sharding)
        
        distances_list = []

        for i_shard in range(num_shards):
            start = i_shard * sharding
            end = min(start + sharding, x.shape[0])
            
            # Extract the batch of points
            x_batch = x[start:end]  # (batch_size, feature_dim)
            
            # Compute distances for this batch
            x_batch_expanded = x_batch.unsqueeze(1)  # (batch_size, 1, feature_dim)
            y_expanded = y.unsqueeze(0)  # (1, num_points_y, feature_dim)
            d_batch = torch.norm(x_batch_expanded - y_expanded, dim=-1, p=2)
            
            distances_list.append(d_batch)
        
        # Concatenate all results
        return torch.cat(distances_list, dim=0)