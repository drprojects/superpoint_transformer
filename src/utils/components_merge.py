from typing import Tuple

import math
import torch

from torch import Tensor
from time import time
from torch_scatter import scatter_sum, scatter_min, scatter_mean
from torch_geometric.utils import coalesce, remove_self_loops

from src.data import NAG, Data, Cluster
from src.utils.scatter import scatter_mean_weighted
from src.utils.neighbors import knn_brute_force
from src.utils.connected_components import wcc_by_max_propagation


def component_graph(I, E, W, no_self_loops=True):
    # Convert node-wise edges to component-wise edges
    E_cp = I[E]

    # Optionally remove self-loops
    if no_self_loops:
        E_cp, W_cp = remove_self_loops(E_cp, edge_attr=W)
    else:
        W_cp = W

    return E_cp, W_cp


def merge_duplicate_edges(E, W, undirected=True, reduce='add'):
    """Merge duplicate edges of a graph. The weight of the resulting
    edge is the sum of the constituting edge weights.
    """
    # Orient all edges i->j so that i<j
    if undirected:
        mask = E[0] > E[1]
        E = torch.cat((E[:, ~mask], E[:, mask].flip(0)), dim=1)
        W = torch.cat((W[~mask], W[mask]))
    return coalesce(E, edge_attr=W, reduce=reduce)


def edge_merge_energy(X, S, E, W, reg, sharding=None):
    """Compute the energy induced by the merge of two connected
    components.
    
    Args:
        X: Node features
        S: Node sizes  
        E: Edge indices
        W: Edge weights
        reg: Regularization parameter
    
    :param sharding: int, float
        Allows mitigating memory use. If `sharding > 1`,
        `edge_index` will be processed into chunks of `sharding`. 
        If `0 < sharding < 1`, then `edge_index` will be 
        divided into parts of `edge_index.shape[1] * sharding` or less
    """
    if sharding is None or sharding <= 0:
        X_source, X_target = X[E]
        S_source, S_target = S[E]
        
        delta_H = (S_source * S_target / (S_source + S_target)
                * ((X_source - X_target)**2).sum(dim=1)
                - reg * W)
        
        return delta_H
    
    else : # Recursive call in case of sharding.
        # Sharding allows limiting the number of edges processed at once.
        # This might alleviate memory use.
        
        # if sharding > 1 : sharding is a number of edges to processed at once
        # else : sharding is a percentage of edges to processed at once
        sharding = int(sharding) if sharding > 1 \
            else math.ceil(E.shape[1] * sharding)
            
        num_shards = math.ceil(E.shape[1] / sharding)
        
        delta_H_list = []
        for i_shard in range(num_shards):
            start = i_shard * sharding
            end = min(start + sharding, E.shape[1])
            
            E_shard = E[:, start:end]
            W_shard = W[start:end]
            
            delta_H_list.append(edge_merge_energy(X, S, E_shard, W_shard, reg, sharding=None))
            
        return torch.cat(delta_H_list)


def find_lowest_energy_merge_candidate(E, delta_H):
    """Identify, for each component, the best adjacent component with
    which to merge. The best candidate is the one which minimizes the
    induced change of energy of the partition.
    """
    # Compute, for each component, its best candidate merge
    num_nodes = E.max() + 1
    source = torch.cat((E[0], E[1]))
    target = torch.cat((E[1], E[0]))
    delta_H = torch.cat((delta_H, delta_H))
    _, argmin = scatter_min(delta_H, source, dim_size=num_nodes)

    # Recover the target component index with which each component.
    # For components that do not have any edge in E, scatter_min() will
    # put num_nodes in argmin
    argmin = argmin[argmin < source.shape[0]]
    E_merge = torch.vstack((source[argmin], target[argmin]))
    delta_H = delta_H[argmin]

    return E_merge, delta_H


def _max_iterations_merge(max_iterations, N):
    """Maximum number of iterations for merging nodes. If max_iterations
    is provided. If not, we set it to the worst possible iterative 
    merging scenario: the number of nodes in the graph.
    """
    return max(N, 1) if max_iterations < 1 else max_iterations


def connect_isolated_nodes(P, E, W, k, w_adjacency):
    """Search for isolated nodes and connect them to the k nearest
    nodes.
    """
    # If k is too small, or if there is fewer than 2 nodes, exit here
    if k < 1 or P.shape[0] < 2:
        return E, W

    # Search for isolated nodes
    mask_isolated = torch.full(
        (P.shape[0],), True, dtype=torch.bool, device=E.device)
    mask_isolated[E.view(-1)] = False
    I_isolated = mask_isolated.nonzero().view(-1)

    # If there are no isolated nodes, exit here
    if I_isolated.shape[0] == 0:
        return E, W

    # Search the nearest
    r_max = (P.max(dim=0).values - P.min(dim=0).values).norm()
    neighbors, distances = knn_brute_force(
        P,
        P[I_isolated],
        k + 1,
        r_max=r_max)

    # Remove self in neighbors
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]

    # Build the new edges and weights
    E_isolated = torch.vstack((
        I_isolated.repeat_interleave(k),
        neighbors.flatten()))
    if w_adjacency > 0:
        W_isolated = (1 / (1 + distances / distances.mean())).flatten()
    else:
        W_isolated = torch.ones_like(distances.flatten(), dtype=torch.float)

    # Combine the new edges and weights to the existing ones
    return (torch.column_stack((E, E_isolated)),
            torch.cat((W, W_isolated)))


def merge_components_by_contour_prior(
        X: Tensor,
        S: Tensor,
        E: Tensor,
        W: Tensor,
        reg: float,
        min_size: int,
        merge_only_small: bool = False,
        P: Tensor = None,
        k: int = -1,
        w_adjacency: float = -1,
        depth: int = 0,
        max_iterations: int = -1,
        sharding: int = None,
        reduce: str = 'add'
) -> Tuple[Tensor, int]:
    """Iteratively merge components based on a greedy energy-based
    contour prior.

    :param X: Node features
    :type X: Tensor
    :param S: Node sizes
    :type S: Tensor
    :param E: Edges of the node-wise graph
    :type E: Tensor
    :param W: Weights of the edges in E
    :type W: Tensor
    :param reg: regularization parameter ruling the importance of edges
        in the energy
    :type reg: float
    :param min_size: All components of smaller than min_size will be
        merged
    :type min_size: int
    :param merge_only_small: If True, only small components will be
        merged. If False, components whose merge allows a decrease in
        the energy will also be merged
    :type merge_only_small: bool
    :param P: Positions of the input nodes in coordinate space. Required
        if `k > 0`
    :type P: Tensor
    :param k: If `k > 0`, the isolated components will be connected to
        their k nearest neighbors in coordinate space using P before
        each merging iteration. By providing k and P, isolated nodes may
        still be merged. If not, it is possible that the algorithm
        returns without meeting the min_size requirements
    :type k: int
    :param w_adjacency: Scalar used to modulate the newly created edge
        weights when `k > 0`. If `w_adjacency <= 0`, all edges
        will have a weight of `1`. Otherwise, edges weights will follow:
        ```1 / (w_adjacency + distance / distance.mean())```
    :type w_adjacency: float
    :param depth: Recursive depth, used to track iterations
    :type depth: int
    :param max_iterations: Maximum number of merging iterations
    :type max_iterations: int
    :param num_batch: If provided, divide edge energy computation into this number of batches
        to reduce memory usage. Useful for large graphs with many edges.
    :type num_batch: int

    :return: New merged parent component index for each input node, depth, 
        and a tuple (X, S, E, W, P) for features, sizes, edges, weights, and positions 
        of the merged components.

    :rtype: Tuple[Tensor, int, Tuple[Tensor, Tensor,Tensor,Tensor,Tensor]]
    
    :param sharding: int, float
        Allows mitigating memory use. If `sharding > 1`,
        `edge_index` will be processed into chunks of `sharding` during 
        the memory bottleneck of the algorithm (i.e., when computing the 
        change of energy of every edge). If `0 < sharding < 1`, then
        `edge_index` will be divided into parts of 
        `edge_index.shape[1] * sharding` or less
    :param reduce: str
        How to reduce duplicate edges. Options: 'add', 'mean', 'max', 
        'min', 'mul'. Default: 'add'
    :type reduce: str
    """
    # Initialization
    num_nodes = X.shape[0]
    device = X.device

    # Return if the maximum number of iterations has been reached
    max_iterations = _max_iterations_merge(max_iterations, num_nodes)
    if depth >= max_iterations:
        return torch.arange(num_nodes, device=device), depth, (X, S, E, W, P)

    # If there is only one component, exit here
    # NB: this assumes indices used in I are compact in [0, I_max]
    if num_nodes < 2:
        return torch.arange(num_nodes, device=device), depth, (X, S, E, W, P)

    # If all components are large enough, and we do not care about
    # merging larger components based on energy, we can exit here
    if merge_only_small and (S >= min_size).all():
        return torch.arange(num_nodes, device=device), depth, (X, S, E, W, P)

    # Remove self-loops in the graph
    E, W = remove_self_loops(E, edge_attr=W)

    # Connect isolated nodes (if k > 0)
    E, W = connect_isolated_nodes(P, E, W, k, w_adjacency)

    # If all components are isolated, exit here
    if E.shape[1] == 0:
        return torch.arange(num_nodes, device=device), depth, (X, S, E, W, P)

    # Merge the duplicate edges. We work on undirected graphs, where
    # i->j and j->i are considered to be the same. By convention, we
    # later only use i->j such that i<j
    # TODO: this is the main bottleneck of the algorithm !
    #  To mitigate this, make sure you pass a small-enough adjacency
    #  graph as input. The more connections each node has, the more
    #  expensive this step. Note that isolated nodes will be properly
    #  handled if k>0 is provided.
    #  Maybe alleviated by only updating the node and edge features
    #  linked to nodes that were merged at the previous iteration ?
    E, W = merge_duplicate_edges(E, W, undirected=True, reduce=reduce)

    # Compute the change of energy induced by the merge of each
    # inter-component edge
    delta_H = edge_merge_energy(X, S, E, W, reg, sharding)

    # Find the components with the lowest merge energy. Components
    # without any neighbor to merge with will be absent from E_merge
    E_merge, delta_H_merge = find_lowest_energy_merge_candidate(E, delta_H)

    # Only keep the best merge for too-small components or merges that
    # reduce the energy
    mask = (S[E_merge[0]] < min_size)
    if not merge_only_small:
        mask = torch.logical_or(mask, (delta_H_merge <= 0))
    E_merge = E_merge[:, mask]

    # If no relevant merge can be found, exit here
    if E_merge.shape[1] == 0:
        return torch.arange(num_nodes, device=device), depth, (X, S, E, W, P)

    # Compute the new components after merging. These are represented by
    # an assignment, just like a new partition level
    I_merged, _ = wcc_by_max_propagation(num_nodes, E_merge)
    
    # Compute the node size and mean feature of each component
    S_merged = scatter_sum(S, I_merged, dim=0)
    X_merged = scatter_mean_weighted(X, I_merged, S)
    P_merged = scatter_mean_weighted(P, I_merged, S) if k > 0 else None

    # Get the superedges between components. Only inter-component edges
    # are preserved (i.e. we remove self-loops)
    E_cp, W_cp = component_graph(I_merged, E, W, no_self_loops=True)
    
    # If there is only one component, exit here
    # NB: we make sure the unique index used is 0
    if I_merged.max() - I_merged.min() == 0:
        return I_merged * 0, depth, (X_merged, S_merged, E_cp, W_cp, P_merged)

    # If no, for some reason, we have the same number of components
    # after the merge, exit here
    if I_merged.max() + 1 == I_merged.shape[0]:
        return torch.arange(I_merged.shape[0], device=device), depth, (X_merged, S_merged, E_cp, W_cp, P_merged)

    # If all components are large enough, and we do not care about
    # merging larger components based on energy, we can exit here
    if merge_only_small and (S_merged >= min_size).all():
        return I_merged, depth, (X_merged, S_merged, E_cp, W_cp, P_merged)

    # Recursively merge the graph of merged components
    I_merged_next, depth, (X_merged, S_merged, E_cp, W_cp, P_merged) = merge_components_by_contour_prior(
        X_merged,
        S_merged,
        E_cp,
        W_cp,
        reg,
        min_size,
        merge_only_small=merge_only_small,
        P=P_merged,
        k=k,
        w_adjacency=w_adjacency,
        depth=depth + 1,
        max_iterations=max_iterations,
        sharding=sharding,
        reduce=reduce)

    # Update the input node assignments with the new merged components
    return I_merged_next[I_merged], depth, (X_merged, S_merged, E_cp, W_cp, P_merged)


def merge_components_by_contour_prior_on_data(
        data,
        reg: float,
        min_size: int,
        merge_only_small: bool = False,
        k: int = -1,
        w_adjacency: float = -1,
        max_iterations: int = -1,
        verbose: bool = False,
        sharding: int = None,
        reduce: str = 'add'
) -> NAG:
    """Compute the weakly connected components of a graph by
    max-propagation.
    
    If the data object has a `super_index` attribute, the components that will
    be merged are the supernodes defined by `super_index`.
    Otherwise, it will merge nodes directly. In both cases, the returned
    data carries the merged components by its `super_index` attribute.
    
    :param data: Data object containing the graph to merge
    :type data: Data
    :param reg: Regularization parameter
    :type reg: float
    :param min_size: Minimum size of the components to merge
    
    :param sharding: int, float
        Allows mitigating memory use. If `sharding > 1`,
        `edge_index` will be processed into chunks of `sharding` during 
        the bottleneck of the algorithm. If `0 < sharding < 1`, then 
        `edge_index` will be divided into parts of 
        `edge_index.shape[1] * sharding` or less
    :param reduce: str
        How to reduce duplicate edges. Options: 'add', 'mean', 'max', 
        'min', 'mul'. Default: 'add'
    :type reduce: str
    """
    
    assert data.x is not None
    assert data.pos is not None
    #assert data.super_index is not None
    assert data.edge_index is not None
    assert data.edge_attr is not None

    if verbose:
        torch.cuda.synchronize()
        start_preprocessing = time()

    # Recover necessary information from the Data object
    N = data.num_nodes
    X = data.x
    P = data.pos
    I = data.super_index if getattr(data, 'super_index', None) is not None \
        else torch.arange(data.num_nodes, device=data.device)
    E = data.edge_index
    W = data.edge_attr
    S = getattr(data, 'node_size', None)
    S = torch.ones_like(I) if S is None else S

    # Compute the node size and mean feature of each component
    S_cp = scatter_sum(S, I, dim=0)
    X_cp = scatter_mean_weighted(X, I, S)
    P_cp = scatter_mean_weighted(P, I, S) if k > 0 else None

    # Get the superedges between components. Only inter-component edges
    # are preserved (i.e. we remove self-loops)
    E_cp, W_cp = component_graph(I, E, W, no_self_loops=False)

    if verbose:
        torch.cuda.synchronize()
        start_merge = time()

    # Merge components
    I_merged, iterations, (X_merged, S_merged, E_cp, W_cp, P_merged) = merge_components_by_contour_prior(
        X_cp,
        S_cp,
        E_cp,
        W_cp,
        reg,
        min_size,
        merge_only_small=merge_only_small,
        P=P_cp,
        k=k,
        w_adjacency=w_adjacency,
        depth=0,
        max_iterations=max_iterations,
        sharding=sharding,
        reduce=reduce)
    
    # If k <= 0, P_merged is not needed in merge_components_by_contour_prior,
    # thus not computed at each step of the recursion. This is why we compute it here.
    P_merged = scatter_mean_weighted(P, I_merged, S) if k <= 0 else P_merged

    if verbose:
        torch.cuda.synchronize()
        num_cp_start = X_cp.shape[0]
        num_cp_end = I_merged.unique().shape[0]
        max_iterations = _max_iterations_merge(max_iterations, N)
        print(
            f"Merging time: {time() - start_merge:0.3f} | "
            f"Preprocessing time: {start_merge - start_preprocessing:0.3f} | "
            f"#CP start: {num_cp_start} | "
            f"#CP end: {num_cp_end} | "
            f"ratio (#CP start)/(#CP end): {num_cp_start / num_cp_end:0.1f} | "
            f"Iterations: {iterations + 1}/{max_iterations}")

    
    data = data.clone()
    super_index = I_merged[I]
    data.super_index = super_index
    
    cluster_size = data.super_index.bincount()
    pointer = torch.cat((torch.tensor([0], dtype=torch.long, device=data.super_index.device), cluster_size.cumsum(dim=0)))
    value = torch.argsort(data.super_index)
    sub = Cluster(pointer, value)
        
    return data, (X_merged, S_merged, E_cp, W_cp, P_merged, sub)