import torch
from torch_scatter import scatter_sum
from torch_graph_components import wcc_by_max_propagation
from torch_graph_components import merge_components_by_contour_prior
from torch_graph_components.merge import component_graph

from src.data import NAG, Data, Cluster
from src.utils.scatter import scatter_mean_weighted


def merge_components_by_contour_prior_on_data(
        data: Data,
        reg: float,
        min_size: int,
        merge_only_small: bool = False,
        k: int = -1,
        w_adjacency: float = -1,
        max_iterations: int = -1,
        sharding: int = None,
        reduce: str = 'add',
        verbose: bool = False,
) -> NAG:
    """Compute the weakly connected components of a graph by
    max-propagation.
    
    If the data object has a `super_index` attribute, the components that will
    be merged are the supernodes defined by `super_index`.
    Otherwise, it will merge nodes directly. In both cases, the returned
    data carries the merged components by its `super_index` attribute.
    
    :param data: Data object containing the graph to merge
    :type data: Data
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
    :param max_iterations: Maximum number of merging iterations
    :type max_iterations: int
    :param sharding: Allows mitigating memory use. If `sharding > 1`,
        `edge_index` will be processed into chunks of `sharding` during
        the memory bottleneck of the algorithm (i.e., when computing the
        change of energy of every edge). If `0 < sharding < 1`, then
        `edge_index` will be divided into parts of
        `edge_index.shape[1] * sharding` or less
    :type sharding: int, float
    :param reduce: str
        How to reduce duplicate edges. Options: 'add', 'mean', 'max',
        'min', 'mul'. Default: 'add'
    :type reduce: str
    :param verbose: Whether to measure speed and return information
        about the algorithm
    :type verbose: bool
    """
    
    assert data.x is not None
    assert data.pos is not None
    # assert data.super_index is not None
    assert data.edge_index is not None
    assert data.edge_attr is not None

    # Recover necessary information from the Data object
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
        reduce=reduce,
        verbose=verbose)
    
    # If k <= 0, P_merged is not needed in merge_components_by_contour_prior,
    # thus not computed at each step of the recursion. This is why we compute it here.
    P_merged = scatter_mean_weighted(P, I_merged, S) if k <= 0 else P_merged
    
    data = data.clone()
    super_index = I_merged[I]
    data.super_index = super_index
    
    cluster_size = data.super_index.bincount()
    pointer = torch.cat((
        torch.tensor([0], dtype=torch.long, device=data.super_index.device),
        cluster_size.cumsum(dim=0)))
    value = torch.argsort(data.super_index)
    sub = Cluster(pointer, value)
        
    return data, (X_merged, S_merged, E_cp, W_cp, P_merged, sub)


def wcc_by_max_propagation_on_data(
        data,
        max_iterations=-1,
        max_max=False,
        verbose=False) -> Data:
    """Compute the weakly connected components of a graph by
    max-propagation.
    """
    # Compute connected components
    I, _ = wcc_by_max_propagation(
        data.num_nodes,
        data.edge_index,
        max_max=max_max,
        depth=0,
        max_iterations=max_iterations,
        verbose=verbose)

    data = data.clone()
    data.super_index = I

    return data
