import torch
from time import time
from torch_scatter import scatter_min, scatter_max, scatter_mean
from torch_geometric.utils import coalesce
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils import remove_self_loops


from src.data import *
from src.transforms.sampling import _group_data

def orient_edges_wrt_max(V, E):
    """Flip edges to always point from larger to lower node values. 

    This allows for max-propagation on a directed graph, without the 
    need to hold onto both (i, j) and (j, i) while knowing one will be 
    useless in the max propagation step.
    """
    mask = torch.where(V[E[1]] > V[E[0]])[0]
    E[:, mask] = E[:, mask].flip(0)
    return E


def max_propagation(V, E):
    """Propagate the max value within each neighborhood."""
    V_max = scatter_max(V[E[0]], E[1], dim=0, dim_size=V.shape[0])[0]

    # Rather than holding onto self-loops in the graph, we manually only 
    # update nodes here if the values have increased
    idx = torch.where(V_max < V)[0]
    V_max[idx] = V[idx]

    return V_max


def max_of_max_propagation(V, V_max):
    """Propagate the max of the max value. 

    This accelerates propagation by setting all nodes with the same 
    value before the neighborhood max-propagation to the same value 
    after the propagation.
    """
    V_max_max_cc = scatter_max(V_max, V, dim=0)[0]
    V_max_max = V_max_max_cc[V]
    return V_max_max


def connected_components_graph(E, V):
    """Compute the graph of connected components by grouping nodes 
    carrying the same values. 

    V is expected to carry dense indices (ie all values in [0, max(V)] 
    are used).
    """
    device = V.device
    N = V.shape[0]

    # For each component, get the smallest node id
    min_idx_per_cc = scatter_min(torch.arange(N, device=device), V)[0]

    # Assign new (dense) indices to the connected components, to be used
    # as nodes for the next (nested) propagation iteration
    I_cc = consecutive_cluster(min_idx_per_cc)[0]

    # Distribute the CC new indices to each input node
    I = I_cc[V]

    # Build the CC graph
    E_cc = I[E]

    # Remove self-loops, the max-propagation will handle this edge case 
    # itself
    E_cc, _ = remove_self_loops(E_cc)

    # Remove duplicate edges to accelerate downstream operations
    # TODO: this is the main bottleneck of the algorithm !
    E_cc = coalesce(E_cc)

    return E_cc, I


def _max_iterations_max_prop(max_iterations, E):
    """Maximum number of iterations for finding the weakly connected
    components using max-propagation. If max_iterations is provided, we
    keep it. If not, we set it to the worst possible graph traversal
    scenario: the number of edges in the graph.
    """
    return max(E.shape[1], 1) if max_iterations < 1 else max_iterations


def _wcc_by_max_propagation_without_isolated_nodes(
        N,
        E,
        max_max=True,
        depth=0,
        max_iterations=-1):
    """Recursively apply max propagation to the graph of connected
    components where self-loops and isolated nodes have been removed.
    Each recursive call is applied only to the smaller graph
    of components.
    """
    max_iterations = _max_iterations_max_prop(max_iterations, E)
    device = E.device

    # Initialize nodes with unique, shuffled, dense indices. Although
    # this introduces stochasticity to the algorithm, this often
    # accelerates convergence compared to using the node indices as
    # propagated values
    V = torch.randperm(N, device=device)

    # Flip edges so that they point from larger to smaller values. This
    # allows holding onto directed graphs while computing weakly
    # connected components, instead of manipulating twice as many edges
    # for undirected graphs
    E = orient_edges_wrt_max(V, E)

    # Propagate the max value within each neighborhood
    V_max = max_propagation(V, E)

    # Max of max propagation
    if max_max:
        V_max = max_of_max_propagation(V, V_max)

    # Compute the connected component assignments. We make sure that
    # connected components are represented by dense indices
    I = consecutive_cluster(V_max)[0]

    # Return if no change after max propagation
    if depth >= max_iterations - 1 or (V != V_max).count_nonzero() == 0:
        return I, depth

    # Compute the graph of connected components on which the recursive
    # propagation will be run
    E_cc, I = connected_components_graph(E, I)

    # Recursive call of the max propagation on the graph of components.
    # This returns the assignment of the current connected components to
    # their parent components after max propagation iteration
    # NB: we recursively call the max-propagation on a graph without
    # assuming non-isolated nodes. This is because the graph E_cc might
    # contain isolated nodes after one merging operation
    I_cc, depth = wcc_by_max_propagation(
        I.max() + 1,
        E_cc,
        max_max=max_max,
        depth=depth + 1,
        max_iterations=max_iterations)

    return I_cc[I], depth


def wcc_by_max_propagation(
        N,
        E,
        max_max=True,
        depth=0,
        max_iterations=-1):
    """Recursively apply max propagation to the graph of connected 
    components. Each recursive call is applied only to the smaller graph
    of components.
    """
    max_iterations = _max_iterations_max_prop(max_iterations, E)
    device = E.device

    # Remove self-loops before the first max-propagation iteration, as 
    # max_propagation() will handle this edge case itself
    # NB: although coalescing the graph may seem like a good idea at 
    # this point, it seems faster in practice to coalesce only after 
    # the first max-propagation and merging operation
    if depth == 0:
        E, _ = remove_self_loops(E)

    # Reduce the problem size by removing the nodes that have no edge
    mask_non_isolated = torch.full((N,), False, dtype=torch.bool, device=device)
    mask_non_isolated[E.view(-1)] = True
    I_sub = mask_non_isolated.nonzero().view(-1)
    I_sub_inv = torch.full((N,), -1, dtype=torch.long, device=device)
    I_sub_inv[I_sub] = torch.arange(I_sub.shape[0], device=device)
    E_sub = I_sub_inv[E]
    N_sub = I_sub.shape[0]
    N_isolated = N - N_sub

    # If all nodes are isolated, exit here
    if N_isolated == N:
        return torch.arange(N, device=device), depth

    # Run the max-propagation on the graph of non-isolated nodes
    I_cc_sub, depth = _wcc_by_max_propagation_without_isolated_nodes(
        N_sub,
        E_sub,
        max_max=max_max,
        depth=depth,
        max_iterations=max_iterations)

    # Combine the index assignments for the connected components on the
    # sub graph with the (trivial) assignment of the isolated nodes to
    # themselves
    if N_isolated > 0:
        I_cc = torch.full((N,), -1, dtype=torch.long, device=device)
        I_cc[I_sub] = I_cc_sub + N_isolated
        I_cc[I_cc == -1] = torch.arange(N_isolated, device=device)
    else:
        I_cc = I_cc_sub

    return I_cc, depth


def wcc_by_max_propagation_on_data(
        data,
        max_iterations=-1,
        max_max=False,
        verbose=False) -> Data:
    """Compute the weakly connected components of a graph by 
    max-propagation.
    """
    if verbose:
        torch.cuda.synchronize()
        start = time()

    # Recover necessary information from the Data object
    N = data.num_nodes
    E = data.edge_index

    # Compute connected components
    I, iterations = wcc_by_max_propagation(
        N,
        E,
        max_max=max_max,
        depth=0,
        max_iterations=max_iterations)

    if verbose:
        torch.cuda.synchronize()
        I_max = max_propagation(I, E)
        num_diff = (I != I_max).count_nonzero().item()
        num_cc = I_max.unique().shape[0]
        max_iterations = _max_iterations_max_prop(max_iterations, E)
        print(
            f"Time: {time() - start:0.3f} | "
            f"Num components: {num_cc} | "
            f"Iterations: {iterations + 1}/{max_iterations} | "
            f"Not converged: {num_diff}/{N}")

    data = data.clone()
    data.super_index = I
    
    return data