"""
Algorithm to find a maximal (not maximum) independent set.

"""
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
__all__ = ['maximal_independent_set']


@not_implemented_for('directed')
@py_random_state(2)
@nx._dispatchable
def maximal_independent_set(G, nodes=None, seed=None):
    """Returns a random maximal independent set guaranteed to contain
    a given set of nodes.

    An independent set is a set of nodes such that the subgraph
    of G induced by these nodes contains no edges. A maximal
    independent set is an independent set such that it is not possible
    to add a new node and still get an independent set.

    Parameters
    ----------
    G : NetworkX graph

    nodes : list or iterable
       Nodes that must be part of the independent set. This set of nodes
       must be independent.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    indep_nodes : list
       List of nodes that are part of a maximal independent set.

    Raises
    ------
    NetworkXUnfeasible
       If the nodes in the provided list are not part of the graph or
       do not form an independent set, an exception is raised.

    NetworkXNotImplemented
        If `G` is directed.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.maximal_independent_set(G)  # doctest: +SKIP
    [4, 0, 2]
    >>> nx.maximal_independent_set(G, [1])  # doctest: +SKIP
    [1, 3]

    Notes
    -----
    This algorithm does not solve the maximum independent set problem.

    """
    import random

    # Check if the graph is directed
    if G.is_directed():
        raise nx.NetworkXNotImplemented("Not implemented for directed graphs.")

    # Initialize the independent set with the given nodes
    if nodes is not None:
        independent_set = set(nodes)
        # Check if the given nodes are in the graph and form an independent set
        if not all(node in G for node in independent_set):
            raise nx.NetworkXUnfeasible("Given nodes are not in the graph.")
        if any(v in G[u] for u in independent_set for v in independent_set if u != v):
            raise nx.NetworkXUnfeasible("Given nodes do not form an independent set.")
    else:
        independent_set = set()

    # Create a set of candidate nodes (all nodes not in the independent set)
    candidates = set(G.nodes()) - independent_set

    # Set the random seed
    random.seed(seed)

    while candidates:
        # Randomly select a node from the candidates
        node = random.choice(list(candidates))
        # Add the node to the independent set
        independent_set.add(node)
        # Remove the node and its neighbors from the candidates
        candidates.remove(node)
        candidates -= set(G[node])

    return list(independent_set)
