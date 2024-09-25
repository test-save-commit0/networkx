"""Functions for generating graphs based on the "duplication" method.

These graph generators start with a small initial graph then duplicate
nodes and (partially) duplicate their edges. These functions are
generally inspired by biological networks.

"""
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import py_random_state
__all__ = ['partial_duplication_graph', 'duplication_divergence_graph']


@py_random_state(4)
@nx._dispatchable(graphs=None, returns_graph=True)
def partial_duplication_graph(N, n, p, q, seed=None):
    """Returns a random graph using the partial duplication model.

    Parameters
    ----------
    N : int
        The total number of nodes in the final graph.

    n : int
        The number of nodes in the initial clique.

    p : float
        The probability of joining each neighbor of a node to the
        duplicate node. Must be a number in the between zero and one,
        inclusive.

    q : float
        The probability of joining the source node to the duplicate
        node. Must be a number in the between zero and one, inclusive.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Notes
    -----
    A graph of nodes is grown by creating a fully connected graph
    of size `n`. The following procedure is then repeated until
    a total of `N` nodes have been reached.

    1. A random node, *u*, is picked and a new node, *v*, is created.
    2. For each neighbor of *u* an edge from the neighbor to *v* is created
       with probability `p`.
    3. An edge from *u* to *v* is created with probability `q`.

    This algorithm appears in [1].

    This implementation allows the possibility of generating
    disconnected graphs.

    References
    ----------
    .. [1] Knudsen Michael, and Carsten Wiuf. "A Markov chain approach to
           randomly grown graphs." Journal of Applied Mathematics 2008.
           <https://doi.org/10.1155/2008/190836>

    """
    if not 0 <= p <= 1 or not 0 <= q <= 1:
        raise NetworkXError("p and q must be probabilities in [0, 1]")
    if n < 1 or N < n:
        raise NetworkXError("n must be at least 1 and N must be at least n")

    G = nx.complete_graph(n)
    for i in range(n, N):
        # Pick a random node
        u = seed.choice(list(G.nodes()))
        # Create a new node
        v = i
        # Add edges from v to u's neighbors with probability p
        for neighbor in G.neighbors(u):
            if seed.random() < p:
                G.add_edge(v, neighbor)
        # Add edge from u to v with probability q
        if seed.random() < q:
            G.add_edge(u, v)
        # Add the new node to the graph
        G.add_node(v)

    return G


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def duplication_divergence_graph(n, p, seed=None):
    """Returns an undirected graph using the duplication-divergence model.

    A graph of `n` nodes is created by duplicating the initial nodes
    and retaining edges incident to the original nodes with a retention
    probability `p`.

    Parameters
    ----------
    n : int
        The desired number of nodes in the graph.
    p : float
        The probability for retaining the edge of the replicated node.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `p` is not a valid probability.
        If `n` is less than 2.

    Notes
    -----
    This algorithm appears in [1].

    This implementation disallows the possibility of generating
    disconnected graphs.

    References
    ----------
    .. [1] I. Ispolatov, P. L. Krapivsky, A. Yuryev,
       "Duplication-divergence model of protein interaction network",
       Phys. Rev. E, 71, 061911, 2005.

    """
    if not 0 <= p <= 1:
        raise NetworkXError("p must be a probability in [0, 1]")
    if n < 2:
        raise NetworkXError("n must be at least 2")

    G = nx.Graph()
    G.add_edge(0, 1)  # Start with two connected nodes

    for new_node in range(2, n):
        # Choose random node to duplicate
        target_node = seed.choice(list(G.nodes()))
        
        # Add new node
        G.add_node(new_node)
        
        # Connect to target's neighbors with probability p
        for neighbor in G.neighbors(target_node):
            if seed.random() < p:
                G.add_edge(new_node, neighbor)
        
        # Always connect to the target node to ensure connectivity
        G.add_edge(new_node, target_node)

    return G
