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
    pass


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
    pass
