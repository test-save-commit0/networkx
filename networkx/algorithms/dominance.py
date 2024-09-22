"""
Dominance algorithms.
"""
from functools import reduce
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['immediate_dominators', 'dominance_frontiers']


@not_implemented_for('undirected')
@nx._dispatchable
def immediate_dominators(G, start):
    """Returns the immediate dominators of all nodes of a directed graph.

    Parameters
    ----------
    G : a DiGraph or MultiDiGraph
        The graph where dominance is to be computed.

    start : node
        The start node of dominance computation.

    Returns
    -------
    idom : dict keyed by nodes
        A dict containing the immediate dominators of each node reachable from
        `start`.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is undirected.

    NetworkXError
        If `start` is not in `G`.

    Notes
    -----
    Except for `start`, the immediate dominators are the parents of their
    corresponding nodes in the dominator tree.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 5), (3, 4), (4, 5)])
    >>> sorted(nx.immediate_dominators(G, 1).items())
    [(1, 1), (2, 1), (3, 1), (4, 3), (5, 1)]

    References
    ----------
    .. [1] K. D. Cooper, T. J. Harvey, and K. Kennedy.
           A simple, fast dominance algorithm.
           Software Practice & Experience, 4:110, 2001.
    """
    pass


@nx._dispatchable
def dominance_frontiers(G, start):
    """Returns the dominance frontiers of all nodes of a directed graph.

    Parameters
    ----------
    G : a DiGraph or MultiDiGraph
        The graph where dominance is to be computed.

    start : node
        The start node of dominance computation.

    Returns
    -------
    df : dict keyed by nodes
        A dict containing the dominance frontiers of each node reachable from
        `start` as lists.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is undirected.

    NetworkXError
        If `start` is not in `G`.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 5), (3, 4), (4, 5)])
    >>> sorted((u, sorted(df)) for u, df in nx.dominance_frontiers(G, 1).items())
    [(1, []), (2, [5]), (3, [5]), (4, [5]), (5, [])]

    References
    ----------
    .. [1] K. D. Cooper, T. J. Harvey, and K. Kennedy.
           A simple, fast dominance algorithm.
           Software Practice & Experience, 4:110, 2001.
    """
    pass
