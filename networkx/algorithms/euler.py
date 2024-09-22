"""
Eulerian circuits and graphs.
"""
from itertools import combinations
import networkx as nx
from ..utils import arbitrary_element, not_implemented_for
__all__ = ['is_eulerian', 'eulerian_circuit', 'eulerize', 'is_semieulerian',
    'has_eulerian_path', 'eulerian_path']


@nx._dispatchable
def is_eulerian(G):
    """Returns True if and only if `G` is Eulerian.

    A graph is *Eulerian* if it has an Eulerian circuit. An *Eulerian
    circuit* is a closed walk that includes each edge of a graph exactly
    once.

    Graphs with isolated vertices (i.e. vertices with zero degree) are not
    considered to have Eulerian circuits. Therefore, if the graph is not
    connected (or not strongly connected, for directed graphs), this function
    returns False.

    Parameters
    ----------
    G : NetworkX graph
       A graph, either directed or undirected.

    Examples
    --------
    >>> nx.is_eulerian(nx.DiGraph({0: [3], 1: [2], 2: [3], 3: [0, 1]}))
    True
    >>> nx.is_eulerian(nx.complete_graph(5))
    True
    >>> nx.is_eulerian(nx.petersen_graph())
    False

    If you prefer to allow graphs with isolated vertices to have Eulerian circuits,
    you can first remove such vertices and then call `is_eulerian` as below example shows.

    >>> G = nx.Graph([(0, 1), (1, 2), (0, 2)])
    >>> G.add_node(3)
    >>> nx.is_eulerian(G)
    False

    >>> G.remove_nodes_from(list(nx.isolates(G)))
    >>> nx.is_eulerian(G)
    True


    """
    pass


@nx._dispatchable
def is_semieulerian(G):
    """Return True iff `G` is semi-Eulerian.

    G is semi-Eulerian if it has an Eulerian path but no Eulerian circuit.

    See Also
    --------
    has_eulerian_path
    is_eulerian
    """
    pass


def _find_path_start(G):
    """Return a suitable starting vertex for an Eulerian path.

    If no path exists, return None.
    """
    pass


@nx._dispatchable
def eulerian_circuit(G, source=None, keys=False):
    """Returns an iterator over the edges of an Eulerian circuit in `G`.

    An *Eulerian circuit* is a closed walk that includes each edge of a
    graph exactly once.

    Parameters
    ----------
    G : NetworkX graph
       A graph, either directed or undirected.

    source : node, optional
       Starting node for circuit.

    keys : bool
       If False, edges generated by this function will be of the form
       ``(u, v)``. Otherwise, edges will be of the form ``(u, v, k)``.
       This option is ignored unless `G` is a multigraph.

    Returns
    -------
    edges : iterator
       An iterator over edges in the Eulerian circuit.

    Raises
    ------
    NetworkXError
       If the graph is not Eulerian.

    See Also
    --------
    is_eulerian

    Notes
    -----
    This is a linear time implementation of an algorithm adapted from [1]_.

    For general information about Euler tours, see [2]_.

    References
    ----------
    .. [1] J. Edmonds, E. L. Johnson.
       Matching, Euler tours and the Chinese postman.
       Mathematical programming, Volume 5, Issue 1 (1973), 111-114.
    .. [2] https://en.wikipedia.org/wiki/Eulerian_path

    Examples
    --------
    To get an Eulerian circuit in an undirected graph::

        >>> G = nx.complete_graph(3)
        >>> list(nx.eulerian_circuit(G))
        [(0, 2), (2, 1), (1, 0)]
        >>> list(nx.eulerian_circuit(G, source=1))
        [(1, 2), (2, 0), (0, 1)]

    To get the sequence of vertices in an Eulerian circuit::

        >>> [u for u, v in nx.eulerian_circuit(G)]
        [0, 2, 1]

    """
    pass


@nx._dispatchable
def has_eulerian_path(G, source=None):
    """Return True iff `G` has an Eulerian path.

    An Eulerian path is a path in a graph which uses each edge of a graph
    exactly once. If `source` is specified, then this function checks
    whether an Eulerian path that starts at node `source` exists.

    A directed graph has an Eulerian path iff:
        - at most one vertex has out_degree - in_degree = 1,
        - at most one vertex has in_degree - out_degree = 1,
        - every other vertex has equal in_degree and out_degree,
        - and all of its vertices belong to a single connected
          component of the underlying undirected graph.

    If `source` is not None, an Eulerian path starting at `source` exists if no
    other node has out_degree - in_degree = 1. This is equivalent to either
    there exists an Eulerian circuit or `source` has out_degree - in_degree = 1
    and the conditions above hold.

    An undirected graph has an Eulerian path iff:
        - exactly zero or two vertices have odd degree,
        - and all of its vertices belong to a single connected component.

    If `source` is not None, an Eulerian path starting at `source` exists if
    either there exists an Eulerian circuit or `source` has an odd degree and the
    conditions above hold.

    Graphs with isolated vertices (i.e. vertices with zero degree) are not considered
    to have an Eulerian path. Therefore, if the graph is not connected (or not strongly
    connected, for directed graphs), this function returns False.

    Parameters
    ----------
    G : NetworkX Graph
        The graph to find an euler path in.

    source : node, optional
        Starting node for path.

    Returns
    -------
    Bool : True if G has an Eulerian path.

    Examples
    --------
    If you prefer to allow graphs with isolated vertices to have Eulerian path,
    you can first remove such vertices and then call `has_eulerian_path` as below example shows.

    >>> G = nx.Graph([(0, 1), (1, 2), (0, 2)])
    >>> G.add_node(3)
    >>> nx.has_eulerian_path(G)
    False

    >>> G.remove_nodes_from(list(nx.isolates(G)))
    >>> nx.has_eulerian_path(G)
    True

    See Also
    --------
    is_eulerian
    eulerian_path
    """
    pass


@nx._dispatchable
def eulerian_path(G, source=None, keys=False):
    """Return an iterator over the edges of an Eulerian path in `G`.

    Parameters
    ----------
    G : NetworkX Graph
        The graph in which to look for an eulerian path.
    source : node or None (default: None)
        The node at which to start the search. None means search over all
        starting nodes.
    keys : Bool (default: False)
        Indicates whether to yield edge 3-tuples (u, v, edge_key).
        The default yields edge 2-tuples

    Yields
    ------
    Edge tuples along the eulerian path.

    Warning: If `source` provided is not the start node of an Euler path
    will raise error even if an Euler Path exists.
    """
    pass


@not_implemented_for('directed')
@nx._dispatchable(returns_graph=True)
def eulerize(G):
    """Transforms a graph into an Eulerian graph.

    If `G` is Eulerian the result is `G` as a MultiGraph, otherwise the result is a smallest
    (in terms of the number of edges) multigraph whose underlying simple graph is `G`.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph

    Returns
    -------
    G : NetworkX multigraph

    Raises
    ------
    NetworkXError
       If the graph is not connected.

    See Also
    --------
    is_eulerian
    eulerian_circuit

    References
    ----------
    .. [1] J. Edmonds, E. L. Johnson.
       Matching, Euler tours and the Chinese postman.
       Mathematical programming, Volume 5, Issue 1 (1973), 111-114.
    .. [2] https://en.wikipedia.org/wiki/Eulerian_path
    .. [3] http://web.math.princeton.edu/math_alive/5/Notes1.pdf

    Examples
    --------
        >>> G = nx.complete_graph(10)
        >>> H = nx.eulerize(G)
        >>> nx.is_eulerian(H)
        True

    """
    pass
