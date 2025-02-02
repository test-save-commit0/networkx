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
    if G.number_of_nodes() == 0:
        return True
    if G.is_directed():
        # For directed graphs, check if in_degree == out_degree for all nodes
        return (G.is_strongly_connected() and
                all(G.in_degree(n) == G.out_degree(n) for n in G))
    else:
        # For undirected graphs, check if all degrees are even
        return (nx.is_connected(G) and
                all(d % 2 == 0 for v, d in G.degree()))


@nx._dispatchable
def is_semieulerian(G):
    """Return True iff `G` is semi-Eulerian.

    G is semi-Eulerian if it has an Eulerian path but no Eulerian circuit.

    See Also
    --------
    has_eulerian_path
    is_eulerian
    """
    if G.number_of_nodes() == 0:
        return False
    if G.is_directed():
        # For directed graphs, check if there's exactly one node with out_degree - in_degree = 1
        # and exactly one node with in_degree - out_degree = 1
        degree_diff = [G.out_degree(n) - G.in_degree(n) for n in G]
        return (G.is_strongly_connected() and
                degree_diff.count(1) == 1 and
                degree_diff.count(-1) == 1 and
                all(d == 0 for d in degree_diff if d not in {-1, 0, 1}))
    else:
        # For undirected graphs, check if there are exactly two nodes with odd degree
        odd_degree_count = sum(1 for v, d in G.degree() if d % 2 != 0)
        return nx.is_connected(G) and odd_degree_count == 2


def _find_path_start(G):
    """Return a suitable starting vertex for an Eulerian path.

    If no path exists, return None.
    """
    if G.is_directed():
        for v in G:
            if G.out_degree(v) - G.in_degree(v) == 1:
                return v
        # If no suitable start found, return an arbitrary node
        return next(iter(G))
    else:
        for v in G:
            if G.degree(v) % 2 != 0:
                return v
        # If all degrees are even, return an arbitrary node
        return next(iter(G))


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
    if not is_eulerian(G):
        raise nx.NetworkXError("Graph is not Eulerian.")
    
    if G.number_of_edges() == 0:
        return []

    if source is None:
        source = arbitrary_element(G)

    if G.is_multigraph():
        G_iter = G.edges
    else:
        G_iter = G.edges

    def get_unused_edge(v):
        for u, w, k in G_iter(v):
            if not used[v][w].get(k, False):
                used[v][w][k] = True
                return w, k
        return None, None

    used = {v: {w: {} for w in G[v]} for v in G}
    vertex_stack = [source]
    last_vertex = None
    while vertex_stack:
        current_vertex = vertex_stack[-1]
        if current_vertex != last_vertex:
            last_vertex = current_vertex
            next_vertex, key = get_unused_edge(current_vertex)
            if next_vertex is not None:
                vertex_stack.append(next_vertex)
                if keys and G.is_multigraph():
                    yield (current_vertex, next_vertex, key)
                else:
                    yield (current_vertex, next_vertex)
            else:
                if len(vertex_stack) > 1:
                    last_vertex = vertex_stack.pop()
                    yield (vertex_stack[-1], last_vertex)
                else:
                    last_vertex = vertex_stack.pop()
        else:
            last_vertex = vertex_stack.pop()
            if len(vertex_stack) > 0:
                yield (vertex_stack[-1], last_vertex)


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
    if G.number_of_nodes() == 0:
        return True
    
    if G.is_directed():
        if not nx.is_strongly_connected(G):
            return False
        
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        diff = {v: out_degree[v] - in_degree[v] for v in G}
        
        if source is not None:
            if diff[source] > 1 or any(d > 0 for v, d in diff.items() if v != source):
                return False
        else:
            if sum(1 for d in diff.values() if d > 0) > 1 or sum(1 for d in diff.values() if d < 0) > 1:
                return False
        
        return all(abs(d) <= 1 for d in diff.values())
    else:
        if not nx.is_connected(G):
            return False
        
        odd_degree_count = sum(1 for v, d in G.degree() if d % 2 != 0)
        
        if source is not None:
            return odd_degree_count == 0 or (odd_degree_count == 2 and G.degree(source) % 2 != 0)
        else:
            return odd_degree_count in (0, 2)


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
    if not has_eulerian_path(G, source):
        raise nx.NetworkXError("Graph has no Eulerian path.")

    if source is None:
        source = _find_path_start(G)

    if G.is_multigraph():
        G_iter = G.edges
    else:
        G_iter = G.edges

    def get_unused_edge(v):
        for u, w, k in G_iter(v):
            if not used[v][w].get(k, False):
                used[v][w][k] = True
                return w, k
        return None, None

    used = {v: {w: {} for w in G[v]} for v in G}
    vertex_stack = [source]
    last_vertex = None

    while vertex_stack:
        current_vertex = vertex_stack[-1]
        if current_vertex != last_vertex:
            last_vertex = current_vertex
            next_vertex, key = get_unused_edge(current_vertex)
            if next_vertex is not None:
                vertex_stack.append(next_vertex)
                if keys and G.is_multigraph():
                    yield (current_vertex, next_vertex, key)
                else:
                    yield (current_vertex, next_vertex)
            else:
                if len(vertex_stack) > 1:
                    last_vertex = vertex_stack.pop()
                    yield (vertex_stack[-1], last_vertex)
                else:
                    last_vertex = vertex_stack.pop()
        else:
            last_vertex = vertex_stack.pop()
            if len(vertex_stack) > 0:
                yield (vertex_stack[-1], last_vertex)


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
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph is not connected.")

    if is_eulerian(G):
        return nx.MultiGraph(G)

    odd_degree_vertices = [v for v, d in G.degree() if d % 2 != 0]
    G_multi = nx.MultiGraph(G)

    if len(odd_degree_vertices) == 0:
        return G_multi

    # Find minimum weight matching
    odd_G = nx.Graph()
    for u, v in combinations(odd_degree_vertices, 2):
        odd_G.add_edge(u, v, weight=nx.shortest_path_length(G, u, v, weight="weight"))

    matching = nx.min_weight_matching(odd_G)

    # Add matched edges to the graph
    for u, v in matching:
        path = nx.shortest_path(G, u, v, weight="weight")
        nx.add_path(G_multi, path)

    return G_multi
