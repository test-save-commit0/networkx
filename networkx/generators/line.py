"""Functions for generating line graphs."""
from collections import defaultdict
from functools import partial
from itertools import combinations
import networkx as nx
from networkx.utils import arbitrary_element
from networkx.utils.decorators import not_implemented_for
__all__ = ['line_graph', 'inverse_line_graph']


@nx._dispatchable(returns_graph=True)
def line_graph(G, create_using=None):
    """Returns the line graph of the graph or digraph `G`.

    The line graph of a graph `G` has a node for each edge in `G` and an
    edge joining those nodes if the two edges in `G` share a common node. For
    directed graphs, nodes are adjacent exactly when the edges they represent
    form a directed path of length two.

    The nodes of the line graph are 2-tuples of nodes in the original graph (or
    3-tuples for multigraphs, with the key of the edge as the third element).

    For information about self-loops and more discussion, see the **Notes**
    section below.

    Parameters
    ----------
    G : graph
        A NetworkX Graph, DiGraph, MultiGraph, or MultiDigraph.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    L : graph
        The line graph of G.

    Examples
    --------
    >>> G = nx.star_graph(3)
    >>> L = nx.line_graph(G)
    >>> print(sorted(map(sorted, L.edges())))  # makes a 3-clique, K3
    [[(0, 1), (0, 2)], [(0, 1), (0, 3)], [(0, 2), (0, 3)]]

    Edge attributes from `G` are not copied over as node attributes in `L`, but
    attributes can be copied manually:

    >>> G = nx.path_graph(4)
    >>> G.add_edges_from((u, v, {"tot": u + v}) for u, v in G.edges)
    >>> G.edges(data=True)
    EdgeDataView([(0, 1, {'tot': 1}), (1, 2, {'tot': 3}), (2, 3, {'tot': 5})])
    >>> H = nx.line_graph(G)
    >>> H.add_nodes_from((node, G.edges[node]) for node in H)
    >>> H.nodes(data=True)
    NodeDataView({(0, 1): {'tot': 1}, (2, 3): {'tot': 5}, (1, 2): {'tot': 3}})

    Notes
    -----
    Graph, node, and edge data are not propagated to the new graph. For
    undirected graphs, the nodes in G must be sortable, otherwise the
    constructed line graph may not be correct.

    *Self-loops in undirected graphs*

    For an undirected graph `G` without multiple edges, each edge can be
    written as a set `\\{u, v\\}`.  Its line graph `L` has the edges of `G` as
    its nodes. If `x` and `y` are two nodes in `L`, then `\\{x, y\\}` is an edge
    in `L` if and only if the intersection of `x` and `y` is nonempty. Thus,
    the set of all edges is determined by the set of all pairwise intersections
    of edges in `G`.

    Trivially, every edge in G would have a nonzero intersection with itself,
    and so every node in `L` should have a self-loop. This is not so
    interesting, and the original context of line graphs was with simple
    graphs, which had no self-loops or multiple edges. The line graph was also
    meant to be a simple graph and thus, self-loops in `L` are not part of the
    standard definition of a line graph. In a pairwise intersection matrix,
    this is analogous to excluding the diagonal entries from the line graph
    definition.

    Self-loops and multiple edges in `G` add nodes to `L` in a natural way, and
    do not require any fundamental changes to the definition. It might be
    argued that the self-loops we excluded before should now be included.
    However, the self-loops are still "trivial" in some sense and thus, are
    usually excluded.

    *Self-loops in directed graphs*

    For a directed graph `G` without multiple edges, each edge can be written
    as a tuple `(u, v)`. Its line graph `L` has the edges of `G` as its
    nodes. If `x` and `y` are two nodes in `L`, then `(x, y)` is an edge in `L`
    if and only if the tail of `x` matches the head of `y`, for example, if `x
    = (a, b)` and `y = (b, c)` for some vertices `a`, `b`, and `c` in `G`.

    Due to the directed nature of the edges, it is no longer the case that
    every edge in `G` should have a self-loop in `L`. Now, the only time
    self-loops arise is if a node in `G` itself has a self-loop.  So such
    self-loops are no longer "trivial" but instead, represent essential
    features of the topology of `G`. For this reason, the historical
    development of line digraphs is such that self-loops are included. When the
    graph `G` has multiple edges, once again only superficial changes are
    required to the definition.

    References
    ----------
    * Harary, Frank, and Norman, Robert Z., "Some properties of line digraphs",
      Rend. Circ. Mat. Palermo, II. Ser. 9 (1960), 161--168.
    * Hemminger, R. L.; Beineke, L. W. (1978), "Line graphs and line digraphs",
      in Beineke, L. W.; Wilson, R. J., Selected Topics in Graph Theory,
      Academic Press Inc., pp. 271--305.

    """
    if G.is_directed():
        L = _lg_directed(G, create_using=create_using)
    else:
        L = _lg_undirected(G, selfloops=False, create_using=create_using)
    return L


def _lg_directed(G, create_using=None):
    """Returns the line graph L of the (multi)digraph G.

    Edges in G appear as nodes in L, represented as tuples of the form (u,v)
    or (u,v,key) if G is a multidigraph. A node in L corresponding to the edge
    (u,v) is connected to every node corresponding to an edge (v,w).

    Parameters
    ----------
    G : digraph
        A directed graph or directed multigraph.
    create_using : NetworkX graph constructor, optional
       Graph type to create. If graph instance, then cleared before populated.
       Default is to use the same graph class as `G`.

    """
    L = nx.empty_graph(0, create_using, default=G.__class__)

    for from_node in G:
        for to_node in G[from_node]:
            if G.is_multigraph():
                for key in G[from_node][to_node]:
                    L.add_node((from_node, to_node, key))
            else:
                L.add_node((from_node, to_node))

    for from_node in G:
        for to_node in G[from_node]:
            if G.is_multigraph():
                for key in G[from_node][to_node]:
                    for next_node in G[to_node]:
                        if G.is_multigraph():
                            for next_key in G[to_node][next_node]:
                                L.add_edge((from_node, to_node, key),
                                           (to_node, next_node, next_key))
                        else:
                            L.add_edge((from_node, to_node, key),
                                       (to_node, next_node))
            else:
                for next_node in G[to_node]:
                    if G.is_multigraph():
                        for next_key in G[to_node][next_node]:
                            L.add_edge((from_node, to_node),
                                       (to_node, next_node, next_key))
                    else:
                        L.add_edge((from_node, to_node),
                                   (to_node, next_node))
    return L


def _lg_undirected(G, selfloops=False, create_using=None):
    """Returns the line graph L of the (multi)graph G.

    Edges in G appear as nodes in L, represented as sorted tuples of the form
    (u,v), or (u,v,key) if G is a multigraph. A node in L corresponding to
    the edge {u,v} is connected to every node corresponding to an edge that
    involves u or v.

    Parameters
    ----------
    G : graph
        An undirected graph or multigraph.
    selfloops : bool
        If `True`, then self-loops are included in the line graph. If `False`,
        they are excluded.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Notes
    -----
    The standard algorithm for line graphs of undirected graphs does not
    produce self-loops.

    """
    L = nx.empty_graph(0, create_using, default=G.__class__)

    for u, v, data in G.edges(data=True):
        # Sort nodes to canonicalize
        (u, v) = sorted([u, v])
        if G.is_multigraph():
            key = data.get('key', None)
            node = (u, v, key)
        else:
            node = (u, v)
        L.add_node(node)

    for u in G:
        for v, w in combinations(G[u], 2):
            if G.is_multigraph():
                for key1 in G[u][v]:
                    for key2 in G[u][w]:
                        node1 = tuple(sorted([u, v]) + [key1])
                        node2 = tuple(sorted([u, w]) + [key2])
                        L.add_edge(node1, node2)
            else:
                node1 = tuple(sorted([u, v]))
                node2 = tuple(sorted([u, w]))
                L.add_edge(node1, node2)

    if selfloops and G.number_of_selfloops() > 0:
        for u, v, data in G.selfloop_edges(data=True):
            if G.is_multigraph():
                key = data.get('key', None)
                node = (u, u, key)
            else:
                node = (u, u)
            L.add_node(node)

    return L


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable(returns_graph=True)
def inverse_line_graph(G):
    """Returns the inverse line graph of graph G.

    If H is a graph, and G is the line graph of H, such that G = L(H).
    Then H is the inverse line graph of G.

    Not all graphs are line graphs and these do not have an inverse line graph.
    In these cases this function raises a NetworkXError.

    Parameters
    ----------
    G : graph
        A NetworkX Graph

    Returns
    -------
    H : graph
        The inverse line graph of G.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed or a multigraph

    NetworkXError
        If G is not a line graph

    Notes
    -----
    This is an implementation of the Roussopoulos algorithm[1]_.

    If G consists of multiple components, then the algorithm doesn't work.
    You should invert every component separately:

    >>> K5 = nx.complete_graph(5)
    >>> P4 = nx.Graph([("a", "b"), ("b", "c"), ("c", "d")])
    >>> G = nx.union(K5, P4)
    >>> root_graphs = []
    >>> for comp in nx.connected_components(G):
    ...     root_graphs.append(nx.inverse_line_graph(G.subgraph(comp)))
    >>> len(root_graphs)
    2

    References
    ----------
    .. [1] Roussopoulos, N.D. , "A max {m, n} algorithm for determining the graph H from
       its line graph G", Information Processing Letters 2, (1973), 108--112, ISSN 0020-0190,
       `DOI link <https://doi.org/10.1016/0020-0190(73)90029-X>`_

    """
    if len(G) == 0:
        return nx.Graph()

    starting_cell = _select_starting_cell(G)
    P = _find_partition(G, starting_cell)

    H = nx.Graph()
    for cell in P:
        if len(cell) == 1:
            H.add_node(cell[0])
        elif len(cell) == 2:
            H.add_edge(*cell)
        else:
            raise nx.NetworkXError("G is not a line graph")

    return H


def _triangles(G, e):
    """Return list of all triangles containing edge e"""
    u, v = e
    return [(u, v, w) for w in set(G[u]) & set(G[v])]


def _odd_triangle(G, T):
    """Test whether T is an odd triangle in G

    Parameters
    ----------
    G : NetworkX Graph
    T : 3-tuple of vertices forming triangle in G

    Returns
    -------
    True is T is an odd triangle
    False otherwise

    Raises
    ------
    NetworkXError
        T is not a triangle in G

    Notes
    -----
    An odd triangle is one in which there exists another vertex in G which is
    adjacent to either exactly one or exactly all three of the vertices in the
    triangle.

    """
    if not nx.is_triangle(G, T):
        raise nx.NetworkXError("T is not a triangle in G")

    for v in G:
        if v not in T:
            adj = sum(1 for u in T if v in G[u])
            if adj == 1 or adj == 3:
                return True
    return False


def _find_partition(G, starting_cell):
    """Find a partition of the vertices of G into cells of complete graphs

    Parameters
    ----------
    G : NetworkX Graph
    starting_cell : tuple of vertices in G which form a cell

    Returns
    -------
    List of tuples of vertices of G

    Raises
    ------
    NetworkXError
        If a cell is not a complete subgraph then G is not a line graph
    """
    cells = [starting_cell]
    for cell in cells:
        for v in G:
            if v not in cell:
                if all(v in G[u] for u in cell):
                    cells.append(tuple(list(cell) + [v]))

    if not all(nx.is_clique(G, cell) for cell in cells):
        raise nx.NetworkXError("G is not a line graph")

    return cells


def _select_starting_cell(G, starting_edge=None):
    """Select a cell to initiate _find_partition

    Parameters
    ----------
    G : NetworkX Graph
    starting_edge: an edge to build the starting cell from

    Returns
    -------
    Tuple of vertices in G

    Raises
    ------
    NetworkXError
        If it is determined that G is not a line graph

    Notes
    -----
    If starting edge not specified then pick an arbitrary edge - doesn't
    matter which. However, this function may call itself requiring a
    specific starting edge. Note that the r, s notation for counting
    triangles is the same as in the Roussopoulos paper cited above.
    """
    if starting_edge is None:
        e = arbitrary_element(G.edges())
    else:
        e = starting_edge

    r = len(_triangles(G, e))
    s = len(set(_triangles(G, e)))

    if r == 0:
        return e
    elif r == 1:
        triangle = _triangles(G, e)[0]
        if _odd_triangle(G, triangle):
            return triangle
        else:
            return e
    elif r == s:
        return e
    else:
        for triangle in _triangles(G, e):
            if not _odd_triangle(G, triangle):
                return _select_starting_cell(G, (triangle[0], triangle[1]))
    
    raise nx.NetworkXError("G is not a line graph")
