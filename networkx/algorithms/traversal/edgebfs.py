"""
=============================
Breadth First Search on Edges
=============================

Algorithms for a breadth-first traversal of edges in a graph.

"""
from collections import deque
import networkx as nx
FORWARD = 'forward'
REVERSE = 'reverse'
__all__ = ['edge_bfs']


@nx._dispatchable
def edge_bfs(G, source=None, orientation=None):
    """A directed, breadth-first-search of edges in `G`, beginning at `source`.

    Yield the edges of G in a breadth-first-search order continuing until
    all edges are generated.

    Parameters
    ----------
    G : graph
        A directed/undirected graph/multigraph.

    source : node, list of nodes
        The node from which the traversal begins. If None, then a source
        is chosen arbitrarily and repeatedly until all edges from each node in
        the graph are searched.

    orientation : None | 'original' | 'reverse' | 'ignore' (default: None)
        For directed graphs and directed multigraphs, edge traversals need not
        respect the original orientation of the edges.
        When set to 'reverse' every edge is traversed in the reverse direction.
        When set to 'ignore', every edge is treated as undirected.
        When set to 'original', every edge is treated as directed.
        In all three cases, the yielded edge tuples add a last entry to
        indicate the direction in which that edge was traversed.
        If orientation is None, the yielded edge has no direction indicated.
        The direction is respected, but not reported.

    Yields
    ------
    edge : directed edge
        A directed edge indicating the path taken by the breadth-first-search.
        For graphs, `edge` is of the form `(u, v)` where `u` and `v`
        are the tail and head of the edge as determined by the traversal.
        For multigraphs, `edge` is of the form `(u, v, key)`, where `key` is
        the key of the edge. When the graph is directed, then `u` and `v`
        are always in the order of the actual directed edge.
        If orientation is not None then the edge tuple is extended to include
        the direction of traversal ('forward' or 'reverse') on that edge.

    Examples
    --------
    >>> nodes = [0, 1, 2, 3]
    >>> edges = [(0, 1), (1, 0), (1, 0), (2, 0), (2, 1), (3, 1)]

    >>> list(nx.edge_bfs(nx.Graph(edges), nodes))
    [(0, 1), (0, 2), (1, 2), (1, 3)]

    >>> list(nx.edge_bfs(nx.DiGraph(edges), nodes))
    [(0, 1), (1, 0), (2, 0), (2, 1), (3, 1)]

    >>> list(nx.edge_bfs(nx.MultiGraph(edges), nodes))
    [(0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (1, 2, 0), (1, 3, 0)]

    >>> list(nx.edge_bfs(nx.MultiDiGraph(edges), nodes))
    [(0, 1, 0), (1, 0, 0), (1, 0, 1), (2, 0, 0), (2, 1, 0), (3, 1, 0)]

    >>> list(nx.edge_bfs(nx.DiGraph(edges), nodes, orientation="ignore"))
    [(0, 1, 'forward'), (1, 0, 'reverse'), (2, 0, 'reverse'), (2, 1, 'reverse'), (3, 1, 'reverse')]

    >>> list(nx.edge_bfs(nx.MultiDiGraph(edges), nodes, orientation="ignore"))
    [(0, 1, 0, 'forward'), (1, 0, 0, 'reverse'), (1, 0, 1, 'reverse'), (2, 0, 0, 'reverse'), (2, 1, 0, 'reverse'), (3, 1, 0, 'reverse')]

    Notes
    -----
    The goal of this function is to visit edges. It differs from the more
    familiar breadth-first-search of nodes, as provided by
    :func:`networkx.algorithms.traversal.breadth_first_search.bfs_edges`, in
    that it does not stop once every node has been visited. In a directed graph
    with edges [(0, 1), (1, 2), (2, 1)], the edge (2, 1) would not be visited
    if not for the functionality provided by this function.

    The naming of this function is very similar to bfs_edges. The difference
    is that 'edge_bfs' yields edges even if they extend back to an already
    explored node while 'bfs_edges' yields the edges of the tree that results
    from a breadth-first-search (BFS) so no edges are reported if they extend
    to already explored nodes. That means 'edge_bfs' reports all edges while
    'bfs_edges' only report those traversed by a node-based BFS. Yet another
    description is that 'bfs_edges' reports the edges traversed during BFS
    while 'edge_bfs' reports all edges in the order they are explored.

    See Also
    --------
    bfs_edges
    bfs_tree
    edge_dfs

    """
    pass
