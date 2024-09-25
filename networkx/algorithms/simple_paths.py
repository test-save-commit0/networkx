from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from networkx.utils import not_implemented_for, pairwise
__all__ = ['all_simple_paths', 'is_simple_path', 'shortest_simple_paths',
    'all_simple_edge_paths']


@nx._dispatchable
def is_simple_path(G, nodes):
    """Returns True if and only if `nodes` form a simple path in `G`.

    A *simple path* in a graph is a nonempty sequence of nodes in which
    no node appears more than once in the sequence, and each adjacent
    pair of nodes in the sequence is adjacent in the graph.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    nodes : list
        A list of one or more nodes in the graph `G`.

    Returns
    -------
    bool
        Whether the given list of nodes represents a simple path in `G`.

    Notes
    -----
    An empty list of nodes is not a path but a list of one node is a
    path. Here's an explanation why.

    This function operates on *node paths*. One could also consider
    *edge paths*. There is a bijection between node paths and edge
    paths.

    The *length of a path* is the number of edges in the path, so a list
    of nodes of length *n* corresponds to a path of length *n* - 1.
    Thus the smallest edge path would be a list of zero edges, the empty
    path. This corresponds to a list of one node.

    To convert between a node path and an edge path, you can use code
    like the following::

        >>> from networkx.utils import pairwise
        >>> nodes = [0, 1, 2, 3]
        >>> edges = list(pairwise(nodes))
        >>> edges
        [(0, 1), (1, 2), (2, 3)]
        >>> nodes = [edges[0][0]] + [v for u, v in edges]
        >>> nodes
        [0, 1, 2, 3]

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> nx.is_simple_path(G, [2, 3, 0])
    True
    >>> nx.is_simple_path(G, [0, 2])
    False

    """
    # Check if the list of nodes is empty
    if not nodes:
        return False
    
    # Check if there are duplicate nodes
    if len(set(nodes)) != len(nodes):
        return False
    
    # Check if each adjacent pair of nodes is connected in the graph
    return all(G.has_edge(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))


@nx._dispatchable
def all_simple_paths(G, source, target, cutoff=None):
    """Generate all simple paths in the graph G from source to target.

    A simple path is a path with no repeated nodes.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : nodes
       Single node or iterable of nodes at which to end path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    path_generator: generator
       A generator that produces lists of simple paths.  If there are no paths
       between the source and target within the given cutoff the generator
       produces no output. If it is possible to traverse the same sequence of
       nodes in multiple ways, namely through parallel edges, then it will be
       returned multiple times (once for each viable edge combination).

    Examples
    --------
    This iterator generates lists of nodes::

        >>> G = nx.complete_graph(4)
        >>> for path in nx.all_simple_paths(G, source=0, target=3):
        ...     print(path)
        ...
        [0, 1, 2, 3]
        [0, 1, 3]
        [0, 2, 1, 3]
        [0, 2, 3]
        [0, 3]

    You can generate only those paths that are shorter than a certain
    length by using the `cutoff` keyword argument::

        >>> paths = nx.all_simple_paths(G, source=0, target=3, cutoff=2)
        >>> print(list(paths))
        [[0, 1, 3], [0, 2, 3], [0, 3]]

    To get each path as the corresponding list of edges, you can use the
    :func:`networkx.utils.pairwise` helper function::

        >>> paths = nx.all_simple_paths(G, source=0, target=3)
        >>> for path in map(nx.utils.pairwise, paths):
        ...     print(list(path))
        [(0, 1), (1, 2), (2, 3)]
        [(0, 1), (1, 3)]
        [(0, 2), (2, 1), (1, 3)]
        [(0, 2), (2, 3)]
        [(0, 3)]

    Pass an iterable of nodes as target to generate all paths ending in any of several nodes::

        >>> G = nx.complete_graph(4)
        >>> for path in nx.all_simple_paths(G, source=0, target=[3, 2]):
        ...     print(path)
        ...
        [0, 1, 2]
        [0, 1, 2, 3]
        [0, 1, 3]
        [0, 1, 3, 2]
        [0, 2]
        [0, 2, 1, 3]
        [0, 2, 3]
        [0, 3]
        [0, 3, 1, 2]
        [0, 3, 2]

    The singleton path from ``source`` to itself is considered a simple path and is
    included in the results:

        >>> G = nx.empty_graph(5)
        >>> list(nx.all_simple_paths(G, source=0, target=0))
        [[0]]

        >>> G = nx.path_graph(3)
        >>> list(nx.all_simple_paths(G, source=0, target={0, 1, 2}))
        [[0], [0, 1], [0, 1, 2]]

    Iterate over each path from the root nodes to the leaf nodes in a
    directed acyclic graph using a functional programming approach::

        >>> from itertools import chain
        >>> from itertools import product
        >>> from itertools import starmap
        >>> from functools import partial
        >>>
        >>> chaini = chain.from_iterable
        >>>
        >>> G = nx.DiGraph([(0, 1), (1, 2), (0, 3), (3, 2)])
        >>> roots = (v for v, d in G.in_degree() if d == 0)
        >>> leaves = (v for v, d in G.out_degree() if d == 0)
        >>> all_paths = partial(nx.all_simple_paths, G)
        >>> list(chaini(starmap(all_paths, product(roots, leaves))))
        [[0, 1, 2], [0, 3, 2]]

    The same list computed using an iterative approach::

        >>> G = nx.DiGraph([(0, 1), (1, 2), (0, 3), (3, 2)])
        >>> roots = (v for v, d in G.in_degree() if d == 0)
        >>> leaves = (v for v, d in G.out_degree() if d == 0)
        >>> all_paths = []
        >>> for root in roots:
        ...     for leaf in leaves:
        ...         paths = nx.all_simple_paths(G, root, leaf)
        ...         all_paths.extend(paths)
        >>> all_paths
        [[0, 1, 2], [0, 3, 2]]

    Iterate over each path from the root nodes to the leaf nodes in a
    directed acyclic graph passing all leaves together to avoid unnecessary
    compute::

        >>> G = nx.DiGraph([(0, 1), (2, 1), (1, 3), (1, 4)])
        >>> roots = (v for v, d in G.in_degree() if d == 0)
        >>> leaves = [v for v, d in G.out_degree() if d == 0]
        >>> all_paths = []
        >>> for root in roots:
        ...     paths = nx.all_simple_paths(G, root, leaves)
        ...     all_paths.extend(paths)
        >>> all_paths
        [[0, 1, 3], [0, 1, 4], [2, 1, 3], [2, 1, 4]]

    If parallel edges offer multiple ways to traverse a given sequence of
    nodes, this sequence of nodes will be returned multiple times:

        >>> G = nx.MultiDiGraph([(0, 1), (0, 1), (1, 2)])
        >>> list(nx.all_simple_paths(G, 0, 2))
        [[0, 1, 2], [0, 1, 2]]

    Notes
    -----
    This algorithm uses a modified depth-first search to generate the
    paths [1]_.  A single path can be found in $O(V+E)$ time but the
    number of simple paths in a graph can be very large, e.g. $O(n!)$ in
    the complete graph of order $n$.

    This function does not check that a path exists between `source` and
    `target`. For large graphs, this may result in very long runtimes.
    Consider using `has_path` to check that a path exists between `source` and
    `target` before calling this function on large graphs.

    References
    ----------
    .. [1] R. Sedgewick, "Algorithms in C, Part 5: Graph Algorithms",
       Addison Wesley Professional, 3rd ed., 2001.

    See Also
    --------
    all_shortest_paths, shortest_path, has_path

    """
    def _all_simple_paths_graph(G, source, target, cutoff=None):
        if source not in G:
            raise nx.NodeNotFound(f"source node {source} not in graph")
        if target in G:
            targets = {target}
        else:
            try:
                targets = set(target)
            except TypeError as e:
                raise nx.NodeNotFound(f"target node {target} not in graph") from e
        if not targets:
            raise nx.NodeNotFound("target is empty")
        if cutoff is None:
            cutoff = len(G) - 1
        if cutoff < 1:
            return
        if source in targets:
            yield [source]
        visited = [source]
        stack = [iter(G[source])]
        while stack:
            children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                visited.pop()
            elif len(visited) < cutoff:
                if child in visited:
                    continue
                if child in targets:
                    yield visited + [child]
                visited.append(child)
                if targets - set(visited):  # expand stack until find all targets
                    stack.append(iter(G[child]))
                else:
                    visited.pop()  # maybe other ways to child
            else:  # len(visited) == cutoff:
                for target in targets - set(visited):
                    if target in children:
                        yield visited + [target]
                stack.pop()
                visited.pop()

    return _all_simple_paths_graph(G, source, target, cutoff)


@nx._dispatchable
def all_simple_edge_paths(G, source, target, cutoff=None):
    """Generate lists of edges for all simple paths in G from source to target.

    A simple path is a path with no repeated nodes.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : nodes
       Single node or iterable of nodes at which to end path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    path_generator: generator
       A generator that produces lists of simple paths.  If there are no paths
       between the source and target within the given cutoff the generator
       produces no output.
       For multigraphs, the list of edges have elements of the form `(u,v,k)`.
       Where `k` corresponds to the edge key.

    Examples
    --------

    Print the simple path edges of a Graph::

        >>> g = nx.Graph([(1, 2), (2, 4), (1, 3), (3, 4)])
        >>> for path in sorted(nx.all_simple_edge_paths(g, 1, 4)):
        ...     print(path)
        [(1, 2), (2, 4)]
        [(1, 3), (3, 4)]

    Print the simple path edges of a MultiGraph. Returned edges come with
    their associated keys::

        >>> mg = nx.MultiGraph()
        >>> mg.add_edge(1, 2, key="k0")
        'k0'
        >>> mg.add_edge(1, 2, key="k1")
        'k1'
        >>> mg.add_edge(2, 3, key="k0")
        'k0'
        >>> for path in sorted(nx.all_simple_edge_paths(mg, 1, 3)):
        ...     print(path)
        [(1, 2, 'k0'), (2, 3, 'k0')]
        [(1, 2, 'k1'), (2, 3, 'k0')]

    When ``source`` is one of the targets, the empty path starting and ending at
    ``source`` without traversing any edge is considered a valid simple edge path
    and is included in the results:

        >>> G = nx.Graph()
        >>> G.add_node(0)
        >>> paths = list(nx.all_simple_edge_paths(G, 0, 0))
        >>> for path in paths:
        ...     print(path)
        []
        >>> len(paths)
        1


    Notes
    -----
    This algorithm uses a modified depth-first search to generate the
    paths [1]_.  A single path can be found in $O(V+E)$ time but the
    number of simple paths in a graph can be very large, e.g. $O(n!)$ in
    the complete graph of order $n$.

    References
    ----------
    .. [1] R. Sedgewick, "Algorithms in C, Part 5: Graph Algorithms",
       Addison Wesley Professional, 3rd ed., 2001.

    See Also
    --------
    all_shortest_paths, shortest_path, all_simple_paths

    """
    def edge_key(u, v):
        return u, v, 0 if G.is_directed() else min(u, v)

    def join_edges(path):
        return list(pairwise(path))

    def join_multigraph_edges(path):
        return [(u, v, min(G[u][v], key=lambda k: G[u][v][k]))
                for u, v in pairwise(path)]

    if G.is_multigraph():
        join = join_multigraph_edges
    else:
        join = join_edges

    if source not in G:
        raise nx.NodeNotFound(f"source node {source} not in graph")
    if target in G:
        targets = {target}
    else:
        try:
            targets = set(target)
        except TypeError as e:
            raise nx.NodeNotFound(f"target node {target} not in graph") from e
    if not targets:
        raise nx.NodeNotFound("target is empty")
    if cutoff is None:
        cutoff = len(G) - 1
    if cutoff < 1:
        return
    if source in targets:
        yield []
    visited = [source]
    stack = [iter(G[source])]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.pop()
        elif len(visited) < cutoff:
            if child not in visited:
                if child in targets:
                    yield join(visited + [child])
                visited.append(child)
                if targets - set(visited):  # expand stack until find all targets
                    stack.append(iter(G[child]))
                else:
                    visited.pop()  # maybe other ways to child
        elif len(visited) == cutoff:
            for target in targets - set(visited):
                if target in children:
                    yield join(visited + [target])
            stack.pop()
            visited.pop()


@not_implemented_for('multigraph')
@nx._dispatchable(edge_attrs='weight')
def shortest_simple_paths(G, source,target, weight=None):
    """Generate all simple paths in the graph G from source to target,
       starting from shortest ones.

    A simple path is a path with no repeated nodes.

    If a weighted shortest path search is to be used, no negative weights
    are allowed.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    weight : string or function
        If it is a string, it is the name of the edge attribute to be
        used as a weight.

        If it is a function, the weight of an edge is the value returned
        by the function. The function must accept exactly three positional
        arguments: the two endpoints of an edge and the dictionary of edge
        attributes for that edge. The function must return a number.

        If None all edges are considered to have unit weight. Default
        value None.

    Returns
    -------
    path_generator: generator
       A generator that produces lists of simple paths, in order from
       shortest to longest.

    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.

    NetworkXError
       If source or target nodes are not in the input graph.

    NetworkXNotImplemented
       If the input graph is a Multi[Di]Graph.

    Examples
    --------

    >>> G = nx.cycle_graph(7)
    >>> paths = list(nx.shortest_simple_paths(G, 0, 3))
    >>> print(paths)
    [[0, 1, 2, 3], [0, 6, 5, 4, 3]]

    You can use this function to efficiently compute the k shortest/best
    paths between two nodes.

    >>> from itertools import islice
    >>> def k_shortest_paths(G, source, target, k, weight=None):
    ...     return list(
    ...         islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    ...     )
    >>> for path in k_shortest_paths(G, 0, 3, 2):
    ...     print(path)
    [0, 1, 2, 3]
    [0, 6, 5, 4, 3]

    Notes
    -----
    This procedure is based on algorithm by Jin Y. Yen [1]_.  Finding
    the first $K$ paths requires $O(KN^3)$ operations.

    See Also
    --------
    all_shortest_paths
    shortest_path
    all_simple_paths

    References
    ----------
    .. [1] Jin Y. Yen, "Finding the K Shortest Loopless Paths in a
       Network", Management Science, Vol. 17, No. 11, Theory Series
       (Jul., 1971), pp. 712-716.

    """
    if source not in G:
        raise nx.NodeNotFound(f"source node {source} not in graph")
    if target not in G:
        raise nx.NodeNotFound(f"target node {target} not in graph")
    if source == target:
        return [[source]]
    if G.is_multigraph():
        raise nx.NetworkXNotImplemented("MultiGraph and MultiDiGraph not supported")

    if weight is None:
        def length_func(path):
            return len(path) - 1
        shortest_path_func = _bidirectional_shortest_path
    else:
        def length_func(path):
            return sum(G[u][v].get(weight, 1) for (u, v) in zip(path, path[1:]))
        shortest_path_func = _bidirectional_dijkstra

    listA = list()
    listB = PathBuffer()
    prev_path = None
    while True:
        if not prev_path:
            length, path = shortest_path_func(G, source, target, weight=weight)
            listB.push(length, path)
        else:
            ignore_nodes = set()
            ignore_edges = set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = length_func(root)
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i-1], path[i]))
                ignore_nodes.add(root[-1])
                try:
                    length, spur = shortest_path_func(G, root[-1], target,
                                                      ignore_nodes=ignore_nodes,
                                                      ignore_edges=ignore_edges,
                                                      weight=weight)
                    path = root[:-1] + spur
                    listB.push(root_length + length, path)
                except nx.NetworkXNoPath:
                    pass

        if listB:
            path = listB.pop()
            yield path
            listA.append(path)
            prev_path = path
        else:
            break


class PathBuffer:

    def __init__(self):
        self.paths = set()
        self.sortedpaths = []
        self.counter = count()

    def __len__(self):
        return len(self.sortedpaths)


def _bidirectional_shortest_path(G, source, target, ignore_nodes=None,
    ignore_edges=None, weight=None):
    """Returns the shortest path between source and target ignoring
       nodes and edges in the containers ignore_nodes and ignore_edges.

    This is a custom modification of the standard bidirectional shortest
    path implementation at networkx.algorithms.unweighted

    Parameters
    ----------
    G : NetworkX graph

    source : node
       starting node for path

    target : node
       ending node for path

    ignore_nodes : container of nodes
       nodes to ignore, optional

    ignore_edges : container of edges
       edges to ignore, optional

    weight : None
       This function accepts a weight argument for convenience of
       shortest_simple_paths function. It will be ignored.

    Returns
    -------
    path: list
       List of nodes in a path from source to target.

    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.

    See Also
    --------
    shortest_path

    """
    pass


def _bidirectional_pred_succ(G, source, target, ignore_nodes=None,
    ignore_edges=None):
    """Bidirectional shortest path helper.
    Returns (pred,succ,w) where
    pred is a dictionary of predecessors from w to the source, and
    succ is a dictionary of successors from w to the target.
    """
    pass


def _bidirectional_dijkstra(G, source, target, weight='weight',
    ignore_nodes=None, ignore_edges=None):
    """Dijkstra's algorithm for shortest paths using bidirectional search.

    This function returns the shortest path between source and target
    ignoring nodes and edges in the containers ignore_nodes and
    ignore_edges.

    This is a custom modification of the standard Dijkstra bidirectional
    shortest path implementation at networkx.algorithms.weighted

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node.

    target : node
       Ending node.

    weight: string, function, optional (default='weight')
       Edge data key or weight function corresponding to the edge weight

    ignore_nodes : container of nodes
       nodes to ignore, optional

    ignore_edges : container of edges
       edges to ignore, optional

    Returns
    -------
    length : number
        Shortest path length.

    Returns a tuple of two dictionaries keyed by node.
    The first dictionary stores distance from the source.
    The second stores the path from the source to that node.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    In practice  bidirectional Dijkstra is much more than twice as fast as
    ordinary Dijkstra.

    Ordinary Dijkstra expands nodes in a sphere-like manner from the
    source. The radius of this sphere will eventually be the length
    of the shortest path. Bidirectional Dijkstra will expand nodes
    from both the source and the target, making two spheres of half
    this radius. Volume of the first sphere is pi*r*r while the
    others are 2*pi*r/2*r/2, making up half the volume.

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    shortest_path
    shortest_path_length
    """
    pass
