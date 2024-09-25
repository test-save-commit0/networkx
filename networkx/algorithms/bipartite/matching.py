"""Provides functions for computing maximum cardinality matchings and minimum
weight full matchings in a bipartite graph.

If you don't care about the particular implementation of the maximum matching
algorithm, simply use the :func:`maximum_matching`. If you do care, you can
import one of the named maximum matching algorithms directly.

For example, to find a maximum matching in the complete bipartite graph with
two vertices on the left and three vertices on the right:

>>> G = nx.complete_bipartite_graph(2, 3)
>>> left, right = nx.bipartite.sets(G)
>>> list(left)
[0, 1]
>>> list(right)
[2, 3, 4]
>>> nx.bipartite.maximum_matching(G)
{0: 2, 1: 3, 2: 0, 3: 1}

The dictionary returned by :func:`maximum_matching` includes a mapping for
vertices in both the left and right vertex sets.

Similarly, :func:`minimum_weight_full_matching` produces, for a complete
weighted bipartite graph, a matching whose cardinality is the cardinality of
the smaller of the two partitions, and for which the sum of the weights of the
edges included in the matching is minimal.

"""
import collections
import itertools
import networkx as nx
from networkx.algorithms.bipartite import sets as bipartite_sets
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
__all__ = ['maximum_matching', 'hopcroft_karp_matching',
    'eppstein_matching', 'to_vertex_cover', 'minimum_weight_full_matching']
INFINITY = float('inf')


@nx._dispatchable
def hopcroft_karp_matching(G, top_nodes=None):
    """Returns the maximum cardinality matching of the bipartite graph `G`.

    A matching is a set of edges that do not share any nodes. A maximum
    cardinality matching is a matching with the most edges possible. It
    is not always unique. Finding a matching in a bipartite graph can be
    treated as a networkx flow problem.

    The functions ``hopcroft_karp_matching`` and ``maximum_matching``
    are aliases of the same function.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    top_nodes : container of nodes

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed. But if more than one solution exists an exception
      will be raised.

    Returns
    -------
    matches : dictionary

      The matching is returned as a dictionary, `matches`, such that
      ``matches[v] == w`` if node `v` is matched to node `w`. Unmatched
      nodes do not occur as a key in `matches`.

    Raises
    ------
    AmbiguousSolution
      Raised if the input bipartite graph is disconnected and no container
      with all nodes in one bipartite set is provided. When determining
      the nodes in each bipartite set more than one valid solution is
      possible if the input graph is disconnected.

    Notes
    -----
    This function is implemented with the `Hopcroft--Karp matching algorithm
    <https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm>`_ for
    bipartite graphs.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    maximum_matching
    hopcroft_karp_matching
    eppstein_matching

    References
    ----------
    .. [1] John E. Hopcroft and Richard M. Karp. "An n^{5 / 2} Algorithm for
       Maximum Matchings in Bipartite Graphs" In: **SIAM Journal of Computing**
       2.4 (1973), pp. 225--231. <https://doi.org/10.1137/0202019>.

    """
    if top_nodes is None:
        try:
            top_nodes = bipartite_sets(G)[0]
        except nx.AmbiguousSolution:
            msg = 'Bipartite graph is disconnected, provide top_nodes explicitly.'
            raise nx.AmbiguousSolution(msg)

    # Initialize matching and expose sets
    matching = {}
    exposed_top = set(top_nodes)
    exposed_bottom = set(G) - set(top_nodes)

    while True:
        # Find an augmenting path
        path = _find_augmenting_path(G, matching, exposed_top, exposed_bottom)
        if not path:
            break

        # Augment the matching along the path
        for i in range(0, len(path) - 1, 2):
            u, v = path[i], path[i + 1]
            matching[u] = v
            matching[v] = u

        # Update exposed sets
        exposed_top -= set(path[::2])
        exposed_bottom -= set(path[1::2])

    return matching

def _find_augmenting_path(G, matching, exposed_top, exposed_bottom):
    """Find an augmenting path in the graph."""
    queue = collections.deque(exposed_top)
    parent = {v: None for v in exposed_top}
    while queue:
        u = queue.popleft()
        if u in exposed_bottom:
            # Found an augmenting path
            path = [u]
            while parent[u] is not None:
                u = parent[u]
                path.append(u)
            return path[::-1]
        for v in G[u]:
            if v not in parent:
                if v in matching:
                    parent[v] = u
                    parent[matching[v]] = v
                    queue.append(matching[v])
                else:
                    parent[v] = u
                    return [u, v]
    return None


@nx._dispatchable
def eppstein_matching(G, top_nodes=None):
    """Returns the maximum cardinality matching of the bipartite graph `G`.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    top_nodes : container

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed. But if more than one solution exists an exception
      will be raised.

    Returns
    -------
    matches : dictionary

      The matching is returned as a dictionary, `matching`, such that
      ``matching[v] == w`` if node `v` is matched to node `w`. Unmatched
      nodes do not occur as a key in `matching`.

    Raises
    ------
    AmbiguousSolution
      Raised if the input bipartite graph is disconnected and no container
      with all nodes in one bipartite set is provided. When determining
      the nodes in each bipartite set more than one valid solution is
      possible if the input graph is disconnected.

    Notes
    -----
    This function is implemented with David Eppstein's version of the algorithm
    Hopcroft--Karp algorithm (see :func:`hopcroft_karp_matching`), which
    originally appeared in the `Python Algorithms and Data Structures library
    (PADS) <http://www.ics.uci.edu/~eppstein/PADS/ABOUT-PADS.txt>`_.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------

    hopcroft_karp_matching

    """
    if top_nodes is None:
        try:
            top_nodes = bipartite_sets(G)[0]
        except nx.AmbiguousSolution:
            msg = 'Bipartite graph is disconnected, provide top_nodes explicitly.'
            raise nx.AmbiguousSolution(msg)

    # Initialize matching and free sets
    matching = {}
    free_top = set(top_nodes)
    free_bottom = set(G) - set(top_nodes)

    while True:
        # Find an augmenting path
        path = _eppstein_augmenting_path(G, matching, free_top, free_bottom)
        if not path:
            break

        # Augment the matching along the path
        for i in range(0, len(path) - 1, 2):
            u, v = path[i], path[i + 1]
            matching[u] = v
            matching[v] = u

        # Update free sets
        free_top -= set(path[::2])
        free_bottom -= set(path[1::2])

    return matching

def _eppstein_augmenting_path(G, matching, free_top, free_bottom):
    """Find an augmenting path in the graph using Eppstein's algorithm."""
    path = []
    used = set()
    
    def dfs(v):
        used.add(v)
        if v in free_bottom:
            return True
        for u in G[v]:
            if u not in used:
                path.append((v, u))
                if u in matching:
                    if dfs(matching[u]):
                        return True
                else:
                    if dfs(u):
                        return True
                path.pop()
        return False

    for v in free_top:
        if dfs(v):
            return [item for pair in path for item in pair]
    return None


def _is_connected_by_alternating_path(G, v, matched_edges, unmatched_edges,
    targets):
    """Returns True if and only if the vertex `v` is connected to one of
    the target vertices by an alternating path in `G`.

    An *alternating path* is a path in which every other edge is in the
    specified maximum matching (and the remaining edges in the path are not in
    the matching). An alternating path may have matched edges in the even
    positions or in the odd positions, as long as the edges alternate between
    'matched' and 'unmatched'.

    `G` is an undirected bipartite NetworkX graph.

    `v` is a vertex in `G`.

    `matched_edges` is a set of edges present in a maximum matching in `G`.

    `unmatched_edges` is a set of edges not present in a maximum
    matching in `G`.

    `targets` is a set of vertices.

    """
    visited = set()
    stack = [(v, True)]  # (vertex, use_matched_edge)

    while stack:
        current, use_matched = stack.pop()
        
        if current in visited:
            continue
        
        visited.add(current)
        
        if current in targets:
            return True
        
        edges_to_check = matched_edges if use_matched else unmatched_edges
        
        for neighbor in G[current]:
            edge = frozenset([current, neighbor])
            if edge in edges_to_check:
                stack.append((neighbor, not use_matched))
    
    return False


def _connected_by_alternating_paths(G, matching, targets):
    """Returns the set of vertices that are connected to one of the target
    vertices by an alternating path in `G` or are themselves a target.

    An *alternating path* is a path in which every other edge is in the
    specified maximum matching (and the remaining edges in the path are not in
    the matching). An alternating path may have matched edges in the even
    positions or in the odd positions, as long as the edges alternate between
    'matched' and 'unmatched'.

    `G` is an undirected bipartite NetworkX graph.

    `matching` is a dictionary representing a maximum matching in `G`, as
    returned by, for example, :func:`maximum_matching`.

    `targets` is a set of vertices.

    """
    matched_edges = {frozenset((v, matching[v])) for v in matching}
    unmatched_edges = {frozenset(e) for e in G.edges() if frozenset(e) not in matched_edges}
    
    connected = set(targets)
    to_explore = set(targets)
    
    while to_explore:
        v = to_explore.pop()
        for neighbor in G[v]:
            if neighbor not in connected:
                if _is_connected_by_alternating_path(G, neighbor, matched_edges, unmatched_edges, targets):
                    connected.add(neighbor)
                    to_explore.add(neighbor)
    
    return connected


@nx._dispatchable
def to_vertex_cover(G, matching, top_nodes=None):
    """Returns the minimum vertex cover corresponding to the given maximum
    matching of the bipartite graph `G`.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    matching : dictionary

      A dictionary whose keys are vertices in `G` and whose values are the
      distinct neighbors comprising the maximum matching for `G`, as returned
      by, for example, :func:`maximum_matching`. The dictionary *must*
      represent the maximum matching.

    top_nodes : container

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed. But if more than one solution exists an exception
      will be raised.

    Returns
    -------
    vertex_cover : :class:`set`

      The minimum vertex cover in `G`.

    Raises
    ------
    AmbiguousSolution
      Raised if the input bipartite graph is disconnected and no container
      with all nodes in one bipartite set is provided. When determining
      the nodes in each bipartite set more than one valid solution is
      possible if the input graph is disconnected.

    Notes
    -----
    This function is implemented using the procedure guaranteed by `Konig's
    theorem
    <https://en.wikipedia.org/wiki/K%C3%B6nig%27s_theorem_%28graph_theory%29>`_,
    which proves an equivalence between a maximum matching and a minimum vertex
    cover in bipartite graphs.

    Since a minimum vertex cover is the complement of a maximum independent set
    for any graph, one can compute the maximum independent set of a bipartite
    graph this way:

    >>> G = nx.complete_bipartite_graph(2, 3)
    >>> matching = nx.bipartite.maximum_matching(G)
    >>> vertex_cover = nx.bipartite.to_vertex_cover(G, matching)
    >>> independent_set = set(G) - vertex_cover
    >>> print(list(independent_set))
    [2, 3, 4]

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    """
    if top_nodes is None:
        try:
            top_nodes = bipartite_sets(G)[0]
        except nx.AmbiguousSolution:
            msg = 'Bipartite graph is disconnected, provide top_nodes explicitly.'
            raise nx.AmbiguousSolution(msg)

    # Initialize the vertex cover with the unmatched vertices on the right side
    vertex_cover = set(G) - set(top_nodes) - set(matching.keys())

    # Add the matched vertices on the left side
    vertex_cover.update(set(top_nodes) & set(matching.keys()))

    # Find alternating paths starting from unmatched vertices on the left side
    unmatched_vertices = set(top_nodes) - set(matching.keys())
    targets = set(G) - set(top_nodes) - set(matching.values())
    connected = _connected_by_alternating_paths(G, matching, targets)

    # Update the vertex cover
    vertex_cover.update(set(top_nodes) - connected)
    vertex_cover.update(set(G) - set(top_nodes) & connected)

    return vertex_cover


maximum_matching = hopcroft_karp_matching


@nx._dispatchable(edge_attrs='weight')
def minimum_weight_full_matching(G, top_nodes=None, weight='weight'):
    """Returns a minimum weight full matching of the bipartite graph `G`.

    Let :math:`G = ((U, V), E)` be a weighted bipartite graph with real weights
    :math:`w : E \\to \\mathbb{R}`. This function then produces a matching
    :math:`M \\subseteq E` with cardinality

    .. math::
       \\lvert M \\rvert = \\min(\\lvert U \\rvert, \\lvert V \\rvert),

    which minimizes the sum of the weights of the edges included in the
    matching, :math:`\\sum_{e \\in M} w(e)`, or raises an error if no such
    matching exists.

    When :math:`\\lvert U \\rvert = \\lvert V \\rvert`, this is commonly
    referred to as a perfect matching; here, since we allow
    :math:`\\lvert U \\rvert` and :math:`\\lvert V \\rvert` to differ, we
    follow Karp [1]_ and refer to the matching as *full*.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    top_nodes : container

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed.

    weight : string, optional (default='weight')

       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    matches : dictionary

      The matching is returned as a dictionary, `matches`, such that
      ``matches[v] == w`` if node `v` is matched to node `w`. Unmatched
      nodes do not occur as a key in `matches`.

    Raises
    ------
    ValueError
      Raised if no full matching exists.

    ImportError
      Raised if SciPy is not available.

    Notes
    -----
    The problem of determining a minimum weight full matching is also known as
    the rectangular linear assignment problem. This implementation defers the
    calculation of the assignment to SciPy.

    References
    ----------
    .. [1] Richard Manning Karp:
       An algorithm to Solve the m x n Assignment Problem in Expected Time
       O(mn log n).
       Networks, 10(2):143–152, 1980.

    """
    try:
        import scipy.optimize
    except ImportError:
        raise ImportError("minimum_weight_full_matching requires SciPy")

    if top_nodes is None:
        try:
            top_nodes = bipartite_sets(G)[0]
        except nx.AmbiguousSolution:
            msg = 'Bipartite graph is disconnected, provide top_nodes explicitly.'
            raise nx.AmbiguousSolution(msg)

    top_nodes = list(top_nodes)
    bottom_nodes = list(set(G) - set(top_nodes))

    # Create the cost matrix
    cost_matrix = biadjacency_matrix(G, row_order=top_nodes,
                                     column_order=bottom_nodes,
                                     weight=weight).toarray()

    # Pad the cost matrix if necessary
    n, m = cost_matrix.shape
    if n > m:
        cost_matrix = np.column_stack([cost_matrix, np.full((n, n - m), np.inf)])
    elif m > n:
        cost_matrix = np.row_stack([cost_matrix, np.full((m - n, m), np.inf)])

    # Solve the assignment problem
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

    # Create the matching dictionary
    matching = {}
    for r, c in zip(row_ind, col_ind):
        if r < len(top_nodes) and c < len(bottom_nodes):
            matching[top_nodes[r]] = bottom_nodes[c]
            matching[bottom_nodes[c]] = top_nodes[r]

    if len(matching) != 2 * min(len(top_nodes), len(bottom_nodes)):
        raise ValueError("No full matching exists.")

    return matching
