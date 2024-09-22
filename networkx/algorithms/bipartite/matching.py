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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass
