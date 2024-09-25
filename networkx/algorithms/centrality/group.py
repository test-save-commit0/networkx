"""Group centrality measures."""
from copy import deepcopy
import networkx as nx
from networkx.algorithms.centrality.betweenness import _accumulate_endpoints, _single_source_dijkstra_path_basic, _single_source_shortest_path_basic
from networkx.utils.decorators import not_implemented_for
__all__ = ['group_betweenness_centrality', 'group_closeness_centrality',
    'group_degree_centrality', 'group_in_degree_centrality',
    'group_out_degree_centrality', 'prominent_group']


@nx._dispatchable(edge_attrs='weight')
def group_betweenness_centrality(G, C, normalized=True, weight=None,
    endpoints=False):
    """Compute the group betweenness centrality for a group of nodes.

    Group betweenness centrality of a group of nodes $C$ is the sum of the
    fraction of all-pairs shortest paths that pass through any vertex in $C$

    .. math::

       c_B(v) =\\sum_{s,t \\in V} \\frac{\\sigma(s, t|v)}{\\sigma(s, t)}

    where $V$ is the set of nodes, $\\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths, and $\\sigma(s, t|C)$ is the number of
    those paths passing through some node in group $C$. Note that
    $(s, t)$ are not members of the group ($V-C$ is the set of nodes
    in $V$ that are not in $C$).

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    C : list or set or list of lists or list of sets
      A group or a list of groups containing nodes which belong to G, for which group betweenness
      centrality is to be calculated.

    normalized : bool, optional (default=True)
      If True, group betweenness is normalized by `1/((|V|-|C|)(|V|-|C|-1))`
      where `|V|` is the number of nodes in G and `|C|` is the number of nodes in C.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      The weight of an edge is treated as the length or distance between the two sides.

    endpoints : bool, optional (default=False)
      If True include the endpoints in the shortest path counts.

    Raises
    ------
    NodeNotFound
       If node(s) in C are not present in G.

    Returns
    -------
    betweenness : list of floats or float
       If C is a single group then return a float. If C is a list with
       several groups then return a list of group betweenness centralities.

    See Also
    --------
    betweenness_centrality

    Notes
    -----
    Group betweenness centrality is described in [1]_ and its importance discussed in [3]_.
    The initial implementation of the algorithm is mentioned in [2]_. This function uses
    an improved algorithm presented in [4]_.

    The number of nodes in the group must be a maximum of n - 2 where `n`
    is the total number of nodes in the graph.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    The total number of paths between source and target is counted
    differently for directed and undirected graphs. Directed paths
    between "u" and "v" are counted as two possible paths (one each
    direction) while undirected paths between "u" and "v" are counted
    as one path. Said another way, the sum in the expression above is
    over all ``s != t`` for directed graphs and for ``s < t`` for undirected graphs.


    References
    ----------
    .. [1] M G Everett and S P Borgatti:
       The Centrality of Groups and Classes.
       Journal of Mathematical Sociology. 23(3): 181-201. 1999.
       http://www.analytictech.com/borgatti/group_centrality.htm
    .. [2] Ulrik Brandes:
       On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.72.9610&rep=rep1&type=pdf
    .. [3] Sourav Medya et. al.:
       Group Centrality Maximization via Network Design.
       SIAM International Conference on Data Mining, SDM 2018, 126–134.
       https://sites.cs.ucsb.edu/~arlei/pubs/sdm18.pdf
    .. [4] Rami Puzis, Yuval Elovici, and Shlomi Dolev.
       "Fast algorithm for successive computation of group betweenness centrality."
       https://journals.aps.org/pre/pdf/10.1103/PhysRevE.76.056709

    """
    from itertools import chain

    # Handle single group and multiple groups
    if isinstance(C[0], (list, set)):
        groups = [set(c) for c in C]
    else:
        groups = [set(C)]

    # Check if all nodes in C are present in G
    for group in groups:
        if not set(group).issubset(G):
            raise nx.NodeNotFound("Node in C not present in G")

    V = set(G)
    betweenness = []

    for group in groups:
        C = set(group)
        if len(C) > len(V) - 2:
            raise nx.NetworkXError("The group C must contain at most n-2 nodes")

        # Initialize betweenness
        bc = {v: 0 for v in V}

        # Compute shortest paths
        if weight is None:
            for s in V - C:
                # Use BFS for unweighted graphs
                S, P, sigma = _single_source_shortest_path_basic(G, s)
                betweenness = _accumulate_endpoints(betweenness, S, P, sigma, s, C, endpoints)
        else:
            for s in V - C:
                # Use Dijkstra for weighted graphs
                S, P, sigma = _single_source_dijkstra_path_basic(G, s, weight)
                betweenness = _accumulate_endpoints(betweenness, S, P, sigma, s, C, endpoints)

        # Normalize
        bc = sum(bc.values())
        if normalized:
            n = len(V)
            bc /= ((n - len(C)) * (n - len(C) - 1))

        betweenness.append(bc)

    return betweenness[0] if len(betweenness) == 1 else betweenness


@nx._dispatchable(edge_attrs='weight')
def prominent_group(G, k, weight=None, C=None, endpoints=False, normalized=
    True, greedy=False):
    """Find the prominent group of size $k$ in graph $G$. The prominence of the
    group is evaluated by the group betweenness centrality.

    Group betweenness centrality of a group of nodes $C$ is the sum of the
    fraction of all-pairs shortest paths that pass through any vertex in $C$

    .. math::

       c_B(v) =\\sum_{s,t \\in V} \\frac{\\sigma(s, t|v)}{\\sigma(s, t)}

    where $V$ is the set of nodes, $\\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths, and $\\sigma(s, t|C)$ is the number of
    those paths passing through some node in group $C$. Note that
    $(s, t)$ are not members of the group ($V-C$ is the set of nodes
    in $V$ that are not in $C$).

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    k : int
       The number of nodes in the group.

    normalized : bool, optional (default=True)
       If True, group betweenness is normalized by ``1/((|V|-|C|)(|V|-|C|-1))``
       where ``|V|`` is the number of nodes in G and ``|C|`` is the number of
       nodes in C.

    weight : None or string, optional (default=None)
       If None, all edge weights are considered equal.
       Otherwise holds the name of the edge attribute used as weight.
       The weight of an edge is treated as the length or distance between the two sides.

    endpoints : bool, optional (default=False)
       If True include the endpoints in the shortest path counts.

    C : list or set, optional (default=None)
       list of nodes which won't be candidates of the prominent group.

    greedy : bool, optional (default=False)
       Using a naive greedy algorithm in order to find non-optimal prominent
       group. For scale free networks the results are negligibly below the optimal
       results.

    Raises
    ------
    NodeNotFound
       If node(s) in C are not present in G.

    Returns
    -------
    max_GBC : float
       The group betweenness centrality of the prominent group.

    max_group : list
        The list of nodes in the prominent group.

    See Also
    --------
    betweenness_centrality, group_betweenness_centrality

    Notes
    -----
    Group betweenness centrality is described in [1]_ and its importance discussed in [3]_.
    The algorithm is described in [2]_ and is based on techniques mentioned in [4]_.

    The number of nodes in the group must be a maximum of ``n - 2`` where ``n``
    is the total number of nodes in the graph.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    The total number of paths between source and target is counted
    differently for directed and undirected graphs. Directed paths
    between "u" and "v" are counted as two possible paths (one each
    direction) while undirected paths between "u" and "v" are counted
    as one path. Said another way, the sum in the expression above is
    over all ``s != t`` for directed graphs and for ``s < t`` for undirected graphs.

    References
    ----------
    .. [1] M G Everett and S P Borgatti:
       The Centrality of Groups and Classes.
       Journal of Mathematical Sociology. 23(3): 181-201. 1999.
       http://www.analytictech.com/borgatti/group_centrality.htm
    .. [2] Rami Puzis, Yuval Elovici, and Shlomi Dolev:
       "Finding the Most Prominent Group in Complex Networks"
       AI communications 20(4): 287-296, 2007.
       https://www.researchgate.net/profile/Rami_Puzis2/publication/220308855
    .. [3] Sourav Medya et. al.:
       Group Centrality Maximization via Network Design.
       SIAM International Conference on Data Mining, SDM 2018, 126–134.
       https://sites.cs.ucsb.edu/~arlei/pubs/sdm18.pdf
    .. [4] Rami Puzis, Yuval Elovici, and Shlomi Dolev.
       "Fast algorithm for successive computation of group betweenness centrality."
       https://journals.aps.org/pre/pdf/10.1103/PhysRevE.76.056709
    """
    import itertools

    if C is None:
        C = set()
    else:
        C = set(C)

    if not set(C).issubset(G):
        raise nx.NodeNotFound("Node in C not present in G")

    V = set(G) - C
    if k > len(V) - 2:
        raise nx.NetworkXError("k must be at most n-2, where n is the number of nodes in G")

    if greedy:
        max_group = []
        remaining_nodes = list(V)
        for _ in range(k):
            max_node = max(remaining_nodes, key=lambda x: group_betweenness_centrality(G, max_group + [x], normalized=normalized, weight=weight, endpoints=endpoints)[0])
            max_group.append(max_node)
            remaining_nodes.remove(max_node)
        max_GBC = group_betweenness_centrality(G, max_group, normalized=normalized, weight=weight, endpoints=endpoints)[0]
    else:
        max_GBC = 0
        max_group = []
        for group in itertools.combinations(V, k):
            GBC = group_betweenness_centrality(G, group, normalized=normalized, weight=weight, endpoints=endpoints)[0]
            if GBC > max_GBC:
                max_GBC = GBC
                max_group = list(group)

    return max_GBC, max_group


@nx._dispatchable(edge_attrs='weight')
def group_closeness_centrality(G, S, weight=None):
    """Compute the group closeness centrality for a group of nodes.

    Group closeness centrality of a group of nodes $S$ is a measure
    of how close the group is to the other nodes in the graph.

    .. math::

       c_{close}(S) = \\frac{|V-S|}{\\sum_{v \\in V-S} d_{S, v}}

       d_{S, v} = min_{u \\in S} (d_{u, v})

    where $V$ is the set of nodes, $d_{S, v}$ is the distance of
    the group $S$ from $v$ defined as above. ($V-S$ is the set of nodes
    in $V$ that are not in $S$).

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    S : list or set
       S is a group of nodes which belong to G, for which group closeness
       centrality is to be calculated.

    weight : None or string, optional (default=None)
       If None, all edge weights are considered equal.
       Otherwise holds the name of the edge attribute used as weight.
       The weight of an edge is treated as the length or distance between the two sides.

    Raises
    ------
    NodeNotFound
       If node(s) in S are not present in G.

    Returns
    -------
    closeness : float
       Group closeness centrality of the group S.

    See Also
    --------
    closeness_centrality

    Notes
    -----
    The measure was introduced in [1]_.
    The formula implemented here is described in [2]_.

    Higher values of closeness indicate greater centrality.

    It is assumed that 1 / 0 is 0 (required in the case of directed graphs,
    or when a shortest path length is 0).

    The number of nodes in the group must be a maximum of n - 1 where `n`
    is the total number of nodes in the graph.

    For directed graphs, the incoming distance is utilized here. To use the
    outward distance, act on `G.reverse()`.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    References
    ----------
    .. [1] M G Everett and S P Borgatti:
       The Centrality of Groups and Classes.
       Journal of Mathematical Sociology. 23(3): 181-201. 1999.
       http://www.analytictech.com/borgatti/group_centrality.htm
    .. [2] J. Zhao et. al.:
       Measuring and Maximizing Group Closeness Centrality over
       Disk Resident Graphs.
       WWWConference Proceedings, 2014. 689-694.
       https://doi.org/10.1145/2567948.2579356
    """
    import numpy as np

    if not set(S).issubset(G):
        raise nx.NodeNotFound("Node in S not present in G")

    n = len(G)
    if len(S) > n - 1:
        raise nx.NetworkXError("S must contain at most n-1 nodes")

    S = set(S)
    V = set(G) - S

    if len(V) == 0:
        return 0.0

    if weight is None:
        path_length = nx.single_source_shortest_path_length
    else:
        path_length = nx.single_source_dijkstra_path_length

    # Calculate distances from each node in S to all other nodes
    distances = {}
    for s in S:
        distances[s] = path_length(G, s, weight=weight)

    # Calculate d_S,v for each v in V-S
    d_S_v = {}
    for v in V:
        d_S_v[v] = min(distances[s].get(v, np.inf) for s in S)

    # Calculate the sum of distances
    sum_distances = sum(d_S_v.values())

    # Avoid division by zero
    if sum_distances == 0:
        return 0.0

    return len(V) / sum_distances


@nx._dispatchable
def group_degree_centrality(G, S):
    """Compute the group degree centrality for a group of nodes.

    Group degree centrality of a group of nodes $S$ is the fraction
    of non-group members connected to group members.

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    S : list or set
       S is a group of nodes which belong to G, for which group degree
       centrality is to be calculated.

    Raises
    ------
    NetworkXError
       If node(s) in S are not in G.

    Returns
    -------
    centrality : float
       Group degree centrality of the group S.

    See Also
    --------
    degree_centrality
    group_in_degree_centrality
    group_out_degree_centrality

    Notes
    -----
    The measure was introduced in [1]_.

    The number of nodes in the group must be a maximum of n - 1 where `n`
    is the total number of nodes in the graph.

    References
    ----------
    .. [1] M G Everett and S P Borgatti:
       The Centrality of Groups and Classes.
       Journal of Mathematical Sociology. 23(3): 181-201. 1999.
       http://www.analytictech.com/borgatti/group_centrality.htm
    """
    if not set(S).issubset(G):
        raise nx.NetworkXError("Node in S not present in G")

    S = set(S)
    V = set(G) - S

    if len(S) == len(G):
        return 0.0

    # Count the number of nodes in V connected to any node in S
    connected_nodes = set()
    for s in S:
        connected_nodes.update(G.neighbors(s))
    connected_nodes -= S

    return len(connected_nodes) / len(V)


@not_implemented_for('undirected')
@nx._dispatchable
def group_in_degree_centrality(G, S):
    """Compute the group in-degree centrality for a group of nodes.

    Group in-degree centrality of a group of nodes $S$ is the fraction
    of non-group members connected to group members by incoming edges.

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    S : list or set
       S is a group of nodes which belong to G, for which group in-degree
       centrality is to be calculated.

    Returns
    -------
    centrality : float
       Group in-degree centrality of the group S.

    Raises
    ------
    NetworkXNotImplemented
       If G is undirected.

    NodeNotFound
       If node(s) in S are not in G.

    See Also
    --------
    degree_centrality
    group_degree_centrality
    group_out_degree_centrality

    Notes
    -----
    The number of nodes in the group must be a maximum of n - 1 where `n`
    is the total number of nodes in the graph.

    `G.neighbors(i)` gives nodes with an outward edge from i, in a DiGraph,
    so for group in-degree centrality, the reverse graph is used.
    """
    if not G.is_directed():
        raise nx.NetworkXNotImplemented("Not defined for undirected graphs.")

    if not set(S).issubset(G):
        raise nx.NodeNotFound("Node in S not present in G")

    S = set(S)
    V = set(G) - S

    if len(S) == len(G):
        return 0.0

    # Use the reverse graph for in-degree
    G_reverse = G.reverse()

    # Count the number of nodes in V connected to any node in S by incoming edges
    connected_nodes = set()
    for s in S:
        connected_nodes.update(G_reverse.neighbors(s))
    connected_nodes -= S

    return len(connected_nodes) / len(V)


@not_implemented_for('undirected')
@nx._dispatchable
def group_out_degree_centrality(G, S):
    """Compute the group out-degree centrality for a group of nodes.

    Group out-degree centrality of a group of nodes $S$ is the fraction
    of non-group members connected to group members by outgoing edges.

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    S : list or set
       S is a group of nodes which belong to G, for which group in-degree
       centrality is to be calculated.

    Returns
    -------
    centrality : float
       Group out-degree centrality of the group S.

    Raises
    ------
    NetworkXNotImplemented
       If G is undirected.

    NodeNotFound
       If node(s) in S are not in G.

    See Also
    --------
    degree_centrality
    group_degree_centrality
    group_in_degree_centrality

    Notes
    -----
    The number of nodes in the group must be a maximum of n - 1 where `n`
    is the total number of nodes in the graph.

    `G.neighbors(i)` gives nodes with an outward edge from i, in a DiGraph,
    so for group out-degree centrality, the graph itself is used.
    """
    if not G.is_directed():
        raise nx.NetworkXNotImplemented("Not defined for undirected graphs.")

    if not set(S).issubset(G):
        raise nx.NodeNotFound("Node in S not present in G")

    S = set(S)
    V = set(G) - S

    if len(S) == len(G):
        return 0.0

    # Count the number of nodes in V connected to any node in S by outgoing edges
    connected_nodes = set()
    for s in S:
        connected_nodes.update(G.neighbors(s))
    connected_nodes -= S

    return len(connected_nodes) / len(V)
