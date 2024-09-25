""" Functions measuring similarity using graph edit distance.

The graph edit distance is the number of edge/node changes needed
to make two graphs isomorphic.

The default algorithm/implementation is sub-optimal for some graphs.
The problem of finding the exact Graph Edit Distance (GED) is NP-hard
so it is often slow. If the simple interface `graph_edit_distance`
takes too long for your graph, try `optimize_graph_edit_distance`
and/or `optimize_edit_paths`.

At the same time, I encourage capable people to investigate
alternative GED algorithms, in order to improve the choices available.
"""
import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
from networkx.utils import np_random_state
__all__ = ['graph_edit_distance', 'optimal_edit_paths',
    'optimize_graph_edit_distance', 'optimize_edit_paths',
    'simrank_similarity', 'panther_similarity', 'generate_random_paths']


@nx._dispatchable(graphs={'G1': 0, 'G2': 1}, preserve_edge_attrs=True,
    preserve_node_attrs=True)
def graph_edit_distance(G1, G2, node_match=None, edge_match=None,
    node_subst_cost=None, node_del_cost=None, node_ins_cost=None,
    edge_subst_cost=None, edge_del_cost=None, edge_ins_cost=None, roots=
    None, upper_bound=None, timeout=None):
    """Returns GED (graph edit distance) between graphs G1 and G2.

    Graph edit distance is a graph similarity measure analogous to
    Levenshtein distance for strings.  It is defined as minimum cost
    of edit path (sequence of node and edge edit operations)
    transforming graph G1 to graph isomorphic to G2.

    Parameters
    ----------
    G1, G2: graphs
        The two graphs G1 and G2 must be of the same type.

    node_match : callable
        A function that returns True if node n1 in G1 and n2 in G2
        should be considered equal during matching.

        The function will be called like

           node_match(G1.nodes[n1], G2.nodes[n2]).

        That is, the function will receive the node attribute
        dictionaries for n1 and n2 as inputs.

        Ignored if node_subst_cost is specified.  If neither
        node_match nor node_subst_cost are specified then node
        attributes are not considered.

    edge_match : callable
        A function that returns True if the edge attribute dictionaries
        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should
        be considered equal during matching.

        The function will be called like

           edge_match(G1[u1][v1], G2[u2][v2]).

        That is, the function will receive the edge attribute
        dictionaries of the edges under consideration.

        Ignored if edge_subst_cost is specified.  If neither
        edge_match nor edge_subst_cost are specified then edge
        attributes are not considered.

    node_subst_cost, node_del_cost, node_ins_cost : callable
        Functions that return the costs of node substitution, node
        deletion, and node insertion, respectively.

        The functions will be called like

           node_subst_cost(G1.nodes[n1], G2.nodes[n2]),
           node_del_cost(G1.nodes[n1]),
           node_ins_cost(G2.nodes[n2]).

        That is, the functions will receive the node attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function node_subst_cost overrides node_match if specified.
        If neither node_match nor node_subst_cost are specified then
        default node substitution cost of 0 is used (node attributes
        are not considered during matching).

        If node_del_cost is not specified then default node deletion
        cost of 1 is used.  If node_ins_cost is not specified then
        default node insertion cost of 1 is used.

    edge_subst_cost, edge_del_cost, edge_ins_cost : callable
        Functions that return the costs of edge substitution, edge
        deletion, and edge insertion, respectively.

        The functions will be called like

           edge_subst_cost(G1[u1][v1], G2[u2][v2]),
           edge_del_cost(G1[u1][v1]),
           edge_ins_cost(G2[u2][v2]).

        That is, the functions will receive the edge attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function edge_subst_cost overrides edge_match if specified.
        If neither edge_match nor edge_subst_cost are specified then
        default edge substitution cost of 0 is used (edge attributes
        are not considered during matching).

        If edge_del_cost is not specified then default edge deletion
        cost of 1 is used.  If edge_ins_cost is not specified then
        default edge insertion cost of 1 is used.

    roots : 2-tuple
        Tuple where first element is a node in G1 and the second
        is a node in G2.
        These nodes are forced to be matched in the comparison to
        allow comparison between rooted graphs.

    upper_bound : numeric
        Maximum edit distance to consider.  Return None if no edit
        distance under or equal to upper_bound exists.

    timeout : numeric
        Maximum number of seconds to execute.
        After timeout is met, the current best GED is returned.

    Examples
    --------
    >>> G1 = nx.cycle_graph(6)
    >>> G2 = nx.wheel_graph(7)
    >>> nx.graph_edit_distance(G1, G2)
    7.0

    >>> G1 = nx.star_graph(5)
    >>> G2 = nx.star_graph(5)
    >>> nx.graph_edit_distance(G1, G2, roots=(0, 0))
    0.0
    >>> nx.graph_edit_distance(G1, G2, roots=(1, 0))
    8.0

    See Also
    --------
    optimal_edit_paths, optimize_graph_edit_distance,

    is_isomorphic: test for graph edit distance of 0

    References
    ----------
    .. [1] Zeina Abu-Aisheh, Romain Raveaux, Jean-Yves Ramel, Patrick
       Martineau. An Exact Graph Edit Distance Algorithm for Solving
       Pattern Recognition Problems. 4th International Conference on
       Pattern Recognition Applications and Methods 2015, Jan 2015,
       Lisbon, Portugal. 2015,
       <10.5220/0005209202710278>. <hal-01168816>
       https://hal.archives-ouvertes.fr/hal-01168816

    """
    from networkx.algorithms.similarity import optimize_graph_edit_distance

    # Set default costs
    if node_subst_cost is None:
        if node_match is None:
            node_subst_cost = lambda n1, n2: 0
        else:
            node_subst_cost = lambda n1, n2: int(not node_match(n1, n2))
    if node_del_cost is None:
        node_del_cost = lambda n: 1
    if node_ins_cost is None:
        node_ins_cost = lambda n: 1
    
    if edge_subst_cost is None:
        if edge_match is None:
            edge_subst_cost = lambda e1, e2: 0
        else:
            edge_subst_cost = lambda e1, e2: int(not edge_match(e1, e2))
    if edge_del_cost is None:
        edge_del_cost = lambda e: 1
    if edge_ins_cost is None:
        edge_ins_cost = lambda e: 1

    # Use optimize_graph_edit_distance to compute GED
    for cost in optimize_graph_edit_distance(
        G1, G2,
        node_subst_cost=node_subst_cost,
        node_del_cost=node_del_cost,
        node_ins_cost=node_ins_cost,
        edge_subst_cost=edge_subst_cost,
        edge_del_cost=edge_del_cost,
        edge_ins_cost=edge_ins_cost,
        roots=roots,
        upper_bound=upper_bound,
        timeout=timeout
    ):
        # Return the last (best) cost found
        best_cost = cost

    return best_cost


@nx._dispatchable(graphs={'G1': 0, 'G2': 1})
def optimal_edit_paths(G1, G2, node_match=None, edge_match=None,
    node_subst_cost=None, node_del_cost=None, node_ins_cost=None,
    edge_subst_cost=None, edge_del_cost=None, edge_ins_cost=None,
    upper_bound=None):
    """Returns all minimum-cost edit paths transforming G1 to G2.

    Graph edit path is a sequence of node and edge edit operations
    transforming graph G1 to graph isomorphic to G2.  Edit operations
    include substitutions, deletions, and insertions.

    Parameters
    ----------
    G1, G2: graphs
        The two graphs G1 and G2 must be of the same type.

    node_match : callable
        A function that returns True if node n1 in G1 and n2 in G2
        should be considered equal during matching.

        The function will be called like

           node_match(G1.nodes[n1], G2.nodes[n2]).

        That is, the function will receive the node attribute
        dictionaries for n1 and n2 as inputs.

        Ignored if node_subst_cost is specified.  If neither
        node_match nor node_subst_cost are specified then node
        attributes are not considered.

    edge_match : callable
        A function that returns True if the edge attribute dictionaries
        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should
        be considered equal during matching.

        The function will be called like

           edge_match(G1[u1][v1], G2[u2][v2]).

        That is, the function will receive the edge attribute
        dictionaries of the edges under consideration.

        Ignored if edge_subst_cost is specified.  If neither
        edge_match nor edge_subst_cost are specified then edge
        attributes are not considered.

    node_subst_cost, node_del_cost, node_ins_cost : callable
        Functions that return the costs of node substitution, node
        deletion, and node insertion, respectively.

        The functions will be called like

           node_subst_cost(G1.nodes[n1], G2.nodes[n2]),
           node_del_cost(G1.nodes[n1]),
           node_ins_cost(G2.nodes[n2]).

        That is, the functions will receive the node attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function node_subst_cost overrides node_match if specified.
        If neither node_match nor node_subst_cost are specified then
        default node substitution cost of 0 is used (node attributes
        are not considered during matching).

        If node_del_cost is not specified then default node deletion
        cost of 1 is used.  If node_ins_cost is not specified then
        default node insertion cost of 1 is used.

    edge_subst_cost, edge_del_cost, edge_ins_cost : callable
        Functions that return the costs of edge substitution, edge
        deletion, and edge insertion, respectively.

        The functions will be called like

           edge_subst_cost(G1[u1][v1], G2[u2][v2]),
           edge_del_cost(G1[u1][v1]),
           edge_ins_cost(G2[u2][v2]).

        That is, the functions will receive the edge attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function edge_subst_cost overrides edge_match if specified.
        If neither edge_match nor edge_subst_cost are specified then
        default edge substitution cost of 0 is used (edge attributes
        are not considered during matching).

        If edge_del_cost is not specified then default edge deletion
        cost of 1 is used.  If edge_ins_cost is not specified then
        default edge insertion cost of 1 is used.

    upper_bound : numeric
        Maximum edit distance to consider.

    Returns
    -------
    edit_paths : list of tuples (node_edit_path, edge_edit_path)
        node_edit_path : list of tuples (u, v)
        edge_edit_path : list of tuples ((u1, v1), (u2, v2))

    cost : numeric
        Optimal edit path cost (graph edit distance). When the cost
        is zero, it indicates that `G1` and `G2` are isomorphic.

    Examples
    --------
    >>> G1 = nx.cycle_graph(4)
    >>> G2 = nx.wheel_graph(5)
    >>> paths, cost = nx.optimal_edit_paths(G1, G2)
    >>> len(paths)
    40
    >>> cost
    5.0

    Notes
    -----
    To transform `G1` into a graph isomorphic to `G2`, apply the node
    and edge edits in the returned ``edit_paths``.
    In the case of isomorphic graphs, the cost is zero, and the paths
    represent different isomorphic mappings (isomorphisms). That is, the
    edits involve renaming nodes and edges to match the structure of `G2`.

    See Also
    --------
    graph_edit_distance, optimize_edit_paths

    References
    ----------
    .. [1] Zeina Abu-Aisheh, Romain Raveaux, Jean-Yves Ramel, Patrick
       Martineau. An Exact Graph Edit Distance Algorithm for Solving
       Pattern Recognition Problems. 4th International Conference on
       Pattern Recognition Applications and Methods 2015, Jan 2015,
       Lisbon, Portugal. 2015,
       <10.5220/0005209202710278>. <hal-01168816>
       https://hal.archives-ouvertes.fr/hal-01168816

    """
    from networkx.algorithms.similarity import optimize_edit_paths

    # Set default costs
    if node_subst_cost is None:
        if node_match is None:
            node_subst_cost = lambda n1, n2: 0
        else:
            node_subst_cost = lambda n1, n2: int(not node_match(n1, n2))
    if node_del_cost is None:
        node_del_cost = lambda n: 1
    if node_ins_cost is None:
        node_ins_cost = lambda n: 1
    
    if edge_subst_cost is None:
        if edge_match is None:
            edge_subst_cost = lambda e1, e2: 0
        else:
            edge_subst_cost = lambda e1, e2: int(not edge_match(e1, e2))
    if edge_del_cost is None:
        edge_del_cost = lambda e: 1
    if edge_ins_cost is None:
        edge_ins_cost = lambda e: 1

    # Use optimize_edit_paths to compute optimal edit paths
    best_paths = []
    best_cost = float('inf')

    for node_edit_path, edge_edit_path, cost in optimize_edit_paths(
        G1, G2,
        node_subst_cost=node_subst_cost,
        node_del_cost=node_del_cost,
        node_ins_cost=node_ins_cost,
        edge_subst_cost=edge_subst_cost,
        edge_del_cost=edge_del_cost,
        edge_ins_cost=edge_ins_cost,
        upper_bound=upper_bound
    ):
        if cost < best_cost:
            best_paths = [(node_edit_path, edge_edit_path)]
            best_cost = cost
        elif cost == best_cost:
            best_paths.append((node_edit_path, edge_edit_path))

    return best_paths, best_cost


@nx._dispatchable(graphs={'G1': 0, 'G2': 1})
def optimize_graph_edit_distance(G1, G2, node_match=None, edge_match=None,
    node_subst_cost=None, node_del_cost=None, node_ins_cost=None,
    edge_subst_cost=None, edge_del_cost=None, edge_ins_cost=None,
    upper_bound=None):
    """Returns consecutive approximations of GED (graph edit distance)
    between graphs G1 and G2.

    Graph edit distance is a graph similarity measure analogous to
    Levenshtein distance for strings.  It is defined as minimum cost
    of edit path (sequence of node and edge edit operations)
    transforming graph G1 to graph isomorphic to G2.

    Parameters
    ----------
    G1, G2: graphs
        The two graphs G1 and G2 must be of the same type.

    node_match : callable
        A function that returns True if node n1 in G1 and n2 in G2
        should be considered equal during matching.

        The function will be called like

           node_match(G1.nodes[n1], G2.nodes[n2]).

        That is, the function will receive the node attribute
        dictionaries for n1 and n2 as inputs.

        Ignored if node_subst_cost is specified.  If neither
        node_match nor node_subst_cost are specified then node
        attributes are not considered.

    edge_match : callable
        A function that returns True if the edge attribute dictionaries
        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should
        be considered equal during matching.

        The function will be called like

           edge_match(G1[u1][v1], G2[u2][v2]).

        That is, the function will receive the edge attribute
        dictionaries of the edges under consideration.

        Ignored if edge_subst_cost is specified.  If neither
        edge_match nor edge_subst_cost are specified then edge
        attributes are not considered.

    node_subst_cost, node_del_cost, node_ins_cost : callable
        Functions that return the costs of node substitution, node
        deletion, and node insertion, respectively.

        The functions will be called like

           node_subst_cost(G1.nodes[n1], G2.nodes[n2]),
           node_del_cost(G1.nodes[n1]),
           node_ins_cost(G2.nodes[n2]).

        That is, the functions will receive the node attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function node_subst_cost overrides node_match if specified.
        If neither node_match nor node_subst_cost are specified then
        default node substitution cost of 0 is used (node attributes
        are not considered during matching).

        If node_del_cost is not specified then default node deletion
        cost of 1 is used.  If node_ins_cost is not specified then
        default node insertion cost of 1 is used.

    edge_subst_cost, edge_del_cost, edge_ins_cost : callable
        Functions that return the costs of edge substitution, edge
        deletion, and edge insertion, respectively.

        The functions will be called like

           edge_subst_cost(G1[u1][v1], G2[u2][v2]),
           edge_del_cost(G1[u1][v1]),
           edge_ins_cost(G2[u2][v2]).

        That is, the functions will receive the edge attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function edge_subst_cost overrides edge_match if specified.
        If neither edge_match nor edge_subst_cost are specified then
        default edge substitution cost of 0 is used (edge attributes
        are not considered during matching).

        If edge_del_cost is not specified then default edge deletion
        cost of 1 is used.  If edge_ins_cost is not specified then
        default edge insertion cost of 1 is used.

    upper_bound : numeric
        Maximum edit distance to consider.

    Returns
    -------
    Generator of consecutive approximations of graph edit distance.

    Examples
    --------
    >>> G1 = nx.cycle_graph(6)
    >>> G2 = nx.wheel_graph(7)
    >>> for v in nx.optimize_graph_edit_distance(G1, G2):
    ...     minv = v
    >>> minv
    7.0

    See Also
    --------
    graph_edit_distance, optimize_edit_paths

    References
    ----------
    .. [1] Zeina Abu-Aisheh, Romain Raveaux, Jean-Yves Ramel, Patrick
       Martineau. An Exact Graph Edit Distance Algorithm for Solving
       Pattern Recognition Problems. 4th International Conference on
       Pattern Recognition Applications and Methods 2015, Jan 2015,
       Lisbon, Portugal. 2015,
       <10.5220/0005209202710278>. <hal-01168816>
       https://hal.archives-ouvertes.fr/hal-01168816
    """
    from networkx.algorithms.similarity import optimize_edit_paths

    # Set default costs
    if node_subst_cost is None:
        if node_match is None:
            node_subst_cost = lambda n1, n2: 0
        else:
            node_subst_cost = lambda n1, n2: int(not node_match(n1, n2))
    if node_del_cost is None:
        node_del_cost = lambda n: 1
    if node_ins_cost is None:
        node_ins_cost = lambda n: 1
    
    if edge_subst_cost is None:
        if edge_match is None:
            edge_subst_cost = lambda e1, e2: 0
        else:
            edge_subst_cost = lambda e1, e2: int(not edge_match(e1, e2))
    if edge_del_cost is None:
        edge_del_cost = lambda e: 1
    if edge_ins_cost is None:
        edge_ins_cost = lambda e: 1

    # Use optimize_edit_paths to generate consecutive approximations
    for _, _, cost in optimize_edit_paths(
        G1, G2,
        node_subst_cost=node_subst_cost,
        node_del_cost=node_del_cost,
        node_ins_cost=node_ins_cost,
        edge_subst_cost=edge_subst_cost,
        edge_del_cost=edge_del_cost,
        edge_ins_cost=edge_ins_cost,
        upper_bound=upper_bound
    ):
        yield cost


@nx._dispatchable(graphs={'G1': 0, 'G2': 1}, preserve_edge_attrs=True,
    preserve_node_attrs=True)
def optimize_edit_paths(G1, G2, node_match=None, edge_match=None,
    node_subst_cost=None, node_del_cost=None, node_ins_cost=None,
    edge_subst_cost=None, edge_del_cost=None, edge_ins_cost=None,
    upper_bound=None, strictly_decreasing=True, roots=None, timeout=None):
    """GED (graph edit distance) calculation: advanced interface.

    Graph edit path is a sequence of node and edge edit operations
    transforming graph G1 to graph isomorphic to G2.  Edit operations
    include substitutions, deletions, and insertions.

    Graph edit distance is defined as minimum cost of edit path.

    Parameters
    ----------
    G1, G2: graphs
        The two graphs G1 and G2 must be of the same type.

    node_match : callable
        A function that returns True if node n1 in G1 and n2 in G2
        should be considered equal during matching.

        The function will be called like

           node_match(G1.nodes[n1], G2.nodes[n2]).

        That is, the function will receive the node attribute
        dictionaries for n1 and n2 as inputs.

        Ignored if node_subst_cost is specified.  If neither
        node_match nor node_subst_cost are specified then node
        attributes are not considered.

    edge_match : callable
        A function that returns True if the edge attribute dictionaries
        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should
        be considered equal during matching.

        The function will be called like

           edge_match(G1[u1][v1], G2[u2][v2]).

        That is, the function will receive the edge attribute
        dictionaries of the edges under consideration.

        Ignored if edge_subst_cost is specified.  If neither
        edge_match nor edge_subst_cost are specified then edge
        attributes are not considered.

    node_subst_cost, node_del_cost, node_ins_cost : callable
        Functions that return the costs of node substitution, node
        deletion, and node insertion, respectively.

        The functions will be called like

           node_subst_cost(G1.nodes[n1], G2.nodes[n2]),
           node_del_cost(G1.nodes[n1]),
           node_ins_cost(G2.nodes[n2]).

        That is, the functions will receive the node attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function node_subst_cost overrides node_match if specified.
        If neither node_match nor node_subst_cost are specified then
        default node substitution cost of 0 is used (node attributes
        are not considered during matching).

        If node_del_cost is not specified then default node deletion
        cost of 1 is used.  If node_ins_cost is not specified then
        default node insertion cost of 1 is used.

    edge_subst_cost, edge_del_cost, edge_ins_cost : callable
        Functions that return the costs of edge substitution, edge
        deletion, and edge insertion, respectively.

        The functions will be called like

           edge_subst_cost(G1[u1][v1], G2[u2][v2]),
           edge_del_cost(G1[u1][v1]),
           edge_ins_cost(G2[u2][v2]).

        That is, the functions will receive the edge attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function edge_subst_cost overrides edge_match if specified.
        If neither edge_match nor edge_subst_cost are specified then
        default edge substitution cost of 0 is used (edge attributes
        are not considered during matching).

        If edge_del_cost is not specified then default edge deletion
        cost of 1 is used.  If edge_ins_cost is not specified then
        default edge insertion cost of 1 is used.

    upper_bound : numeric
        Maximum edit distance to consider.

    strictly_decreasing : bool
        If True, return consecutive approximations of strictly
        decreasing cost.  Otherwise, return all edit paths of cost
        less than or equal to the previous minimum cost.

    roots : 2-tuple
        Tuple where first element is a node in G1 and the second
        is a node in G2.
        These nodes are forced to be matched in the comparison to
        allow comparison between rooted graphs.

    timeout : numeric
        Maximum number of seconds to execute.
        After timeout is met, the current best GED is returned.

    Returns
    -------
    Generator of tuples (node_edit_path, edge_edit_path, cost)
        node_edit_path : list of tuples (u, v)
        edge_edit_path : list of tuples ((u1, v1), (u2, v2))
        cost : numeric

    See Also
    --------
    graph_edit_distance, optimize_graph_edit_distance, optimal_edit_paths

    References
    ----------
    .. [1] Zeina Abu-Aisheh, Romain Raveaux, Jean-Yves Ramel, Patrick
       Martineau. An Exact Graph Edit Distance Algorithm for Solving
       Pattern Recognition Problems. 4th International Conference on
       Pattern Recognition Applications and Methods 2015, Jan 2015,
       Lisbon, Portugal. 2015,
       <10.5220/0005209202710278>. <hal-01168816>
       https://hal.archives-ouvertes.fr/hal-01168816

    """
    pass


@nx._dispatchable
def simrank_similarity(G, source=None, target=None, importance_factor=0.9,
    max_iterations=1000, tolerance=0.0001):
    """Returns the SimRank similarity of nodes in the graph ``G``.

    SimRank is a similarity metric that says "two objects are considered
    to be similar if they are referenced by similar objects." [1]_.

    The pseudo-code definition from the paper is::

        def simrank(G, u, v):
            in_neighbors_u = G.predecessors(u)
            in_neighbors_v = G.predecessors(v)
            scale = C / (len(in_neighbors_u) * len(in_neighbors_v))
            return scale * sum(
                simrank(G, w, x) for w, x in product(in_neighbors_u, in_neighbors_v)
            )

    where ``G`` is the graph, ``u`` is the source, ``v`` is the target,
    and ``C`` is a float decay or importance factor between 0 and 1.

    The SimRank algorithm for determining node similarity is defined in
    [2]_.

    Parameters
    ----------
    G : NetworkX graph
        A NetworkX graph

    source : node
        If this is specified, the returned dictionary maps each node
        ``v`` in the graph to the similarity between ``source`` and
        ``v``.

    target : node
        If both ``source`` and ``target`` are specified, the similarity
        value between ``source`` and ``target`` is returned. If
        ``target`` is specified but ``source`` is not, this argument is
        ignored.

    importance_factor : float
        The relative importance of indirect neighbors with respect to
        direct neighbors.

    max_iterations : integer
        Maximum number of iterations.

    tolerance : float
        Error tolerance used to check convergence. When an iteration of
        the algorithm finds that no similarity value changes more than
        this amount, the algorithm halts.

    Returns
    -------
    similarity : dictionary or float
        If ``source`` and ``target`` are both ``None``, this returns a
        dictionary of dictionaries, where keys are node pairs and value
        are similarity of the pair of nodes.

        If ``source`` is not ``None`` but ``target`` is, this returns a
        dictionary mapping node to the similarity of ``source`` and that
        node.

        If neither ``source`` nor ``target`` is ``None``, this returns
        the similarity value for the given pair of nodes.

    Raises
    ------
    ExceededMaxIterations
        If the algorithm does not converge within ``max_iterations``.

    NodeNotFound
        If either ``source`` or ``target`` is not in `G`.

    Examples
    --------
    >>> G = nx.cycle_graph(2)
    >>> nx.simrank_similarity(G)
    {0: {0: 1.0, 1: 0.0}, 1: {0: 0.0, 1: 1.0}}
    >>> nx.simrank_similarity(G, source=0)
    {0: 1.0, 1: 0.0}
    >>> nx.simrank_similarity(G, source=0, target=0)
    1.0

    The result of this function can be converted to a numpy array
    representing the SimRank matrix by using the node order of the
    graph to determine which row and column represent each node.
    Other ordering of nodes is also possible.

    >>> import numpy as np
    >>> sim = nx.simrank_similarity(G)
    >>> np.array([[sim[u][v] for v in G] for u in G])
    array([[1., 0.],
           [0., 1.]])
    >>> sim_1d = nx.simrank_similarity(G, source=0)
    >>> np.array([sim[0][v] for v in G])
    array([1., 0.])

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/SimRank
    .. [2] G. Jeh and J. Widom.
           "SimRank: a measure of structural-context similarity",
           In KDD'02: Proceedings of the Eighth ACM SIGKDD
           International Conference on Knowledge Discovery and Data Mining,
           pp. 538--543. ACM Press, 2002.
    """
    import numpy as np
    from itertools import product

    if source is not None and source not in G:
        raise nx.NodeNotFound(f"Source node {source} is not in G")
    if target is not None and target not in G:
        raise nx.NodeNotFound(f"Target node {target} is not in G")

    nodes = list(G.nodes())
    node_indices = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    sim_prev = np.zeros((n, n))
    sim = np.identity(n)

    for _ in range(max_iterations):
        if np.allclose(sim, sim_prev, atol=tolerance):
            break
        sim_prev = sim.copy()
        for u, v in product(range(n), repeat=2):
            if u == v:
                continue
            u_nb = list(G.predecessors(nodes[u]))
            v_nb = list(G.predecessors(nodes[v]))
            if not u_nb or not v_nb:
                sim[u][v] = 0
            else:
                s = sum(sim_prev[node_indices[w]][node_indices[x]]
                        for w, x in product(u_nb, v_nb))
                sim[u][v] = (importance_factor * s) / (len(u_nb) * len(v_nb))
    else:
        raise nx.ExceededMaxIterations(max_iterations)

    if source is not None:
        if target is not None:
            return sim[node_indices[source]][node_indices[target]]
        else:
            return {v: sim[node_indices[source]][node_indices[v]] for v in nodes}
    else:
        return {u: {v: sim[node_indices[u]][node_indices[v]] for v in nodes} for u in nodes}


def _simrank_similarity_python(G, source=None, target=None,
    importance_factor=0.9, max_iterations=1000, tolerance=0.0001):
    """Returns the SimRank similarity of nodes in the graph ``G``.

    This pure Python version is provided for pedagogical purposes.

    Examples
    --------
    >>> G = nx.cycle_graph(2)
    >>> nx.similarity._simrank_similarity_python(G)
    {0: {0: 1, 1: 0.0}, 1: {0: 0.0, 1: 1}}
    >>> nx.similarity._simrank_similarity_python(G, source=0)
    {0: 1, 1: 0.0}
    >>> nx.similarity._simrank_similarity_python(G, source=0, target=0)
    1
    """
    pass


def _simrank_similarity_numpy(G, source=None, target=None,
    importance_factor=0.9, max_iterations=1000, tolerance=0.0001):
    """Calculate SimRank of nodes in ``G`` using matrices with ``numpy``.

    The SimRank algorithm for determining node similarity is defined in
    [1]_.

    Parameters
    ----------
    G : NetworkX graph
        A NetworkX graph

    source : node
        If this is specified, the returned dictionary maps each node
        ``v`` in the graph to the similarity between ``source`` and
        ``v``.

    target : node
        If both ``source`` and ``target`` are specified, the similarity
        value between ``source`` and ``target`` is returned. If
        ``target`` is specified but ``source`` is not, this argument is
        ignored.

    importance_factor : float
        The relative importance of indirect neighbors with respect to
        direct neighbors.

    max_iterations : integer
        Maximum number of iterations.

    tolerance : float
        Error tolerance used to check convergence. When an iteration of
        the algorithm finds that no similarity value changes more than
        this amount, the algorithm halts.

    Returns
    -------
    similarity : numpy array or float
        If ``source`` and ``target`` are both ``None``, this returns a
        2D array containing SimRank scores of the nodes.

        If ``source`` is not ``None`` but ``target`` is, this returns an
        1D array containing SimRank scores of ``source`` and that
        node.

        If neither ``source`` nor ``target`` is ``None``, this returns
        the similarity value for the given pair of nodes.

    Examples
    --------
    >>> G = nx.cycle_graph(2)
    >>> nx.similarity._simrank_similarity_numpy(G)
    array([[1., 0.],
           [0., 1.]])
    >>> nx.similarity._simrank_similarity_numpy(G, source=0)
    array([1., 0.])
    >>> nx.similarity._simrank_similarity_numpy(G, source=0, target=0)
    1.0

    References
    ----------
    .. [1] G. Jeh and J. Widom.
           "SimRank: a measure of structural-context similarity",
           In KDD'02: Proceedings of the Eighth ACM SIGKDD
           International Conference on Knowledge Discovery and Data Mining,
           pp. 538--543. ACM Press, 2002.
    """
    pass


@nx._dispatchable(edge_attrs='weight')
def panther_similarity(G, source, k=5, path_length=5, c=0.5, delta=0.1, eps
    =None, weight='weight'):
    """Returns the Panther similarity of nodes in the graph `G` to node ``v``.

    Panther is a similarity metric that says "two objects are considered
    to be similar if they frequently appear on the same paths." [1]_.

    Parameters
    ----------
    G : NetworkX graph
        A NetworkX graph
    source : node
        Source node for which to find the top `k` similar other nodes
    k : int (default = 5)
        The number of most similar nodes to return.
    path_length : int (default = 5)
        How long the randomly generated paths should be (``T`` in [1]_)
    c : float (default = 0.5)
        A universal positive constant used to scale the number
        of sample random paths to generate.
    delta : float (default = 0.1)
        The probability that the similarity $S$ is not an epsilon-approximation to (R, phi),
        where $R$ is the number of random paths and $\\phi$ is the probability
        that an element sampled from a set $A \\subseteq D$, where $D$ is the domain.
    eps : float or None (default = None)
        The error bound. Per [1]_, a good value is ``sqrt(1/|E|)``. Therefore,
        if no value is provided, the recommended computed value will be used.
    weight : string or None, optional (default="weight")
        The name of an edge attribute that holds the numerical value
        used as a weight. If None then each edge has weight 1.

    Returns
    -------
    similarity : dictionary
        Dictionary of nodes to similarity scores (as floats). Note:
        the self-similarity (i.e., ``v``) will not be included in
        the returned dictionary. So, for ``k = 5``, a dictionary of
        top 4 nodes and their similarity scores will be returned.

    Raises
    ------
    NetworkXUnfeasible
        If `source` is an isolated node.

    NodeNotFound
        If `source` is not in `G`.

    Notes
    -----
        The isolated nodes in `G` are ignored.

    Examples
    --------
    >>> G = nx.star_graph(10)
    >>> sim = nx.panther_similarity(G, 0)

    References
    ----------
    .. [1] Zhang, J., Tang, J., Ma, C., Tong, H., Jing, Y., & Li, J.
           Panther: Fast top-k similarity search on large networks.
           In Proceedings of the ACM SIGKDD International Conference
           on Knowledge Discovery and Data Mining (Vol. 2015-August, pp. 1445–1454).
           Association for Computing Machinery. https://doi.org/10.1145/2783258.2783267.
    """
    import math
    from collections import Counter
    from networkx.utils import not_implemented_for

    @not_implemented_for('multigraph')
    def panther_similarity_impl(G, source, k, path_length, c, delta, eps, weight):
        if source not in G:
            raise nx.NodeNotFound(f"Source node {source} is not in G")

        if G.degree(source) == 0:
            raise nx.NetworkXUnfeasible("Source node is isolated.")

        if eps is None:
            eps = math.sqrt(1 / G.number_of_edges())

        # Calculate the number of random paths to generate
        R = int((c / (eps ** 2)) * math.log(2 / delta))

        # Generate random paths
        paths = list(generate_random_paths(G, R, path_length, weight=weight))

        # Count occurrences of nodes in paths containing the source node
        node_counts = Counter()
        for path in paths:
            if source in path:
                node_counts.update(set(path) - {source})

        # Calculate similarity scores
        similarity = {node: count / R for node, count in node_counts.items()}

        # Sort by similarity score and return top k-1 (excluding the source node)
        return dict(sorted(similarity.items(), key=lambda x: x[1], reverse=True)[:k-1])

    return panther_similarity_impl(G, source, k, path_length, c, delta, eps, weight)


@np_random_state(5)
@nx._dispatchable(edge_attrs='weight')
def generate_random_paths(G, sample_size, path_length=5, index_map=None,
    weight='weight', seed=None):
    """Randomly generate `sample_size` paths of length `path_length`.

    Parameters
    ----------
    G : NetworkX graph
        A NetworkX graph
    sample_size : integer
        The number of paths to generate. This is ``R`` in [1]_.
    path_length : integer (default = 5)
        The maximum size of the path to randomly generate.
        This is ``T`` in [1]_. According to the paper, ``T >= 5`` is
        recommended.
    index_map : dictionary, optional
        If provided, this will be populated with the inverted
        index of nodes mapped to the set of generated random path
        indices within ``paths``.
    weight : string or None, optional (default="weight")
        The name of an edge attribute that holds the numerical value
        used as a weight. If None then each edge has weight 1.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    paths : generator of lists
        Generator of `sample_size` paths each with length `path_length`.

    Examples
    --------
    Note that the return value is the list of paths:

    >>> G = nx.star_graph(3)
    >>> random_path = nx.generate_random_paths(G, 2)

    By passing a dictionary into `index_map`, it will build an
    inverted index mapping of nodes to the paths in which that node is present:

    >>> G = nx.star_graph(3)
    >>> index_map = {}
    >>> random_path = nx.generate_random_paths(G, 3, index_map=index_map)
    >>> paths_containing_node_0 = [
    ...     random_path[path_idx] for path_idx in index_map.get(0, [])
    ... ]

    References
    ----------
    .. [1] Zhang, J., Tang, J., Ma, C., Tong, H., Jing, Y., & Li, J.
           Panther: Fast top-k similarity search on large networks.
           In Proceedings of the ACM SIGKDD International Conference
           on Knowledge Discovery and Data Mining (Vol. 2015-August, pp. 1445–1454).
           Association for Computing Machinery. https://doi.org/10.1145/2783258.2783267.
    """
    import random

    if seed is not None:
        random.seed(seed)

    def generate_path():
        path = []
        current_node = random.choice(list(G.nodes()))
        for _ in range(path_length):
            path.append(current_node)
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
            current_node = random.choices(neighbors, weights=[G[current_node][neighbor].get(weight, 1) for neighbor in neighbors])[0]
        return path

    paths = []
    for i in range(sample_size):
        path = generate_path()
        paths.append(path)
        if index_map is not None:
            for node in set(path):
                if node not in index_map:
                    index_map[node] = set()
                index_map[node].add(i)

    return paths
