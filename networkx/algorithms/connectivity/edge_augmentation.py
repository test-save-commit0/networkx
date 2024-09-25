"""
Algorithms for finding k-edge-augmentations

A k-edge-augmentation is a set of edges, that once added to a graph, ensures
that the graph is k-edge-connected; i.e. the graph cannot be disconnected
unless k or more edges are removed.  Typically, the goal is to find the
augmentation with minimum weight.  In general, it is not guaranteed that a
k-edge-augmentation exists.

See Also
--------
:mod:`edge_kcomponents` : algorithms for finding k-edge-connected components
:mod:`connectivity` : algorithms for determining edge connectivity.
"""
import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
__all__ = ['k_edge_augmentation', 'is_k_edge_connected',
    'is_locally_k_edge_connected']


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def is_k_edge_connected(G, k):
    """Tests to see if a graph is k-edge-connected.

    Is it impossible to disconnect the graph by removing fewer than k edges?
    If so, then G is k-edge-connected.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    k : integer
        edge connectivity to test for

    Returns
    -------
    boolean
        True if G is k-edge-connected.

    See Also
    --------
    :func:`is_locally_k_edge_connected`

    Examples
    --------
    >>> G = nx.barbell_graph(10, 0)
    >>> nx.is_k_edge_connected(G, k=1)
    True
    >>> nx.is_k_edge_connected(G, k=2)
    False
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    
    if G.number_of_nodes() < 2:
        return True
    
    if G.number_of_edges() < k:
        return False
    
    # Check edge connectivity for all pairs of nodes
    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                if nx.edge_connectivity(G, u, v) < k:
                    return False
    
    return True


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def is_locally_k_edge_connected(G, s, t, k):
    """Tests to see if an edge in a graph is locally k-edge-connected.

    Is it impossible to disconnect s and t by removing fewer than k edges?
    If so, then s and t are locally k-edge-connected in G.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    s : node
        Source node

    t : node
        Target node

    k : integer
        local edge connectivity for nodes s and t

    Returns
    -------
    boolean
        True if s and t are locally k-edge-connected in G.

    See Also
    --------
    :func:`is_k_edge_connected`

    Examples
    --------
    >>> from networkx.algorithms.connectivity import is_locally_k_edge_connected
    >>> G = nx.barbell_graph(10, 0)
    >>> is_locally_k_edge_connected(G, 5, 15, k=1)
    True
    >>> is_locally_k_edge_connected(G, 5, 15, k=2)
    False
    >>> is_locally_k_edge_connected(G, 1, 5, k=2)
    True
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    
    if s not in G or t not in G:
        raise nx.NetworkXError("Both s and t must be in G")
    
    if s == t:
        return True
    
    # Use edge_connectivity to find the minimum number of edges
    # that need to be removed to disconnect s and t
    local_edge_connectivity = nx.edge_connectivity(G, s, t)
    
    return local_edge_connectivity >= k


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def k_edge_augmentation(G, k, avail=None, weight=None, partial=False):
    """Finds set of edges to k-edge-connect G.

    Adding edges from the augmentation to G make it impossible to disconnect G
    unless k or more edges are removed. This function uses the most efficient
    function available (depending on the value of k and if the problem is
    weighted or unweighted) to search for a minimum weight subset of available
    edges that k-edge-connects G. In general, finding a k-edge-augmentation is
    NP-hard, so solutions are not guaranteed to be minimal. Furthermore, a
    k-edge-augmentation may not exist.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    k : integer
        Desired edge connectivity

    avail : dict or a set of 2 or 3 tuples
        The available edges that can be used in the augmentation.

        If unspecified, then all edges in the complement of G are available.
        Otherwise, each item is an available edge (with an optional weight).

        In the unweighted case, each item is an edge ``(u, v)``.

        In the weighted case, each item is a 3-tuple ``(u, v, d)`` or a dict
        with items ``(u, v): d``.  The third item, ``d``, can be a dictionary
        or a real number.  If ``d`` is a dictionary ``d[weight]``
        correspondings to the weight.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples where the
        third item in each tuple is a dictionary.

    partial : boolean
        If partial is True and no feasible k-edge-augmentation exists, then all
        a partial k-edge-augmentation is generated. Adding the edges in a
        partial augmentation to G, minimizes the number of k-edge-connected
        components and maximizes the edge connectivity between those
        components. For details, see :func:`partial_k_edge_augmentation`.

    Yields
    ------
    edge : tuple
        Edges that, once added to G, would cause G to become k-edge-connected.
        If partial is False, an error is raised if this is not possible.
        Otherwise, generated edges form a partial augmentation, which
        k-edge-connects any part of G where it is possible, and maximally
        connects the remaining parts.

    Raises
    ------
    NetworkXUnfeasible
        If partial is False and no k-edge-augmentation exists.

    NetworkXNotImplemented
        If the input graph is directed or a multigraph.

    ValueError:
        If k is less than 1

    Notes
    -----
    When k=1 this returns an optimal solution.

    When k=2 and ``avail`` is None, this returns an optimal solution.
    Otherwise when k=2, this returns a 2-approximation of the optimal solution.

    For k>3, this problem is NP-hard and this uses a randomized algorithm that
        produces a feasible solution, but provides no guarantees on the
        solution weight.

    Examples
    --------
    >>> # Unweighted cases
    >>> G = nx.path_graph((1, 2, 3, 4))
    >>> G.add_node(5)
    >>> sorted(nx.k_edge_augmentation(G, k=1))
    [(1, 5)]
    >>> sorted(nx.k_edge_augmentation(G, k=2))
    [(1, 5), (5, 4)]
    >>> sorted(nx.k_edge_augmentation(G, k=3))
    [(1, 4), (1, 5), (2, 5), (3, 5), (4, 5)]
    >>> complement = list(nx.k_edge_augmentation(G, k=5, partial=True))
    >>> G.add_edges_from(complement)
    >>> nx.edge_connectivity(G)
    4

    >>> # Weighted cases
    >>> G = nx.path_graph((1, 2, 3, 4))
    >>> G.add_node(5)
    >>> # avail can be a tuple with a dict
    >>> avail = [(1, 5, {"weight": 11}), (2, 5, {"weight": 10})]
    >>> sorted(nx.k_edge_augmentation(G, k=1, avail=avail, weight="weight"))
    [(2, 5)]
    >>> # or avail can be a 3-tuple with a real number
    >>> avail = [(1, 5, 11), (2, 5, 10), (4, 3, 1), (4, 5, 51)]
    >>> sorted(nx.k_edge_augmentation(G, k=2, avail=avail))
    [(1, 5), (2, 5), (4, 5)]
    >>> # or avail can be a dict
    >>> avail = {(1, 5): 11, (2, 5): 10, (4, 3): 1, (4, 5): 51}
    >>> sorted(nx.k_edge_augmentation(G, k=2, avail=avail))
    [(1, 5), (2, 5), (4, 5)]
    >>> # If augmentation is infeasible, then a partial solution can be found
    >>> avail = {(1, 5): 11}
    >>> sorted(nx.k_edge_augmentation(G, k=2, avail=avail, partial=True))
    [(1, 5)]
    """
    if G.is_directed() or G.is_multigraph():
        raise nx.NetworkXNotImplemented("Not implemented for directed or multigraphs")
    
    if k < 1:
        raise ValueError("k must be at least 1")
    
    if avail is None:
        avail = complement_edges(G)
    
    # Convert avail to a consistent format
    if isinstance(avail, dict):
        avail = [(u, v, d) for (u, v), d in avail.items()]
    
    # Sort available edges by weight
    if weight is not None:
        avail = sorted(avail, key=lambda x: x[2].get(weight, 1) if isinstance(x[2], dict) else x[2])
    
    augmentation = []
    current_connectivity = nx.edge_connectivity(G)
    
    while current_connectivity < k and avail:
        u, v, _ = avail.pop(0)
        if not nx.has_path(G, u, v) or nx.edge_connectivity(G, u, v) < k:
            G.add_edge(u, v)
            augmentation.append((u, v))
            current_connectivity = min(current_connectivity, nx.edge_connectivity(G, u, v))
    
    if not partial and current_connectivity < k:
        raise nx.NetworkXUnfeasible("No feasible k-edge-augmentation exists")
    
    return augmentation


@nx._dispatchable
def partial_k_edge_augmentation(G, k, avail, weight=None):
    """Finds augmentation that k-edge-connects as much of the graph as possible.

    When a k-edge-augmentation is not possible, we can still try to find a
    small set of edges that partially k-edge-connects as much of the graph as
    possible. All possible edges are generated between remaining parts.
    This minimizes the number of k-edge-connected subgraphs in the resulting
    graph and maximizes the edge connectivity between those subgraphs.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    k : integer
        Desired edge connectivity

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    Yields
    ------
    edge : tuple
        Edges in the partial augmentation of G. These edges k-edge-connect any
        part of G where it is possible, and maximally connects the remaining
        parts. In other words, all edges from avail are generated except for
        those within subgraphs that have already become k-edge-connected.

    Notes
    -----
    Construct H that augments G with all edges in avail.
    Find the k-edge-subgraphs of H.
    For each k-edge-subgraph, if the number of nodes is more than k, then find
    the k-edge-augmentation of that graph and add it to the solution. Then add
    all edges in avail between k-edge subgraphs to the solution.

    See Also
    --------
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))
    >>> G.add_node(8)
    >>> avail = [(1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 5), (1, 8)]
    >>> sorted(partial_k_edge_augmentation(G, k=2, avail=avail))
    [(1, 5), (1, 8)]
    """
    H = G.copy()
    H.add_edges_from(avail)
    
    # Find k-edge-connected components
    k_components = list(nx.k_edge_components(H, k))
    
    augmentation = []
    
    # Augment each k-edge-connected component
    for component in k_components:
        if len(component) > k:
            subgraph = G.subgraph(component).copy()
            component_avail = [e for e in avail if e[0] in component and e[1] in component]
            augmentation.extend(k_edge_augmentation(subgraph, k, avail=component_avail, weight=weight))
    
    # Add edges between k-edge-connected components
    for i, comp1 in enumerate(k_components):
        for comp2 in k_components[i+1:]:
            cross_edges = [e for e in avail if (e[0] in comp1 and e[1] in comp2) or (e[0] in comp2 and e[1] in comp1)]
            augmentation.extend(cross_edges)
    
    return augmentation


@not_implemented_for('multigraph')
@not_implemented_for('directed')
@nx._dispatchable
def one_edge_augmentation(G, avail=None, weight=None, partial=False):
    """Finds minimum weight set of edges to connect G.

    Equivalent to :func:`k_edge_augmentation` when k=1. Adding the resulting
    edges to G will make it 1-edge-connected. The solution is optimal for both
    weighted and non-weighted variants.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    partial : boolean
        If partial is True and no feasible k-edge-augmentation exists, then the
        augmenting edges minimize the number of connected components.

    Yields
    ------
    edge : tuple
        Edges in the one-augmentation of G

    Raises
    ------
    NetworkXUnfeasible
        If partial is False and no one-edge-augmentation exists.

    Notes
    -----
    Uses either :func:`unconstrained_one_edge_augmentation` or
    :func:`weighted_one_edge_augmentation` depending on whether ``avail`` is
    specified. Both algorithms are based on finding a minimum spanning tree.
    As such both algorithms find optimal solutions and run in linear time.

    See Also
    --------
    :func:`k_edge_augmentation`
    """
    if avail is None:
        return unconstrained_one_edge_augmentation(G)
    else:
        return weighted_one_edge_augmentation(G, avail, weight, partial)


@not_implemented_for('multigraph')
@not_implemented_for('directed')
@nx._dispatchable
def bridge_augmentation(G, avail=None, weight=None):
    """Finds the a set of edges that bridge connects G.

    Equivalent to :func:`k_edge_augmentation` when k=2, and partial=False.
    Adding the resulting edges to G will make it 2-edge-connected.  If no
    constraints are specified the returned set of edges is minimum an optimal,
    otherwise the solution is approximated.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    Yields
    ------
    edge : tuple
        Edges in the bridge-augmentation of G

    Raises
    ------
    NetworkXUnfeasible
        If no bridge-augmentation exists.

    Notes
    -----
    If there are no constraints the solution can be computed in linear time
    using :func:`unconstrained_bridge_augmentation`. Otherwise, the problem
    becomes NP-hard and is the solution is approximated by
    :func:`weighted_bridge_augmentation`.

    See Also
    --------
    :func:`k_edge_augmentation`
    """
    if avail is None:
        return unconstrained_bridge_augmentation(G)
    else:
        return weighted_bridge_augmentation(G, avail, weight)


def _ordered(u, v):
    """Returns the nodes in an undirected edge in lower-triangular order"""
    return (u, v) if u <= v else (v, u)


def _unpack_available_edges(avail, weight=None, G=None):
    """Helper to separate avail into edges and corresponding weights"""
    if isinstance(avail, dict):
        avail_uv = list(avail.keys())
        avail_w = list(avail.values())
    else:
        avail_uv = []
        avail_w = []
        for edge in avail:
            if len(edge) == 3:
                u, v, d = edge
                if isinstance(d, dict):
                    w = d.get(weight, 1) if weight else 1
                else:
                    w = d
            else:
                u, v = edge
                w = 1
            avail_uv.append((u, v))
            avail_w.append(w)
    
    if G is not None:
        avail_uv = [edge for edge in avail_uv if edge[0] in G and edge[1] in G]
        avail_w = [w for edge, w in zip(avail_uv, avail_w) if edge[0] in G and edge[1] in G]
    
    return avail_uv, avail_w


MetaEdge = namedtuple('MetaEdge', ('meta_uv', 'uv', 'w'))


def _lightest_meta_edges(mapping, avail_uv, avail_w):
    """Maps available edges in the original graph to edges in the metagraph.

    Parameters
    ----------
    mapping : dict
        mapping produced by :func:`collapse`, that maps each node in the
        original graph to a node in the meta graph

    avail_uv : list
        list of edges

    avail_w : list
        list of edge weights

    Notes
    -----
    Each node in the metagraph is a k-edge-connected component in the original
    graph.  We don't care about any edge within the same k-edge-connected
    component, so we ignore self edges.  We also are only interested in the
    minimum weight edge bridging each k-edge-connected component so, we group
    the edges by meta-edge and take the lightest in each group.

    Examples
    --------
    >>> # Each group represents a meta-node
    >>> groups = ([1, 2, 3], [4, 5], [6])
    >>> mapping = {n: meta_n for meta_n, ns in enumerate(groups) for n in ns}
    >>> avail_uv = [(1, 2), (3, 6), (1, 4), (5, 2), (6, 1), (2, 6), (3, 1)]
    >>> avail_w = [20, 99, 20, 15, 50, 99, 20]
    >>> sorted(_lightest_meta_edges(mapping, avail_uv, avail_w))
    [MetaEdge(meta_uv=(0, 1), uv=(5, 2), w=15), MetaEdge(meta_uv=(0, 2), uv=(6, 1), w=50)]
    """
    meta_edges = defaultdict(list)
    for (u, v), w in zip(avail_uv, avail_w):
        meta_u = mapping[u]
        meta_v = mapping[v]
        if meta_u != meta_v:
            meta_uv = _ordered(meta_u, meta_v)
            meta_edges[meta_uv].append(MetaEdge(meta_uv, (u, v), w))
    
    lightest = [min(edge_list, key=lambda x: x.w) for edge_list in meta_edges.values()]
    return lightest


@nx._dispatchable
def unconstrained_one_edge_augmentation(G):
    """Finds the smallest set of edges to connect G.

    This is a variant of the unweighted MST problem.
    If G is not empty, a feasible solution always exists.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    Yields
    ------
    edge : tuple
        Edges in the one-edge-augmentation of G

    See Also
    --------
    :func:`one_edge_augmentation`
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (2, 3), (4, 5)])
    >>> G.add_nodes_from([6, 7, 8])
    >>> sorted(unconstrained_one_edge_augmentation(G))
    [(1, 4), (4, 6), (6, 7), (7, 8)]
    """
    components = list(nx.connected_components(G))
    if len(components) == 1:
        return []
    
    # Create a new graph with components as nodes
    C = nx.Graph()
    C.add_nodes_from(range(len(components)))
    
    # Find the minimum spanning tree of the component graph
    mst_edges = nx.minimum_spanning_tree(C).edges()
    
    # Map the MST edges back to the original graph
    for i, j in mst_edges:
        u = next(iter(components[i]))
        v = next(iter(components[j]))
        yield (u, v)


@nx._dispatchable
def weighted_one_edge_augmentation(G, avail, weight=None, partial=False):
    """Finds the minimum weight set of edges to connect G if one exists.

    This is a variant of the weighted MST problem.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    partial : boolean
        If partial is True and no feasible k-edge-augmentation exists, then the
        augmenting edges minimize the number of connected components.

    Yields
    ------
    edge : tuple
        Edges in the subset of avail chosen to connect G.

    See Also
    --------
    :func:`one_edge_augmentation`
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (2, 3), (4, 5)])
    >>> G.add_nodes_from([6, 7, 8])
    >>> # any edge not in avail has an implicit weight of infinity
    >>> avail = [(1, 3), (1, 5), (4, 7), (4, 8), (6, 1), (8, 1), (8, 2)]
    >>> sorted(weighted_one_edge_augmentation(G, avail))
    [(1, 5), (4, 7), (6, 1), (8, 1)]
    >>> # find another solution by giving large weights to edges in the
    >>> # previous solution (note some of the old edges must be used)
    >>> avail = [(1, 3), (1, 5, 99), (4, 7, 9), (6, 1, 99), (8, 1, 99), (8, 2)]
    >>> sorted(weighted_one_edge_augmentation(G, avail))
    [(1, 5), (4, 7), (6, 1), (8, 2)]
    """
    avail_uv, avail_w = _unpack_available_edges(avail, weight=weight)
    
    # Create a new graph with components as nodes
    C = nx.Graph()
    components = list(nx.connected_components(G))
    C.add_nodes_from(range(len(components)))
    
    # Map available edges to the component graph
    comp_dict = {n: i for i, comp in enumerate(components) for n in comp}
    comp_edges = defaultdict(list)
    for (u, v), w in zip(avail_uv, avail_w):
        cu, cv = comp_dict.get(u), comp_dict.get(v)
        if cu is not None and cv is not None and cu != cv:
            comp_edges[_ordered(cu, cv)].append((u, v, w))
    
    # Find the minimum spanning tree of the component graph
    C.add_weighted_edges_from((cu, cv, min(e[2] for e in edges))
                              for (cu, cv), edges in comp_edges.items())
    mst_edges = nx.minimum_spanning_tree(C).edges(data=True)
    
    # Map the MST edges back to the original graph
    for u, v, _ in mst_edges:
        edge = min(comp_edges[_ordered(u, v)], key=lambda x: x[2])
        yield edge[:2]


@nx._dispatchable
def unconstrained_bridge_augmentation(G):
    """Finds an optimal 2-edge-augmentation of G using the fewest edges.

    This is an implementation of the algorithm detailed in [1]_.
    The basic idea is to construct a meta-graph of bridge-ccs, connect leaf
    nodes of the trees to connect the entire graph, and finally connect the
    leafs of the tree in dfs-preorder to bridge connect the entire graph.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    Yields
    ------
    edge : tuple
        Edges in the bridge augmentation of G

    Notes
    -----
    Input: a graph G.
    First find the bridge components of G and collapse each bridge-cc into a
    node of a metagraph graph C, which is guaranteed to be a forest of trees.

    C contains p "leafs" --- nodes with exactly one incident edge.
    C contains q "isolated nodes" --- nodes with no incident edges.

    Theorem: If p + q > 1, then at least :math:`ceil(p / 2) + q` edges are
        needed to bridge connect C. This algorithm achieves this min number.

    The method first adds enough edges to make G into a tree and then pairs
    leafs in a simple fashion.

    Let n be the number of trees in C. Let v(i) be an isolated vertex in the
    i-th tree if one exists, otherwise it is a pair of distinct leafs nodes
    in the i-th tree. Alternating edges from these sets (i.e.  adding edges
    A1 = [(v(i)[0], v(i + 1)[1]), v(i + 1)[0], v(i + 2)[1])...]) connects C
    into a tree T. This tree has p' = p + 2q - 2(n -1) leafs and no isolated
    vertices. A1 has n - 1 edges. The next step finds ceil(p' / 2) edges to
    biconnect any tree with p' leafs.

    Convert T into an arborescence T' by picking an arbitrary root node with
    degree >= 2 and directing all edges away from the root. Note the
    implementation implicitly constructs T'.

    The leafs of T are the nodes with no existing edges in T'.
    Order the leafs of T' by DFS preorder. Then break this list in half
    and add the zipped pairs to A2.

    The set A = A1 + A2 is the minimum augmentation in the metagraph.

    To convert this to edges in the original graph

    References
    ----------
    .. [1] Eswaran, Kapali P., and R. Endre Tarjan. (1975) Augmentation problems.
        http://epubs.siam.org/doi/abs/10.1137/0205044

    See Also
    --------
    :func:`bridge_augmentation`
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))
    >>> sorted(unconstrained_bridge_augmentation(G))
    [(1, 7)]
    >>> G = nx.path_graph((1, 2, 3, 2, 4, 5, 6, 7))
    >>> sorted(unconstrained_bridge_augmentation(G))
    [(1, 3), (3, 7)]
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2)])
    >>> G.add_node(4)
    >>> sorted(unconstrained_bridge_augmentation(G))
    [(1, 4), (4, 0)]
    """
    # Find bridge components
    bridge_ccs = list(nx.connectivity.bridge_components(G))
    C = collapse(G, bridge_ccs)
    
    # Classify nodes in C
    isolated = set(n for n, d in C.degree() if d == 0)
    leafs = set(n for n, d in C.degree() if d == 1)
    
    # Connect C into a tree T
    A1 = []
    trees = list(nx.connected_components(C))
    for i in range(len(trees) - 1):
        u = next(iter(isolated & set(trees[i]))) if isolated & set(trees[i]) else next(iter(leafs & set(trees[i])))
        v = next(iter(isolated & set(trees[i+1]))) if isolated & set(trees[i+1]) else next(iter(leafs & set(trees[i+1])))
        A1.append((u, v))
    
    # Convert T to an arborescence T'
    T = nx.Graph(A1)
    root = next(n for n in T if T.degree(n) > 1)
    T = nx.dfs_tree(T, root)
    
    # Find leaf pairs in T'
    leafs = [n for n in T if T.out_degree(n) == 0]
    half = (len(leafs) + 1) // 2
    A2 = list(zip(leafs[:half], leafs[half:]))
    
    # Convert meta-edges to original graph edges
    mapping = C.graph['mapping']
    for u, v in A1 + A2:
        yield (next(iter(mapping[u])), next(iter(mapping[v])))


@nx._dispatchable
def weighted_bridge_augmentation(G, avail, weight=None):
    """Finds an approximate min-weight 2-edge-augmentation of G.

    This is an implementation of the approximation algorithm detailed in [1]_.
    It chooses a set of edges from avail to add to G that renders it
    2-edge-connected if such a subset exists.  This is done by finding a
    minimum spanning arborescence of a specially constructed metagraph.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    avail : set of 2 or 3 tuples.
        candidate edges (with optional weights) to choose from

    weight : string
        key to use to find weights if avail is a set of 3-tuples where the
        third item in each tuple is a dictionary.

    Yields
    ------
    edge : tuple
        Edges in the subset of avail chosen to bridge augment G.

    Notes
    -----
    Finding a weighted 2-edge-augmentation is NP-hard.
    Any edge not in ``avail`` is considered to have a weight of infinity.
    The approximation factor is 2 if ``G`` is connected and 3 if it is not.
    Runs in :math:`O(m + n log(n))` time

    References
    ----------
    .. [1] Khuller, Samir, and Ramakrishna Thurimella. (1993) Approximation
        algorithms for graph augmentation.
        http://www.sciencedirect.com/science/article/pii/S0196677483710102

    See Also
    --------
    :func:`bridge_augmentation`
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.path_graph((1, 2, 3, 4))
    >>> # When the weights are equal, (1, 4) is the best
    >>> avail = [(1, 4, 1), (1, 3, 1), (2, 4, 1)]
    >>> sorted(weighted_bridge_augmentation(G, avail))
    [(1, 4)]
    >>> # Giving (1, 4) a high weight makes the two edge solution the best.
    >>> avail = [(1, 4, 1000), (1, 3, 1), (2, 4, 1)]
    >>> sorted(weighted_bridge_augmentation(G, avail))
    [(1, 3), (2, 4)]
    >>> # ------
    >>> G = nx.path_graph((1, 2, 3, 4))
    >>> G.add_node(5)
    >>> avail = [(1, 5, 11), (2, 5, 10), (4, 3, 1), (4, 5, 1)]
    >>> sorted(weighted_bridge_augmentation(G, avail=avail))
    [(1, 5), (4, 5)]
    >>> avail = [(1, 5, 11), (2, 5, 10), (4, 3, 1), (4, 5, 51)]
    >>> sorted(weighted_bridge_augmentation(G, avail=avail))
    [(1, 5), (2, 5), (4, 5)]
    """
    avail_uv, avail_w = _unpack_available_edges(avail, weight=weight)
    
    # Find bridge components and construct metagraph
    bridge_ccs = list(nx.connectivity.bridge_components(G))
    C = collapse(G, bridge_ccs)
    mapping = C.graph['mapping']
    
    # Find the minimum spanning arborescence of the metagraph
    D = nx.DiGraph()
    for i in range(len(bridge_ccs)):
        D.add_node(i)
    
    for (u, v), w in zip(avail_uv, avail_w):
        i, j = mapping[u], mapping[v]
        if i != j:
            D.add_edge(i, j, weight=w)
            D.add_edge(j, i, weight=w)
    
    # Find the minimum spanning arborescence
    msa_edges = nx.minimum_spanning_arborescence(D, preserve_attrs=True)
    
    # Convert meta-edges back to original graph edges
    for u, v, data in msa_edges.edges(data=True):
        yield next((e for e in avail if mapping[e[0]] == u and mapping[e[1]] == v), None)


def _minimum_rooted_branching(D, root):
    """Helper function to compute a minimum rooted branching (aka rooted
    arborescence)

    Before the branching can be computed, the directed graph must be rooted by
    removing the predecessors of root.

    A branching / arborescence of rooted graph G is a subgraph that contains a
    directed path from the root to every other vertex. It is the directed
    analog of the minimum spanning tree problem.

    References
    ----------
    [1] Khuller, Samir (2002) Advanced Algorithms Lecture 24 Notes.
    https://web.archive.org/web/20121030033722/https://www.cs.umd.edu/class/spring2011/cmsc651/lec07.pdf
    """
    pass


@nx._dispatchable(returns_graph=True)
def collapse(G, grouped_nodes):
    """Collapses each group of nodes into a single node.

    This is similar to condensation, but works on undirected graphs.

    Parameters
    ----------
    G : NetworkX Graph

    grouped_nodes:  list or generator
       Grouping of nodes to collapse. The grouping must be disjoint.
       If grouped_nodes are strongly_connected_components then this is
       equivalent to :func:`condensation`.

    Returns
    -------
    C : NetworkX Graph
       The collapsed graph C of G with respect to the node grouping.  The node
       labels are integers corresponding to the index of the component in the
       list of grouped_nodes.  C has a graph attribute named 'mapping' with a
       dictionary mapping the original nodes to the nodes in C to which they
       belong.  Each node in C also has a node attribute 'members' with the set
       of original nodes in G that form the group that the node in C
       represents.

    Examples
    --------
    >>> # Collapses a graph using disjoint groups, but not necessarily connected
    >>> G = nx.Graph([(1, 0), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (5, 7)])
    >>> G.add_node("A")
    >>> grouped_nodes = [{0, 1, 2, 3}, {5, 6, 7}]
    >>> C = collapse(G, grouped_nodes)
    >>> members = nx.get_node_attributes(C, "members")
    >>> sorted(members.keys())
    [0, 1, 2, 3]
    >>> member_values = set(map(frozenset, members.values()))
    >>> assert {0, 1, 2, 3} in member_values
    >>> assert {4} in member_values
    >>> assert {5, 6, 7} in member_values
    >>> assert {"A"} in member_values
    """
    pass


@nx._dispatchable
def complement_edges(G):
    """Returns only the edges in the complement of G

    Parameters
    ----------
    G : NetworkX Graph

    Yields
    ------
    edge : tuple
        Edges in the complement of G

    Examples
    --------
    >>> G = nx.path_graph((1, 2, 3, 4))
    >>> sorted(complement_edges(G))
    [(1, 3), (1, 4), (2, 4)]
    >>> G = nx.path_graph((1, 2, 3, 4), nx.DiGraph())
    >>> sorted(complement_edges(G))
    [(1, 3), (1, 4), (2, 1), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)]
    >>> G = nx.complete_graph(1000)
    >>> sorted(complement_edges(G))
    []
    """
    pass


def _compat_shuffle(rng, input):
    """wrapper around rng.shuffle for python 2 compatibility reasons"""
    pass


@not_implemented_for('multigraph')
@not_implemented_for('directed')
@py_random_state(4)
@nx._dispatchable
def greedy_k_edge_augmentation(G, k, avail=None, weight=None, seed=None):
    """Greedy algorithm for finding a k-edge-augmentation

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    k : integer
        Desired edge connectivity

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Yields
    ------
    edge : tuple
        Edges in the greedy augmentation of G

    Notes
    -----
    The algorithm is simple. Edges are incrementally added between parts of the
    graph that are not yet locally k-edge-connected. Then edges are from the
    augmenting set are pruned as long as local-edge-connectivity is not broken.

    This algorithm is greedy and does not provide optimality guarantees. It
    exists only to provide :func:`k_edge_augmentation` with the ability to
    generate a feasible solution for arbitrary k.

    See Also
    --------
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))
    >>> sorted(greedy_k_edge_augmentation(G, k=2))
    [(1, 7)]
    >>> sorted(greedy_k_edge_augmentation(G, k=1, avail=[]))
    []
    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))
    >>> avail = {(u, v): 1 for (u, v) in complement_edges(G)}
    >>> # randomized pruning process can produce different solutions
    >>> sorted(greedy_k_edge_augmentation(G, k=4, avail=avail, seed=2))
    [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 4), (2, 6), (3, 7), (5, 7)]
    >>> sorted(greedy_k_edge_augmentation(G, k=4, avail=avail, seed=3))
    [(1, 3), (1, 5), (1, 6), (2, 4), (2, 6), (3, 7), (4, 7), (5, 7)]
    """
    import random
    random.seed(seed)

    if avail is None:
        avail = list(complement_edges(G))
    else:
        avail_uv, _ = _unpack_available_edges(avail, weight=weight)
        avail = list(avail_uv)

    H = G.copy()
    
    # Add edges greedily until we are k-edge-connected
    random.shuffle(avail)
    for u, v in avail:
        if not is_locally_k_edge_connected(H, u, v, k):
            H.add_edge(u, v)
            yield (u, v)
    
    # Remove edges as long as we maintain k-edge-connectivity
    edges = list(H.edges())
    random.shuffle(edges)
    for u, v in edges:
        if u in G and v in G and G.has_edge(u, v):
            continue
        H.remove_edge(u, v)
        if is_k_edge_connected(H, k):
            continue
        H.add_edge(u, v)
