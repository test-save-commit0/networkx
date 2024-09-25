"""Flow based node and edge disjoint paths."""
import networkx as nx
from networkx.algorithms.flow import edmonds_karp, preflow_push, shortest_augmenting_path
from networkx.exception import NetworkXNoPath
default_flow_func = edmonds_karp
from itertools import filterfalse as _filterfalse
from .utils import build_auxiliary_edge_connectivity, build_auxiliary_node_connectivity
__all__ = ['edge_disjoint_paths', 'node_disjoint_paths']


@nx._dispatchable(graphs={'G': 0, 'auxiliary?': 5}, preserve_edge_attrs={
    'auxiliary': {'capacity': float('inf')}})
def edge_disjoint_paths(G, s, t, flow_func=None, cutoff=None, auxiliary=
    None, residual=None):
    """Returns the edges disjoint paths between source and target.

    Edge disjoint paths are paths that do not share any edge. The
    number of edge disjoint paths between source and target is equal
    to their edge connectivity.

    Parameters
    ----------
    G : NetworkX graph

    s : node
        Source node for the flow.

    t : node
        Sink node for the flow.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. The choice of the default function
        may change from version to version and should not be relied on.
        Default value: None.

    cutoff : integer or None (default: None)
        Maximum number of paths to yield. If specified, the maximum flow
        algorithm will terminate when the flow value reaches or exceeds the
        cutoff. This only works for flows that support the cutoff parameter
        (most do) and is ignored otherwise.

    auxiliary : NetworkX DiGraph
        Auxiliary digraph to compute flow based edge connectivity. It has
        to have a graph attribute called mapping with a dictionary mapping
        node names in G and in the auxiliary digraph. If provided
        it will be reused instead of recreated. Default value: None.

    residual : NetworkX DiGraph
        Residual network to compute maximum flow. If provided it will be
        reused instead of recreated. Default value: None.

    Returns
    -------
    paths : generator
        A generator of edge independent paths.

    Raises
    ------
    NetworkXNoPath
        If there is no path between source and target.

    NetworkXError
        If source or target are not in the graph G.

    See also
    --------
    :meth:`node_disjoint_paths`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    Examples
    --------
    We use in this example the platonic icosahedral graph, which has node
    edge connectivity 5, thus there are 5 edge disjoint paths between any
    pair of nodes.

    >>> G = nx.icosahedral_graph()
    >>> len(list(nx.edge_disjoint_paths(G, 0, 6)))
    5


    If you need to compute edge disjoint paths on several pairs of
    nodes in the same graph, it is recommended that you reuse the
    data structures that NetworkX uses in the computation: the
    auxiliary digraph for edge connectivity, and the residual
    network for the underlying maximum flow computation.

    Example of how to compute edge disjoint paths among all pairs of
    nodes of the platonic icosahedral graph reusing the data
    structures.

    >>> import itertools
    >>> # You also have to explicitly import the function for
    >>> # building the auxiliary digraph from the connectivity package
    >>> from networkx.algorithms.connectivity import build_auxiliary_edge_connectivity
    >>> H = build_auxiliary_edge_connectivity(G)
    >>> # And the function for building the residual network from the
    >>> # flow package
    >>> from networkx.algorithms.flow import build_residual_network
    >>> # Note that the auxiliary digraph has an edge attribute named capacity
    >>> R = build_residual_network(H, "capacity")
    >>> result = {n: {} for n in G}
    >>> # Reuse the auxiliary digraph and the residual network by passing them
    >>> # as arguments
    >>> for u, v in itertools.combinations(G, 2):
    ...     k = len(list(nx.edge_disjoint_paths(G, u, v, auxiliary=H, residual=R)))
    ...     result[u][v] = k
    >>> all(result[u][v] == 5 for u, v in itertools.combinations(G, 2))
    True

    You can also use alternative flow algorithms for computing edge disjoint
    paths. For instance, in dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better than
    the default :meth:`edmonds_karp` which is faster for sparse
    networks with highly skewed degree distributions. Alternative flow
    functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> len(list(nx.edge_disjoint_paths(G, 0, 6, flow_func=shortest_augmenting_path)))
    5

    Notes
    -----
    This is a flow based implementation of edge disjoint paths. We compute
    the maximum flow between source and target on an auxiliary directed
    network. The saturated edges in the residual network after running the
    maximum flow algorithm correspond to edge disjoint paths between source
    and target in the original network. This function handles both directed
    and undirected graphs, and can use all flow algorithms from NetworkX flow
    package.

    """
    if auxiliary is None:
        H = build_auxiliary_edge_connectivity(G)
    else:
        H = auxiliary

    # The edge connectivity algorithm uses an auxiliary digraph.
    # H has a graph attribute mapping with a dict mapping node
    # names in G and in H
    mapping = H.graph['mapping']
    if (s not in mapping) or (t not in mapping):
        raise nx.NetworkXError('node %s or %s not in graph' % (s, t))
    
    # Maximum flow algorithm
    if flow_func is None:
        flow_func = default_flow_func

    # Compute maximum flow between source and target
    R = flow_func(H, mapping[s], mapping[t], capacity='capacity',
                  residual=residual, cutoff=cutoff, value_only=False)

    # Saturated edges in the residual network correspond to edge disjoint paths
    # between source and target in the original graph.
    cutset = [(u, v) for (u, v, d) in R.edges(data=True)
              if d['flow'] == d['capacity']]

    # Rebuild the edge disjoint paths from the cutset
    paths_found = []
    H_copy = H.copy()
    for u, v in cutset:
        path = [mapping[u], mapping[v]]
        H_copy.remove_edge(u, v)
        while path[0] != mapping[s]:
            prev = [e[0] for e in H_copy.in_edges(path[0])]
            path.insert(0, mapping[prev[0]])
        while path[-1] != mapping[t]:
            succ = [e[1] for e in H_copy.out_edges(path[-1])]
            path.append(mapping[succ[0]])
        paths_found.append(path)

    return _unique_everseen(paths_found)


@nx._dispatchable(graphs={'G': 0, 'auxiliary?': 5}, preserve_node_attrs={
    'auxiliary': {'id': None}}, preserve_graph_attrs={'auxiliary'})
def node_disjoint_paths(G, s, t, flow_func=None, cutoff=None, auxiliary=
    None, residual=None):
    """Computes node disjoint paths between source and target.

    Node disjoint paths are paths that only share their first and last
    nodes. The number of node independent paths between two nodes is
    equal to their local node connectivity.

    Parameters
    ----------
    G : NetworkX graph

    s : node
        Source node.

    t : node
        Target node.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The choice
        of the default function may change from version to version and
        should not be relied on. Default value: None.

    cutoff : integer or None (default: None)
        Maximum number of paths to yield. If specified, the maximum flow
        algorithm will terminate when the flow value reaches or exceeds the
        cutoff. This only works for flows that support the cutoff parameter
        (most do) and is ignored otherwise.

    auxiliary : NetworkX DiGraph
        Auxiliary digraph to compute flow based node connectivity. It has
        to have a graph attribute called mapping with a dictionary mapping
        node names in G and in the auxiliary digraph. If provided
        it will be reused instead of recreated. Default value: None.

    residual : NetworkX DiGraph
        Residual network to compute maximum flow. If provided it will be
        reused instead of recreated. Default value: None.

    Returns
    -------
    paths : generator
        Generator of node disjoint paths.

    Raises
    ------
    NetworkXNoPath
        If there is no path between source and target.

    NetworkXError
        If source or target are not in the graph G.

    Examples
    --------
    We use in this example the platonic icosahedral graph, which has node
    connectivity 5, thus there are 5 node disjoint paths between any pair
    of non neighbor nodes.

    >>> G = nx.icosahedral_graph()
    >>> len(list(nx.node_disjoint_paths(G, 0, 6)))
    5

    If you need to compute node disjoint paths between several pairs of
    nodes in the same graph, it is recommended that you reuse the
    data structures that NetworkX uses in the computation: the
    auxiliary digraph for node connectivity and node cuts, and the
    residual network for the underlying maximum flow computation.

    Example of how to compute node disjoint paths reusing the data
    structures:

    >>> # You also have to explicitly import the function for
    >>> # building the auxiliary digraph from the connectivity package
    >>> from networkx.algorithms.connectivity import build_auxiliary_node_connectivity
    >>> H = build_auxiliary_node_connectivity(G)
    >>> # And the function for building the residual network from the
    >>> # flow package
    >>> from networkx.algorithms.flow import build_residual_network
    >>> # Note that the auxiliary digraph has an edge attribute named capacity
    >>> R = build_residual_network(H, "capacity")
    >>> # Reuse the auxiliary digraph and the residual network by passing them
    >>> # as arguments
    >>> len(list(nx.node_disjoint_paths(G, 0, 6, auxiliary=H, residual=R)))
    5

    You can also use alternative flow algorithms for computing node disjoint
    paths. For instance, in dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better than
    the default :meth:`edmonds_karp` which is faster for sparse
    networks with highly skewed degree distributions. Alternative flow
    functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> len(list(nx.node_disjoint_paths(G, 0, 6, flow_func=shortest_augmenting_path)))
    5

    Notes
    -----
    This is a flow based implementation of node disjoint paths. We compute
    the maximum flow between source and target on an auxiliary directed
    network. The saturated edges in the residual network after running the
    maximum flow algorithm correspond to node disjoint paths between source
    and target in the original network. This function handles both directed
    and undirected graphs, and can use all flow algorithms from NetworkX flow
    package.

    See also
    --------
    :meth:`edge_disjoint_paths`
    :meth:`node_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    """
    if auxiliary is None:
        H = build_auxiliary_node_connectivity(G)
    else:
        H = auxiliary

    # The node connectivity algorithm uses an auxiliary digraph.
    # H has a graph attribute mapping with a dict mapping node
    # names in G and in H
    mapping = H.graph['mapping']
    if (s not in mapping) or (t not in mapping):
        raise nx.NetworkXError('node %s or %s not in graph' % (s, t))

    # Maximum flow algorithm
    if flow_func is None:
        flow_func = default_flow_func

    # Compute maximum flow between source and target
    R = flow_func(H, f'{mapping[s]}B', f'{mapping[t]}A', capacity='capacity',
                  residual=residual, cutoff=cutoff, value_only=False)

    # Saturated edges in the residual network correspond to node disjoint paths
    # between source and target in the original graph.
    cutset = [(u, v) for (u, v, d) in R.edges(data=True)
              if d['flow'] == d['capacity']]

    # Rebuild the node disjoint paths from the cutset
    paths_found = []
    H_copy = H.copy()
    for u, v in cutset:
        path = []
        if v.endswith('A'):  # We have reached the target
            path = [mapping[v[:-1]]]
        else:
            path = [mapping[u[:-1]], mapping[v[:-1]]]
        H_copy.remove_edge(u, v)
        
        # Extend the path to source
        curr = u
        while not curr.endswith('B'):
            prev = [e[0] for e in H_copy.in_edges(curr)][0]
            path.insert(0, mapping[prev[:-1]])
            curr = prev
        
        # Extend the path to target
        curr = v
        while not curr.endswith('A'):
            succ = [e[1] for e in H_copy.out_edges(curr)][0]
            path.append(mapping[succ[:-1]])
            curr = succ
        
        paths_found.append(path)

    return _unique_everseen(paths_found)


def _unique_everseen(iterable):
    """List unique elements, preserving order. Remember all elements ever seen."""
    seen = set()
    for element in iterable:
        if element not in seen:
            seen.add(element)
            yield element
