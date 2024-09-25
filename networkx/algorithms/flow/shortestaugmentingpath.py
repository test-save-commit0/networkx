"""
Shortest augmenting path algorithm for maximum flow problems.
"""
from collections import deque
import networkx as nx
from .edmondskarp import edmonds_karp_core
from .utils import CurrentEdge, build_residual_network
__all__ = ['shortest_augmenting_path']


def shortest_augmenting_path_impl(G, s, t, capacity, residual, two_phase,
    cutoff):
    """Implementation of the shortest augmenting path algorithm."""
    if residual is None:
        R = build_residual_network(G, capacity)
    else:
        R = residual

    # Initialize flow to zero
    for u in R:
        for e in R[u].values():
            e['flow'] = 0

    if cutoff is None:
        cutoff = float('inf')

    R_nodes = R.nodes
    R_succ = R.succ

    def augment(path):
        """Augment flow along a path from s to t."""
        # Find minimum residual capacity along the path
        flow = min(R_succ[u][v]['capacity'] - R_succ[u][v]['flow']
                   for u, v in zip(path, path[1:]))
        # Augment flow along the path
        for u, v in zip(path, path[1:]):
            edge = R_succ[u][v]
            edge['flow'] += flow
            R_succ[v][u]['flow'] -= flow
        return flow

    def bidirectional_bfs():
        """Bidirectional breadth-first search for an augmenting path."""
        pred = {s: None}
        succ = {t: None}
        forward = {s: 0}
        backward = {t: 0}
        forward_fringe = deque([(s, 0)])
        backward_fringe = deque([(t, 0)])
        while forward_fringe and backward_fringe:
            if len(forward_fringe) <= len(backward_fringe):
                u, d = forward_fringe.popleft()
                for v, edge in R_succ[u].items():
                    if v not in forward:
                        if edge['flow'] < edge['capacity']:
                            forward[v] = d + 1
                            pred[v] = u
                            forward_fringe.append((v, d + 1))
                            if v in backward:
                                return v, pred, succ
            else:
                u, d = backward_fringe.popleft()
                for v, edge in R.pred[u].items():
                    if v not in backward:
                        if edge['flow'] < edge['capacity']:
                            backward[v] = d + 1
                            succ[v] = u
                            backward_fringe.append((v, d + 1))
                            if v in forward:
                                return v, pred, succ
        return None, None, None

    flow_value = 0
    while flow_value < cutoff:
        v, pred, succ = bidirectional_bfs()
        if pred is None:
            break
        path = [v]
        u = v
        while u != s:
            u = pred[u]
            path.append(u)
        path.reverse()
        u = v
        while u != t:
            u = succ[u]
            path.append(u)
        flow_value += augment(path)

    R.graph['flow_value'] = flow_value
    return R


@nx._dispatchable(edge_attrs={'capacity': float('inf')}, returns_graph=True)
def shortest_augmenting_path(G, s, t, capacity='capacity', residual=None,
    value_only=False, two_phase=False, cutoff=None):
    """Find a maximum single-commodity flow using the shortest augmenting path
    algorithm.

    This function returns the residual network resulting after computing
    the maximum flow. See below for details about the conventions
    NetworkX uses for defining residual networks.

    This algorithm has a running time of $O(n^2 m)$ for $n$ nodes and $m$
    edges.

    Parameters
    ----------
    G : NetworkX graph
        Edges of the graph are expected to have an attribute called
        'capacity'. If this attribute is not present, the edge is
        considered to have infinite capacity.

    s : node
        Source node for the flow.

    t : node
        Sink node for the flow.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    residual : NetworkX graph
        Residual network on which the algorithm is to be executed. If None, a
        new residual network is created. Default value: None.

    value_only : bool
        If True compute only the value of the maximum flow. This parameter
        will be ignored by this algorithm because it is not applicable.

    two_phase : bool
        If True, a two-phase variant is used. The two-phase variant improves
        the running time on unit-capacity networks from $O(nm)$ to
        $O(\\min(n^{2/3}, m^{1/2}) m)$. Default value: False.

    cutoff : integer, float
        If specified, the algorithm will terminate when the flow value reaches
        or exceeds the cutoff. In this case, it may be unable to immediately
        determine a minimum cut. Default value: None.

    Returns
    -------
    R : NetworkX DiGraph
        Residual network after computing the maximum flow.

    Raises
    ------
    NetworkXError
        The algorithm does not support MultiGraph and MultiDiGraph. If
        the input graph is an instance of one of these two classes, a
        NetworkXError is raised.

    NetworkXUnbounded
        If the graph has a path of infinite capacity, the value of a
        feasible flow on the graph is unbounded above and the function
        raises a NetworkXUnbounded.

    See also
    --------
    :meth:`maximum_flow`
    :meth:`minimum_cut`
    :meth:`edmonds_karp`
    :meth:`preflow_push`

    Notes
    -----
    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. If :samp:`cutoff` is not
    specified, reachability to :samp:`t` using only edges :samp:`(u, v)` such
    that :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    Examples
    --------
    >>> from networkx.algorithms.flow import shortest_augmenting_path

    The functions that implement flow algorithms and output a residual
    network, such as this one, are not imported to the base NetworkX
    namespace, so you have to explicitly import them from the flow package.

    >>> G = nx.DiGraph()
    >>> G.add_edge("x", "a", capacity=3.0)
    >>> G.add_edge("x", "b", capacity=1.0)
    >>> G.add_edge("a", "c", capacity=3.0)
    >>> G.add_edge("b", "c", capacity=5.0)
    >>> G.add_edge("b", "d", capacity=4.0)
    >>> G.add_edge("d", "e", capacity=2.0)
    >>> G.add_edge("c", "y", capacity=2.0)
    >>> G.add_edge("e", "y", capacity=3.0)
    >>> R = shortest_augmenting_path(G, "x", "y")
    >>> flow_value = nx.maximum_flow_value(G, "x", "y")
    >>> flow_value
    3.0
    >>> flow_value == R.graph["flow_value"]
    True

    """
    if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
        raise nx.NetworkXError("Shortest augmenting path algorithm does not support MultiGraph and MultiDiGraph.")

    if s not in G:
        raise nx.NetworkXError(f"Source node {s} not in graph")
    if t not in G:
        raise nx.NetworkXError(f"Sink node {t} not in graph")

    if s == t:
        raise nx.NetworkXError("Source and sink are the same node")

    R = shortest_augmenting_path_impl(G, s, t, capacity, residual, two_phase, cutoff)

    return R
