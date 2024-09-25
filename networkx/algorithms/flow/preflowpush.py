"""
Highest-label preflow-push algorithm for maximum flow problems.
"""
from collections import deque
from itertools import islice
import networkx as nx
from ...utils import arbitrary_element
from .utils import CurrentEdge, GlobalRelabelThreshold, Level, build_residual_network, detect_unboundedness
__all__ = ['preflow_push']


def preflow_push_impl(G, s, t, capacity, residual, global_relabel_freq,
    value_only):
    """Implementation of the highest-label preflow-push algorithm."""
    def push(u, v):
        flow = min(R.nodes[u]['excess'], R[u][v]['capacity'] - R[u][v]['flow'])
        if flow > 0:
            R[u][v]['flow'] += flow
            R[v][u]['flow'] -= flow
            R.nodes[u]['excess'] -= flow
            R.nodes[v]['excess'] += flow
            return flow
        return 0

    def relabel(u):
        min_height = float('inf')
        for v in R[u]:
            if R[u][v]['flow'] < R[u][v]['capacity']:
                min_height = min(min_height, R.nodes[v]['height'])
        R.nodes[u]['height'] = min_height + 1

    def discharge(u):
        while R.nodes[u]['excess'] > 0:
            if not R.nodes[u]['current_edge']:
                relabel(u)
                R.nodes[u]['current_edge'] = CurrentEdge(R[u])
            else:
                v = R.nodes[u]['current_edge']
                if (R[u][v]['flow'] < R[u][v]['capacity'] and
                    R.nodes[u]['height'] > R.nodes[v]['height']):
                    push(u, v)
                else:
                    R.nodes[u]['current_edge'].move_to_next()

    def global_relabeling():
        heights = {node: float('inf') for node in R}
        heights[t] = 0
        q = deque([t])
        seen = set([t])
        while q:
            u = q.popleft()
            height = heights[u] + 1
            for v in R.predecessors(u):
                if v not in seen and R[v][u]['flow'] < R[v][u]['capacity']:
                    heights[v] = height
                    seen.add(v)
                    q.append(v)
        for node in R:
            R.nodes[node]['height'] = heights[node]
            R.nodes[node]['current_edge'] = CurrentEdge(R[node])

    if not isinstance(residual, nx.DiGraph):
        R = build_residual_network(G, capacity)
    else:
        R = residual

    # Initialize preflow
    R.graph['flow_value'] = 0
    for u in R:
        R.nodes[u]['excess'] = 0
        R.nodes[u]['height'] = 0
        R.nodes[u]['current_edge'] = CurrentEdge(R[u])
    R.nodes[s]['height'] = len(R)

    for u, v in R.out_edges(s):
        flow = R[u][v]['capacity']
        R[u][v]['flow'] = flow
        R[v][u]['flow'] = -flow
        R.nodes[u]['excess'] -= flow
        R.nodes[v]['excess'] += flow

    if global_relabel_freq is not None:
        global_relabel_threshold = GlobalRelabelThreshold(R, global_relabel_freq)

    # Main loop
    active = {u for u in R if u not in (s, t) and R.nodes[u]['excess'] > 0}
    while active:
        u = max(active, key=lambda x: R.nodes[x]['height'])
        discharge(u)
        active = {u for u in R if u not in (s, t) and R.nodes[u]['excess'] > 0}
        if global_relabel_freq is not None:
            if global_relabel_threshold():
                global_relabeling()

    if not value_only:
        return R
    else:
        return R.nodes[t]['excess']


@nx._dispatchable(edge_attrs={'capacity': float('inf')}, returns_graph=True)
def preflow_push(G, s, t, capacity='capacity', residual=None,
    global_relabel_freq=1, value_only=False):
    """Find a maximum single-commodity flow using the highest-label
    preflow-push algorithm.

    This function returns the residual network resulting after computing
    the maximum flow. See below for details about the conventions
    NetworkX uses for defining residual networks.

    This algorithm has a running time of $O(n^2 \\sqrt{m})$ for $n$ nodes and
    $m$ edges.


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

    global_relabel_freq : integer, float
        Relative frequency of applying the global relabeling heuristic to speed
        up the algorithm. If it is None, the heuristic is disabled. Default
        value: 1.

    value_only : bool
        If False, compute a maximum flow; otherwise, compute a maximum preflow
        which is enough for computing the maximum flow value. Default value:
        False.

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
    :meth:`shortest_augmenting_path`

    Notes
    -----
    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`. For each node :samp:`u` in :samp:`R`,
    :samp:`R.nodes[u]['excess']` represents the difference between flow into
    :samp:`u` and flow out of :samp:`u`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. Reachability to :samp:`t` using
    only edges :samp:`(u, v)` such that
    :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    Examples
    --------
    >>> from networkx.algorithms.flow import preflow_push

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
    >>> R = preflow_push(G, "x", "y")
    >>> flow_value = nx.maximum_flow_value(G, "x", "y")
    >>> flow_value == R.graph["flow_value"]
    True
    >>> # preflow_push also stores the maximum flow value
    >>> # in the excess attribute of the sink node t
    >>> flow_value == R.nodes["y"]["excess"]
    True
    >>> # For some problems, you might only want to compute a
    >>> # maximum preflow.
    >>> R = preflow_push(G, "x", "y", value_only=True)
    >>> flow_value == R.graph["flow_value"]
    True
    >>> flow_value == R.nodes["y"]["excess"]
    True

    """
    R = preflow_push_impl(G, s, t, capacity, residual, global_relabel_freq, value_only)
    
    if value_only:
        # R is the flow value in this case
        return R
    
    # Detect infinite-capacity paths
    if detect_unboundedness(R, s, t):
        raise nx.NetworkXUnbounded(
            "Infinite capacity path, flow unbounded above."
        )
    
    # Add the flow value to the graph
    R.graph['flow_value'] = R.nodes[t]['excess']
    
    return R
