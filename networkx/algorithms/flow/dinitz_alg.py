"""
Dinitz' algorithm for maximum flow problems.
"""
from collections import deque
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
from networkx.utils import pairwise
__all__ = ['dinitz']


@nx._dispatchable(edge_attrs={'capacity': float('inf')}, returns_graph=True)
def dinitz(G, s, t, capacity='capacity', residual=None, value_only=False,
    cutoff=None):
    """Find a maximum single-commodity flow using Dinitz' algorithm.

    This function returns the residual network resulting after computing
    the maximum flow. See below for details about the conventions
    NetworkX uses for defining residual networks.

    This algorithm has a running time of $O(n^2 m)$ for $n$ nodes and $m$
    edges [1]_.


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
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

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
    >>> from networkx.algorithms.flow import dinitz

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
    >>> R = dinitz(G, "x", "y")
    >>> flow_value = nx.maximum_flow_value(G, "x", "y")
    >>> flow_value
    3.0
    >>> flow_value == R.graph["flow_value"]
    True

    References
    ----------
    .. [1] Dinitz' Algorithm: The Original Version and Even's Version.
           2006. Yefim Dinitz. In Theoretical Computer Science. Lecture
           Notes in Computer Science. Volume 3895. pp 218-240.
           https://doi.org/10.1007/11685654_10

    """
    if not nx.is_directed(G):
        raise nx.NetworkXError("Dinitz algorithm works only for directed graphs.")

    if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
        raise nx.NetworkXError("Dinitz algorithm does not support MultiGraph and MultiDiGraph.")

    if s not in G:
        raise nx.NetworkXError(f"Source {s} is not in graph")
    if t not in G:
        raise nx.NetworkXError(f"Sink {t} is not in graph")

    if residual is None:
        R = build_residual_network(G, capacity)
    else:
        R = residual

    # Initialize flow to 0
    nx.set_edge_attributes(R, 0, 'flow')

    def bfs():
        level = {s: 0}
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v, attr in R[u].items():
                if v not in level and attr['capacity'] > attr['flow']:
                    level[v] = level[u] + 1
                    queue.append(v)
                    if v == t:
                        return level
        return None

    def dfs(u, flow):
        if u == t:
            return flow
        for v, attr in R[u].items():
            if level[v] == level[u] + 1 and attr['capacity'] > attr['flow']:
                bottleneck = dfs(v, min(flow, attr['capacity'] - attr['flow']))
                if bottleneck > 0:
                    R[u][v]['flow'] += bottleneck
                    R[v][u]['flow'] -= bottleneck
                    return bottleneck
        return 0

    flow_value = 0
    while True:
        level = bfs()
        if level is None:
            break
        while True:
            flow = dfs(s, float('inf'))
            if flow == 0:
                break
            flow_value += flow
            if cutoff is not None and flow_value >= cutoff:
                break
        if cutoff is not None and flow_value >= cutoff:
            break

    R.graph['flow_value'] = flow_value
    return R
