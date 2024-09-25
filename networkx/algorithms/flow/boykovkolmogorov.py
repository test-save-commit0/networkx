"""
Boykov-Kolmogorov algorithm for maximum flow problems.
"""
from collections import deque
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
__all__ = ['boykov_kolmogorov']


@nx._dispatchable(edge_attrs={'capacity': float('inf')}, returns_graph=True)
def boykov_kolmogorov(G, s, t, capacity='capacity', residual=None,
    value_only=False, cutoff=None):
    """Find a maximum single-commodity flow using Boykov-Kolmogorov algorithm.

    This function returns the residual network resulting after computing
    the maximum flow. See below for details about the conventions
    NetworkX uses for defining residual networks.

    This algorithm has worse case complexity $O(n^2 m |C|)$ for $n$ nodes, $m$
    edges, and $|C|$ the cost of the minimum cut [1]_. This implementation
    uses the marking heuristic defined in [2]_ which improves its running
    time in many practical problems.

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
    >>> from networkx.algorithms.flow import boykov_kolmogorov

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
    >>> R = boykov_kolmogorov(G, "x", "y")
    >>> flow_value = nx.maximum_flow_value(G, "x", "y")
    >>> flow_value
    3.0
    >>> flow_value == R.graph["flow_value"]
    True

    A nice feature of the Boykov-Kolmogorov algorithm is that a partition
    of the nodes that defines a minimum cut can be easily computed based
    on the search trees used during the algorithm. These trees are stored
    in the graph attribute `trees` of the residual network.

    >>> source_tree, target_tree = R.graph["trees"]
    >>> partition = (set(source_tree), set(G) - set(source_tree))

    Or equivalently:

    >>> partition = (set(G) - set(target_tree), set(target_tree))

    References
    ----------
    .. [1] Boykov, Y., & Kolmogorov, V. (2004). An experimental comparison
           of min-cut/max-flow algorithms for energy minimization in vision.
           Pattern Analysis and Machine Intelligence, IEEE Transactions on,
           26(9), 1124-1137.
           https://doi.org/10.1109/TPAMI.2004.60

    .. [2] Vladimir Kolmogorov. Graph-based Algorithms for Multi-camera
           Reconstruction Problem. PhD thesis, Cornell University, CS Department,
           2003. pp. 109-114.
           https://web.archive.org/web/20170809091249/https://pub.ist.ac.at/~vnk/papers/thesis.pdf

    """
    if not isinstance(G, nx.Graph):
        raise nx.NetworkXError("Boykov-Kolmogorov algorithm not applicable for MultiGraph or MultiDiGraph ")

    if s not in G:
        raise nx.NetworkXError(f"Source {s} not in graph")
    if t not in G:
        raise nx.NetworkXError(f"Sink {t} not in graph")

    if residual is None:
        R = build_residual_network(G, capacity)
    else:
        R = residual

    # Initialize the residual network
    for u in R:
        for e in R[u].values():
            e['flow'] = 0

    inf = R.graph['inf']

    def grow():
        """Bidirectional breadth-first search for the growth stage."""
        while active:
            v = active.popleft()
            if v in source_tree:
                for w, attr in R[v].items():
                    if w not in source_tree and attr['capacity'] > attr['flow']:
                        source_tree[w] = v
                        active.append(w)
                        if w in target_tree:
                            return w  # Found an augmenting path
            else:  # v in target_tree
                for w in R.predecessors(v):
                    attr = R[w][v]
                    if w not in target_tree and attr['flow'] > 0:
                        target_tree[w] = v
                        active.append(w)
                        if w in source_tree:
                            return w  # Found an augmenting path
        return None

    def augment(v):
        """Augment flow along the path found by grow()."""
        path = [v]
        u = v
        while u != s:
            u = source_tree[u]
            path.append(u)
        u = v
        while u != t:
            u = target_tree[u]
            path.append(u)
        path.reverse()
        flow = inf
        for u, v in zip(path[:-1], path[1:]):
            if v in R[u]:
                flow = min(flow, R[u][v]['capacity'] - R[u][v]['flow'])
            else:
                flow = min(flow, R[v][u]['flow'])
        for u, v in zip(path[:-1], path[1:]):
            if v in R[u]:
                R[u][v]['flow'] += flow
                R[v][u]['flow'] -= flow
            else:
                R[v][u]['flow'] -= flow
                R[u][v]['flow'] += flow
        return flow

    def adopt():
        """Adopt orphans to maintain valid search trees."""
        while orphans:
            v = orphans.popleft()
            if v in source_tree:
                found = False
                for u in R.predecessors(v):
                    if u in source_tree and R[u][v]['capacity'] > R[u][v]['flow']:
                        source_tree[v] = u
                        found = True
                        break
                if not found:
                    for w in R[v]:
                        if w in source_tree:
                            if w != s:
                                orphans.append(w)
                            del source_tree[w]
                    del source_tree[v]
            else:  # v in target_tree
                found = False
                for u in R[v]:
                    if u in target_tree and R[v][u]['flow'] > 0:
                        target_tree[v] = u
                        found = True
                        break
                if not found:
                    for w, attr in R[v].items():
                        if w in target_tree:
                            if w != t:
                                orphans.append(w)
                            del target_tree[w]
                    del target_tree[v]

    # Initialize search trees and active set
    source_tree = {s: None}
    target_tree = {t: None}
    active = deque([s, t])
    orphans = deque()

    flow_value = 0
    while True:
        v = grow()
        if v is None:
            break
        flow_value += augment(v)
        adopt()
        if cutoff is not None and flow_value >= cutoff:
            break

    R.graph['flow_value'] = flow_value
    R.graph['trees'] = (source_tree, target_tree)
    return R
