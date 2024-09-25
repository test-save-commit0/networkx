"""
Utility classes and functions for network flow algorithms.
"""
from collections import deque
import networkx as nx
__all__ = ['CurrentEdge', 'Level', 'GlobalRelabelThreshold',
    'build_residual_network', 'detect_unboundedness', 'build_flow_dict']


class CurrentEdge:
    """Mechanism for iterating over out-edges incident to a node in a circular
    manner. StopIteration exception is raised when wraparound occurs.
    """
    __slots__ = '_edges', '_it', '_curr'

    def __init__(self, edges):
        self._edges = edges
        if self._edges:
            self._rewind()


class Level:
    """Active and inactive nodes in a level."""
    __slots__ = 'active', 'inactive'

    def __init__(self):
        self.active = set()
        self.inactive = set()


class GlobalRelabelThreshold:
    """Measurement of work before the global relabeling heuristic should be
    applied.
    """

    def __init__(self, n, m, freq):
        self._threshold = (n + m) / freq if freq else float('inf')
        self._work = 0


@nx._dispatchable(edge_attrs={'capacity': float('inf')}, returns_graph=True)
def build_residual_network(G, capacity):
    """Build a residual network and initialize a zero flow.

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

    """
    R = nx.DiGraph()
    R.add_nodes_from(G)

    inf = float('inf')
    R.graph['inf'] = inf

    for u, v, attr in G.edges(data=True):
        if u != v:
            r_capacity = attr.get(capacity, inf)
            if r_capacity == inf:
                r_capacity = R.graph['inf']

            if R.has_edge(u, v):
                R[u][v]['capacity'] += r_capacity
            else:
                R.add_edge(u, v, capacity=r_capacity, flow=0)

            if not R.has_edge(v, u):
                R.add_edge(v, u, capacity=0, flow=0)

    R.graph['flow_value'] = 0
    return R


@nx._dispatchable(graphs='R', preserve_edge_attrs={'R': {'capacity': float(
    'inf')}}, preserve_graph_attrs=True)
def detect_unboundedness(R, s, t):
    """Detect an infinite-capacity s-t path in R."""
    inf = R.graph['inf']
    queue = deque([(s, [])])
    visited = set()

    while queue:
        node, path = queue.popleft()
        if node == t:
            return path
        if node in visited:
            continue
        visited.add(node)

        for _, v, attr in R.edges(node, data=True):
            if attr['capacity'] == inf and v not in visited:
                new_path = path + [(node, v)]
                queue.append((v, new_path))

    return None


@nx._dispatchable(graphs={'G': 0, 'R': 1}, preserve_edge_attrs={'R': {
    'flow': None}})
def build_flow_dict(G, R):
    """Build a flow dictionary from a residual network."""
    flow_dict = {}

    for u in G:
        flow_dict[u] = {}
        for v, attr in G[u].items():
            if R.has_edge(u, v):
                flow_dict[u][v] = max(0, R[u][v]['flow'])
            else:
                flow_dict[u][v] = 0

    return flow_dict
