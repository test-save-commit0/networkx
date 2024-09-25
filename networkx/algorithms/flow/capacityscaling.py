"""
Capacity scaling minimum cost flow algorithm.
"""
__all__ = ['capacity_scaling']
from itertools import chain
from math import log
import networkx as nx
from ...utils import BinaryHeap, arbitrary_element, not_implemented_for


def _detect_unboundedness(R):
    """Detect infinite-capacity negative cycles."""
    for cycle in nx.simple_cycles(R):
        if all(R[u][v].get('capacity', float('inf')) == float('inf') for u, v in zip(cycle, cycle[1:] + [cycle[0]])):
            if sum(R[u][v].get('weight', 0) for u, v in zip(cycle, cycle[1:] + [cycle[0]])) < 0:
                return True
    return False


@not_implemented_for('undirected')
def _build_residual_network(G, demand, capacity, weight):
    """Build a residual network and initialize a zero flow."""
    R = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        cap = data.get(capacity, float('inf'))
        w = data.get(weight, 0)
        R.add_edge(u, v, capacity=cap, weight=w)
        R.add_edge(v, u, capacity=0, weight=-w)

    for node, node_demand in G.nodes(data=demand):
        R.nodes[node]['demand'] = node_demand

    return R


def _build_flow_dict(G, R, capacity, weight):
    """Build a flow dictionary from a residual network."""
    flow_dict = {n: {} for n in G}
    for u, v, data in G.edges(data=True):
        if R.has_edge(u, v):
            flow_dict[u][v] = max(0, data.get(capacity, float('inf')) - R[u][v].get('capacity', 0))
        else:
            flow_dict[u][v] = data.get(capacity, float('inf'))
    return flow_dict


@nx._dispatchable(node_attrs='demand', edge_attrs={'capacity': float('inf'),
    'weight': 0})
def capacity_scaling(G, demand='demand', capacity='capacity', weight='weight', heap=BinaryHeap):
    """Find a minimum cost flow satisfying all demands in digraph G.

    This is a capacity scaling successive shortest augmenting path algorithm.

    G is a digraph with edge costs and capacities and in which nodes
    have demand, i.e., they want to send or receive some amount of
    flow. A negative demand means that the node wants to send flow, a
    positive demand means that the node want to receive flow. A flow on
    the digraph G satisfies all demand if the net flow into each node
    is equal to the demand of that node.

    Parameters
    ----------
    G : NetworkX graph
        DiGraph or MultiDiGraph on which a minimum cost flow satisfying all
        demands is to be found.

    demand : string
        Nodes of the graph G are expected to have an attribute demand
        that indicates how much flow a node wants to send (negative
        demand) or receive (positive demand). Note that the sum of the
        demands should be 0 otherwise the problem in not feasible. If
        this attribute is not present, a node is considered to have 0
        demand. Default value: 'demand'.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    weight : string
        Edges of the graph G are expected to have an attribute weight
        that indicates the cost incurred by sending one unit of flow on
        that edge. If not present, the weight is considered to be 0.
        Default value: 'weight'.

    heap : class
        Type of heap to be used in the algorithm. It should be a subclass of
        :class:`MinHeap` or implement a compatible interface.

        If a stock heap implementation is to be used, :class:`BinaryHeap` is
        recommended over :class:`PairingHeap` for Python implementations without
        optimized attribute accesses (e.g., CPython) despite a slower
        asymptotic running time. For Python implementations with optimized
        attribute accesses (e.g., PyPy), :class:`PairingHeap` provides better
        performance. Default value: :class:`BinaryHeap`.

    Returns
    -------
    flowCost : integer
        Cost of a minimum cost flow satisfying all demands.

    flowDict : dictionary
        If G is a digraph, a dict-of-dicts keyed by nodes such that
        flowDict[u][v] is the flow on edge (u, v).
        If G is a MultiDiGraph, a dict-of-dicts-of-dicts keyed by nodes
        so that flowDict[u][v][key] is the flow on edge (u, v, key).

    Raises
    ------
    NetworkXError
        This exception is raised if the input graph is not directed,
        not connected.

    NetworkXUnfeasible
        This exception is raised in the following situations:

            * The sum of the demands is not zero. Then, there is no
              flow satisfying all demands.
            * There is no flow satisfying all demand.

    NetworkXUnbounded
        This exception is raised if the digraph G has a cycle of
        negative cost and infinite capacity. Then, the cost of a flow
        satisfying all demands is unbounded below.

    Notes
    -----
    This algorithm does not work if edge weights are floating-point numbers.

    See also
    --------
    :meth:`network_simplex`

    Examples
    --------
    A simple example of a min cost flow problem.

    >>> G = nx.DiGraph()
    >>> G.add_node("a", demand=-5)
    >>> G.add_node("d", demand=5)
    >>> G.add_edge("a", "b", weight=3, capacity=4)
    >>> G.add_edge("a", "c", weight=6, capacity=10)
    >>> G.add_edge("b", "d", weight=1, capacity=9)
    >>> G.add_edge("c", "d", weight=2, capacity=5)
    >>> flowCost, flowDict = nx.capacity_scaling(G)
    >>> flowCost
    24
    >>> flowDict
    {'a': {'b': 4, 'c': 1}, 'd': {}, 'b': {'d': 4}, 'c': {'d': 1}}

    It is possible to change the name of the attributes used for the
    algorithm.

    >>> G = nx.DiGraph()
    >>> G.add_node("p", spam=-4)
    >>> G.add_node("q", spam=2)
    >>> G.add_node("a", spam=-2)
    >>> G.add_node("d", spam=-1)
    >>> G.add_node("t", spam=2)
    >>> G.add_node("w", spam=3)
    >>> G.add_edge("p", "q", cost=7, vacancies=5)
    >>> G.add_edge("p", "a", cost=1, vacancies=4)
    >>> G.add_edge("q", "d", cost=2, vacancies=3)
    >>> G.add_edge("t", "q", cost=1, vacancies=2)
    >>> G.add_edge("a", "t", cost=2, vacancies=4)
    >>> G.add_edge("d", "w", cost=3, vacancies=4)
    >>> G.add_edge("t", "w", cost=4, vacancies=1)
    >>> flowCost, flowDict = nx.capacity_scaling(
    ...     G, demand="spam", capacity="vacancies", weight="cost"
    ... )
    >>> flowCost
    37
    >>> flowDict
    {'p': {'q': 2, 'a': 2}, 'q': {'d': 1}, 'a': {'t': 4}, 'd': {'w': 2}, 't': {'q': 1, 'w': 1}, 'w': {}}
    """
    if not nx.is_directed(G):
        raise nx.NetworkXError("Capacity scaling algorithm works only for directed graphs.")

    if not nx.is_weakly_connected(G):
        raise nx.NetworkXError("Graph is not connected.")

    R = _build_residual_network(G, demand, capacity, weight)

    if _detect_unboundedness(R):
        raise nx.NetworkXUnbounded("Negative cost cycle of infinite capacity found. Flow cost is unbounded below.")

    inf = float('inf')
    f = {u: {v: 0 for v in G[u]} for u in G}
    c = sum(abs(R.nodes[n]['demand']) for n in R if R.nodes[n]['demand'] != 0)
    U = 2 ** int(log(c, 2))

    while U >= 1:
        delta = {}
        for u in R:
            for v, e in R[u].items():
                cap = e.get('capacity', inf)
                if cap >= U:
                    delta[u, v] = e.get(weight, 0)

        while True:
            T = nx.DiGraph()
            for u, v in delta:
                T.add_edge(u, v, weight=delta[u, v])

            try:
                path = nx.shortest_path(T, weight='weight')
                path_edges = list(zip(path[:-1], path[1:]))
                flow = min(R[u][v]['capacity'] for u, v in path_edges)
                for u, v in path_edges:
                    if (v, u) in f[v]:
                        f[v][u] -= flow
                    else:
                        f[u][v] += flow
                    R[u][v]['capacity'] -= flow
                    R[v][u]['capacity'] += flow
                    if R[u][v]['capacity'] < U:
                        del delta[u, v]
            except nx.NetworkXNoPath:
                break

        U //= 2

    if sum(R.nodes[n]['demand'] for n in R) != 0:
        raise nx.NetworkXUnfeasible("Total node demand is not zero. No flow satisfies all demands.")

    flowDict = _build_flow_dict(G, R, capacity, weight)
    flowCost = sum(flowDict[u][v] * G[u][v].get(weight, 0) for u in flowDict for v in flowDict[u])

    return flowCost, flowDict
