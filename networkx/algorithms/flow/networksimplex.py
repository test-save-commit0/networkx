"""
Minimum cost flow algorithms on directed connected graphs.
"""
__all__ = ['network_simplex']
from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for


class _DataEssentialsAndFunctions:

    def __init__(self, G, multigraph, demand='demand', capacity='capacity',
        weight='weight'):
        self.node_list = list(G)
        self.node_indices = {u: i for i, u in enumerate(self.node_list)}
        self.node_demands = [G.nodes[u].get(demand, 0) for u in self.node_list]
        self.edge_sources = []
        self.edge_targets = []
        if multigraph:
            self.edge_keys = []
        self.edge_indices = {}
        self.edge_capacities = []
        self.edge_weights = []
        if not multigraph:
            edges = G.edges(data=True)
        else:
            edges = G.edges(data=True, keys=True)
        inf = float('inf')
        edges = (e for e in edges if e[0] != e[1] and e[-1].get(capacity,
            inf) != 0)
        for i, e in enumerate(edges):
            self.edge_sources.append(self.node_indices[e[0]])
            self.edge_targets.append(self.node_indices[e[1]])
            if multigraph:
                self.edge_keys.append(e[2])
            self.edge_indices[e[:-1]] = i
            self.edge_capacities.append(e[-1].get(capacity, inf))
            self.edge_weights.append(e[-1].get(weight, 0))
        self.edge_count = None
        self.edge_flow = None
        self.node_potentials = None
        self.parent = None
        self.parent_edge = None
        self.subtree_size = None
        self.next_node_dft = None
        self.prev_node_dft = None
        self.last_descendent_dft = None
        self._spanning_tree_initialized = False

    def find_apex(self, p, q):
        """
        Find the lowest common ancestor of nodes p and q in the spanning tree.
        """
        while p != q:
            if self.subtree_size[p] < self.subtree_size[q]:
                p = self.parent[p]
            else:
                q = self.parent[q]
        return p

    def trace_path(self, p, w):
        """
        Returns the nodes and edges on the path from node p to its ancestor w.
        """
        Wn = [p]
        We = []
        while p != w:
            e = self.parent_edge[p]
            We.append(e)
            p = self.parent[p]
            Wn.append(p)
        return Wn, We

    def find_cycle(self, i, p, q):
        """
        Returns the nodes and edges on the cycle containing edge i == (p, q)
        when the latter is added to the spanning tree.

        The cycle is oriented in the direction from p to q.
        """
        w = self.find_apex(p, q)
        Wn, We = self.trace_path(p, w)
        Wn.reverse()
        We.reverse()
        Wn2, We2 = self.trace_path(q, w)
        Wn += Wn2[1:]
        We.append(i)
        We += We2
        return Wn, We

    def augment_flow(self, Wn, We, f):
        """
        Augment f units of flow along a cycle represented by Wn and We.
        """
        for i, (p, q) in enumerate(zip(Wn, Wn[1:])):
            if p == self.parent[q]:
                self.edge_flow[We[i]] += f
            else:
                self.edge_flow[We[i]] -= f

    def trace_subtree(self, p):
        """
        Yield the nodes in the subtree rooted at a node p.
        """
        yield p
        for q in self.node_list:
            if self.parent[q] == p:
                yield from self.trace_subtree(q)

    def remove_edge(self, s, t):
        """
        Remove an edge (s, t) where parent[t] == s from the spanning tree.
        """
        size_t = self.subtree_size[t]
        for q in self.trace_subtree(t):
            self.subtree_size[self.parent[q]] -= size_t
            if q != t:
                self.last_descendent_dft[self.parent[q]] = self.last_descendent_dft[q]
        self.parent[t] = None
        self.parent_edge[t] = None

    def make_root(self, q):
        """
        Make a node q the root of its containing subtree.
        """
        p = q
        while self.parent[p] is not None:
            r = self.parent[p]
            size_p = self.subtree_size[p]
            self.subtree_size[r] -= size_p
            self.subtree_size[p] += self.subtree_size[r]
            i = self.parent_edge[p]
            self.parent_edge[p] = self.parent_edge[r]
            self.parent_edge[r] = i
            self.parent[p] = self.parent[r]
            self.parent[r] = p
            p = r

    def add_edge(self, i, p, q):
        """
        Add an edge (p, q) to the spanning tree where q is the root of a subtree.
        """
        self.parent[q] = p
        self.parent_edge[q] = i
        self.subtree_size[p] += self.subtree_size[q]
        self.last_descendent_dft[p] = self.last_descendent_dft[q]

    def update_potentials(self, i, p, q):
        """
        Update the potentials of the nodes in the subtree rooted at a node
        q connected to its parent p by an edge i.
        """
        if p == self.edge_sources[i]:
            d = self.edge_weights[i] - self.node_potentials[p] + self.node_potentials[q]
        else:
            d = -(self.edge_weights[i] - self.node_potentials[q] + self.node_potentials[p])
        for r in self.trace_subtree(q):
            self.node_potentials[r] += d

    def reduced_cost(self, i):
        """Returns the reduced cost of an edge i."""
        return (self.edge_weights[i] - self.node_potentials[self.edge_sources[i]] +
                self.node_potentials[self.edge_targets[i]])

    def find_entering_edges(self):
        """Yield entering edges until none can be found."""
        for i in range(len(self.edge_sources)):
            c = self.reduced_cost(i)
            if (c < 0 and self.edge_flow[i] < self.edge_capacities[i]) or (c > 0 and self.edge_flow[i] > 0):
                yield i

    def residual_capacity(self, i, p):
        """Returns the residual capacity of an edge i in the direction away
        from its endpoint p.
        """
        if p == self.edge_sources[i]:
            return self.edge_capacities[i] - self.edge_flow[i]
        else:
            return self.edge_flow[i]

    def find_leaving_edge(self, Wn, We):
        """Returns the leaving edge in a cycle represented by Wn and We."""
        j, theta = None, float('inf')
        for i, (p, q) in enumerate(zip(Wn, Wn[1:])):
            if p == self.parent[q]:
                f = self.residual_capacity(We[i], p)
            else:
                f = self.residual_capacity(We[i], q)
            if f < theta:
                j, theta = i, f
        return We[j], theta


@not_implemented_for('undirected')
@nx._dispatchable(node_attrs='demand', edge_attrs={'capacity': float('inf'),
    'weight': 0})
def network_simplex(G, demand='demand', capacity='capacity', weight='weight'):
    """Find a minimum cost flow satisfying all demands in digraph G.

    This is a primal network simplex algorithm that uses the leaving
    arc rule to prevent cycling.

    G is a digraph with edge costs and capacities and in which nodes
    have demand, i.e., they want to send or receive some amount of
    flow. A negative demand means that the node wants to send flow, a
    positive demand means that the node want to receive flow. A flow on
    the digraph G satisfies all demand if the net flow into each node
    is equal to the demand of that node.

    Parameters
    ----------
    G : NetworkX graph
        DiGraph on which a minimum cost flow satisfying all demands is
        to be found.

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

    Returns
    -------
    flowCost : integer, float
        Cost of a minimum cost flow satisfying all demands.

    flowDict : dictionary
        Dictionary of dictionaries keyed by nodes such that
        flowDict[u][v] is the flow edge (u, v).

    Raises
    ------
    NetworkXError
        This exception is raised if the input graph is not directed or
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
    This algorithm is not guaranteed to work if edge weights or demands
    are floating point numbers (overflows and roundoff errors can
    cause problems). As a workaround you can use integer numbers by
    multiplying the relevant edge attributes by a convenient
    constant factor (eg 100).

    See also
    --------
    cost_of_flow, max_flow_min_cost, min_cost_flow, min_cost_flow_cost

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
    >>> flowCost, flowDict = nx.network_simplex(G)
    >>> flowCost
    24
    >>> flowDict
    {'a': {'b': 4, 'c': 1}, 'd': {}, 'b': {'d': 4}, 'c': {'d': 1}}

    The mincost flow algorithm can also be used to solve shortest path
    problems. To find the shortest path between two nodes u and v,
    give all edges an infinite capacity, give node u a demand of -1 and
    node v a demand a 1. Then run the network simplex. The value of a
    min cost flow will be the distance between u and v and edges
    carrying positive flow will indicate the path.

    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from(
    ...     [
    ...         ("s", "u", 10),
    ...         ("s", "x", 5),
    ...         ("u", "v", 1),
    ...         ("u", "x", 2),
    ...         ("v", "y", 1),
    ...         ("x", "u", 3),
    ...         ("x", "v", 5),
    ...         ("x", "y", 2),
    ...         ("y", "s", 7),
    ...         ("y", "v", 6),
    ...     ]
    ... )
    >>> G.add_node("s", demand=-1)
    >>> G.add_node("v", demand=1)
    >>> flowCost, flowDict = nx.network_simplex(G)
    >>> flowCost == nx.shortest_path_length(G, "s", "v", weight="weight")
    True
    >>> sorted([(u, v) for u in flowDict for v in flowDict[u] if flowDict[u][v] > 0])
    [('s', 'x'), ('u', 'v'), ('x', 'u')]
    >>> nx.shortest_path(G, "s", "v", weight="weight")
    ['s', 'x', 'u', 'v']

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
    >>> flowCost, flowDict = nx.network_simplex(
    ...     G, demand="spam", capacity="vacancies", weight="cost"
    ... )
    >>> flowCost
    37
    >>> flowDict
    {'p': {'q': 2, 'a': 2}, 'q': {'d': 1}, 'a': {'t': 4}, 'd': {'w': 2}, 't': {'q': 1, 'w': 1}, 'w': {}}

    References
    ----------
    .. [1] Z. Kiraly, P. Kovacs.
           Efficient implementation of minimum-cost flow algorithms.
           Acta Universitatis Sapientiae, Informatica 4(1):67--118. 2012.
    .. [2] R. Barr, F. Glover, D. Klingman.
           Enhancement of spanning tree labeling procedures for network
           optimization.
           INFOR 17(1):16--34. 1979.
    """
    pass
