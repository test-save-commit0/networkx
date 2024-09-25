"""Generates graphs resembling the Internet Autonomous System network"""
import networkx as nx
from networkx.utils import py_random_state
__all__ = ['random_internet_as_graph']


def uniform_int_from_avg(a, m, seed):
    """Pick a random integer with uniform probability.

    Returns a random integer uniformly taken from a distribution with
    minimum value 'a' and average value 'm', X~U(a,b), E[X]=m, X in N where
    b = 2*m - a.

    Notes
    -----
    p = (b-floor(b))/2
    X = X1 + X2; X1~U(a,floor(b)), X2~B(p)
    E[X] = E[X1] + E[X2] = (floor(b)+a)/2 + (b-floor(b))/2 = (b+a)/2 = m
    """
    b = 2 * m - a
    floor_b = int(b)
    p = (b - floor_b) / 2
    x1 = seed.randint(a, floor_b)
    x2 = int(seed.random() < p)
    return x1 + x2


def choose_pref_attach(degs, seed):
    """Pick a random value, with a probability given by its weight.

    Returns a random choice among degs keys, each of which has a
    probability proportional to the corresponding dictionary value.

    Parameters
    ----------
    degs: dictionary
        It contains the possible values (keys) and the corresponding
        probabilities (values)
    seed: random state

    Returns
    -------
    v: object
        A key of degs or None if degs is empty
    """
    if not degs:
        return None
    
    total = sum(degs.values())
    r = seed.random() * total
    for k, v in degs.items():
        r -= v
        if r <= 0:
            return k


class AS_graph_generator:
    """Generates random internet AS graphs."""

    def __init__(self, n, seed):
        """Initializes variables. Immediate numbers are taken from [1].

        Parameters
        ----------
        n: integer
            Number of graph nodes
        seed: random state
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        Returns
        -------
        GG: AS_graph_generator object

        References
        ----------
        [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of
        BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas
        in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.
        """
        self.seed = seed
        self.n_t = min(n, round(self.seed.random() * 2 + 4))
        self.n_m = round(0.15 * n)
        self.n_cp = round(0.05 * n)
        self.n_c = max(0, n - self.n_t - self.n_m - self.n_cp)
        self.d_m = 2 + 2.5 * n / 10000
        self.d_cp = 2 + 1.5 * n / 10000
        self.d_c = 1 + 5 * n / 100000
        self.p_m_m = 1 + 2 * n / 10000
        self.p_cp_m = 0.2 + 2 * n / 10000
        self.p_cp_cp = 0.05 + 2 * n / 100000
        self.t_m = 0.375
        self.t_cp = 0.375
        self.t_c = 0.125

    def t_graph(self):
        """Generates the core mesh network of tier one nodes of a AS graph.

        Returns
        -------
        G: Networkx Graph
            Core network
        """
        G = nx.Graph()
        for i in range(self.n_t):
            G.add_node(i, type='T')
        for i in range(self.n_t):
            for j in range(i + 1, self.n_t):
                G.add_edge(i, j, type='peer', customer='none')
        return G

    def choose_peer_pref_attach(self, node_list):
        """Pick a node with a probability weighted by its peer degree.

        Pick a node from node_list with preferential attachment
        computed only on their peer degree
        """
        peer_degs = {n: sum(1 for _, v, d in self.G.edges(n, data=True) if d['type'] == 'peer') for n in node_list}
        return choose_pref_attach(peer_degs, self.seed)

    def choose_node_pref_attach(self, node_list):
        """Pick a node with a probability weighted by its degree.

        Pick a node from node_list with preferential attachment
        computed on their degree
        """
        degs = {n: self.G.degree(n) for n in node_list}
        return choose_pref_attach(degs, self.seed)

    def add_customer(self, i, j):
        """Keep the dictionaries 'customers' and 'providers' consistent."""
        if i not in self.customers:
            self.customers[i] = set()
        if j not in self.providers:
            self.providers[j] = set()
        self.customers[i].add(j)
        self.providers[j].add(i)

    def add_node(self, i, kind, reg2prob, avg_deg, t_edge_prob):
        """Add a node and its customer transit edges to the graph.

        Parameters
        ----------
        i: object
            Identifier of the new node
        kind: string
            Type of the new node. Options are: 'M' for middle node, 'CP' for
            content provider and 'C' for customer.
        reg2prob: float
            Probability the new node can be in two different regions.
        avg_deg: float
            Average number of transit nodes of which node i is customer.
        t_edge_prob: float
            Probability node i establish a customer transit edge with a tier
            one (T) node

        Returns
        -------
        i: object
            Identifier of the new node
        """
        self.G.add_node(i, type=kind)
        self.node_region[i] = self.seed.choice(range(self.n_regions))
        if self.seed.random() < reg2prob:
            self.node_region[i] = set([self.node_region[i], self.seed.choice(range(self.n_regions))])
        
        deg = uniform_int_from_avg(1, avg_deg, self.seed)
        providers = []
        
        if self.seed.random() < t_edge_prob:
            t_node = self.seed.choice([n for n, d in self.G.nodes(data=True) if d['type'] == 'T'])
            providers.append(t_node)
            deg -= 1
        
        while deg > 0:
            provider = self.choose_node_pref_attach([n for n in self.G.nodes() if n not in providers])
            if provider is not None:
                providers.append(provider)
                deg -= 1
        
        for p in providers:
            self.G.add_edge(i, p, type='transit', customer=i)
            self.add_customer(i, p)
        
        return i

    def add_m_peering_link(self, m, to_kind):
        """Add a peering link between two middle tier (M) nodes.

        Target node j is drawn considering a preferential attachment based on
        other M node peering degree.

        Parameters
        ----------
        m: object
            Node identifier
        to_kind: string
            type for target node j (must be always M)

        Returns
        -------
        success: boolean
        """
        assert to_kind == 'M'
        candidates = [n for n, d in self.G.nodes(data=True) 
                      if d['type'] == 'M' and n != m and not self.G.has_edge(m, n)]
        if not candidates:
            return False
        j = self.choose_peer_pref_attach(candidates)
        self.G.add_edge(m, j, type='peer', customer='none')
        return True

    def add_cp_peering_link(self, cp, to_kind):
        """Add a peering link to a content provider (CP) node.

        Target node j can be CP or M and it is drawn uniformly among the nodes
        belonging to the same region as cp.

        Parameters
        ----------
        cp: object
            Node identifier
        to_kind: string
            type for target node j (must be M or CP)

        Returns
        -------
        success: boolean
        """
        assert to_kind in ['M', 'CP']
        cp_region = self.node_region[cp]
        candidates = [n for n, d in self.G.nodes(data=True) 
                      if d['type'] == to_kind and n != cp and not self.G.has_edge(cp, n)
                      and (isinstance(cp_region, set) and self.node_region[n] in cp_region
                           or self.node_region[n] == cp_region)]
        if not candidates:
            return False
        j = self.seed.choice(candidates)
        self.G.add_edge(cp, j, type='peer', customer='none')
        return True

    def graph_regions(self, rn):
        """Initializes AS network regions.

        Parameters
        ----------
        rn: integer
            Number of regions
        """
        self.n_regions = rn
        self.node_region = {}

    def add_peering_links(self, from_kind, to_kind):
        """Utility function to add peering links among node groups."""
        nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == from_kind]
        for node in nodes:
            if from_kind == 'M':
                self.add_m_peering_link(node, to_kind)
            elif from_kind == 'CP':
                self.add_cp_peering_link(node, to_kind)

    def generate(self):
        """Generates a random AS network graph as described in [1].

        Returns
        -------
        G: Graph object

        Notes
        -----
        The process steps are the following: first we create the core network
        of tier one nodes, then we add the middle tier (M), the content
        provider (CP) and the customer (C) nodes along with their transit edges
        (link i,j means i is customer of j). Finally we add peering links
        between M nodes, between M and CP nodes and between CP node couples.
        For a detailed description of the algorithm, please refer to [1].

        References
        ----------
        [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of
        BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas
        in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.
        """
        self.G = self.t_graph()
        self.customers = {}
        self.providers = {}
        self.graph_regions(3)

        n = self.n_t
        for _ in range(self.n_m):
            self.add_node(n, 'M', 0.5, self.d_m, self.t_m)
            n += 1
        for _ in range(self.n_cp):
            self.add_node(n, 'CP', 0.5, self.d_cp, self.t_cp)
            n += 1
        for _ in range(self.n_c):
            self.add_node(n, 'C', 0.5, self.d_c, self.t_c)
            n += 1

        self.add_peering_links('M', 'M')
        self.add_peering_links('CP', 'M')
        self.add_peering_links('CP', 'CP')

        return self.G


@py_random_state(1)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_internet_as_graph(n, seed=None):
    """Generates a random undirected graph resembling the Internet AS network

    Parameters
    ----------
    n: integer in [1000, 10000]
        Number of graph nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G: Networkx Graph object
        A randomly generated undirected graph

    Notes
    -----
    This algorithm returns an undirected graph resembling the Internet
    Autonomous System (AS) network, it uses the approach by Elmokashfi et al.
    [1]_ and it grants the properties described in the related paper [1]_.

    Each node models an autonomous system, with an attribute 'type' specifying
    its kind; tier-1 (T), mid-level (M), customer (C) or content-provider (CP).
    Each edge models an ADV communication link (hence, bidirectional) with
    attributes:

      - type: transit|peer, the kind of commercial agreement between nodes;
      - customer: <node id>, the identifier of the node acting as customer
        ('none' if type is peer).

    References
    ----------
    .. [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of
       BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas
       in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.
    """
    if not 1000 <= n <= 10000:
        raise nx.NetworkXError("n must be between 1000 and 10000")
    
    generator = AS_graph_generator(n, seed)
    return generator.generate()
