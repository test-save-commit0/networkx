"""
Generators for random graphs.

"""
import itertools
import math
from collections import defaultdict
import networkx as nx
from networkx.utils import py_random_state
from .classic import complete_graph, empty_graph, path_graph, star_graph
from .degree_seq import degree_sequence_tree
__all__ = ['fast_gnp_random_graph', 'gnp_random_graph',
    'dense_gnm_random_graph', 'gnm_random_graph', 'erdos_renyi_graph',
    'binomial_graph', 'newman_watts_strogatz_graph', 'watts_strogatz_graph',
    'connected_watts_strogatz_graph', 'random_regular_graph',
    'barabasi_albert_graph', 'dual_barabasi_albert_graph',
    'extended_barabasi_albert_graph', 'powerlaw_cluster_graph',
    'random_lobster', 'random_shell_graph', 'random_powerlaw_tree',
    'random_powerlaw_tree_sequence', 'random_kernel_graph']


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def fast_gnp_random_graph(n, p, seed=None, directed=False):
    """Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph or
    a binomial graph.

    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        Probability for edge creation.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True, this function returns a directed graph.

    Notes
    -----
    The $G_{n,p}$ graph algorithm chooses each of the $[n (n - 1)] / 2$
    (undirected) or $n (n - 1)$ (directed) possible edges with probability $p$.

    This algorithm [1]_ runs in $O(n + m)$ time, where `m` is the expected number of
    edges, which equals $p n (n - 1) / 2$. This should be faster than
    :func:`gnp_random_graph` when $p$ is small and the expected number of edges
    is small (that is, the graph is sparse).

    See Also
    --------
    gnp_random_graph

    References
    ----------
    .. [1] Vladimir Batagelj and Ulrik Brandes,
       "Efficient generation of large random networks",
       Phys. Rev. E, 71, 036113, 2005.
    """
    if p <= 0 or p >= 1:
        return nx.gnp_random_graph(n, p, seed=seed, directed=directed)

    G = nx.empty_graph(n)
    G.name = f"fast_gnp_random_graph({n}, {p})"

    if directed:
        G = nx.DiGraph(G)

    v = 1
    w = -1
    lp = math.log(1.0 - p)

    while v < n:
        lr = math.log(1.0 - seed.random())
        w = w + 1 + int(lr / lp)
        while w >= v and v < n:
            w = w - v
            v = v + 1
        if v < n:
            G.add_edge(v, w)

    return G


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def gnp_random_graph(n, p, seed=None, directed=False):
    """Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
    or a binomial graph.

    The $G_{n,p}$ model chooses each of the possible edges with probability $p$.

    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        Probability for edge creation.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True, this function returns a directed graph.

    See Also
    --------
    fast_gnp_random_graph

    Notes
    -----
    This algorithm [2]_ runs in $O(n^2)$ time.  For sparse graphs (that is, for
    small values of $p$), :func:`fast_gnp_random_graph` is a faster algorithm.

    :func:`binomial_graph` and :func:`erdos_renyi_graph` are
    aliases for :func:`gnp_random_graph`.

    >>> nx.binomial_graph is nx.gnp_random_graph
    True
    >>> nx.erdos_renyi_graph is nx.gnp_random_graph
    True

    References
    ----------
    .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
    .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
    """
    G = nx.empty_graph(n)
    G.name = f"gnp_random_graph({n}, {p})"

    if directed:
        G = nx.DiGraph(G)
        edges = itertools.permutations(range(n), 2)
    else:
        edges = itertools.combinations(range(n), 2)

    G.add_edges_from(e for e in edges if seed.random() < p)
    return G


binomial_graph = gnp_random_graph
erdos_renyi_graph = gnp_random_graph


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def dense_gnm_random_graph(n, m, seed=None):
    """Returns a $G_{n,m}$ random graph.

    In the $G_{n,m}$ model, a graph is chosen uniformly at random from the set
    of all graphs with $n$ nodes and $m$ edges.

    This algorithm should be faster than :func:`gnm_random_graph` for dense
    graphs.

    Parameters
    ----------
    n : int
        The number of nodes.
    m : int
        The number of edges.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    gnm_random_graph

    Notes
    -----
    Algorithm by Keith M. Briggs Mar 31, 2006.
    Inspired by Knuth's Algorithm S (Selection sampling technique),
    in section 3.4.2 of [1]_.

    References
    ----------
    .. [1] Donald E. Knuth, The Art of Computer Programming,
        Volume 2/Seminumerical algorithms, Third Edition, Addison-Wesley, 1997.
    """
    mmax = n * (n - 1) // 2
    if m >= mmax:
        return nx.complete_graph(n)

    G = nx.empty_graph(n)
    G.name = f"dense_gnm_random_graph({n}, {m})"

    if n == 1 or m >= mmax:
        return G

    u = 0
    v = 1
    t = 0
    k = 0
    while True:
        if seed.random() * (mmax - t) < m - k:
            G.add_edge(u, v)
            k += 1
            if k == m:
                return G
        t += 1
        v += 1
        if v == n:  # go to next row
            u += 1
            v = u + 1


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def gnm_random_graph(n, m, seed=None, directed=False):
    """Returns a $G_{n,m}$ random graph.

    In the $G_{n,m}$ model, a graph is chosen uniformly at random from the set
    of all graphs with $n$ nodes and $m$ edges.

    This algorithm should be faster than :func:`dense_gnm_random_graph` for
    sparse graphs.

    Parameters
    ----------
    n : int
        The number of nodes.
    m : int
        The number of edges.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True return a directed graph

    See also
    --------
    dense_gnm_random_graph

    """
    G = nx.empty_graph(n)
    G.name = f"gnm_random_graph({n}, {m})"

    if directed:
        G = nx.DiGraph(G)
        max_edges = n * (n - 1)
    else:
        max_edges = n * (n - 1) // 2

    if m >= max_edges:
        return nx.complete_graph(n, create_using=G)

    nlist = list(G.nodes())
    edge_count = 0
    while edge_count < m:
        u = seed.choice(nlist)
        v = seed.choice(nlist)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
            edge_count += 1

    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def newman_watts_strogatz_graph(n, k, p, seed=None):
    """Returns a Newman–Watts–Strogatz small-world graph.

    Parameters
    ----------
    n : int
        The number of nodes.
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of adding a new edge for each edge.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Notes
    -----
    First create a ring over $n$ nodes [1]_.  Then each node in the ring is
    connected with its $k$ nearest neighbors (or $k - 1$ neighbors if $k$
    is odd).  Then shortcuts are created by adding new edges as follows: for
    each edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest
    neighbors" with probability $p$ add a new edge $(u, w)$ with
    randomly-chosen existing node $w$.  In contrast with
    :func:`watts_strogatz_graph`, no edges are removed.

    See Also
    --------
    watts_strogatz_graph

    References
    ----------
    .. [1] M. E. J. Newman and D. J. Watts,
       Renormalization group analysis of the small-world network model,
       Physics Letters A, 263, 341, 1999.
       https://doi.org/10.1016/S0375-9601(99)00757-4
    """
    if k >= n:
        raise nx.NetworkXError("k>=n, choose smaller k or larger n")

    G = nx.empty_graph(n)
    G.name = f"newman_watts_strogatz_graph({n}, {k}, {p})"
    nodes = list(G.nodes())
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        G.add_edges_from(zip(nodes, targets))
    # add new edges
    for u, v in list(G.edges()):
        if seed.random() < p:
            w = seed.choice(nodes)
            # Enforce no self-loops or multiple edges
            while w == u or G.has_edge(u, w):
                w = seed.choice(nodes)
            G.add_edge(u, w)
    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def watts_strogatz_graph(n, k, p, seed=None):
    """Returns a Watts–Strogatz small-world graph.

    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of rewiring each edge
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    newman_watts_strogatz_graph
    connected_watts_strogatz_graph

    Notes
    -----
    First create a ring over $n$ nodes [1]_.  Then each node in the ring is joined
    to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).
    Then shortcuts are created by replacing some edges as follows: for each
    edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors"
    with probability $p$ replace it with a new edge $(u, w)$ with uniformly
    random choice of existing node $w$.

    In contrast with :func:`newman_watts_strogatz_graph`, the random rewiring
    does not increase the number of edges. The rewired graph is not guaranteed
    to be connected as in :func:`connected_watts_strogatz_graph`.

    References
    ----------
    .. [1] Duncan J. Watts and Steven H. Strogatz,
       Collective dynamics of small-world networks,
       Nature, 393, pp. 440--442, 1998.
    """
    if k >= n:
        raise nx.NetworkXError("k>=n, choose smaller k or larger n")

    G = nx.empty_graph(n)
    G.name = f"watts_strogatz_graph({n}, {k}, {p})"
    nodes = list(G.nodes())
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        G.add_edges_from(zip(nodes, targets))
    # rewire edges from each node
    for u, v in list(G.edges()):
        if seed.random() < p:
            w = seed.choice(nodes)
            # Enforce no self-loops or multiple edges
            while w == u or G.has_edge(u, w):
                w = seed.choice(nodes)
            G.remove_edge(u, v)
            G.add_edge(u, w)
    return G


@py_random_state(4)
@nx._dispatchable(graphs=None, returns_graph=True)
def connected_watts_strogatz_graph(n, k, p, tries=100, seed=None):
    """Returns a connected Watts–Strogatz small-world graph.

    Attempts to generate a connected graph by repeated generation of
    Watts–Strogatz small-world graphs.  An exception is raised if the maximum
    number of tries is exceeded.

    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of rewiring each edge
    tries : int
        Number of attempts to generate a connected graph.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Notes
    -----
    First create a ring over $n$ nodes [1]_.  Then each node in the ring is joined
    to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).
    Then shortcuts are created by replacing some edges as follows: for each
    edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors"
    with probability $p$ replace it with a new edge $(u, w)$ with uniformly
    random choice of existing node $w$.
    The entire process is repeated until a connected graph results.

    See Also
    --------
    newman_watts_strogatz_graph
    watts_strogatz_graph

    References
    ----------
    .. [1] Duncan J. Watts and Steven H. Strogatz,
       Collective dynamics of small-world networks,
       Nature, 393, pp. 440--442, 1998.
    """
    for i in range(tries):
        G = watts_strogatz_graph(n, k, p, seed)
        if nx.is_connected(G):
            return G
    raise nx.NetworkXError(f"Failed to generate connected graph in {tries} tries")


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_regular_graph(d, n, seed=None):
    """Returns a random $d$-regular graph on $n$ nodes.

    A regular graph is a graph where each node has the same number of neighbors.

    The resulting graph has no self-loops or parallel edges.

    Parameters
    ----------
    d : int
      The degree of each node.
    n : integer
      The number of nodes. The value of $n \\times d$ must be even.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Notes
    -----
    The nodes are numbered from $0$ to $n - 1$.

    Kim and Vu's paper [2]_ shows that this algorithm samples in an
    asymptotically uniform way from the space of random graphs when
    $d = O(n^{1 / 3 - \\epsilon})$.

    Raises
    ------

    NetworkXError
        If $n \\times d$ is odd or $d$ is greater than or equal to $n$.

    References
    ----------
    .. [1] A. Steger and N. Wormald,
       Generating random regular graphs quickly,
       Probability and Computing 8 (1999), 377-396, 1999.
       https://doi.org/10.1017/S0963548399003867

    .. [2] Jeong Han Kim and Van H. Vu,
       Generating random regular graphs,
       Proceedings of the thirty-fifth ACM symposium on Theory of computing,
       San Diego, CA, USA, pp 213--222, 2003.
       http://portal.acm.org/citation.cfm?id=780542.780576
    """
    if (n * d) % 2 != 0:
        raise nx.NetworkXError("n * d must be even")
    if d >= n:
        raise nx.NetworkXError("d must be less than n")

    def _suitable(edges, potential_edges):
        # Helper function to check if there are suitable edges remaining
        # If False, the generation of the graph has failed
        if not potential_edges:
            return True
        for u, v in potential_edges:
            if v not in edges[u]:
                return True
        return False

    def _try_creation():
        edges = {i: set() for i in range(n)}
        stubs = list(range(n)) * d
        seed.shuffle(stubs)
        while stubs:
            potential_edges = [(stubs[0], stubs[1])]
            for i in range(2, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v or v in edges[u]:
                    potential_edges.append((u, v))
                else:
                    edges[u].add(v)
                    edges[v].add(u)
            if not _suitable(edges, potential_edges):
                return None
            stubs = [u for u, v in potential_edges]
        return edges

    # Try to create the graph, if it fails, try again
    for _ in range(100):  # Arbitrary limit on number of tries
        edges = _try_creation()
        if edges is not None:
            G = nx.Graph(edges)
            G.name = f"random_regular_graph({d}, {n})"
            return G
    raise nx.NetworkXError("Failed to generate graph")


def _random_subset(seq, m, rng):
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    pass


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def barabasi_albert_graph(n, m, seed=None, initial_graph=None):
    """Returns a random graph using Barabási–Albert preferential attachment

    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    initial_graph : Graph or None (default)
        Initial network for Barabási–Albert algorithm.
        It should be a connected graph for most use cases.
        A copy of `initial_graph` is used.
        If None, starts from a star graph on (m+1) nodes.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n``, or
        the initial graph number of nodes m0 does not satisfy ``m <= m0 <= n``.

    References
    ----------
    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """
    if m < 1 or m >= n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1 and m < n, m = %d, n = %d" % (m, n))

    if initial_graph is None:
        # Default initial graph : star graph on (m + 1) nodes
        G = nx.star_graph(m)
    else:
        G = initial_graph.copy()

    if len(G) < m or len(G) > n:
        raise nx.NetworkXError(f"Initial graph must have m <= n0 <= n nodes, n0 = {len(G)}")

    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = [n for n, d in G.degree() for _ in range(d)]
    # Start adding the other n-m nodes. The first node is m.
    source = len(G)
    while source < n:
        # Add edges to m nodes from the existing nodes
        targets = _random_subset(repeated_nodes, m, seed)
        # Add the edges
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created
        repeated_nodes.extend(targets)
        # And the new node itself
        repeated_nodes.extend([source] * m)
        source += 1
    return G


@py_random_state(4)
@nx._dispatchable(graphs=None, returns_graph=True)
def dual_barabasi_albert_graph(n, m1, m2, p, seed=None, initial_graph=None):
    """Returns a random graph using dual Barabási–Albert preferential attachment

    A graph of $n$ nodes is grown by attaching new nodes each with either $m_1$
    edges (with probability $p$) or $m_2$ edges (with probability $1-p$) that
    are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m1 : int
        Number of edges to link each new node to existing nodes with probability $p$
    m2 : int
        Number of edges to link each new node to existing nodes with probability $1-p$
    p : float
        The probability of attaching $m_1$ edges (as opposed to $m_2$ edges)
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    initial_graph : Graph or None (default)
        Initial network for Barabási–Albert algorithm.
        A copy of `initial_graph` is used.
        It should be connected for most use cases.
        If None, starts from an star graph on max(m1, m2) + 1 nodes.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m1` and `m2` do not satisfy ``1 <= m1,m2 < n``, or
        `p` does not satisfy ``0 <= p <= 1``, or
        the initial graph number of nodes m0 does not satisfy m1, m2 <= m0 <= n.

    References
    ----------
    .. [1] N. Moshiri "The dual-Barabasi-Albert model", arXiv:1810.10538.
    """
    pass


@py_random_state(4)
@nx._dispatchable(graphs=None, returns_graph=True)
def extended_barabasi_albert_graph(n, m, p, q, seed=None):
    """Returns an extended Barabási–Albert model graph.

    An extended Barabási–Albert model graph is a random graph constructed
    using preferential attachment. The extended model allows new edges,
    rewired edges or new nodes. Based on the probabilities $p$ and $q$
    with $p + q < 1$, the growing behavior of the graph is determined as:

    1) With $p$ probability, $m$ new edges are added to the graph,
    starting from randomly chosen existing nodes and attached preferentially at the other end.

    2) With $q$ probability, $m$ existing edges are rewired
    by randomly choosing an edge and rewiring one end to a preferentially chosen node.

    3) With $(1 - p - q)$ probability, $m$ new nodes are added to the graph
    with edges attached preferentially.

    When $p = q = 0$, the model behaves just like the Barabási–Alber model.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges with which a new node attaches to existing nodes
    p : float
        Probability value for adding an edge between existing nodes. p + q < 1
    q : float
        Probability value of rewiring of existing edges. p + q < 1
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n`` or ``1 >= p + q``

    References
    ----------
    .. [1] Albert, R., & Barabási, A. L. (2000)
       Topology of evolving networks: local events and universality
       Physical review letters, 85(24), 5234.
    """
    pass


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def powerlaw_cluster_graph(n, m, p, seed=None):
    """Holme and Kim algorithm for growing graphs with powerlaw
    degree distribution and approximate average clustering.

    Parameters
    ----------
    n : int
        the number of nodes
    m : int
        the number of random edges to add for each new node
    p : float,
        Probability of adding a triangle after adding a random edge
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Notes
    -----
    The average clustering has a hard time getting above a certain
    cutoff that depends on `m`.  This cutoff is often quite low.  The
    transitivity (fraction of triangles to possible triangles) seems to
    decrease with network size.

    It is essentially the Barabási–Albert (BA) growth model with an
    extra step that each random edge is followed by a chance of
    making an edge to one of its neighbors too (and thus a triangle).

    This algorithm improves on BA in the sense that it enables a
    higher average clustering to be attained if desired.

    It seems possible to have a disconnected graph with this algorithm
    since the initial `m` nodes may not be all linked to a new node
    on the first iteration like the BA model.

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m <= n`` or `p` does not
        satisfy ``0 <= p <= 1``.

    References
    ----------
    .. [1] P. Holme and B. J. Kim,
       "Growing scale-free networks with tunable clustering",
       Phys. Rev. E, 65, 026107, 2002.
    """
    pass


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_lobster(n, p1, p2, seed=None):
    """Returns a random lobster graph.

    A lobster is a tree that reduces to a caterpillar when pruning all
    leaf nodes. A caterpillar is a tree that reduces to a path graph
    when pruning all leaf nodes; setting `p2` to zero produces a caterpillar.

    This implementation iterates on the probabilities `p1` and `p2` to add
    edges at levels 1 and 2, respectively. Graphs are therefore constructed
    iteratively with uniform randomness at each level rather than being selected
    uniformly at random from the set of all possible lobsters.

    Parameters
    ----------
    n : int
        The expected number of nodes in the backbone
    p1 : float
        Probability of adding an edge to the backbone
    p2 : float
        Probability of adding an edge one level beyond backbone
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Raises
    ------
    NetworkXError
        If `p1` or `p2` parameters are >= 1 because the while loops would never finish.
    """
    pass


@py_random_state(1)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_shell_graph(constructor, seed=None):
    """Returns a random shell graph for the constructor given.

    Parameters
    ----------
    constructor : list of three-tuples
        Represents the parameters for a shell, starting at the center
        shell.  Each element of the list must be of the form `(n, m,
        d)`, where `n` is the number of nodes in the shell, `m` is
        the number of edges in the shell, and `d` is the ratio of
        inter-shell (next) edges to intra-shell edges. If `d` is zero,
        there will be no intra-shell edges, and if `d` is one there
        will be all possible intra-shell edges.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Examples
    --------
    >>> constructor = [(10, 20, 0.8), (20, 40, 0.8)]
    >>> G = nx.random_shell_graph(constructor)

    """
    pass


@py_random_state(2)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_powerlaw_tree(n, gamma=3, seed=None, tries=100):
    """Returns a tree with a power law degree distribution.

    Parameters
    ----------
    n : int
        The number of nodes.
    gamma : float
        Exponent of the power law.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    tries : int
        Number of attempts to adjust the sequence to make it a tree.

    Raises
    ------
    NetworkXError
        If no valid sequence is found within the maximum number of
        attempts.

    Notes
    -----
    A trial power law degree sequence is chosen and then elements are
    swapped with new elements from a powerlaw distribution until the
    sequence makes a tree (by checking, for example, that the number of
    edges is one smaller than the number of nodes).

    """
    pass


@py_random_state(2)
@nx._dispatchable(graphs=None)
def random_powerlaw_tree_sequence(n, gamma=3, seed=None, tries=100):
    """Returns a degree sequence for a tree with a power law distribution.

    Parameters
    ----------
    n : int,
        The number of nodes.
    gamma : float
        Exponent of the power law.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    tries : int
        Number of attempts to adjust the sequence to make it a tree.

    Raises
    ------
    NetworkXError
        If no valid sequence is found within the maximum number of
        attempts.

    Notes
    -----
    A trial power law degree sequence is chosen and then elements are
    swapped with new elements from a power law distribution until
    the sequence makes a tree (by checking, for example, that the number of
    edges is one smaller than the number of nodes).

    """
    pass


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_kernel_graph(n, kernel_integral, kernel_root=None, seed=None):
    """Returns an random graph based on the specified kernel.

    The algorithm chooses each of the $[n(n-1)]/2$ possible edges with
    probability specified by a kernel $\\kappa(x,y)$ [1]_.  The kernel
    $\\kappa(x,y)$ must be a symmetric (in $x,y$), non-negative,
    bounded function.

    Parameters
    ----------
    n : int
        The number of nodes
    kernel_integral : function
        Function that returns the definite integral of the kernel $\\kappa(x,y)$,
        $F(y,a,b) := \\int_a^b \\kappa(x,y)dx$
    kernel_root: function (optional)
        Function that returns the root $b$ of the equation $F(y,a,b) = r$.
        If None, the root is found using :func:`scipy.optimize.brentq`
        (this requires SciPy).
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Notes
    -----
    The kernel is specified through its definite integral which must be
    provided as one of the arguments. If the integral and root of the
    kernel integral can be found in $O(1)$ time then this algorithm runs in
    time $O(n+m)$ where m is the expected number of edges [2]_.

    The nodes are set to integers from $0$ to $n-1$.

    Examples
    --------
    Generate an Erdős–Rényi random graph $G(n,c/n)$, with kernel
    $\\kappa(x,y)=c$ where $c$ is the mean expected degree.

    >>> def integral(u, w, z):
    ...     return c * (z - w)
    >>> def root(u, w, r):
    ...     return r / c + w
    >>> c = 1
    >>> graph = nx.random_kernel_graph(1000, integral, root)

    See Also
    --------
    gnp_random_graph
    expected_degree_graph

    References
    ----------
    .. [1] Bollobás, Béla,  Janson, S. and Riordan, O.
       "The phase transition in inhomogeneous random graphs",
       *Random Structures Algorithms*, 31, 3--122, 2007.

    .. [2] Hagberg A, Lemons N (2015),
       "Fast Generation of Sparse Random Kernel Graphs".
       PLoS ONE 10(9): e0135177, 2015. doi:10.1371/journal.pone.0135177
    """
    pass
