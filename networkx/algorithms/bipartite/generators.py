"""
Generators and functions for bipartite graphs.
"""
import math
import numbers
from functools import reduce
import networkx as nx
from networkx.utils import nodes_or_number, py_random_state
__all__ = ['configuration_model', 'havel_hakimi_graph',
    'reverse_havel_hakimi_graph', 'alternating_havel_hakimi_graph',
    'preferential_attachment_graph', 'random_graph', 'gnmk_random_graph',
    'complete_bipartite_graph']


@nx._dispatchable(graphs=None, returns_graph=True)
@nodes_or_number([0, 1])
def complete_bipartite_graph(n1, n2, create_using=None):
    """Returns the complete bipartite graph `K_{n_1,n_2}`.

    The graph is composed of two partitions with nodes 0 to (n1 - 1)
    in the first and nodes n1 to (n1 + n2 - 1) in the second.
    Each node in the first is connected to each node in the second.

    Parameters
    ----------
    n1, n2 : integer or iterable container of nodes
        If integers, nodes are from `range(n1)` and `range(n1, n1 + n2)`.
        If a container, the elements are the nodes.
    create_using : NetworkX graph instance, (default: nx.Graph)
       Return graph of this type.

    Notes
    -----
    Nodes are the integers 0 to `n1 + n2 - 1` unless either n1 or n2 are
    containers of nodes. If only one of n1 or n2 are integers, that
    integer is replaced by `range` of that integer.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.complete_bipartite_graph
    """
    if create_using is None:
        create_using = nx.Graph()
    elif not create_using.is_directed():
        create_using = nx.Graph(create_using)
    else:
        create_using = nx.DiGraph(create_using)

    if isinstance(n1, numbers.Integral):
        n1 = range(n1)
    if isinstance(n2, numbers.Integral):
        n2 = range(n1, n1 + n2)

    G = create_using
    G.add_nodes_from(n1, bipartite=0)
    G.add_nodes_from(n2, bipartite=1)
    G.add_edges_from((u, v) for u in n1 for v in n2)
    return G


@py_random_state(3)
@nx._dispatchable(name='bipartite_configuration_model', graphs=None,
    returns_graph=True)
def configuration_model(aseq, bseq, create_using=None, seed=None):
    """Returns a random bipartite graph from two given degree sequences.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    bseq : list
       Degree sequence for node set B.
    create_using : NetworkX graph instance, optional
       Return graph of this type.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes len(aseq) to (len(bseq) - 1).
    Nodes from set A are connected to nodes in set B by choosing
    randomly from the possible free stubs, one in A and one in B.

    Notes
    -----
    The sum of the two sequences must be equal: sum(aseq)=sum(bseq)
    If no graph type is specified use MultiGraph with parallel edges.
    If you want a graph with no parallel edges use create_using=Graph()
    but then the resulting degree sequences might not be exact.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.configuration_model
    """
    if create_using is None:
        create_using = nx.MultiGraph()
    elif not create_using.is_multigraph():
        raise nx.NetworkXError("create_using must be a multigraph")

    G = create_using
    G.clear()

    if sum(aseq) != sum(bseq):
        raise nx.NetworkXError("Degree sequences must have equal sums")

    n = len(aseq)
    m = len(bseq)
    
    stubs = []
    for i, d in enumerate(aseq):
        stubs.extend([i] * d)
        G.add_node(i, bipartite=0)
    for i, d in enumerate(bseq, start=n):
        stubs.extend([i] * d)
        G.add_node(i, bipartite=1)

    rng = seed if isinstance(seed, nx.utils.RandomState) else nx.utils.RandomState(seed)
    rng.shuffle(stubs)

    while stubs:
        u, v = stubs.pop(), stubs.pop()
        G.add_edge(u, v)

    return G


@nx._dispatchable(name='bipartite_havel_hakimi_graph', graphs=None,
    returns_graph=True)
def havel_hakimi_graph(aseq, bseq, create_using=None):
    """Returns a bipartite graph from two given degree sequences using a
    Havel-Hakimi style construction.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes len(aseq) to (len(bseq) - 1).
    Nodes from the set A are connected to nodes in the set B by
    connecting the highest degree nodes in set A to the highest degree
    nodes in set B until all stubs are connected.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    bseq : list
       Degree sequence for node set B.
    create_using : NetworkX graph instance, optional
       Return graph of this type.

    Notes
    -----
    The sum of the two sequences must be equal: sum(aseq)=sum(bseq)
    If no graph type is specified use MultiGraph with parallel edges.
    If you want a graph with no parallel edges use create_using=Graph()
    but then the resulting degree sequences might not be exact.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.havel_hakimi_graph
    """
    if create_using is None:
        create_using = nx.MultiGraph()
    elif not create_using.is_multigraph():
        raise nx.NetworkXError("create_using must be a multigraph")

    G = create_using
    G.clear()

    if sum(aseq) != sum(bseq):
        raise nx.NetworkXError("Degree sequences must have equal sums")

    n = len(aseq)
    m = len(bseq)

    for i in range(n):
        G.add_node(i, bipartite=0)
    for i in range(n, n + m):
        G.add_node(i, bipartite=1)

    A = sorted([(d, i) for i, d in enumerate(aseq)], reverse=True)
    B = sorted([(d, i) for i, d in enumerate(bseq, start=n)], reverse=True)

    while A and B:
        da, a = A.pop(0)
        while da and B:
            db, b = B.pop(0)
            G.add_edge(a, b)
            da -= 1
            db -= 1
            if db:
                B.append((db, b))
                B.sort(reverse=True)
        if da:
            A.append((da, a))
            A.sort(reverse=True)

    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def reverse_havel_hakimi_graph(aseq, bseq, create_using=None):
    """Returns a bipartite graph from two given degree sequences using a
    Havel-Hakimi style construction.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes len(aseq) to (len(bseq) - 1).
    Nodes from set A are connected to nodes in the set B by connecting
    the highest degree nodes in set A to the lowest degree nodes in
    set B until all stubs are connected.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    bseq : list
       Degree sequence for node set B.
    create_using : NetworkX graph instance, optional
       Return graph of this type.

    Notes
    -----
    The sum of the two sequences must be equal: sum(aseq)=sum(bseq)
    If no graph type is specified use MultiGraph with parallel edges.
    If you want a graph with no parallel edges use create_using=Graph()
    but then the resulting degree sequences might not be exact.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.reverse_havel_hakimi_graph
    """
    if create_using is None:
        create_using = nx.MultiGraph()
    elif not create_using.is_multigraph():
        raise nx.NetworkXError("create_using must be a multigraph")

    G = create_using
    G.clear()

    if sum(aseq) != sum(bseq):
        raise nx.NetworkXError("Degree sequences must have equal sums")

    n = len(aseq)
    m = len(bseq)

    for i in range(n):
        G.add_node(i, bipartite=0)
    for i in range(n, n + m):
        G.add_node(i, bipartite=1)

    A = sorted([(d, i) for i, d in enumerate(aseq)], reverse=True)
    B = sorted([(d, i) for i, d in enumerate(bseq, start=n)])

    while A and B:
        da, a = A.pop(0)
        while da and B:
            db, b = B.pop(0)
            G.add_edge(a, b)
            da -= 1
            db -= 1
            if db:
                B.append((db, b))
                B.sort()
        if da:
            A.append((da, a))
            A.sort(reverse=True)

    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def alternating_havel_hakimi_graph(aseq, bseq, create_using=None):
    """Returns a bipartite graph from two given degree sequences using
    an alternating Havel-Hakimi style construction.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes len(aseq) to (len(bseq) - 1).
    Nodes from the set A are connected to nodes in the set B by
    connecting the highest degree nodes in set A to alternatively the
    highest and the lowest degree nodes in set B until all stubs are
    connected.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    bseq : list
       Degree sequence for node set B.
    create_using : NetworkX graph instance, optional
       Return graph of this type.

    Notes
    -----
    The sum of the two sequences must be equal: sum(aseq)=sum(bseq)
    If no graph type is specified use MultiGraph with parallel edges.
    If you want a graph with no parallel edges use create_using=Graph()
    but then the resulting degree sequences might not be exact.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.alternating_havel_hakimi_graph
    """
    if create_using is None:
        create_using = nx.MultiGraph()
    elif not create_using.is_multigraph():
        raise nx.NetworkXError("create_using must be a multigraph")

    G = create_using
    G.clear()

    if sum(aseq) != sum(bseq):
        raise nx.NetworkXError("Degree sequences must have equal sums")

    n = len(aseq)
    m = len(bseq)

    for i in range(n):
        G.add_node(i, bipartite=0)
    for i in range(n, n + m):
        G.add_node(i, bipartite=1)

    A = sorted([(d, i) for i, d in enumerate(aseq)], reverse=True)
    B = sorted([(d, i) for i, d in enumerate(bseq, start=n)], reverse=True)

    alternate = True
    while A and B:
        da, a = A.pop(0)
        while da and B:
            if alternate:
                db, b = B.pop(0)
            else:
                db, b = B.pop()
            G.add_edge(a, b)
            da -= 1
            db -= 1
            if db:
                B.append((db, b))
                B.sort(reverse=True)
            alternate = not alternate
        if da:
            A.append((da, a))
            A.sort(reverse=True)

    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def preferential_attachment_graph(aseq, p, create_using=None, seed=None):
    """Create a bipartite graph with a preferential attachment model from
    a given single degree sequence.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes starting with node len(aseq).
    The number of nodes in set B is random.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    p :  float
       Probability that a new bottom node is added.
    create_using : NetworkX graph instance, optional
       Return graph of this type.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    References
    ----------
    .. [1] Guillaume, J.L. and Latapy, M.,
       Bipartite graphs as models of complex networks.
       Physica A: Statistical Mechanics and its Applications,
       2006, 371(2), pp.795-813.
    .. [2] Jean-Loup Guillaume and Matthieu Latapy,
       Bipartite structure of all complex networks,
       Inf. Process. Lett. 90, 2004, pg. 215-221
       https://doi.org/10.1016/j.ipl.2004.03.007

    Notes
    -----
    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.preferential_attachment_graph
    """
    if create_using is None:
        create_using = nx.Graph()
    elif create_using.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")

    G = create_using
    G.clear()

    if p < 0 or p > 1:
        raise nx.NetworkXError("Probability p must be in [0,1]")

    n = len(aseq)
    m = 0  # Number of nodes in set B

    rng = seed if isinstance(seed, nx.utils.RandomState) else nx.utils.RandomState(seed)

    for i in range(n):
        G.add_node(i, bipartite=0)

    stubs = list(range(n)) * aseq[0]
    for i in range(1, n):
        for _ in range(aseq[i]):
            if rng.random() < p:
                # Add a new node to set B
                new_node = n + m
                G.add_node(new_node, bipartite=1)
                G.add_edge(i, new_node)
                stubs.append(new_node)
                m += 1
            else:
                # Connect to an existing node in set B
                j = rng.choice(stubs)
                G.add_edge(i, j)
            stubs.append(i)

    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_graph(n, m, p, seed=None, directed=False):
    """Returns a bipartite random graph.

    This is a bipartite version of the binomial (Erdős-Rényi) graph.
    The graph is composed of two partitions. Set A has nodes 0 to
    (n - 1) and set B has nodes n to (n + m - 1).

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set.
    m : int
        The number of nodes in the second bipartite set.
    p : float
        Probability for edge creation.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True return a directed graph

    Notes
    -----
    The bipartite random graph algorithm chooses each of the n*m (undirected)
    or 2*nm (directed) possible edges with probability p.

    This algorithm is $O(n+m)$ where $m$ is the expected number of edges.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.random_graph

    See Also
    --------
    gnp_random_graph, configuration_model

    References
    ----------
    .. [1] Vladimir Batagelj and Ulrik Brandes,
       "Efficient generation of large random networks",
       Phys. Rev. E, 71, 036113, 2005.
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(range(n), bipartite=0)
    G.add_nodes_from(range(n, n + m), bipartite=1)

    rng = seed if isinstance(seed, nx.utils.RandomState) else nx.utils.RandomState(seed)

    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_bipartite_graph(n, m)

    lp = math.log(1.0 - p)

    v = 0
    w = -1
    while v < n:
        lr = math.log(1.0 - rng.random())
        w = w + 1 + int(lr / lp)
        while w >= m and v < n:
            w = w - m
            v = v + 1
        if v < n:
            G.add_edge(v, n + w)

    if directed:
        # Add edges in the reverse direction
        v = 0
        w = -1
        while v < n:
            lr = math.log(1.0 - rng.random())
            w = w + 1 + int(lr / lp)
            while w >= m and v < n:
                w = w - m
                v = v + 1
            if v < n:
                G.add_edge(n + w, v)

    return G


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def gnmk_random_graph(n, m, k, seed=None, directed=False):
    """Returns a random bipartite graph G_{n,m,k}.

    Produces a bipartite graph chosen randomly out of the set of all graphs
    with n top nodes, m bottom nodes, and k edges.
    The graph is composed of two sets of nodes.
    Set A has nodes 0 to (n - 1) and set B has nodes n to (n + m - 1).

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set.
    m : int
        The number of nodes in the second bipartite set.
    k : int
        The number of edges
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True return a directed graph

    Examples
    --------
    from nx.algorithms import bipartite
    G = bipartite.gnmk_random_graph(10,20,50)

    See Also
    --------
    gnm_random_graph

    Notes
    -----
    If k > m * n then a complete bipartite graph is returned.

    This graph is a bipartite version of the `G_{nm}` random graph model.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.gnmk_random_graph
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(range(n), bipartite=0)
    G.add_nodes_from(range(n, n + m), bipartite=1)

    if k > n * m:
        return nx.complete_bipartite_graph(n, m)

    rng = seed if isinstance(seed, nx.utils.RandomState) else nx.utils.RandomState(seed)

    edge_count = 0
    while edge_count < k:
        u = rng.randint(0, n - 1)
        v = rng.randint(n, n + m - 1)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            edge_count += 1

    return G
