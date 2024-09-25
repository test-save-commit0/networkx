"""Algorithms to characterize the number of triangles in a graph."""
from collections import Counter
from itertools import chain, combinations
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['triangles', 'average_clustering', 'clustering', 'transitivity',
    'square_clustering', 'generalized_degree']


@not_implemented_for('directed')
@nx._dispatchable
def triangles(G, nodes=None):
    """Compute the number of triangles.

    Finds the number of triangles that include a node as one vertex.

    Parameters
    ----------
    G : graph
       A networkx graph

    nodes : node, iterable of nodes, or None (default=None)
        If a singleton node, return the number of triangles for that node.
        If an iterable, compute the number of triangles for each of those nodes.
        If `None` (the default) compute the number of triangles for all nodes in `G`.

    Returns
    -------
    out : dict or int
       If `nodes` is a container of nodes, returns number of triangles keyed by node (dict).
       If `nodes` is a specific node, returns number of triangles for the node (int).

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.triangles(G, 0))
    6
    >>> print(nx.triangles(G))
    {0: 6, 1: 6, 2: 6, 3: 6, 4: 6}
    >>> print(list(nx.triangles(G, [0, 1]).values()))
    [6, 6]

    Notes
    -----
    Self loops are ignored.

    """
    if nodes in G:
        return sum(1 for _ in nx.triangles_iter(G, nodes)) // 2
    elif nodes is None:
        return {n: sum(1 for _ in nx.triangles_iter(G, n)) // 2 for n in G}
    else:
        return {n: sum(1 for _ in nx.triangles_iter(G, n)) // 2 for n in nodes if n in G}


@not_implemented_for('multigraph')
def _triangles_and_degree_iter(G, nodes=None):
    """Return an iterator of (node, degree, triangles, generalized degree).

    This double counts triangles so you may want to divide by 2.
    See degree(), triangles() and generalized_degree() for definitions
    and details.

    """
    if nodes is None:
        nodes = G
    for v in nodes:
        deg = G.degree(v)
        ntriangles = sum(1 for _ in nx.triangles_iter(G, v))
        gen_deg = Counter(len(set(G[u]) & set(G[w])) for u, w in combinations(G[v], 2))
        yield (v, deg, ntriangles, gen_deg)


@not_implemented_for('multigraph')
def _weighted_triangles_and_degree_iter(G, nodes=None, weight='weight'):
    """Return an iterator of (node, degree, weighted_triangles).

    Used for weighted clustering.
    Note: this returns the geometric average weight of edges in the triangle.
    Also, each triangle is counted twice (each direction).
    So you may want to divide by 2.

    """
    if nodes is None:
        nodes = G
    for v in nodes:
        deg = G.degree(v, weight=weight)
        wtriangles = 0
        for u, w in combinations(G[v], 2):
            if w in G[u]:
                wtriangles += (G[v][u].get(weight, 1) *
                               G[v][w].get(weight, 1) *
                               G[u][w].get(weight, 1)) ** (1/3)
        yield (v, deg, wtriangles)


@not_implemented_for('multigraph')
def _directed_triangles_and_degree_iter(G, nodes=None):
    """Return an iterator of
    (node, total_degree, reciprocal_degree, directed_triangles).

    Used for directed clustering.
    Note that unlike `_triangles_and_degree_iter()`, this function counts
    directed triangles so does not count triangles twice.

    """
    if nodes is None:
        nodes = G
    for v in nodes:
        dtriangles = 0
        for u, w in permutations(G.successors(v), 2):
            if G.has_edge(u, w):
                dtriangles += 1
        total_degree = G.in_degree(v) + G.out_degree(v)
        reciprocal_degree = sum(1 for u in G.successors(v) if G.has_edge(u, v))
        yield (v, total_degree, reciprocal_degree, dtriangles)


@not_implemented_for('multigraph')
def _directed_weighted_triangles_and_degree_iter(G, nodes=None, weight='weight'):
    """Return an iterator of
    (node, total_degree, reciprocal_degree, directed_weighted_triangles).

    Used for directed weighted clustering.
    Note that unlike `_weighted_triangles_and_degree_iter()`, this function counts
    directed triangles so does not count triangles twice.

    """
    if nodes is None:
        nodes = G
    for v in nodes:
        dwtriangles = 0
        for u, w in permutations(G.successors(v), 2):
            if G.has_edge(u, w):
                dwtriangles += (G[v][u].get(weight, 1) *
                                G[v][w].get(weight, 1) *
                                G[u][w].get(weight, 1)) ** (1/3)
        total_degree = sum(d.get(weight, 1) for u, d in G.pred[v].items()) + \
                       sum(d.get(weight, 1) for u, d in G.succ[v].items())
        reciprocal_degree = sum(1 for u in G.successors(v) if G.has_edge(u, v))
        yield (v, total_degree, reciprocal_degree, dwtriangles)


@nx._dispatchable(edge_attrs='weight')
def average_clustering(G, nodes=None, weight=None, count_zeros=True):
    """Compute the average clustering coefficient for the graph G.

    The clustering coefficient for the graph is the average,

    .. math::

       C = \\frac{1}{n}\\sum_{v \\in G} c_v,

    where :math:`n` is the number of nodes in `G`.

    Parameters
    ----------
    G : graph

    nodes : container of nodes, optional (default=all nodes in G)
       Compute average clustering for nodes in this container.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    count_zeros : bool
       If False include only the nodes with nonzero clustering in the average.

    Returns
    -------
    avg : float
       Average clustering

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.average_clustering(G))
    1.0

    Notes
    -----
    This is a space saving routine; it might be faster
    to use the clustering function to get a list and then take the average.

    Self loops are ignored.

    References
    ----------
    .. [1] Generalizations of the clustering coefficient to weighted
       complex networks by J. Saramäki, M. Kivelä, J.-P. Onnela,
       K. Kaski, and J. Kertész, Physical Review E, 75 027105 (2007).
       http://jponnela.com/web_documents/a9.pdf
    .. [2] Marcus Kaiser,  Mean clustering coefficients: the role of isolated
       nodes and leafs on clustering measures for small-world networks.
       https://arxiv.org/abs/0802.2512
    """
    c = clustering(G, nodes, weight)
    if not count_zeros:
        c = {n: v for n, v in c.items() if v != 0}
    return sum(c.values()) / len(c) if len(c) > 0 else 0.0


@nx._dispatchable(edge_attrs='weight')
def clustering(G, nodes=None, weight=None):
    """Compute the clustering coefficient for nodes.

    For unweighted graphs, the clustering of a node :math:`u`
    is the fraction of possible triangles through that node that exist,

    .. math::

      c_u = \\frac{2 T(u)}{deg(u)(deg(u)-1)},

    where :math:`T(u)` is the number of triangles through node :math:`u` and
    :math:`deg(u)` is the degree of :math:`u`.

    For weighted graphs, there are several ways to define clustering [1]_.
    the one used here is defined
    as the geometric average of the subgraph edge weights [2]_,

    .. math::

       c_u = \\frac{1}{deg(u)(deg(u)-1))}
             \\sum_{vw} (\\hat{w}_{uv} \\hat{w}_{uw} \\hat{w}_{vw})^{1/3}.

    The edge weights :math:`\\hat{w}_{uv}` are normalized by the maximum weight
    in the network :math:`\\hat{w}_{uv} = w_{uv}/\\max(w)`.

    The value of :math:`c_u` is assigned to 0 if :math:`deg(u) < 2`.

    Additionally, this weighted definition has been generalized to support negative edge weights [3]_.

    For directed graphs, the clustering is similarly defined as the fraction
    of all possible directed triangles or geometric average of the subgraph
    edge weights for unweighted and weighted directed graph respectively [4]_.

    .. math::

       c_u = \\frac{T(u)}{2(deg^{tot}(u)(deg^{tot}(u)-1) - 2deg^{\\leftrightarrow}(u))},

    where :math:`T(u)` is the number of directed triangles through node
    :math:`u`, :math:`deg^{tot}(u)` is the sum of in degree and out degree of
    :math:`u` and :math:`deg^{\\leftrightarrow}(u)` is the reciprocal degree of
    :math:`u`.


    Parameters
    ----------
    G : graph

    nodes : node, iterable of nodes, or None (default=None)
        If a singleton node, return the number of triangles for that node.
        If an iterable, compute the number of triangles for each of those nodes.
        If `None` (the default) compute the number of triangles for all nodes in `G`.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    Returns
    -------
    out : float, or dictionary
       Clustering coefficient at specified nodes

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.clustering(G, 0))
    1.0
    >>> print(nx.clustering(G))
    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    Notes
    -----
    Self loops are ignored.

    References
    ----------
    .. [1] Generalizations of the clustering coefficient to weighted
       complex networks by J. Saramäki, M. Kivelä, J.-P. Onnela,
       K. Kaski, and J. Kertész, Physical Review E, 75 027105 (2007).
       http://jponnela.com/web_documents/a9.pdf
    .. [2] Intensity and coherence of motifs in weighted complex
       networks by J. P. Onnela, J. Saramäki, J. Kertész, and K. Kaski,
       Physical Review E, 71(6), 065103 (2005).
    .. [3] Generalization of Clustering Coefficients to Signed Correlation Networks
       by G. Costantini and M. Perugini, PloS one, 9(2), e88669 (2014).
    .. [4] Clustering in complex directed networks by G. Fagiolo,
       Physical Review E, 76(2), 026107 (2007).
    """
    if G.is_directed():
        if weight is not None:
            td_iter = _directed_weighted_triangles_and_degree_iter(G, nodes, weight)
            clusterc = {v: 0 if t == 0 else t / ((dt * (dt - 1) - 2 * db) * 2) for
                        v, dt, db, t in td_iter}
        else:
            td_iter = _directed_triangles_and_degree_iter(G, nodes)
            clusterc = {v: 0 if t == 0 else t / ((dt * (dt - 1) - 2 * db) * 2) for
                        v, dt, db, t in td_iter}
    else:
        if weight is not None:
            td_iter = _weighted_triangles_and_degree_iter(G, nodes, weight)
            clusterc = {v: 0 if d < 2 else t / (d * (d - 1)) for
                        v, d, t in td_iter}
        else:
            td_iter = _triangles_and_degree_iter(G, nodes)
            clusterc = {v: 0 if d < 2 else t / (d * (d - 1)) for
                        v, d, t, _ in td_iter}
    if nodes in G:
        return clusterc[nodes]
    return clusterc


@nx._dispatchable
def transitivity(G):
    """Compute graph transitivity, the fraction of all possible triangles
    present in G.

    Possible triangles are identified by the number of "triads"
    (two edges with a shared vertex).

    The transitivity is

    .. math::

        T = 3\\frac{\\#triangles}{\\#triads}.

    Parameters
    ----------
    G : graph

    Returns
    -------
    out : float
       Transitivity

    Notes
    -----
    Self loops are ignored.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.transitivity(G))
    1.0
    """
    triangles = sum(nx.triangles(G).values()) / 3
    contri = sum(d * (d - 1) for d in dict(G.degree()).values())
    return 0 if triangles == 0 else triangles * 6.0 / contri


@nx._dispatchable
def square_clustering(G, nodes=None):
    """Compute the squares clustering coefficient for nodes.

    For each node return the fraction of possible squares that exist at
    the node [1]_

    .. math::
       C_4(v) = \\frac{ \\sum_{u=1}^{k_v}
       \\sum_{w=u+1}^{k_v} q_v(u,w) }{ \\sum_{u=1}^{k_v}
       \\sum_{w=u+1}^{k_v} [a_v(u,w) + q_v(u,w)]},

    where :math:`q_v(u,w)` are the number of common neighbors of :math:`u` and
    :math:`w` other than :math:`v` (ie squares), and :math:`a_v(u,w) = (k_u -
    (1+q_v(u,w)+\\theta_{uv})) + (k_w - (1+q_v(u,w)+\\theta_{uw}))`, where
    :math:`\\theta_{uw} = 1` if :math:`u` and :math:`w` are connected and 0
    otherwise. [2]_

    Parameters
    ----------
    G : graph

    nodes : container of nodes, optional (default=all nodes in G)
       Compute clustering for nodes in this container.

    Returns
    -------
    c4 : dictionary
       A dictionary keyed by node with the square clustering coefficient value.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.square_clustering(G, 0))
    1.0
    >>> print(nx.square_clustering(G))
    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    Notes
    -----
    While :math:`C_3(v)` (triangle clustering) gives the probability that
    two neighbors of node v are connected with each other, :math:`C_4(v)` is
    the probability that two neighbors of node v share a common
    neighbor different from v. This algorithm can be applied to both
    bipartite and unipartite networks.

    References
    ----------
    .. [1] Pedro G. Lind, Marta C. González, and Hans J. Herrmann. 2005
        Cycles and clustering in bipartite networks.
        Physical Review E (72) 056127.
    .. [2] Zhang, Peng et al. Clustering Coefficient and Community Structure of
        Bipartite Networks. Physica A: Statistical Mechanics and its Applications 387.27 (2008): 6869–6875.
        https://arxiv.org/abs/0710.0117v1
    """
    if nodes is None:
        nodes = G
    clustering = {}
    for v in nodes:
        clustering[v] = 0
        potential = 0
        for u, w in combinations(G[v], 2):
            squares = len((set(G[u]) & set(G[w])) - {v})
            clustering[v] += squares
            degm = squares + 1
            if w in G[u]:
                degm += 1
            potential += (len(G[u]) - degm) + (len(G[w]) - degm) + squares
        if potential > 0:
            clustering[v] /= potential
    if nodes in G:
        return clustering[nodes]
    return clustering


@not_implemented_for('directed')
@nx._dispatchable
def generalized_degree(G, nodes=None):
    """Compute the generalized degree for nodes.

    For each node, the generalized degree shows how many edges of given
    triangle multiplicity the node is connected to. The triangle multiplicity
    of an edge is the number of triangles an edge participates in. The
    generalized degree of node :math:`i` can be written as a vector
    :math:`\\mathbf{k}_i=(k_i^{(0)}, \\dotsc, k_i^{(N-2)})` where
    :math:`k_i^{(j)}` is the number of edges attached to node :math:`i` that
    participate in :math:`j` triangles.

    Parameters
    ----------
    G : graph

    nodes : container of nodes, optional (default=all nodes in G)
       Compute the generalized degree for nodes in this container.

    Returns
    -------
    out : Counter, or dictionary of Counters
       Generalized degree of specified nodes. The Counter is keyed by edge
       triangle multiplicity.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.generalized_degree(G, 0))
    Counter({3: 4})
    >>> print(nx.generalized_degree(G))
    {0: Counter({3: 4}), 1: Counter({3: 4}), 2: Counter({3: 4}), 3: Counter({3: 4}), 4: Counter({3: 4})}

    To recover the number of triangles attached to a node:

    >>> k1 = nx.generalized_degree(G, 0)
    >>> sum([k * v for k, v in k1.items()]) / 2 == nx.triangles(G, 0)
    True

    Notes
    -----
    Self loops are ignored.

    In a network of N nodes, the highest triangle multiplicity an edge can have
    is N-2.

    The return value does not include a `zero` entry if no edges of a
    particular triangle multiplicity are present.

    The number of triangles node :math:`i` is attached to can be recovered from
    the generalized degree :math:`\\mathbf{k}_i=(k_i^{(0)}, \\dotsc,
    k_i^{(N-2)})` by :math:`(k_i^{(1)}+2k_i^{(2)}+\\dotsc +(N-2)k_i^{(N-2)})/2`.

    References
    ----------
    .. [1] Networks with arbitrary edge multiplicities by V. Zlatić,
        D. Garlaschelli and G. Caldarelli, EPL (Europhysics Letters),
        Volume 97, Number 2 (2012).
        https://iopscience.iop.org/article/10.1209/0295-5075/97/28005
    """
    if nodes in G:
        return next(generalized_degree_iter(G, nodes))[1]
    return {v: gd for v, gd in generalized_degree_iter(G, nodes)}
