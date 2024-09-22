"""Functions for analyzing triads of a graph."""
from collections import defaultdict
from itertools import combinations, permutations
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
__all__ = ['triadic_census', 'is_triad', 'all_triplets', 'all_triads',
    'triads_by_type', 'triad_type', 'random_triad']
TRICODES = (1, 2, 2, 3, 2, 4, 6, 8, 2, 6, 5, 7, 3, 8, 7, 11, 2, 6, 4, 8, 5,
    9, 9, 13, 6, 10, 9, 14, 7, 14, 12, 15, 2, 5, 6, 7, 6, 9, 10, 14, 4, 9, 
    9, 12, 8, 13, 14, 15, 3, 7, 8, 11, 7, 12, 14, 15, 8, 14, 13, 15, 11, 15,
    15, 16)
TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U',
    '030T', '030C', '201', '120D', '120U', '120C', '210', '300')
TRICODE_TO_NAME = {i: TRIAD_NAMES[code - 1] for i, code in enumerate(TRICODES)}


def _tricode(G, v, u, w):
    """Returns the integer code of the given triad.

    This is some fancy magic that comes from Batagelj and Mrvar's paper. It
    treats each edge joining a pair of `v`, `u`, and `w` as a bit in
    the binary representation of an integer.

    """
    pass


@not_implemented_for('undirected')
@nx._dispatchable
def triadic_census(G, nodelist=None):
    """Determines the triadic census of a directed graph.

    The triadic census is a count of how many of the 16 possible types of
    triads are present in a directed graph. If a list of nodes is passed, then
    only those triads are taken into account which have elements of nodelist in them.

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph
    nodelist : list
        List of nodes for which you want to calculate triadic census

    Returns
    -------
    census : dict
       Dictionary with triad type as keys and number of occurrences as values.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4), (4, 1), (4, 2)])
    >>> triadic_census = nx.triadic_census(G)
    >>> for key, value in triadic_census.items():
    ...     print(f"{key}: {value}")
    003: 0
    012: 0
    102: 0
    021D: 0
    021U: 0
    021C: 0
    111D: 0
    111U: 0
    030T: 2
    030C: 2
    201: 0
    120D: 0
    120U: 0
    120C: 0
    210: 0
    300: 0

    Notes
    -----
    This algorithm has complexity $O(m)$ where $m$ is the number of edges in
    the graph.

    For undirected graphs, the triadic census can be computed by first converting
    the graph into a directed graph using the ``G.to_directed()`` method.
    After this conversion, only the triad types 003, 102, 201 and 300 will be
    present in the undirected scenario.

    Raises
    ------
    ValueError
        If `nodelist` contains duplicate nodes or nodes not in `G`.
        If you want to ignore this you can preprocess with `set(nodelist) & G.nodes`

    See also
    --------
    triad_graph

    References
    ----------
    .. [1] Vladimir Batagelj and Andrej Mrvar, A subquadratic triad census
        algorithm for large sparse networks with small maximum degree,
        University of Ljubljana,
        http://vlado.fmf.uni-lj.si/pub/networks/doc/triads/triads.pdf

    """
    pass


@nx._dispatchable
def is_triad(G):
    """Returns True if the graph G is a triad, else False.

    Parameters
    ----------
    G : graph
       A NetworkX Graph

    Returns
    -------
    istriad : boolean
       Whether G is a valid triad

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    >>> nx.is_triad(G)
    True
    >>> G.add_edge(0, 1)
    >>> nx.is_triad(G)
    False
    """
    pass


@not_implemented_for('undirected')
@nx._dispatchable
def all_triplets(G):
    """Returns a generator of all possible sets of 3 nodes in a DiGraph.

    .. deprecated:: 3.3

       all_triplets is deprecated and will be removed in NetworkX version 3.5.
       Use `itertools.combinations` instead::

          all_triplets = itertools.combinations(G, 3)

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph

    Returns
    -------
    triplets : generator of 3-tuples
       Generator of tuples of 3 nodes

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    >>> list(nx.all_triplets(G))
    [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]

    """
    pass


@not_implemented_for('undirected')
@nx._dispatchable(returns_graph=True)
def all_triads(G):
    """A generator of all possible triads in G.

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph

    Returns
    -------
    all_triads : generator of DiGraphs
       Generator of triads (order-3 DiGraphs)

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4), (4, 1), (4, 2)])
    >>> for triad in nx.all_triads(G):
    ...     print(triad.edges)
    [(1, 2), (2, 3), (3, 1)]
    [(1, 2), (4, 1), (4, 2)]
    [(3, 1), (3, 4), (4, 1)]
    [(2, 3), (3, 4), (4, 2)]

    """
    pass


@not_implemented_for('undirected')
@nx._dispatchable
def triads_by_type(G):
    """Returns a list of all triads for each triad type in a directed graph.
    There are exactly 16 different types of triads possible. Suppose 1, 2, 3 are three
    nodes, they will be classified as a particular triad type if their connections
    are as follows:

    - 003: 1, 2, 3
    - 012: 1 -> 2, 3
    - 102: 1 <-> 2, 3
    - 021D: 1 <- 2 -> 3
    - 021U: 1 -> 2 <- 3
    - 021C: 1 -> 2 -> 3
    - 111D: 1 <-> 2 <- 3
    - 111U: 1 <-> 2 -> 3
    - 030T: 1 -> 2 -> 3, 1 -> 3
    - 030C: 1 <- 2 <- 3, 1 -> 3
    - 201: 1 <-> 2 <-> 3
    - 120D: 1 <- 2 -> 3, 1 <-> 3
    - 120U: 1 -> 2 <- 3, 1 <-> 3
    - 120C: 1 -> 2 -> 3, 1 <-> 3
    - 210: 1 -> 2 <-> 3, 1 <-> 3
    - 300: 1 <-> 2 <-> 3, 1 <-> 3

    Refer to the :doc:`example gallery </auto_examples/graph/plot_triad_types>`
    for visual examples of the triad types.

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph

    Returns
    -------
    tri_by_type : dict
       Dictionary with triad types as keys and lists of triads as values.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3), (3, 1), (5, 6), (5, 4), (6, 7)])
    >>> dict = nx.triads_by_type(G)
    >>> dict["120C"][0].edges()
    OutEdgeView([(1, 2), (1, 3), (2, 3), (3, 1)])
    >>> dict["012"][0].edges()
    OutEdgeView([(1, 2)])

    References
    ----------
    .. [1] Snijders, T. (2012). "Transitivity and triads." University of
        Oxford.
        https://web.archive.org/web/20170830032057/http://www.stats.ox.ac.uk/~snijders/Trans_Triads_ha.pdf
    """
    pass


@not_implemented_for('undirected')
@nx._dispatchable
def triad_type(G):
    """Returns the sociological triad type for a triad.

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph with 3 nodes

    Returns
    -------
    triad_type : str
       A string identifying the triad type

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    >>> nx.triad_type(G)
    '030C'
    >>> G.add_edge(1, 3)
    >>> nx.triad_type(G)
    '120C'

    Notes
    -----
    There can be 6 unique edges in a triad (order-3 DiGraph) (so 2^^6=64 unique
    triads given 3 nodes). These 64 triads each display exactly 1 of 16
    topologies of triads (topologies can be permuted). These topologies are
    identified by the following notation:

    {m}{a}{n}{type} (for example: 111D, 210, 102)

    Here:

    {m}     = number of mutual ties (takes 0, 1, 2, 3); a mutual tie is (0,1)
              AND (1,0)
    {a}     = number of asymmetric ties (takes 0, 1, 2, 3); an asymmetric tie
              is (0,1) BUT NOT (1,0) or vice versa
    {n}     = number of null ties (takes 0, 1, 2, 3); a null tie is NEITHER
              (0,1) NOR (1,0)
    {type}  = a letter (takes U, D, C, T) corresponding to up, down, cyclical
              and transitive. This is only used for topologies that can have
              more than one form (eg: 021D and 021U).

    References
    ----------
    .. [1] Snijders, T. (2012). "Transitivity and triads." University of
        Oxford.
        https://web.archive.org/web/20170830032057/http://www.stats.ox.ac.uk/~snijders/Trans_Triads_ha.pdf
    """
    pass


@not_implemented_for('undirected')
@py_random_state(1)
@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def random_triad(G, seed=None):
    """Returns a random triad from a directed graph.

    .. deprecated:: 3.3

       random_triad is deprecated and will be removed in version 3.5.
       Use random sampling directly instead::

          G.subgraph(random.sample(list(G), 3))

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G2 : subgraph
       A randomly selected triad (order-3 NetworkX DiGraph)

    Raises
    ------
    NetworkXError
        If the input Graph has less than 3 nodes.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3), (3, 1), (5, 6), (5, 4), (6, 7)])
    >>> triad = nx.random_triad(G, seed=1)
    >>> triad.edges
    OutEdgeView([(1, 2)])

    """
    pass
