"""Provides explicit constructions of expander graphs.

"""
import itertools
import networkx as nx
__all__ = ['margulis_gabber_galil_graph', 'chordal_cycle_graph',
    'paley_graph', 'maybe_regular_expander', 'is_regular_expander',
    'random_regular_expander_graph']


@nx._dispatchable(graphs=None, returns_graph=True)
def margulis_gabber_galil_graph(n, create_using=None):
    """Returns the Margulis-Gabber-Galil undirected MultiGraph on `n^2` nodes.

    The undirected MultiGraph is regular with degree `8`. Nodes are integer
    pairs. The second-largest eigenvalue of the adjacency matrix of the graph
    is at most `5 \\sqrt{2}`, regardless of `n`.

    Parameters
    ----------
    n : int
        Determines the number of nodes in the graph: `n^2`.
    create_using : NetworkX graph constructor, optional (default MultiGraph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : graph
        The constructed undirected multigraph.

    Raises
    ------
    NetworkXError
        If the graph is directed or not a multigraph.

    """
    pass


@nx._dispatchable(graphs=None, returns_graph=True)
def chordal_cycle_graph(p, create_using=None):
    """Returns the chordal cycle graph on `p` nodes.

    The returned graph is a cycle graph on `p` nodes with chords joining each
    vertex `x` to its inverse modulo `p`. This graph is a (mildly explicit)
    3-regular expander [1]_.

    `p` *must* be a prime number.

    Parameters
    ----------
    p : a prime number

        The number of vertices in the graph. This also indicates where the
        chordal edges in the cycle will be created.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : graph
        The constructed undirected multigraph.

    Raises
    ------
    NetworkXError

        If `create_using` indicates directed or not a multigraph.

    References
    ----------

    .. [1] Theorem 4.4.2 in A. Lubotzky. "Discrete groups, expanding graphs and
           invariant measures", volume 125 of Progress in Mathematics.
           Birkhäuser Verlag, Basel, 1994.

    """
    pass


@nx._dispatchable(graphs=None, returns_graph=True)
def paley_graph(p, create_using=None):
    """Returns the Paley $\\frac{(p-1)}{2}$ -regular graph on $p$ nodes.

    The returned graph is a graph on $\\mathbb{Z}/p\\mathbb{Z}$ with edges between $x$ and $y$
    if and only if $x-y$ is a nonzero square in $\\mathbb{Z}/p\\mathbb{Z}$.

    If $p \\equiv 1  \\pmod 4$, $-1$ is a square in $\\mathbb{Z}/p\\mathbb{Z}$ and therefore $x-y$ is a square if and
    only if $y-x$ is also a square, i.e the edges in the Paley graph are symmetric.

    If $p \\equiv 3 \\pmod 4$, $-1$ is not a square in $\\mathbb{Z}/p\\mathbb{Z}$ and therefore either $x-y$ or $y-x$
    is a square in $\\mathbb{Z}/p\\mathbb{Z}$ but not both.

    Note that a more general definition of Paley graphs extends this construction
    to graphs over $q=p^n$ vertices, by using the finite field $F_q$ instead of $\\mathbb{Z}/p\\mathbb{Z}$.
    This construction requires to compute squares in general finite fields and is
    not what is implemented here (i.e `paley_graph(25)` does not return the true
    Paley graph associated with $5^2$).

    Parameters
    ----------
    p : int, an odd prime number.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : graph
        The constructed directed graph.

    Raises
    ------
    NetworkXError
        If the graph is a multigraph.

    References
    ----------
    Chapter 13 in B. Bollobas, Random Graphs. Second edition.
    Cambridge Studies in Advanced Mathematics, 73.
    Cambridge University Press, Cambridge (2001).
    """
    pass


@nx.utils.decorators.np_random_state('seed')
@nx._dispatchable(graphs=None, returns_graph=True)
def maybe_regular_expander(n, d, *, create_using=None, max_tries=100, seed=None
    ):
    """Utility for creating a random regular expander.

    Returns a random $d$-regular graph on $n$ nodes which is an expander
    graph with very good probability.

    Parameters
    ----------
    n : int
      The number of nodes.
    d : int
      The degree of each node.
    create_using : Graph Instance or Constructor
      Indicator of type of graph to return.
      If a Graph-type instance, then clear and use it.
      If a constructor, call it to create an empty graph.
      Use the Graph constructor by default.
    max_tries : int. (default: 100)
      The number of allowed loops when generating each independent cycle
    seed : (default: None)
      Seed used to set random number generation state. See :ref`Randomness<randomness>`.

    Notes
    -----
    The nodes are numbered from $0$ to $n - 1$.

    The graph is generated by taking $d / 2$ random independent cycles.

    Joel Friedman proved that in this model the resulting
    graph is an expander with probability
    $1 - O(n^{-\\tau})$ where $\\tau = \\lceil (\\sqrt{d - 1}) / 2 \\rceil - 1$. [1]_

    Examples
    --------
    >>> G = nx.maybe_regular_expander(n=200, d=6, seed=8020)

    Returns
    -------
    G : graph
        The constructed undirected graph.

    Raises
    ------
    NetworkXError
        If $d % 2 != 0$ as the degree must be even.
        If $n - 1$ is less than $ 2d $ as the graph is complete at most.
        If max_tries is reached

    See Also
    --------
    is_regular_expander
    random_regular_expander_graph

    References
    ----------
    .. [1] Joel Friedman,
       A Proof of Alon’s Second Eigenvalue Conjecture and Related Problems, 2004
       https://arxiv.org/abs/cs/0405020

    """
    pass


@nx.utils.not_implemented_for('directed')
@nx.utils.not_implemented_for('multigraph')
@nx._dispatchable(preserve_edge_attrs={'G': {'weight': 1}})
def is_regular_expander(G, *, epsilon=0):
    """Determines whether the graph G is a regular expander. [1]_

    An expander graph is a sparse graph with strong connectivity properties.

    More precisely, this helper checks whether the graph is a
    regular $(n, d, \\lambda)$-expander with $\\lambda$ close to
    the Alon-Boppana bound and given by
    $\\lambda = 2 \\sqrt{d - 1} + \\epsilon$. [2]_

    In the case where $\\epsilon = 0$ then if the graph successfully passes the test
    it is a Ramanujan graph. [3]_

    A Ramanujan graph has spectral gap almost as large as possible, which makes them
    excellent expanders.

    Parameters
    ----------
    G : NetworkX graph
    epsilon : int, float, default=0

    Returns
    -------
    bool
        Whether the given graph is a regular $(n, d, \\lambda)$-expander
        where $\\lambda = 2 \\sqrt{d - 1} + \\epsilon$.

    Examples
    --------
    >>> G = nx.random_regular_expander_graph(20, 4)
    >>> nx.is_regular_expander(G)
    True

    See Also
    --------
    maybe_regular_expander
    random_regular_expander_graph

    References
    ----------
    .. [1] Expander graph, https://en.wikipedia.org/wiki/Expander_graph
    .. [2] Alon-Boppana bound, https://en.wikipedia.org/wiki/Alon%E2%80%93Boppana_bound
    .. [3] Ramanujan graphs, https://en.wikipedia.org/wiki/Ramanujan_graph

    """
    pass


@nx.utils.decorators.np_random_state('seed')
@nx._dispatchable(graphs=None, returns_graph=True)
def random_regular_expander_graph(n, d, *, epsilon=0, create_using=None,
    max_tries=100, seed=None):
    """Returns a random regular expander graph on $n$ nodes with degree $d$.

    An expander graph is a sparse graph with strong connectivity properties. [1]_

    More precisely the returned graph is a $(n, d, \\lambda)$-expander with
    $\\lambda = 2 \\sqrt{d - 1} + \\epsilon$, close to the Alon-Boppana bound. [2]_

    In the case where $\\epsilon = 0$ it returns a Ramanujan graph.
    A Ramanujan graph has spectral gap almost as large as possible,
    which makes them excellent expanders. [3]_

    Parameters
    ----------
    n : int
      The number of nodes.
    d : int
      The degree of each node.
    epsilon : int, float, default=0
    max_tries : int, (default: 100)
      The number of allowed loops, also used in the maybe_regular_expander utility
    seed : (default: None)
      Seed used to set random number generation state. See :ref`Randomness<randomness>`.

    Raises
    ------
    NetworkXError
        If max_tries is reached

    Examples
    --------
    >>> G = nx.random_regular_expander_graph(20, 4)
    >>> nx.is_regular_expander(G)
    True

    Notes
    -----
    This loops over `maybe_regular_expander` and can be slow when
    $n$ is too big or $\\epsilon$ too small.

    See Also
    --------
    maybe_regular_expander
    is_regular_expander

    References
    ----------
    .. [1] Expander graph, https://en.wikipedia.org/wiki/Expander_graph
    .. [2] Alon-Boppana bound, https://en.wikipedia.org/wiki/Alon%E2%80%93Boppana_bound
    .. [3] Ramanujan graphs, https://en.wikipedia.org/wiki/Ramanujan_graph

    """
    pass
