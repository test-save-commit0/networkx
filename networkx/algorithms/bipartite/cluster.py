"""Functions for computing clustering of pairs

"""
import itertools
import networkx as nx
__all__ = ['clustering', 'average_clustering', 'latapy_clustering',
    'robins_alexander_clustering']
modes = {'dot': cc_dot, 'min': cc_min, 'max': cc_max}


@nx._dispatchable
def latapy_clustering(G, nodes=None, mode='dot'):
    """Compute a bipartite clustering coefficient for nodes.

    The bipartite clustering coefficient is a measure of local density
    of connections defined as [1]_:

    .. math::

       c_u = \\frac{\\sum_{v \\in N(N(u))} c_{uv} }{|N(N(u))|}

    where `N(N(u))` are the second order neighbors of `u` in `G` excluding `u`,
    and `c_{uv}` is the pairwise clustering coefficient between nodes
    `u` and `v`.

    The mode selects the function for `c_{uv}` which can be:

    `dot`:

    .. math::

       c_{uv}=\\frac{|N(u)\\cap N(v)|}{|N(u) \\cup N(v)|}

    `min`:

    .. math::

       c_{uv}=\\frac{|N(u)\\cap N(v)|}{min(|N(u)|,|N(v)|)}

    `max`:

    .. math::

       c_{uv}=\\frac{|N(u)\\cap N(v)|}{max(|N(u)|,|N(v)|)}


    Parameters
    ----------
    G : graph
        A bipartite graph

    nodes : list or iterable (optional)
        Compute bipartite clustering for these nodes. The default
        is all nodes in G.

    mode : string
        The pairwise bipartite clustering method to be used in the computation.
        It must be "dot", "max", or "min".

    Returns
    -------
    clustering : dictionary
        A dictionary keyed by node with the clustering coefficient value.


    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)  # path graphs are bipartite
    >>> c = bipartite.clustering(G)
    >>> c[0]
    0.5
    >>> c = bipartite.clustering(G, mode="min")
    >>> c[0]
    1.0

    See Also
    --------
    robins_alexander_clustering
    average_clustering
    networkx.algorithms.cluster.square_clustering

    References
    ----------
    .. [1] Latapy, Matthieu, Clémence Magnien, and Nathalie Del Vecchio (2008).
       Basic notions for the analysis of large two-mode networks.
       Social Networks 30(1), 31--48.
    """
    if nodes is None:
        nodes = G
    
    if mode not in modes:
        raise nx.NetworkXError("Mode for bipartite clustering must be 'dot', 'min', or 'max'")

    cc_func = modes[mode]
    clustering = {}

    for v in nodes:
        cc = 0.0
        nbrs = set(G[v])
        if len(nbrs) > 1:
            second_order = set.union(*[set(G[u]) for u in nbrs]) - set([v])
            for u in second_order:
                cc += cc_func(nbrs, set(G[u]))
            cc /= len(second_order)
        clustering[v] = cc

    return clustering


clustering = latapy_clustering


@nx._dispatchable(name='bipartite_average_clustering')
def average_clustering(G, nodes=None, mode='dot'):
    """Compute the average bipartite clustering coefficient.

    A clustering coefficient for the whole graph is the average,

    .. math::

       C = \\frac{1}{n}\\sum_{v \\in G} c_v,

    where `n` is the number of nodes in `G`.

    Similar measures for the two bipartite sets can be defined [1]_

    .. math::

       C_X = \\frac{1}{|X|}\\sum_{v \\in X} c_v,

    where `X` is a bipartite set of `G`.

    Parameters
    ----------
    G : graph
        a bipartite graph

    nodes : list or iterable, optional
        A container of nodes to use in computing the average.
        The nodes should be either the entire graph (the default) or one of the
        bipartite sets.

    mode : string
        The pairwise bipartite clustering method.
        It must be "dot", "max", or "min"

    Returns
    -------
    clustering : float
       The average bipartite clustering for the given set of nodes or the
       entire graph if no nodes are specified.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.star_graph(3)  # star graphs are bipartite
    >>> bipartite.average_clustering(G)
    0.75
    >>> X, Y = bipartite.sets(G)
    >>> bipartite.average_clustering(G, X)
    0.0
    >>> bipartite.average_clustering(G, Y)
    1.0

    See Also
    --------
    clustering

    Notes
    -----
    The container of nodes passed to this function must contain all of the nodes
    in one of the bipartite sets ("top" or "bottom") in order to compute
    the correct average bipartite clustering coefficients.
    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.


    References
    ----------
    .. [1] Latapy, Matthieu, Clémence Magnien, and Nathalie Del Vecchio (2008).
        Basic notions for the analysis of large two-mode networks.
        Social Networks 30(1), 31--48.
    """
    if nodes is None:
        nodes = G
    clustering = latapy_clustering(G, nodes=nodes, mode=mode)
    return sum(clustering.values()) / len(clustering)


@nx._dispatchable
def robins_alexander_clustering(G):
    """Compute the bipartite clustering of G.

    Robins and Alexander [1]_ defined bipartite clustering coefficient as
    four times the number of four cycles `C_4` divided by the number of
    three paths `L_3` in a bipartite graph:

    .. math::

       CC_4 = \\frac{4 * C_4}{L_3}

    Parameters
    ----------
    G : graph
        a bipartite graph

    Returns
    -------
    clustering : float
       The Robins and Alexander bipartite clustering for the input graph.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.davis_southern_women_graph()
    >>> print(round(bipartite.robins_alexander_clustering(G), 3))
    0.468

    See Also
    --------
    latapy_clustering
    networkx.algorithms.cluster.square_clustering

    References
    ----------
    .. [1] Robins, G. and M. Alexander (2004). Small worlds among interlocking
           directors: Network structure and distance in bipartite graphs.
           Computational & Mathematical Organization Theory 10(1), 69–94.

    """
    from itertools import combinations
    
    def four_cycles():
        cycles = 0
        for u, v in G.edges():
            for w in G[u]:
                if w != v:
                    for x in G[v]:
                        if x != u and x in G[w]:
                            cycles += 1
        return cycles // 4  # Each cycle is counted 4 times

    def three_paths():
        paths = 0
        for n in G:
            nbrs = list(G[n])
            paths += sum(len(list(G[u])) - 1 for u in nbrs)
        return paths

    C_4 = four_cycles()
    L_3 = three_paths()
    
    if L_3 == 0:
        return 0.0
    else:
        return (4.0 * C_4) / L_3
