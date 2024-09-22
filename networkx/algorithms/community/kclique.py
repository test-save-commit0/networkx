from collections import defaultdict
import networkx as nx
__all__ = ['k_clique_communities']


@nx._dispatchable
def k_clique_communities(G, k, cliques=None):
    """Find k-clique communities in graph using the percolation method.

    A k-clique community is the union of all cliques of size k that
    can be reached through adjacent (sharing k-1 nodes) k-cliques.

    Parameters
    ----------
    G : NetworkX graph

    k : int
       Size of smallest clique

    cliques: list or generator
       Precomputed cliques (use networkx.find_cliques(G))

    Returns
    -------
    Yields sets of nodes, one for each k-clique community.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> K5 = nx.convert_node_labels_to_integers(G, first_label=2)
    >>> G.add_edges_from(K5.edges())
    >>> c = list(nx.community.k_clique_communities(G, 4))
    >>> sorted(list(c[0]))
    [0, 1, 2, 3, 4, 5, 6]
    >>> list(nx.community.k_clique_communities(G, 6))
    []

    References
    ----------
    .. [1] Gergely Palla, Imre Derényi, Illés Farkas1, and Tamás Vicsek,
       Uncovering the overlapping community structure of complex networks
       in nature and society Nature 435, 814-818, 2005,
       doi:10.1038/nature03607
    """
    pass
