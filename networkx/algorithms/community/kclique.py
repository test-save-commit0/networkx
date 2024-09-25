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
    if k < 2:
        raise nx.NetworkXError(f"k={k}, k must be 2 or greater.")
    if cliques is None:
        cliques = nx.find_cliques(G)

    cliques = [frozenset(c) for c in cliques if len(c) >= k]

    # First index which nodes are in which cliques
    membership_dict = defaultdict(list)
    for i, c in enumerate(cliques):
        for node in c:
            membership_dict[node].append(i)

    # For each clique, see which adjacent cliques percolate
    perc_graph = nx.Graph()
    perc_graph.add_nodes_from(range(len(cliques)))
    for i, clique in enumerate(cliques):
        for j in range(i + 1, len(cliques)):
            if len(clique.intersection(cliques[j])) >= (k - 1):
                perc_graph.add_edge(i, j)

    # Connected components of clique graph with perc edges
    # are the k-clique communities
    for component in nx.connected_components(perc_graph):
        yield set.union(*[cliques[i] for i in component])
