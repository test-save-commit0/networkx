"""Functions for computing reaching centrality of a node or a graph."""
import networkx as nx
from networkx.utils import pairwise
__all__ = ['global_reaching_centrality', 'local_reaching_centrality']


def _average_weight(G, path, weight=None):
    """Returns the average weight of an edge in a weighted path.

    Parameters
    ----------
    G : graph
      A networkx graph.

    path: list
      A list of vertices that define the path.

    weight : None or string, optional (default=None)
      If None, edge weights are ignored.  Then the average weight of an edge
      is assumed to be the multiplicative inverse of the length of the path.
      Otherwise holds the name of the edge attribute used as weight.
    """
    if len(path) < 2:
        return 0
    if weight is None:
        return 1 / (len(path) - 1)
    return sum(G[u][v].get(weight, 1) for u, v in pairwise(path)) / (len(path) - 1)


@nx._dispatchable(edge_attrs='weight')
def global_reaching_centrality(G, weight=None, normalized=True):
    """Returns the global reaching centrality of a directed graph.

    The *global reaching centrality* of a weighted directed graph is the
    average over all nodes of the difference between the local reaching
    centrality of the node and the greatest local reaching centrality of
    any node in the graph [1]_. For more information on the local
    reaching centrality, see :func:`local_reaching_centrality`.
    Informally, the local reaching centrality is the proportion of the
    graph that is reachable from the neighbors of the node.

    Parameters
    ----------
    G : DiGraph
        A networkx DiGraph.

    weight : None or string, optional (default=None)
        Attribute to use for edge weights. If ``None``, each edge weight
        is assumed to be one. A higher weight implies a stronger
        connection between nodes and a *shorter* path length.

    normalized : bool, optional (default=True)
        Whether to normalize the edge weights by the total sum of edge
        weights.

    Returns
    -------
    h : float
        The global reaching centrality of the graph.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge(1, 2)
    >>> G.add_edge(1, 3)
    >>> nx.global_reaching_centrality(G)
    1.0
    >>> G.add_edge(3, 2)
    >>> nx.global_reaching_centrality(G)
    0.75

    See also
    --------
    local_reaching_centrality

    References
    ----------
    .. [1] Mones, Enys, Lilla Vicsek, and Tamás Vicsek.
           "Hierarchy Measure for Complex Networks."
           *PLoS ONE* 7.3 (2012): e33799.
           https://doi.org/10.1371/journal.pone.0033799
    """
    local_reach = {v: local_reaching_centrality(G, v, weight=weight, normalized=normalized)
                   for v in G}
    if not local_reach:
        return 0.0
    max_reach = max(local_reach.values())
    return sum(max_reach - c for c in local_reach.values()) / len(G)


@nx._dispatchable(edge_attrs='weight')
def local_reaching_centrality(G, v, paths=None, weight=None, normalized=True):
    """Returns the local reaching centrality of a node in a directed
    graph.

    The *local reaching centrality* of a node in a directed graph is the
    proportion of other nodes reachable from that node [1]_.

    Parameters
    ----------
    G : DiGraph
        A NetworkX DiGraph.

    v : node
        A node in the directed graph `G`.

    paths : dictionary (default=None)
        If this is not `None` it must be a dictionary representation
        of single-source shortest paths, as computed by, for example,
        :func:`networkx.shortest_path` with source node `v`. Use this
        keyword argument if you intend to invoke this function many
        times but don't want the paths to be recomputed each time.

    weight : None or string, optional (default=None)
        Attribute to use for edge weights.  If `None`, each edge weight
        is assumed to be one. A higher weight implies a stronger
        connection between nodes and a *shorter* path length.

    normalized : bool, optional (default=True)
        Whether to normalize the edge weights by the total sum of edge
        weights.

    Returns
    -------
    h : float
        The local reaching centrality of the node ``v`` in the graph
        ``G``.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3)])
    >>> nx.local_reaching_centrality(G, 3)
    0.0
    >>> G.add_edge(3, 2)
    >>> nx.local_reaching_centrality(G, 3)
    0.5

    See also
    --------
    global_reaching_centrality

    References
    ----------
    .. [1] Mones, Enys, Lilla Vicsek, and Tamás Vicsek.
           "Hierarchy Measure for Complex Networks."
           *PLoS ONE* 7.3 (2012): e33799.
           https://doi.org/10.1371/journal.pone.0033799
    """
    if paths is None:
        paths = nx.shortest_path(G, source=v, weight=weight)
    
    n = len(G) - 1  # excluding the node itself
    if n == 0:
        return 0.0
    
    reachable = sum(1 for p in paths.values() if len(p) > 1)
    if not normalized:
        return reachable / n
    
    total_weight = sum(_average_weight(G, p, weight) for p in paths.values() if len(p) > 1)
    max_weight = n * _average_weight(G, [v, list(G.neighbors(v))[0]], weight) if G.neighbors(v) else 0
    
    return total_weight / max_weight if max_weight > 0 else 0.0
