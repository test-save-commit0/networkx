"""Load centrality."""
from operator import itemgetter
import networkx as nx
__all__ = ['load_centrality', 'edge_load_centrality']


@nx._dispatchable(edge_attrs='weight')
def newman_betweenness_centrality(G, v=None, cutoff=None, normalized=True,
    weight=None):
    """Compute load centrality for nodes.

    The load centrality of a node is the fraction of all shortest
    paths that pass through that node.

    Parameters
    ----------
    G : graph
      A networkx graph.

    normalized : bool, optional (default=True)
      If True the betweenness values are normalized by b=b/(n-1)(n-2) where
      n is the number of nodes in G.

    weight : None or string, optional (default=None)
      If None, edge weights are ignored.
      Otherwise holds the name of the edge attribute used as weight.
      The weight of an edge is treated as the length or distance between the two sides.

    cutoff : bool, optional (default=None)
      If specified, only consider paths of length <= cutoff.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with centrality as the value.

    See Also
    --------
    betweenness_centrality

    Notes
    -----
    Load centrality is slightly different than betweenness. It was originally
    introduced by [2]_. For this load algorithm see [1]_.

    References
    ----------
    .. [1] Mark E. J. Newman:
       Scientific collaboration networks. II.
       Shortest paths, weighted networks, and centrality.
       Physical Review E 64, 016132, 2001.
       http://journals.aps.org/pre/abstract/10.1103/PhysRevE.64.016132
    .. [2] Kwang-Il Goh, Byungnam Kahng and Doochul Kim
       Universal behavior of Load Distribution in Scale-Free Networks.
       Physical Review Letters 87(27):1–4, 2001.
       https://doi.org/10.1103/PhysRevLett.87.278701
    """
    betweenness = dict.fromkeys(G, 0.0)
    nodes = G.nodes() if v is None else [v]
    for s in nodes:
        betweenness.update(_node_betweenness(G, s, cutoff, normalized, weight))
    
    # Normalize the betweenness values
    if normalized:
        n = len(G)
        if n <= 2:
            return betweenness  # No normalization for graphs with 0 or 1 node
        scale = 1 / ((n - 1) * (n - 2))
        for v in betweenness:
            betweenness[v] *= scale
    
    return betweenness


def _node_betweenness(G, source, cutoff=False, normalized=True, weight=None):
    """Node betweenness_centrality helper:

    See betweenness_centrality for what you probably want.
    This actually computes "load" and not betweenness.
    See https://networkx.lanl.gov/ticket/103

    This calculates the load of each node for paths from a single source.
    (The fraction of number of shortests paths from source that go
    through each node.)

    To get the load for a node you need to do all-pairs shortest paths.

    If weight is not None then use Dijkstra for finding shortest paths.
    """
    betweenness = dict.fromkeys(G, 0.0)
    if weight is None:
        paths = nx.single_source_shortest_path_length(G, source, cutoff)
        pred = {v: [] for v in G}
        for v, p in nx.shortest_path(G, source).items():
            if len(p) > 1:
                pred[p[-1]].append(v)
    else:
        paths = nx.single_source_dijkstra_path_length(G, source, cutoff=cutoff, weight=weight)
        pred = nx.single_source_dijkstra_path(G, source, cutoff=cutoff, weight=weight)
        pred = {v: [k for k, p in pred.items() if p[-1] == v] for v in G}

    delta = dict.fromkeys(G, 0)
    for t in G:
        if t == source:
            continue
        coeff = 1 / len(pred[t])
        for p in pred[t]:
            if p != source:
                delta[p] += coeff
                betweenness[p] += coeff

    for v in delta:
        betweenness[v] += delta[v]

    return betweenness


load_centrality = newman_betweenness_centrality


@nx._dispatchable
def edge_load_centrality(G, cutoff=False):
    """Compute edge load.

    WARNING: This concept of edge load has not been analysed
    or discussed outside of NetworkX that we know of.
    It is based loosely on load_centrality in the sense that
    it counts the number of shortest paths which cross each edge.
    This function is for demonstration and testing purposes.

    Parameters
    ----------
    G : graph
        A networkx graph

    cutoff : bool, optional (default=False)
        If specified, only consider paths of length <= cutoff.

    Returns
    -------
    A dict keyed by edge 2-tuple to the number of shortest paths
    which use that edge. Where more than one path is shortest
    the count is divided equally among paths.
    """
    edge_load = dict.fromkeys(G.edges(), 0.0)
    for s in G:
        edge_load.update(_edge_betweenness(G, s, cutoff=cutoff))
    return edge_load


def _edge_betweenness(G, source, nodes=None, cutoff=False):
    """Edge betweenness helper."""
    betweenness = dict.fromkeys(G.edges(), 0.0)
    if cutoff is False:
        paths = nx.single_source_shortest_path(G, source)
    else:
        paths = nx.single_source_shortest_path(G, source, cutoff)
    
    for target in paths:
        if target == source:
            continue
        sp = paths[target]
        for i in range(len(sp) - 1):
            edge = (sp[i], sp[i+1])
            if edge not in betweenness:
                edge = (sp[i+1], sp[i])
            betweenness[edge] += 1 / len(paths[target])
    
    return betweenness
