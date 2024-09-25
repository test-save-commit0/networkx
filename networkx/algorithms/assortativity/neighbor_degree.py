import networkx as nx
__all__ = ['average_neighbor_degree']


@nx._dispatchable(edge_attrs='weight')
def average_neighbor_degree(G, source='out', target='out', nodes=None,
    weight=None):
    """Returns the average degree of the neighborhood of each node.

    In an undirected graph, the neighborhood `N(i)` of node `i` contains the
    nodes that are connected to `i` by an edge.

    For directed graphs, `N(i)` is defined according to the parameter `source`:

        - if source is 'in', then `N(i)` consists of predecessors of node `i`.
        - if source is 'out', then `N(i)` consists of successors of node `i`.
        - if source is 'in+out', then `N(i)` is both predecessors and successors.

    The average neighborhood degree of a node `i` is

    .. math::

        k_{nn,i} = \\frac{1}{|N(i)|} \\sum_{j \\in N(i)} k_j

    where `N(i)` are the neighbors of node `i` and `k_j` is
    the degree of node `j` which belongs to `N(i)`. For weighted
    graphs, an analogous measure can be defined [1]_,

    .. math::

        k_{nn,i}^{w} = \\frac{1}{s_i} \\sum_{j \\in N(i)} w_{ij} k_j

    where `s_i` is the weighted degree of node `i`, `w_{ij}`
    is the weight of the edge that links `i` and `j` and
    `N(i)` are the neighbors of node `i`.


    Parameters
    ----------
    G : NetworkX graph

    source : string ("in"|"out"|"in+out"), optional (default="out")
       Directed graphs only.
       Use "in"- or "out"-neighbors of source node.

    target : string ("in"|"out"|"in+out"), optional (default="out")
       Directed graphs only.
       Use "in"- or "out"-degree for target node.

    nodes : list or iterable, optional (default=G.nodes)
        Compute neighbor degree only for specified nodes.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    Returns
    -------
    d: dict
       A dictionary keyed by node to the average degree of its neighbors.

    Raises
    ------
    NetworkXError
        If either `source` or `target` are not one of 'in', 'out', or 'in+out'.
        If either `source` or `target` is passed for an undirected graph.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> G.edges[0, 1]["weight"] = 5
    >>> G.edges[2, 3]["weight"] = 3

    >>> nx.average_neighbor_degree(G)
    {0: 2.0, 1: 1.5, 2: 1.5, 3: 2.0}
    >>> nx.average_neighbor_degree(G, weight="weight")
    {0: 2.0, 1: 1.1666666666666667, 2: 1.25, 3: 2.0}

    >>> G = nx.DiGraph()
    >>> nx.add_path(G, [0, 1, 2, 3])
    >>> nx.average_neighbor_degree(G, source="in", target="in")
    {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0}

    >>> nx.average_neighbor_degree(G, source="out", target="out")
    {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}

    See Also
    --------
    average_degree_connectivity

    References
    ----------
    .. [1] A. Barrat, M. Barthélemy, R. Pastor-Satorras, and A. Vespignani,
       "The architecture of complex weighted networks".
       PNAS 101 (11): 3747–3752 (2004).
    """
    if G.is_directed():
        if source not in ('in', 'out', 'in+out'):
            raise nx.NetworkXError("source must be 'in', 'out' or 'in+out'")
        if target not in ('in', 'out', 'in+out'):
            raise nx.NetworkXError("target must be 'in', 'out' or 'in+out'")
    else:
        if source != 'out' or target != 'out':
            raise nx.NetworkXError("source and target arguments are only supported for directed graphs")

    if nodes is None:
        nodes = G.nodes()

    def get_neighbors(node):
        if G.is_directed():
            if source == 'in':
                return G.predecessors(node)
            elif source == 'out':
                return G.successors(node)
            else:  # source == 'in+out'
                return set(G.predecessors(node)) | set(G.successors(node))
        else:
            return G.neighbors(node)

    def get_degree(node):
        if G.is_directed():
            if target == 'in':
                return G.in_degree(node, weight=weight)
            elif target == 'out':
                return G.out_degree(node, weight=weight)
            else:  # target == 'in+out'
                return G.in_degree(node, weight=weight) + G.out_degree(node, weight=weight)
        else:
            return G.degree(node, weight=weight)

    avg_nbr_degree = {}
    for node in nodes:
        neighbors = list(get_neighbors(node))
        if len(neighbors) > 0:
            if weight is None:
                avg_nbr_degree[node] = sum(get_degree(nbr) for nbr in neighbors) / len(neighbors)
            else:
                total_weighted_degree = sum(G[node][nbr].get(weight, 1) * get_degree(nbr) for nbr in neighbors)
                total_weight = sum(G[node][nbr].get(weight, 1) for nbr in neighbors)
                avg_nbr_degree[node] = total_weighted_degree / total_weight if total_weight > 0 else 0
        else:
            avg_nbr_degree[node] = 0

    return avg_nbr_degree
