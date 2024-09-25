"""
Vitality measures.
"""
from functools import partial
import networkx as nx
__all__ = ['closeness_vitality']


@nx._dispatchable(edge_attrs='weight')
def closeness_vitality(G, node=None, weight=None, wiener_index=None):
    """Returns the closeness vitality for nodes in the graph.

    The *closeness vitality* of a node, defined in Section 3.6.2 of [1],
    is the change in the sum of distances between all node pairs when
    excluding that node.

    Parameters
    ----------
    G : NetworkX graph
        A strongly-connected graph.

    weight : string
        The name of the edge attribute used as weight. This is passed
        directly to the :func:`~networkx.wiener_index` function.

    node : object
        If specified, only the closeness vitality for this node will be
        returned. Otherwise, a dictionary mapping each node to its
        closeness vitality will be returned.

    Other parameters
    ----------------
    wiener_index : number
        If you have already computed the Wiener index of the graph
        `G`, you can provide that value here. Otherwise, it will be
        computed for you.

    Returns
    -------
    dictionary or float
        If `node` is None, this function returns a dictionary
        with nodes as keys and closeness vitality as the
        value. Otherwise, it returns only the closeness vitality for the
        specified `node`.

        The closeness vitality of a node may be negative infinity if
        removing that node would disconnect the graph.

    Examples
    --------
    >>> G = nx.cycle_graph(3)
    >>> nx.closeness_vitality(G)
    {0: 2.0, 1: 2.0, 2: 2.0}

    See Also
    --------
    closeness_centrality

    References
    ----------
    .. [1] Ulrik Brandes, Thomas Erlebach (eds.).
           *Network Analysis: Methodological Foundations*.
           Springer, 2005.
           <http://books.google.com/books?id=TTNhSm7HYrIC>

    """
    if wiener_index is None:
        wiener_index = nx.wiener_index(G, weight=weight)

    if node is not None:
        return _single_node_closeness_vitality(G, node, weight, wiener_index)

    vitality = {}
    for n in G:
        vitality[n] = _single_node_closeness_vitality(G, n, weight, wiener_index)
    return vitality

def _single_node_closeness_vitality(G, node, weight, wiener_index):
    if G.number_of_nodes() <= 1:
        return 0.0

    try:
        H = G.copy()
        H.remove_node(node)
        new_wiener_index = nx.wiener_index(H, weight=weight)
        return wiener_index - new_wiener_index
    except nx.NetworkXError:
        # The graph is disconnected after removing the node
        return float('-inf')
