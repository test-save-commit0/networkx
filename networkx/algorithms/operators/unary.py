"""Unary operations on graphs"""
import networkx as nx
__all__ = ['complement', 'reverse']


@nx._dispatchable(returns_graph=True)
def complement(G):
    """Returns the graph complement of G.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    Returns
    -------
    GC : A new graph.

    Notes
    -----
    Note that `complement` does not create self-loops and also
    does not produce parallel edges for MultiGraphs.

    Graph, node, and edge data are not propagated to the new graph.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (3, 5)])
    >>> G_complement = nx.complement(G)
    >>> G_complement.edges()  # This shows the edges of the complemented graph
    EdgeView([(1, 4), (1, 5), (2, 4), (2, 5), (4, 5)])

    """
    if G.is_directed():
        GC = G.__class__()
    else:
        GC = G.__class__()
    GC.add_nodes_from(G.nodes())
    
    for u in G:
        for v in G:
            if u != v and not G.has_edge(u, v):
                GC.add_edge(u, v)
    
    return GC


@nx._dispatchable(returns_graph=True)
def reverse(G, copy=True):
    """Returns the reverse directed graph of G.

    Parameters
    ----------
    G : directed graph
        A NetworkX directed graph
    copy : bool
        If True, then a new graph is returned. If False, then the graph is
        reversed in place.

    Returns
    -------
    H : directed graph
        The reversed G.

    Raises
    ------
    NetworkXError
        If graph is undirected.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3), (3, 4), (3, 5)])
    >>> G_reversed = nx.reverse(G)
    >>> G_reversed.edges()
    OutEdgeView([(2, 1), (3, 1), (3, 2), (4, 3), (5, 3)])

    """
    if not G.is_directed():
        raise nx.NetworkXError("Cannot reverse an undirected graph.")
    
    if copy:
        H = G.__class__()
        H.add_nodes_from(G.nodes(data=True))
        H.add_edges_from((v, u, d) for (u, v, d) in G.edges(data=True))
        H.graph.update(G.graph)
        return H
    else:
        G.reverse(copy=False)
        return G
