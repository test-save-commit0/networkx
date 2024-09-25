"""
Flow Hierarchy.
"""
import networkx as nx
__all__ = ['flow_hierarchy']


@nx._dispatchable(edge_attrs='weight')
def flow_hierarchy(G, weight=None):
    """Returns the flow hierarchy of a directed network.

    Flow hierarchy is defined as the fraction of edges not participating
    in cycles in a directed graph [1]_.

    Parameters
    ----------
    G : DiGraph or MultiDiGraph
       A directed graph

    weight : string, optional (default=None)
       Attribute to use for edge weights. If None the weight defaults to 1.

    Returns
    -------
    h : float
       Flow hierarchy value

    Notes
    -----
    The algorithm described in [1]_ computes the flow hierarchy through
    exponentiation of the adjacency matrix.  This function implements an
    alternative approach that finds strongly connected components.
    An edge is in a cycle if and only if it is in a strongly connected
    component, which can be found in $O(m)$ time using Tarjan's algorithm.

    References
    ----------
    .. [1] Luo, J.; Magee, C.L. (2011),
       Detecting evolving patterns of self-organizing networks by flow
       hierarchy measurement, Complexity, Volume 16 Issue 6 53-61.
       DOI: 10.1002/cplx.20368
       http://web.mit.edu/~cmagee/www/documents/28-DetectingEvolvingPatterns_FlowHierarchy.pdf
    """
    if not G.is_directed():
        raise nx.NetworkXError("Flow hierarchy is not defined for undirected graphs.")

    # Find strongly connected components
    scc = nx.strongly_connected_components(G)
    scc_subgraphs = [G.subgraph(c) for c in scc]

    # Count edges in cycles (i.e., in strongly connected components with more than one node)
    edges_in_cycles = sum(sg.number_of_edges() for sg in scc_subgraphs if len(sg) > 1)

    # Count total number of edges
    total_edges = G.number_of_edges()

    # Calculate flow hierarchy
    if total_edges == 0:
        return 1.0  # By convention, an empty graph has perfect hierarchy

    return 1 - (edges_in_cycles / total_edges)
