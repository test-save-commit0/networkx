"""Functions for computing and verifying regular graphs."""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['is_regular', 'is_k_regular', 'k_factor']


@nx._dispatchable
def is_regular(G):
    """Determines whether the graph ``G`` is a regular graph.

    A regular graph is a graph where each vertex has the same degree. A
    regular digraph is a graph where the indegree and outdegree of each
    vertex are equal.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    bool
        Whether the given graph or digraph is regular.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 1)])
    >>> nx.is_regular(G)
    True

    """
    pass


@not_implemented_for('directed')
@nx._dispatchable
def is_k_regular(G, k):
    """Determines whether the graph ``G`` is a k-regular graph.

    A k-regular graph is a graph where each vertex has degree k.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    bool
        Whether the given graph is k-regular.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 1)])
    >>> nx.is_k_regular(G, k=3)
    False

    """
    pass


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable(preserve_edge_attrs=True, returns_graph=True)
def k_factor(G, k, matching_weight='weight'):
    """Compute a k-factor of G

    A k-factor of a graph is a spanning k-regular subgraph.
    A spanning k-regular subgraph of G is a subgraph that contains
    each vertex of G and a subset of the edges of G such that each
    vertex has degree k.

    Parameters
    ----------
    G : NetworkX graph
      Undirected graph

    matching_weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.
       Used for finding the max-weighted perfect matching.
       If key not found, uses 1 as weight.

    Returns
    -------
    G2 : NetworkX graph
        A k-factor of G

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 1)])
    >>> G2 = nx.k_factor(G, k=1)
    >>> G2.edges()
    EdgeView([(1, 2), (3, 4)])

    References
    ----------
    .. [1] "An algorithm for computing simple k-factors.",
       Meijer, Henk, Yurai Núñez-Rodríguez, and David Rappaport,
       Information processing letters, 2009.
    """
    pass
