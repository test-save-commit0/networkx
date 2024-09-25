"""Functions for computing dominating sets in a graph."""
from itertools import chain
import networkx as nx
from networkx.utils import arbitrary_element
__all__ = ['dominating_set', 'is_dominating_set']


@nx._dispatchable
def dominating_set(G, start_with=None):
    """Finds a dominating set for the graph G.

    A *dominating set* for a graph with node set *V* is a subset *D* of
    *V* such that every node not in *D* is adjacent to at least one
    member of *D* [1]_.

    Parameters
    ----------
    G : NetworkX graph

    start_with : node (default=None)
        Node to use as a starting point for the algorithm.

    Returns
    -------
    D : set
        A dominating set for G.

    Notes
    -----
    This function is an implementation of algorithm 7 in [2]_ which
    finds some dominating set, not necessarily the smallest one.

    See also
    --------
    is_dominating_set

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dominating_set

    .. [2] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    """
    if len(G) == 0:
        return set()
    
    if start_with is None:
        start_with = arbitrary_element(G)
    
    dom_set = {start_with}
    undominated = set(G) - set(G[start_with]) - {start_with}
    
    while undominated:
        # Find the node that dominates the most undominated nodes
        best_node = max(G, key=lambda n: len(set(G[n]) & undominated))
        dom_set.add(best_node)
        undominated -= set(G[best_node]) | {best_node}
    
    return dom_set


@nx._dispatchable
def is_dominating_set(G, nbunch):
    """Checks if `nbunch` is a dominating set for `G`.

    A *dominating set* for a graph with node set *V* is a subset *D* of
    *V* such that every node not in *D* is adjacent to at least one
    member of *D* [1]_.

    Parameters
    ----------
    G : NetworkX graph

    nbunch : iterable
        An iterable of nodes in the graph `G`.

    See also
    --------
    dominating_set

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dominating_set

    """
    nbunch = set(nbunch)
    if not nbunch.issubset(G):
        raise nx.NetworkXError("nbunch is not a subset of the nodes of G")
    
    return set(G) <= set(chain.from_iterable(G[n] for n in nbunch)) | nbunch
