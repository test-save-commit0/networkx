"""
Kanevsky all minimum node k cutsets algorithm.
"""
import copy
from collections import defaultdict
from itertools import combinations
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow import build_residual_network, edmonds_karp, shortest_augmenting_path
from .utils import build_auxiliary_node_connectivity
default_flow_func = edmonds_karp
__all__ = ['all_node_cuts']


@nx._dispatchable
def all_node_cuts(G, k=None, flow_func=None):
    """Returns all minimum k cutsets of an undirected graph G.

    This implementation is based on Kanevsky's algorithm [1]_ for finding all
    minimum-size node cut-sets of an undirected graph G; ie the set (or sets)
    of nodes of cardinality equal to the node connectivity of G. Thus if
    removed, would break G into two or more connected components.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    k : Integer
        Node connectivity of the input graph. If k is None, then it is
        computed. Default value: None.

    flow_func : function
        Function to perform the underlying flow computations. Default value is
        :func:`~networkx.algorithms.flow.edmonds_karp`. This function performs
        better in sparse graphs with right tailed degree distributions.
        :func:`~networkx.algorithms.flow.shortest_augmenting_path` will
        perform better in denser graphs.


    Returns
    -------
    cuts : a generator of node cutsets
        Each node cutset has cardinality equal to the node connectivity of
        the input graph.

    Examples
    --------
    >>> # A two-dimensional grid graph has 4 cutsets of cardinality 2
    >>> G = nx.grid_2d_graph(5, 5)
    >>> cutsets = list(nx.all_node_cuts(G))
    >>> len(cutsets)
    4
    >>> all(2 == len(cutset) for cutset in cutsets)
    True
    >>> nx.node_connectivity(G)
    2

    Notes
    -----
    This implementation is based on the sequential algorithm for finding all
    minimum-size separating vertex sets in a graph [1]_. The main idea is to
    compute minimum cuts using local maximum flow computations among a set
    of nodes of highest degree and all other non-adjacent nodes in the Graph.
    Once we find a minimum cut, we add an edge between the high degree
    node and the target node of the local maximum flow computation to make
    sure that we will not find that minimum cut again.

    See also
    --------
    node_connectivity
    edmonds_karp
    shortest_augmenting_path

    References
    ----------
    .. [1]  Kanevsky, A. (1993). Finding all minimum-size separating vertex
            sets in a graph. Networks 23(6), 533--541.
            http://onlinelibrary.wiley.com/doi/10.1002/net.3230230604/abstract

    """
    if not nx.is_connected(G):
        raise nx.NetworkXError("Input graph is not connected")

    if flow_func is None:
        flow_func = default_flow_func

    if k is None:
        k = nx.node_connectivity(G, flow_func=flow_func)

    # Special cases
    if k == 0:
        return []
    if k == 1:
        return (set([node]) for node in nx.articulation_points(G))

    # General case
    H = build_auxiliary_node_connectivity(G)
    R = build_residual_network(H, 'capacity')
    kwargs = dict(flow_func=flow_func, residual=R)

    # Sort nodes by degree in descending order
    nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    
    # Set to store the cutsets we've found
    cutsets = set()

    for source in nodes:
        for target in G.nodes():
            if source == target or G.has_edge(source, target):
                continue

            # Find the minimum cut
            cut_value, partition = nx.minimum_cut(H, source, target, **kwargs)
            
            if cut_value == k:
                # We found a minimum cut
                reachable, non_reachable = partition
                cutset = set(reachable) & set(non_reachable)
                if len(cutset) == k and cutset not in cutsets:
                    cutsets.add(frozenset(cutset))
                    yield set(cutset)

                # Add an edge to make sure we don't find this cut again
                H.add_edge(source, target, capacity=H.number_of_edges())

    # Clean up
    H.clear()
    R.clear()


def _is_separating_set(G, cut):
    """Assumes that the input graph is connected"""
    if len(cut) == len(G) - 1:
        return True
    H = G.copy()
    H.remove_nodes_from(cut)
    return not nx.is_connected(H)
