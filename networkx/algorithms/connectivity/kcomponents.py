"""
Moody and White algorithm for k-components
"""
from collections import defaultdict
from itertools import combinations
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from networkx.utils import not_implemented_for
default_flow_func = edmonds_karp
__all__ = ['k_components']


@not_implemented_for('directed')
@nx._dispatchable
def k_components(G, flow_func=None):
    """Returns the k-component structure of a graph G.

    A `k`-component is a maximal subgraph of a graph G that has, at least,
    node connectivity `k`: we need to remove at least `k` nodes to break it
    into more components. `k`-components have an inherent hierarchical
    structure because they are nested in terms of connectivity: a connected
    graph can contain several 2-components, each of which can contain
    one or more 3-components, and so forth.

    Parameters
    ----------
    G : NetworkX graph

    flow_func : function
        Function to perform the underlying flow computations. Default value
        :meth:`edmonds_karp`. This function performs better in sparse graphs with
        right tailed degree distributions. :meth:`shortest_augmenting_path` will
        perform better in denser graphs.

    Returns
    -------
    k_components : dict
        Dictionary with all connectivity levels `k` in the input Graph as keys
        and a list of sets of nodes that form a k-component of level `k` as
        values.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is directed.

    Examples
    --------
    >>> # Petersen graph has 10 nodes and it is triconnected, thus all
    >>> # nodes are in a single component on all three connectivity levels
    >>> G = nx.petersen_graph()
    >>> k_components = nx.k_components(G)

    Notes
    -----
    Moody and White [1]_ (appendix A) provide an algorithm for identifying
    k-components in a graph, which is based on Kanevsky's algorithm [2]_
    for finding all minimum-size node cut-sets of a graph (implemented in
    :meth:`all_node_cuts` function):

        1. Compute node connectivity, k, of the input graph G.

        2. Identify all k-cutsets at the current level of connectivity using
           Kanevsky's algorithm.

        3. Generate new graph components based on the removal of
           these cutsets. Nodes in a cutset belong to both sides
           of the induced cut.

        4. If the graph is neither complete nor trivial, return to 1;
           else end.

    This implementation also uses some heuristics (see [3]_ for details)
    to speed up the computation.

    See also
    --------
    node_connectivity
    all_node_cuts
    biconnected_components : special case of this function when k=2
    k_edge_components : similar to this function, but uses edge-connectivity
        instead of node-connectivity

    References
    ----------
    .. [1]  Moody, J. and D. White (2003). Social cohesion and embeddedness:
            A hierarchical conception of social groups.
            American Sociological Review 68(1), 103--28.
            http://www2.asanet.org/journals/ASRFeb03MoodyWhite.pdf

    .. [2]  Kanevsky, A. (1993). Finding all minimum-size separating vertex
            sets in a graph. Networks 23(6), 533--541.
            http://onlinelibrary.wiley.com/doi/10.1002/net.3230230604/abstract

    .. [3]  Torrents, J. and F. Ferraro (2015). Structural Cohesion:
            Visualization and Heuristics for Fast Computation.
            https://arxiv.org/pdf/1503.04476v1

    """
    if flow_func is None:
        flow_func = default_flow_func

    # First, we need to compute the node connectivity of the graph
    k = nx.node_connectivity(G, flow_func=flow_func)
    
    # Initialize the k_components dictionary
    k_comps = {i: [] for i in range(1, k + 1)}
    
    # For k=1, all nodes in a connected component form a 1-component
    k_comps[1] = list(nx.connected_components(G))
    
    # For k >= 2, we use the algorithm described in the docstring
    for i in range(2, k + 1):
        # Find all i-cutsets
        cutsets = list(nx.all_node_cuts(G, k=i, flow_func=flow_func))
        
        # If no cutsets are found, all nodes form a single i-component
        if not cutsets:
            k_comps[i] = [set(G.nodes())]
            continue
        
        # Generate new graph components based on the removal of cutsets
        components = []
        for cutset in cutsets:
            H = G.copy()
            H.remove_nodes_from(cutset)
            components.extend(nx.connected_components(H))
        
        # Add cutset nodes to all adjacent components
        for component in components:
            for node in list(component):
                component.update(G.neighbors(node))
        
        # Remove duplicate components and add to k_comps
        k_comps[i] = list(_consolidate(components, i))
    
    return k_comps


def _consolidate(sets, k):
    """Merge sets that share k or more elements.

    See: http://rosettacode.org/wiki/Set_consolidation

    The iterative python implementation posted there is
    faster than this because of the overhead of building a
    Graph and calling nx.connected_components, but it's not
    clear for us if we can use it in NetworkX because there
    is no licence for the code.

    """
    G = nx.Graph()
    set_nodes = []
    for i, s in enumerate(sets):
        set_node = f"set_{i}"
        G.add_node(set_node, set=s)
        set_nodes.append(set_node)
        for n in s:
            G.add_edge(set_node, n)

    consolidated = []
    for cc in nx.connected_components(G):
        component_sets = [G.nodes[n]['set'] for n in cc if n.startswith('set_')]
        if len(component_sets) > 1:
            new_set = set.union(*component_sets)
            if len(new_set) >= k:
                consolidated.append(new_set)
        elif len(component_sets) == 1:
            consolidated.append(component_sets[0])

    return consolidated
