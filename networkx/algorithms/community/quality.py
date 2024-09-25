"""Functions for measuring the quality of a partition (into
communities).

"""
from itertools import combinations
import networkx as nx
from networkx import NetworkXError
from networkx.algorithms.community.community_utils import is_partition
from networkx.utils.decorators import argmap
__all__ = ['modularity', 'partition_quality']


class NotAPartition(NetworkXError):
    """Raised if a given collection is not a partition."""

    def __init__(self, G, collection):
        msg = f'{collection} is not a valid partition of the graph {G}'
        super().__init__(msg)


def _require_partition(G, partition):
    """Decorator to check that a valid partition is input to a function

    Raises :exc:`networkx.NetworkXError` if the partition is not valid.

    This decorator should be used on functions whose first two arguments
    are a graph and a partition of the nodes of that graph (in that
    order)::

        >>> @require_partition
        ... def foo(G, partition):
        ...     print("partition is valid!")
        ...
        >>> G = nx.complete_graph(5)
        >>> partition = [{0, 1}, {2, 3}, {4}]
        >>> foo(G, partition)
        partition is valid!
        >>> partition = [{0}, {2, 3}, {4}]
        >>> foo(G, partition)
        Traceback (most recent call last):
          ...
        networkx.exception.NetworkXError: `partition` is not a valid partition of the nodes of G
        >>> partition = [{0, 1}, {1, 2, 3}, {4}]
        >>> foo(G, partition)
        Traceback (most recent call last):
          ...
        networkx.exception.NetworkXError: `partition` is not a valid partition of the nodes of G

    """
    if not is_partition(G, partition):
        raise NotAPartition(G, partition)


require_partition = argmap(_require_partition, (0, 1))


@nx._dispatchable
def intra_community_edges(G, partition):
    """Returns the number of intra-community edges for a partition of `G`.

    Parameters
    ----------
    G : NetworkX graph.

    partition : iterable of sets of nodes
        This must be a partition of the nodes of `G`.

    The "intra-community edges" are those edges joining a pair of nodes
    in the same block of the partition.

    """
    return sum(G.subgraph(community).number_of_edges() for community in partition)


@nx._dispatchable
def inter_community_edges(G, partition):
    """Returns the number of inter-community edges for a partition of `G`.
    according to the given
    partition of the nodes of `G`.

    Parameters
    ----------
    G : NetworkX graph.

    partition : iterable of sets of nodes
        This must be a partition of the nodes of `G`.

    The *inter-community edges* are those edges joining a pair of nodes
    in different blocks of the partition.

    Implementation note: this function creates an intermediate graph
    that may require the same amount of memory as that of `G`.

    """
    # Create a graph with communities as nodes
    community_graph = nx.Graph()
    community_graph.add_nodes_from(range(len(partition)))

    for i, community1 in enumerate(partition):
        for j, community2 in enumerate(partition[i+1:], start=i+1):
            if any(G.has_edge(u, v) for u in community1 for v in community2):
                community_graph.add_edge(i, j)

    return community_graph.number_of_edges()


@nx._dispatchable
def inter_community_non_edges(G, partition):
    """Returns the number of inter-community non-edges according to the
    given partition of the nodes of `G`.

    Parameters
    ----------
    G : NetworkX graph.

    partition : iterable of sets of nodes
        This must be a partition of the nodes of `G`.

    A *non-edge* is a pair of nodes (undirected if `G` is undirected)
    that are not adjacent in `G`. The *inter-community non-edges* are
    those non-edges on a pair of nodes in different blocks of the
    partition.

    Implementation note: this function creates two intermediate graphs,
    which may require up to twice the amount of memory as required to
    store `G`.

    """
    # Create a complete graph with the same nodes as G
    H = nx.complete_graph(G.nodes())
    
    # Remove edges that exist in G
    H.remove_edges_from(G.edges())
    
    # Count inter-community non-edges
    return sum(1 for u, v in H.edges() if any(u in c1 and v in c2 for c1 in partition for c2 in partition if c1 != c2))


@nx._dispatchable(edge_attrs='weight')
def modularity(G, communities, weight='weight', resolution=1):
    """Returns the modularity of the given partition of the graph.

    Modularity is defined in [1]_ as

    .. math::
        Q = \\frac{1}{2m} \\sum_{ij} \\left( A_{ij} - \\gamma\\frac{k_ik_j}{2m}\\right)
            \\delta(c_i,c_j)

    where $m$ is the number of edges (or sum of all edge weights as in [5]_),
    $A$ is the adjacency matrix of `G`, $k_i$ is the (weighted) degree of $i$,
    $\\gamma$ is the resolution parameter, and $\\delta(c_i, c_j)$ is 1 if $i$ and
    $j$ are in the same community else 0.

    According to [2]_ (and verified by some algebra) this can be reduced to

    .. math::
       Q = \\sum_{c=1}^{n}
       \\left[ \\frac{L_c}{m} - \\gamma\\left( \\frac{k_c}{2m} \\right) ^2 \\right]

    where the sum iterates over all communities $c$, $m$ is the number of edges,
    $L_c$ is the number of intra-community links for community $c$,
    $k_c$ is the sum of degrees of the nodes in community $c$,
    and $\\gamma$ is the resolution parameter.

    The resolution parameter sets an arbitrary tradeoff between intra-group
    edges and inter-group edges. More complex grouping patterns can be
    discovered by analyzing the same network with multiple values of gamma
    and then combining the results [3]_. That said, it is very common to
    simply use gamma=1. More on the choice of gamma is in [4]_.

    The second formula is the one actually used in calculation of the modularity.
    For directed graphs the second formula replaces $k_c$ with $k^{in}_c k^{out}_c$.

    Parameters
    ----------
    G : NetworkX Graph

    communities : list or iterable of set of nodes
        These node sets must represent a partition of G's nodes.

    weight : string or None, optional (default="weight")
        The edge attribute that holds the numerical value used
        as a weight. If None or an edge does not have that attribute,
        then that edge has weight 1.

    resolution : float (default=1)
        If resolution is less than 1, modularity favors larger communities.
        Greater than 1 favors smaller communities.

    Returns
    -------
    Q : float
        The modularity of the partition.

    Raises
    ------
    NotAPartition
        If `communities` is not a partition of the nodes of `G`.

    Examples
    --------
    >>> G = nx.barbell_graph(3, 0)
    >>> nx.community.modularity(G, [{0, 1, 2}, {3, 4, 5}])
    0.35714285714285715
    >>> nx.community.modularity(G, nx.community.label_propagation_communities(G))
    0.35714285714285715

    References
    ----------
    .. [1] M. E. J. Newman "Networks: An Introduction", page 224.
       Oxford University Press, 2011.
    .. [2] Clauset, Aaron, Mark EJ Newman, and Cristopher Moore.
       "Finding community structure in very large networks."
       Phys. Rev. E 70.6 (2004). <https://arxiv.org/abs/cond-mat/0408187>
    .. [3] Reichardt and Bornholdt "Statistical Mechanics of Community Detection"
       Phys. Rev. E 74, 016110, 2006. https://doi.org/10.1103/PhysRevE.74.016110
    .. [4] M. E. J. Newman, "Equivalence between modularity optimization and
       maximum likelihood methods for community detection"
       Phys. Rev. E 94, 052315, 2016. https://doi.org/10.1103/PhysRevE.94.052315
    .. [5] Blondel, V.D. et al. "Fast unfolding of communities in large
       networks" J. Stat. Mech 10008, 1-12 (2008).
       https://doi.org/10.1088/1742-5468/2008/10/P10008
    """
    if not isinstance(communities, list):
        communities = list(communities)
    if not is_partition(G, communities):
        raise NotAPartition(G, communities)

    directed = G.is_directed()
    m = G.size(weight=weight)
    if m == 0:
        return 0.0

    Q = 0.0
    for community in communities:
        community_edges = G.subgraph(community).size(weight=weight)
        community_degree = sum(dict(G.degree(community, weight=weight)).values())
        if directed:
            in_degree = sum(dict(G.in_degree(community, weight=weight)).values())
            out_degree = sum(dict(G.out_degree(community, weight=weight)).values())
            Q += community_edges / m - resolution * ((in_degree * out_degree) / (m * m))
        else:
            Q += community_edges / m - resolution * ((community_degree / (2 * m)) ** 2)

    return Q


@require_partition
@nx._dispatchable
def partition_quality(G, partition):
    """Returns the coverage and performance of a partition of G.

    The *coverage* of a partition is the ratio of the number of
    intra-community edges to the total number of edges in the graph.

    The *performance* of a partition is the number of
    intra-community edges plus inter-community non-edges divided by the total
    number of potential edges.

    This algorithm has complexity $O(C^2 + L)$ where C is the number of communities and L is the number of links.

    Parameters
    ----------
    G : NetworkX graph

    partition : sequence
        Partition of the nodes of `G`, represented as a sequence of
        sets of nodes (blocks). Each block of the partition represents a
        community.

    Returns
    -------
    (float, float)
        The (coverage, performance) tuple of the partition, as defined above.

    Raises
    ------
    NetworkXError
        If `partition` is not a valid partition of the nodes of `G`.

    Notes
    -----
    If `G` is a multigraph;
        - for coverage, the multiplicity of edges is counted
        - for performance, the result is -1 (total number of possible edges is not defined)

    References
    ----------
    .. [1] Santo Fortunato.
           "Community Detection in Graphs".
           *Physical Reports*, Volume 486, Issue 3--5 pp. 75--174
           <https://arxiv.org/abs/0906.0612>
    """
    node_community = {}
    for i, community in enumerate(partition):
        for node in community:
            node_community[node] = i

    num_intra_edges = sum(1 for u, v in G.edges() if node_community[u] == node_community[v])
    num_edges = G.number_of_edges()
    coverage = num_intra_edges / num_edges if num_edges > 0 else 0.0

    if G.is_multigraph():
        return coverage, -1.0

    num_nodes = G.number_of_nodes()
    num_possible_edges = num_nodes * (num_nodes - 1) // 2
    num_inter_non_edges = sum(1 for u, v in combinations(G.nodes(), 2)
                              if not G.has_edge(u, v) and node_community[u] != node_community[v])
    performance = (num_intra_edges + num_inter_non_edges) / num_possible_edges

    return coverage, performance
