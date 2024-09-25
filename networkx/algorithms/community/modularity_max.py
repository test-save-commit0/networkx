"""Functions for detecting communities based on modularity."""
from collections import defaultdict
import networkx as nx
from networkx.algorithms.community.quality import modularity
from networkx.utils import not_implemented_for
from networkx.utils.mapped_queue import MappedQueue
__all__ = ['greedy_modularity_communities',
    'naive_greedy_modularity_communities']


def _greedy_modularity_communities_generator(G, weight=None, resolution=1):
    """Yield community partitions of G and the modularity change at each step.

    This function performs Clauset-Newman-Moore greedy modularity maximization [2]_
    At each step of the process it yields the change in modularity that will occur in
    the next step followed by yielding the new community partition after that step.

    Greedy modularity maximization begins with each node in its own community
    and repeatedly joins the pair of communities that lead to the largest
    modularity until one community contains all nodes (the partition has one set).

    This function maximizes the generalized modularity, where `resolution`
    is the resolution parameter, often expressed as $\\gamma$.
    See :func:`~networkx.algorithms.community.quality.modularity`.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or None, optional (default=None)
        The name of an edge attribute that holds the numerical value used
        as a weight.  If None, then each edge has weight 1.
        The degree is the sum of the edge weights adjacent to the node.

    resolution : float (default=1)
        If resolution is less than 1, modularity favors larger communities.
        Greater than 1 favors smaller communities.

    Yields
    ------
    Alternating yield statements produce the following two objects:

    communities: dict_values
        A dict_values of frozensets of nodes, one for each community.
        This represents a partition of the nodes of the graph into communities.
        The first yield is the partition with each node in its own community.

    dq: float
        The change in modularity when merging the next two communities
        that leads to the largest modularity.

    See Also
    --------
    modularity

    References
    ----------
    .. [1] Newman, M. E. J. "Networks: An Introduction", page 224
       Oxford University Press 2011.
    .. [2] Clauset, A., Newman, M. E., & Moore, C.
       "Finding community structure in very large networks."
       Physical Review E 70(6), 2004.
    .. [3] Reichardt and Bornholdt "Statistical Mechanics of Community
       Detection" Phys. Rev. E74, 2006.
    .. [4] Newman, M. E. J."Analysis of weighted networks"
       Physical Review E 70(5 Pt 2):056131, 2004.
    """
    # Initialize each node to its own community
    communities = {node: frozenset([node]) for node in G}
    degrees = dict(G.degree(weight=weight))
    m = sum(degrees.values()) / 2

    # Calculate initial modularity
    Q = modularity(G, communities.values(), weight=weight, resolution=resolution)

    # Initialize data structures for efficient updates
    community_edges = defaultdict(int)
    community_degrees = defaultdict(int)
    for u, v, w in G.edges(data=weight, default=1):
        c1, c2 = communities[u], communities[v]
        community_edges[c1, c2] += w
        community_edges[c2, c1] += w
        community_degrees[c1] += w
        if u != v:
            community_degrees[c2] += w

    # Main loop
    while len(communities) > 1:
        best_merge = None
        best_dq = -1

        # Find the best merge
        for c1, c2 in community_edges:
            if c1 != c2:
                dq = 2 * (community_edges[c1, c2] - resolution * community_degrees[c1] * community_degrees[c2] / (2 * m))
                if dq > best_dq:
                    best_dq = dq
                    best_merge = (c1, c2)

        if best_merge is None:
            break

        # Perform the merge
        c1, c2 = best_merge
        new_community = c1.union(c2)
        del communities[list(c2)[0]]
        for node in c2:
            communities[node] = new_community

        # Update data structures
        for other_c in set(community_edges):
            if other_c != c1 and other_c != c2:
                community_edges[new_community, other_c] = community_edges[c1, other_c] + community_edges[c2, other_c]
                community_edges[other_c, new_community] = community_edges[new_community, other_c]
        community_degrees[new_community] = community_degrees[c1] + community_degrees[c2]
        del community_degrees[c2]

        # Clean up old entries
        for k in list(community_edges.keys()):
            if c1 in k or c2 in k:
                del community_edges[k]

        # Yield results
        yield best_dq
        yield communities.values()

    # Yield final partition
    yield 0
    yield communities.values()


@nx._dispatchable(edge_attrs='weight')
def greedy_modularity_communities(G, weight=None, resolution=1, cutoff=1,
    best_n=None):
    """Find communities in G using greedy modularity maximization.

    This function uses Clauset-Newman-Moore greedy modularity maximization [2]_
    to find the community partition with the largest modularity.

    Greedy modularity maximization begins with each node in its own community
    and repeatedly joins the pair of communities that lead to the largest
    modularity until no further increase in modularity is possible (a maximum).
    Two keyword arguments adjust the stopping condition. `cutoff` is a lower
    limit on the number of communities so you can stop the process before
    reaching a maximum (used to save computation time). `best_n` is an upper
    limit on the number of communities so you can make the process continue
    until at most n communities remain even if the maximum modularity occurs
    for more. To obtain exactly n communities, set both `cutoff` and `best_n` to n.

    This function maximizes the generalized modularity, where `resolution`
    is the resolution parameter, often expressed as $\\gamma$.
    See :func:`~networkx.algorithms.community.quality.modularity`.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or None, optional (default=None)
        The name of an edge attribute that holds the numerical value used
        as a weight.  If None, then each edge has weight 1.
        The degree is the sum of the edge weights adjacent to the node.

    resolution : float, optional (default=1)
        If resolution is less than 1, modularity favors larger communities.
        Greater than 1 favors smaller communities.

    cutoff : int, optional (default=1)
        A minimum number of communities below which the merging process stops.
        The process stops at this number of communities even if modularity
        is not maximized. The goal is to let the user stop the process early.
        The process stops before the cutoff if it finds a maximum of modularity.

    best_n : int or None, optional (default=None)
        A maximum number of communities above which the merging process will
        not stop. This forces community merging to continue after modularity
        starts to decrease until `best_n` communities remain.
        If ``None``, don't force it to continue beyond a maximum.

    Raises
    ------
    ValueError : If the `cutoff` or `best_n`  value is not in the range
        ``[1, G.number_of_nodes()]``, or if `best_n` < `cutoff`.

    Returns
    -------
    communities: list
        A list of frozensets of nodes, one for each community.
        Sorted by length with largest communities first.

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> c = nx.community.greedy_modularity_communities(G)
    >>> sorted(c[0])
    [8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

    See Also
    --------
    modularity

    References
    ----------
    .. [1] Newman, M. E. J. "Networks: An Introduction", page 224
       Oxford University Press 2011.
    .. [2] Clauset, A., Newman, M. E., & Moore, C.
       "Finding community structure in very large networks."
       Physical Review E 70(6), 2004.
    .. [3] Reichardt and Bornholdt "Statistical Mechanics of Community
       Detection" Phys. Rev. E74, 2006.
    .. [4] Newman, M. E. J."Analysis of weighted networks"
       Physical Review E 70(5 Pt 2):056131, 2004.
    """
    # Input validation
    n = G.number_of_nodes()
    if cutoff not in range(1, n + 1):
        raise ValueError(f"cutoff must be in [1, {n}]")
    if best_n is not None:
        if best_n not in range(1, n + 1):
            raise ValueError(f"best_n must be in [1, {n}]")
        if best_n < cutoff:
            raise ValueError("best_n must be greater than or equal to cutoff")

    # Run the generator
    communities = None
    modularity = -1
    for dq, partition in _greedy_modularity_communities_generator(G, weight, resolution):
        if len(partition) < cutoff:
            break
        if dq < 0 and best_n is None:
            break
        communities = partition
        modularity += dq
        if best_n is not None and len(communities) <= best_n:
            break

    # If no valid partition was found, return trivial partition
    if communities is None:
        communities = [frozenset([n]) for n in G]

    return sorted(communities, key=len, reverse=True)


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable(edge_attrs='weight')
def naive_greedy_modularity_communities(G, resolution=1, weight=None):
    """Find communities in G using greedy modularity maximization.

    This implementation is O(n^4), much slower than alternatives, but it is
    provided as an easy-to-understand reference implementation.

    Greedy modularity maximization begins with each node in its own community
    and joins the pair of communities that most increases modularity until no
    such pair exists.

    This function maximizes the generalized modularity, where `resolution`
    is the resolution parameter, often expressed as $\\gamma$.
    See :func:`~networkx.algorithms.community.quality.modularity`.

    Parameters
    ----------
    G : NetworkX graph
        Graph must be simple and undirected.

    resolution : float (default=1)
        If resolution is less than 1, modularity favors larger communities.
        Greater than 1 favors smaller communities.

    weight : string or None, optional (default=None)
        The name of an edge attribute that holds the numerical value used
        as a weight.  If None, then each edge has weight 1.
        The degree is the sum of the edge weights adjacent to the node.

    Returns
    -------
    list
        A list of sets of nodes, one for each community.
        Sorted by length with largest communities first.

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> c = nx.community.naive_greedy_modularity_communities(G)
    >>> sorted(c[0])
    [8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

    See Also
    --------
    greedy_modularity_communities
    modularity
    """
    # Start with each node in its own community
    communities = [{node} for node in G.nodes()]
    
    while len(communities) > 1:
        best_merge = None
        best_increase = 0
        
        for i, comm1 in enumerate(communities):
            for j, comm2 in enumerate(communities[i+1:], start=i+1):
                new_comm = comm1.union(comm2)
                old_modularity = modularity(G, communities, resolution=resolution, weight=weight)
                new_communities = [c for k, c in enumerate(communities) if k != i and k != j]
                new_communities.append(new_comm)
                new_modularity = modularity(G, new_communities, resolution=resolution, weight=weight)
                increase = new_modularity - old_modularity
                
                if increase > best_increase:
                    best_increase = increase
                    best_merge = (i, j)
        
        if best_merge is None:
            break
        
        i, j = best_merge
        communities[i] = communities[i].union(communities[j])
        communities.pop(j)
    
    return sorted(communities, key=len, reverse=True)
