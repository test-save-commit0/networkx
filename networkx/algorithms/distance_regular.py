"""
=======================
Distance-regular graphs
=======================
"""
import networkx as nx
from networkx.utils import not_implemented_for
from .distance_measures import diameter
__all__ = ['is_distance_regular', 'is_strongly_regular',
    'intersection_array', 'global_parameters']


@nx._dispatchable
def is_distance_regular(G):
    """Returns True if the graph is distance regular, False otherwise.

    A connected graph G is distance-regular if for any nodes x,y
    and any integers i,j=0,1,...,d (where d is the graph
    diameter), the number of vertices at distance i from x and
    distance j from y depends only on i,j and the graph distance
    between x and y, independently of the choice of x and y.

    Parameters
    ----------
    G: Networkx graph (undirected)

    Returns
    -------
    bool
      True if the graph is Distance Regular, False otherwise

    Examples
    --------
    >>> G = nx.hypercube_graph(6)
    >>> nx.is_distance_regular(G)
    True

    See Also
    --------
    intersection_array, global_parameters

    Notes
    -----
    For undirected and simple graphs only

    References
    ----------
    .. [1] Brouwer, A. E.; Cohen, A. M.; and Neumaier, A.
        Distance-Regular Graphs. New York: Springer-Verlag, 1989.
    .. [2] Weisstein, Eric W. "Distance-Regular Graph."
        http://mathworld.wolfram.com/Distance-RegularGraph.html

    """
    if not nx.is_connected(G):
        return False
    
    d = diameter(G)
    n = G.number_of_nodes()
    
    for u in G.nodes():
        distances = nx.single_source_shortest_path_length(G, u)
        level_sizes = [0] * (d + 1)
        for v, dist in distances.items():
            level_sizes[dist] += 1
        
        for v in G.nodes():
            if u != v:
                v_distances = nx.single_source_shortest_path_length(G, v)
                v_level_sizes = [0] * (d + 1)
                for w, dist in v_distances.items():
                    v_level_sizes[dist] += 1
                
                if level_sizes != v_level_sizes:
                    return False
    
    return True


def global_parameters(b, c):
    """Returns global parameters for a given intersection array.

    Given a distance-regular graph G with integers b_i, c_i,i = 0,....,d
    such that for any 2 vertices x,y in G at a distance i=d(x,y), there
    are exactly c_i neighbors of y at a distance of i-1 from x and b_i
    neighbors of y at a distance of i+1 from x.

    Thus, a distance regular graph has the global parameters,
    [[c_0,a_0,b_0],[c_1,a_1,b_1],......,[c_d,a_d,b_d]] for the
    intersection array  [b_0,b_1,.....b_{d-1};c_1,c_2,.....c_d]
    where a_i+b_i+c_i=k , k= degree of every vertex.

    Parameters
    ----------
    b : list

    c : list

    Returns
    -------
    iterable
       An iterable over three tuples.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> b, c = nx.intersection_array(G)
    >>> list(nx.global_parameters(b, c))
    [(0, 0, 3), (1, 0, 2), (1, 1, 1), (1, 1, 1), (2, 0, 1), (3, 0, 0)]

    References
    ----------
    .. [1] Weisstein, Eric W. "Global Parameters."
       From MathWorld--A Wolfram Web Resource.
       http://mathworld.wolfram.com/GlobalParameters.html

    See Also
    --------
    intersection_array
    """
    d = len(b)
    k = b[0]
    for i in range(d + 1):
        c_i = c[i] if i > 0 else 0
        b_i = b[i] if i < d else 0
        a_i = k - b_i - c_i
        yield (c_i, a_i, b_i)


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def intersection_array(G):
    """Returns the intersection array of a distance-regular graph.

    Given a distance-regular graph G with integers b_i, c_i,i = 0,....,d
    such that for any 2 vertices x,y in G at a distance i=d(x,y), there
    are exactly c_i neighbors of y at a distance of i-1 from x and b_i
    neighbors of y at a distance of i+1 from x.

    A distance regular graph's intersection array is given by,
    [b_0,b_1,.....b_{d-1};c_1,c_2,.....c_d]

    Parameters
    ----------
    G: Networkx graph (undirected)

    Returns
    -------
    b,c: tuple of lists

    Examples
    --------
    >>> G = nx.icosahedral_graph()
    >>> nx.intersection_array(G)
    ([5, 2, 1], [1, 2, 5])

    References
    ----------
    .. [1] Weisstein, Eric W. "Intersection Array."
       From MathWorld--A Wolfram Web Resource.
       http://mathworld.wolfram.com/IntersectionArray.html

    See Also
    --------
    global_parameters
    """
    if not is_distance_regular(G):
        raise nx.NetworkXError("Graph is not distance regular.")
    
    d = diameter(G)
    n = G.number_of_nodes()
    k = G.degree(list(G.nodes())[0])
    
    b = [0] * d
    c = [0] * (d + 1)
    
    for i in range(d):
        u = list(G.nodes())[0]
        distances = nx.single_source_shortest_path_length(G, u)
        nodes_at_dist_i = [v for v, dist in distances.items() if dist == i]
        nodes_at_dist_i_plus_1 = [v for v, dist in distances.items() if dist == i + 1]
        
        if i == 0:
            b[i] = len(nodes_at_dist_i_plus_1)
        else:
            v = nodes_at_dist_i[0]
            b[i] = sum(1 for w in G.neighbors(v) if distances[w] == i + 1)
            c[i] = sum(1 for w in G.neighbors(v) if distances[w] == i - 1)
    
    c[d] = k - b[d-1]
    
    return b, c[1:]


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def is_strongly_regular(G):
    """Returns True if and only if the given graph is strongly
    regular.

    An undirected graph is *strongly regular* if

    * it is regular,
    * each pair of adjacent vertices has the same number of neighbors in
      common,
    * each pair of nonadjacent vertices has the same number of neighbors
      in common.

    Each strongly regular graph is a distance-regular graph.
    Conversely, if a distance-regular graph has diameter two, then it is
    a strongly regular graph. For more information on distance-regular
    graphs, see :func:`is_distance_regular`.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    bool
        Whether `G` is strongly regular.

    Examples
    --------

    The cycle graph on five vertices is strongly regular. It is
    two-regular, each pair of adjacent vertices has no shared neighbors,
    and each pair of nonadjacent vertices has one shared neighbor::

        >>> G = nx.cycle_graph(5)
        >>> nx.is_strongly_regular(G)
        True

    """
    if not nx.is_regular(G):
        return False

    n = G.number_of_nodes()
    if n < 3:
        return True

    degrees = list(dict(G.degree()).values())
    k = degrees[0]

    # Check if the graph is regular
    if not all(d == k for d in degrees):
        return False

    # Check common neighbors for adjacent and non-adjacent pairs
    adj_common = set()
    non_adj_common = set()

    for u in G:
        for v in G:
            if u != v:
                common = len(set(G.neighbors(u)) & set(G.neighbors(v)))
                if G.has_edge(u, v):
                    adj_common.add(common)
                else:
                    non_adj_common.add(common)

    # For strongly regular graphs, adj_common and non_adj_common should each have only one value
    return len(adj_common) == 1 and len(non_adj_common) == 1
