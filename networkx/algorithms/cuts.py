"""Functions for finding and evaluating cuts in a graph.

"""
from itertools import chain
import networkx as nx
__all__ = ['boundary_expansion', 'conductance', 'cut_size',
    'edge_expansion', 'mixing_expansion', 'node_expansion',
    'normalized_cut_size', 'volume']


@nx._dispatchable(edge_attrs='weight')
def cut_size(G, S, T=None, weight=None):
    """Returns the size of the cut between two sets of nodes.

    A *cut* is a partition of the nodes of a graph into two sets. The
    *cut size* is the sum of the weights of the edges "between" the two
    sets of nodes.

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    T : collection
        A collection of nodes in `G`. If not specified, this is taken to
        be the set complement of `S`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        Total weight of all edges from nodes in set `S` to nodes in
        set `T` (and, in the case of directed graphs, all edges from
        nodes in `T` to nodes in `S`).

    Examples
    --------
    In the graph with two cliques joined by a single edges, the natural
    bipartition of the graph into two blocks, one for each clique,
    yields a cut of weight one::

        >>> G = nx.barbell_graph(3, 0)
        >>> S = {0, 1, 2}
        >>> T = {3, 4, 5}
        >>> nx.cut_size(G, S, T)
        1

    Each parallel edge in a multigraph is counted when determining the
    cut size::

        >>> G = nx.MultiGraph(["ab", "ab"])
        >>> S = {"a"}
        >>> T = {"b"}
        >>> nx.cut_size(G, S, T)
        2

    Notes
    -----
    In a multigraph, the cut size is the total weight of edges including
    multiplicity.

    """
    if T is None:
        T = set(G.nodes()) - set(S)
    
    cut_edges = ((u, v) for u in S for v in T if G.has_edge(u, v))
    
    if G.is_directed():
        cut_edges = chain(cut_edges, ((u, v) for u in T for v in S if G.has_edge(u, v)))
    
    if weight is None:
        return sum(1 for _ in cut_edges)
    else:
        return sum(G[u][v].get(weight, 1) for u, v in cut_edges)


@nx._dispatchable(edge_attrs='weight')
def volume(G, S, weight=None):
    """Returns the volume of a set of nodes.

    The *volume* of a set *S* is the sum of the (out-)degrees of nodes
    in *S* (taking into account parallel edges in multigraphs). [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The volume of the set of nodes represented by `S` in the graph
        `G`.

    See also
    --------
    conductance
    cut_size
    edge_expansion
    edge_boundary
    normalized_cut_size

    References
    ----------
    .. [1] David Gleich.
           *Hierarchical Directed Spectral Graph Partitioning*.
           <https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202005%20-%20hierarchical%20directed%20spectral.pdf>

    """
    if G.is_directed():
        degree = G.out_degree
    else:
        degree = G.degree
    
    if weight is None:
        return sum(dict(degree(S)).values())
    else:
        return sum(dict(degree(S, weight=weight)).values())


@nx._dispatchable(edge_attrs='weight')
def normalized_cut_size(G, S, T=None, weight=None):
    """Returns the normalized size of the cut between two sets of nodes.

    The *normalized cut size* is the cut size times the sum of the
    reciprocal sizes of the volumes of the two sets. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    T : collection
        A collection of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The normalized cut size between the two sets `S` and `T`.

    Notes
    -----
    In a multigraph, the cut size is the total weight of edges including
    multiplicity.

    See also
    --------
    conductance
    cut_size
    edge_expansion
    volume

    References
    ----------
    .. [1] David Gleich.
           *Hierarchical Directed Spectral Graph Partitioning*.
           <https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202005%20-%20hierarchical%20directed%20spectral.pdf>

    """
    if T is None:
        T = set(G.nodes()) - set(S)
    
    cut = cut_size(G, S, T, weight)
    vol_S = volume(G, S, weight)
    vol_T = volume(G, T, weight)
    
    if vol_S == 0 or vol_T == 0:
        return float('inf')
    
    return cut * (1 / vol_S + 1 / vol_T)


@nx._dispatchable(edge_attrs='weight')
def conductance(G, S, T=None, weight=None):
    """Returns the conductance of two sets of nodes.

    The *conductance* is the quotient of the cut size and the smaller of
    the volumes of the two sets. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    T : collection
        A collection of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The conductance between the two sets `S` and `T`.

    See also
    --------
    cut_size
    edge_expansion
    normalized_cut_size
    volume

    References
    ----------
    .. [1] David Gleich.
           *Hierarchical Directed Spectral Graph Partitioning*.
           <https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202005%20-%20hierarchical%20directed%20spectral.pdf>

    """
    if T is None:
        T = set(G.nodes()) - set(S)
    
    cut = cut_size(G, S, T, weight)
    vol_S = volume(G, S, weight)
    vol_T = volume(G, T, weight)
    
    if vol_S == 0 or vol_T == 0:
        return float('inf')
    
    return cut / min(vol_S, vol_T)


@nx._dispatchable(edge_attrs='weight')
def edge_expansion(G, S, T=None, weight=None):
    """Returns the edge expansion between two node sets.

    The *edge expansion* is the quotient of the cut size and the smaller
    of the cardinalities of the two sets. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    T : collection
        A collection of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The edge expansion between the two sets `S` and `T`.

    See also
    --------
    boundary_expansion
    mixing_expansion
    node_expansion

    References
    ----------
    .. [1] Fan Chung.
           *Spectral Graph Theory*.
           (CBMS Regional Conference Series in Mathematics, No. 92),
           American Mathematical Society, 1997, ISBN 0-8218-0315-8
           <http://www.math.ucsd.edu/~fan/research/revised.html>

    """
    if T is None:
        T = set(G.nodes()) - set(S)
    
    cut = cut_size(G, S, T, weight)
    return cut / min(len(S), len(T))


@nx._dispatchable(edge_attrs='weight')
def mixing_expansion(G, S, T=None, weight=None):
    """Returns the mixing expansion between two node sets.

    The *mixing expansion* is the quotient of the cut size and twice the
    number of edges in the graph. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    T : collection
        A collection of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The mixing expansion between the two sets `S` and `T`.

    See also
    --------
    boundary_expansion
    edge_expansion
    node_expansion

    References
    ----------
    .. [1] Vadhan, Salil P.
           "Pseudorandomness."
           *Foundations and Trends
           in Theoretical Computer Science* 7.1–3 (2011): 1–336.
           <https://doi.org/10.1561/0400000010>

    """
    if T is None:
        T = set(G.nodes()) - set(S)
    
    cut = cut_size(G, S, T, weight)
    total_edges = G.number_of_edges()
    
    if weight is not None:
        total_edges = sum(d[weight] for u, v, d in G.edges(data=True))
    
    return cut / (2 * total_edges)


@nx._dispatchable
def node_expansion(G, S):
    """Returns the node expansion of the set `S`.

    The *node expansion* is the quotient of the size of the node
    boundary of *S* and the cardinality of *S*. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    Returns
    -------
    number
        The node expansion of the set `S`.

    See also
    --------
    boundary_expansion
    edge_expansion
    mixing_expansion

    References
    ----------
    .. [1] Vadhan, Salil P.
           "Pseudorandomness."
           *Foundations and Trends
           in Theoretical Computer Science* 7.1–3 (2011): 1–336.
           <https://doi.org/10.1561/0400000010>

    """
    S = set(S)
    node_boundary = set(n for s in S for n in G[s]) - S
    return len(node_boundary) / len(S)


@nx._dispatchable
def boundary_expansion(G, S):
    """Returns the boundary expansion of the set `S`.

    The *boundary expansion* is the quotient of the size
    of the node boundary and the cardinality of *S*. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    Returns
    -------
    number
        The boundary expansion of the set `S`.

    See also
    --------
    edge_expansion
    mixing_expansion
    node_expansion

    References
    ----------
    .. [1] Vadhan, Salil P.
           "Pseudorandomness."
           *Foundations and Trends in Theoretical Computer Science*
           7.1–3 (2011): 1–336.
           <https://doi.org/10.1561/0400000010>

    """
    S = set(S)
    node_boundary = set(n for s in S for n in G[s]) - S
    return len(node_boundary) / len(S)
