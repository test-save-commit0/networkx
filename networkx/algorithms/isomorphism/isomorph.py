"""
Graph isomorphism functions.
"""
import networkx as nx
from networkx.exception import NetworkXError
__all__ = ['could_be_isomorphic', 'fast_could_be_isomorphic',
    'faster_could_be_isomorphic', 'is_isomorphic']


@nx._dispatchable(graphs={'G1': 0, 'G2': 1})
def could_be_isomorphic(G1, G2):
    """Returns False if graphs are definitely not isomorphic.
    True does NOT guarantee isomorphism.

    Parameters
    ----------
    G1, G2 : graphs
       The two graphs G1 and G2 must be the same type.

    Notes
    -----
    Checks for matching degree, triangle, and number of cliques sequences.
    The triangle sequence contains the number of triangles each node is part of.
    The clique sequence contains for each node the number of maximal cliques
    involving that node.

    """
    # Check if the graphs have the same number of nodes and edges
    if G1.number_of_nodes() != G2.number_of_nodes() or G1.number_of_edges() != G2.number_of_edges():
        return False

    # Check degree sequence
    degree_seq1 = sorted(d for n, d in G1.degree())
    degree_seq2 = sorted(d for n, d in G2.degree())
    if degree_seq1 != degree_seq2:
        return False

    # Check triangle sequence
    triangle_seq1 = sorted(nx.triangles(G1).values())
    triangle_seq2 = sorted(nx.triangles(G2).values())
    if triangle_seq1 != triangle_seq2:
        return False

    # Check clique sequence
    clique_seq1 = sorted(len(list(nx.cliques_containing_node(G1, n))) for n in G1)
    clique_seq2 = sorted(len(list(nx.cliques_containing_node(G2, n))) for n in G2)
    if clique_seq1 != clique_seq2:
        return False

    return True


graph_could_be_isomorphic = could_be_isomorphic


@nx._dispatchable(graphs={'G1': 0, 'G2': 1})
def fast_could_be_isomorphic(G1, G2):
    """Returns False if graphs are definitely not isomorphic.

    True does NOT guarantee isomorphism.

    Parameters
    ----------
    G1, G2 : graphs
       The two graphs G1 and G2 must be the same type.

    Notes
    -----
    Checks for matching degree and triangle sequences. The triangle
    sequence contains the number of triangles each node is part of.
    """
    # Check if the graphs have the same number of nodes and edges
    if G1.number_of_nodes() != G2.number_of_nodes() or G1.number_of_edges() != G2.number_of_edges():
        return False

    # Check degree sequence
    degree_seq1 = sorted(d for n, d in G1.degree())
    degree_seq2 = sorted(d for n, d in G2.degree())
    if degree_seq1 != degree_seq2:
        return False

    # Check triangle sequence
    triangle_seq1 = sorted(nx.triangles(G1).values())
    triangle_seq2 = sorted(nx.triangles(G2).values())
    if triangle_seq1 != triangle_seq2:
        return False

    return True


fast_graph_could_be_isomorphic = fast_could_be_isomorphic


@nx._dispatchable(graphs={'G1': 0, 'G2': 1})
def faster_could_be_isomorphic(G1, G2):
    """Returns False if graphs are definitely not isomorphic.

    True does NOT guarantee isomorphism.

    Parameters
    ----------
    G1, G2 : graphs
       The two graphs G1 and G2 must be the same type.

    Notes
    -----
    Checks for matching degree sequences.
    """
    # Check if the graphs have the same number of nodes and edges
    if G1.number_of_nodes() != G2.number_of_nodes() or G1.number_of_edges() != G2.number_of_edges():
        return False

    # Check degree sequence
    degree_seq1 = sorted(d for n, d in G1.degree())
    degree_seq2 = sorted(d for n, d in G2.degree())
    if degree_seq1 != degree_seq2:
        return False

    return True


faster_graph_could_be_isomorphic = faster_could_be_isomorphic


@nx._dispatchable(graphs={'G1': 0, 'G2': 1}, preserve_edge_attrs=
    'edge_match', preserve_node_attrs='node_match')
def is_isomorphic(G1, G2, node_match=None, edge_match=None):
    """Returns True if the graphs G1 and G2 are isomorphic and False otherwise.

    Parameters
    ----------
    G1, G2: graphs
        The two graphs G1 and G2 must be the same type.

    node_match : callable
        A function that returns True if node n1 in G1 and n2 in G2 should
        be considered equal during the isomorphism test.
        If node_match is not specified then node attributes are not considered.

        The function will be called like

           node_match(G1.nodes[n1], G2.nodes[n2]).

        That is, the function will receive the node attribute dictionaries
        for n1 and n2 as inputs.

    edge_match : callable
        A function that returns True if the edge attribute dictionary
        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should
        be considered equal during the isomorphism test.  If edge_match is
        not specified then edge attributes are not considered.

        The function will be called like

           edge_match(G1[u1][v1], G2[u2][v2]).

        That is, the function will receive the edge attribute dictionaries
        of the edges under consideration.

    Notes
    -----
    Uses the vf2 algorithm [1]_.

    Examples
    --------
    >>> import networkx.algorithms.isomorphism as iso

    For digraphs G1 and G2, using 'weight' edge attribute (default: 1)

    >>> G1 = nx.DiGraph()
    >>> G2 = nx.DiGraph()
    >>> nx.add_path(G1, [1, 2, 3, 4], weight=1)
    >>> nx.add_path(G2, [10, 20, 30, 40], weight=2)
    >>> em = iso.numerical_edge_match("weight", 1)
    >>> nx.is_isomorphic(G1, G2)  # no weights considered
    True
    >>> nx.is_isomorphic(G1, G2, edge_match=em)  # match weights
    False

    For multidigraphs G1 and G2, using 'fill' node attribute (default: '')

    >>> G1 = nx.MultiDiGraph()
    >>> G2 = nx.MultiDiGraph()
    >>> G1.add_nodes_from([1, 2, 3], fill="red")
    >>> G2.add_nodes_from([10, 20, 30, 40], fill="red")
    >>> nx.add_path(G1, [1, 2, 3, 4], weight=3, linewidth=2.5)
    >>> nx.add_path(G2, [10, 20, 30, 40], weight=3)
    >>> nm = iso.categorical_node_match("fill", "red")
    >>> nx.is_isomorphic(G1, G2, node_match=nm)
    True

    For multidigraphs G1 and G2, using 'weight' edge attribute (default: 7)

    >>> G1.add_edge(1, 2, weight=7)
    1
    >>> G2.add_edge(10, 20)
    1
    >>> em = iso.numerical_multiedge_match("weight", 7, rtol=1e-6)
    >>> nx.is_isomorphic(G1, G2, edge_match=em)
    True

    For multigraphs G1 and G2, using 'weight' and 'linewidth' edge attributes
    with default values 7 and 2.5. Also using 'fill' node attribute with
    default value 'red'.

    >>> em = iso.numerical_multiedge_match(["weight", "linewidth"], [7, 2.5])
    >>> nm = iso.categorical_node_match("fill", "red")
    >>> nx.is_isomorphic(G1, G2, edge_match=em, node_match=nm)
    True

    See Also
    --------
    numerical_node_match, numerical_edge_match, numerical_multiedge_match
    categorical_node_match, categorical_edge_match, categorical_multiedge_match

    References
    ----------
    .. [1]  L. P. Cordella, P. Foggia, C. Sansone, M. Vento,
       "An Improved Algorithm for Matching Large Graphs",
       3rd IAPR-TC15 Workshop  on Graph-based Representations in
       Pattern Recognition, Cuen, pp. 149-159, 2001.
       https://www.researchgate.net/publication/200034365_An_Improved_Algorithm_for_Matching_Large_Graphs
    """
    if G1.is_directed() != G2.is_directed():
        return False

    if G1.number_of_nodes() != G2.number_of_nodes():
        return False

    if G1.number_of_edges() != G2.number_of_edges():
        return False

    GM = nx.isomorphism.GraphMatcher(G1, G2, node_match=node_match, edge_match=edge_match)
    return GM.is_isomorphic()
