"""
==========================
Bipartite Graph Algorithms
==========================
"""
import networkx as nx
from networkx.algorithms.components import connected_components
from networkx.exception import AmbiguousSolution
__all__ = ['is_bipartite', 'is_bipartite_node_set', 'color', 'sets',
    'density', 'degrees']


@nx._dispatchable
def color(G):
    """Returns a two-coloring of the graph.

    Raises an exception if the graph is not bipartite.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    color : dictionary
        A dictionary keyed by node with a 1 or 0 as data for each node color.

    Raises
    ------
    NetworkXError
        If the graph is not two-colorable.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> c = bipartite.color(G)
    >>> print(c)
    {0: 1, 1: 0, 2: 1, 3: 0}

    You can use this to set a node attribute indicating the bipartite set:

    >>> nx.set_node_attributes(G, c, "bipartite")
    >>> print(G.nodes[0]["bipartite"])
    1
    >>> print(G.nodes[1]["bipartite"])
    0
    """
    color = {}
    for component in nx.connected_components(G):
        try:
            start = next(iter(component))
            color[start] = 1
            queue = [start]
            while queue:
                node = queue.pop(0)
                node_color = color[node]
                for neighbor in G[node]:
                    if neighbor not in color:
                        color[neighbor] = 1 - node_color
                        queue.append(neighbor)
                    elif color[neighbor] == node_color:
                        raise nx.NetworkXError("Graph is not bipartite.")
        except StopIteration:
            pass
    return color


@nx._dispatchable
def is_bipartite(G):
    """Returns True if graph G is bipartite, False if not.

    Parameters
    ----------
    G : NetworkX graph

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> print(bipartite.is_bipartite(G))
    True

    See Also
    --------
    color, is_bipartite_node_set
    """
    try:
        color(G)
        return True
    except nx.NetworkXError:
        return False


@nx._dispatchable
def is_bipartite_node_set(G, nodes):
    """Returns True if nodes and G/nodes are a bipartition of G.

    Parameters
    ----------
    G : NetworkX graph

    nodes: list or container
      Check if nodes are a one of a bipartite set.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> X = set([1, 3])
    >>> bipartite.is_bipartite_node_set(G, X)
    True

    Notes
    -----
    An exception is raised if the input nodes are not distinct, because in this
    case some bipartite algorithms will yield incorrect results.
    For connected graphs the bipartite sets are unique.  This function handles
    disconnected graphs.
    """
    S = set(nodes)
    if len(S) != len(nodes):
        raise nx.NetworkXError("Input nodes are not distinct.")

    T = set(G) - S
    for node in S:
        if any(neighbor in S for neighbor in G[node]):
            return False
    for node in T:
        if any(neighbor in T for neighbor in G[node]):
            return False
    return True


@nx._dispatchable
def sets(G, top_nodes=None):
    """Returns bipartite node sets of graph G.

    Raises an exception if the graph is not bipartite or if the input
    graph is disconnected and thus more than one valid solution exists.
    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    Parameters
    ----------
    G : NetworkX graph

    top_nodes : container, optional
      Container with all nodes in one bipartite node set. If not supplied
      it will be computed. But if more than one solution exists an exception
      will be raised.

    Returns
    -------
    X : set
      Nodes from one side of the bipartite graph.
    Y : set
      Nodes from the other side.

    Raises
    ------
    AmbiguousSolution
      Raised if the input bipartite graph is disconnected and no container
      with all nodes in one bipartite set is provided. When determining
      the nodes in each bipartite set more than one valid solution is
      possible if the input graph is disconnected.
    NetworkXError
      Raised if the input graph is not bipartite.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> X, Y = bipartite.sets(G)
    >>> list(X)
    [0, 2]
    >>> list(Y)
    [1, 3]

    See Also
    --------
    color

    """
    if not is_bipartite(G):
        raise nx.NetworkXError("Graph is not bipartite.")

    if top_nodes is not None:
        X = set(top_nodes)
        Y = set(G) - X
        if is_bipartite_node_set(G, X):
            return (X, Y)
        else:
            raise nx.NetworkXError("Graph is not bipartite.")

    cc = list(connected_components(G))
    if len(cc) > 1:
        raise AmbiguousSolution("Graph is disconnected.")

    node_color = color(G)
    X = {n for n, c in node_color.items() if c == 0}
    Y = {n for n, c in node_color.items() if c == 1}
    return (X, Y)


@nx._dispatchable(graphs='B')
def density(B, nodes):
    """Returns density of bipartite graph B.

    Parameters
    ----------
    B : NetworkX graph

    nodes: list or container
      Nodes in one node set of the bipartite graph.

    Returns
    -------
    d : float
       The bipartite density

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.complete_bipartite_graph(3, 2)
    >>> X = set([0, 1, 2])
    >>> bipartite.density(G, X)
    1.0
    >>> Y = set([3, 4])
    >>> bipartite.density(G, Y)
    1.0

    Notes
    -----
    The container of nodes passed as argument must contain all nodes
    in one of the two bipartite node sets to avoid ambiguity in the
    case of disconnected graphs.
    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    color
    """
    n = len(nodes)
    m = len(set(B) - set(nodes))
    if n == 0 or m == 0:
        return 0.0
    edges = B.edges()
    return len(edges) / (n * m)


@nx._dispatchable(graphs='B', edge_attrs='weight')
def degrees(B, nodes, weight=None):
    """Returns the degrees of the two node sets in the bipartite graph B.

    Parameters
    ----------
    B : NetworkX graph

    nodes: list or container
      Nodes in one node set of the bipartite graph.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    Returns
    -------
    (degX,degY) : tuple of dictionaries
       The degrees of the two bipartite sets as dictionaries keyed by node.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.complete_bipartite_graph(3, 2)
    >>> Y = set([3, 4])
    >>> degX, degY = bipartite.degrees(G, Y)
    >>> dict(degX)
    {0: 2, 1: 2, 2: 2}

    Notes
    -----
    The container of nodes passed as argument must contain all nodes
    in one of the two bipartite node sets to avoid ambiguity in the
    case of disconnected graphs.
    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    color, density
    """
    X = set(nodes)
    Y = set(B) - X
    degX = {x: B.degree(x, weight=weight) for x in X}
    degY = {y: B.degree(y, weight=weight) for y in Y}
    return (degX, degY)
