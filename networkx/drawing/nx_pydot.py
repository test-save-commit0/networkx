"""
*****
Pydot
*****

Import and export NetworkX graphs in Graphviz dot format using pydot.

Either this module or nx_agraph can be used to interface with graphviz.

Examples
--------
>>> G = nx.complete_graph(5)
>>> PG = nx.nx_pydot.to_pydot(G)
>>> H = nx.nx_pydot.from_pydot(PG)

See Also
--------
 - pydot:         https://github.com/erocarrera/pydot
 - Graphviz:      https://www.graphviz.org
 - DOT Language:  http://www.graphviz.org/doc/info/lang.html
"""
from locale import getpreferredencoding
import networkx as nx
from networkx.utils import open_file
__all__ = ['write_dot', 'read_dot', 'graphviz_layout', 'pydot_layout',
    'to_pydot', 'from_pydot']


@open_file(1, mode='w')
def write_dot(G, path):
    """Write NetworkX graph G to Graphviz dot format on path.

    Path can be a string or a file handle.
    """
    import pydot
    P = to_pydot(G)
    path.write(P.to_string())


@open_file(0, mode='r')
@nx._dispatchable(name='pydot_read_dot', graphs=None, returns_graph=True)
def read_dot(path):
    """Returns a NetworkX :class:`MultiGraph` or :class:`MultiDiGraph` from the
    dot file with the passed path.

    If this file contains multiple graphs, only the first such graph is
    returned. All graphs _except_ the first are silently ignored.

    Parameters
    ----------
    path : str or file
        Filename or file handle.

    Returns
    -------
    G : MultiGraph or MultiDiGraph
        A :class:`MultiGraph` or :class:`MultiDiGraph`.

    Notes
    -----
    Use `G = nx.Graph(nx.nx_pydot.read_dot(path))` to return a :class:`Graph` instead of a
    :class:`MultiGraph`.
    """
    import pydot
    data = path.read()
    P = pydot.graph_from_dot_data(data)
    return from_pydot(P)


@nx._dispatchable(graphs=None, returns_graph=True)
def from_pydot(P):
    """Returns a NetworkX graph from a Pydot graph.

    Parameters
    ----------
    P : Pydot graph
      A graph created with Pydot

    Returns
    -------
    G : NetworkX multigraph
        A MultiGraph or MultiDiGraph.

    Examples
    --------
    >>> K5 = nx.complete_graph(5)
    >>> A = nx.nx_pydot.to_pydot(K5)
    >>> G = nx.nx_pydot.from_pydot(A)  # return MultiGraph

    # make a Graph instead of MultiGraph
    >>> G = nx.Graph(nx.nx_pydot.from_pydot(A))

    """
    if P.get_strict(None):  # Directed graphs are automatically strict.
        G = nx.MultiDiGraph()
    else:
        G = nx.MultiGraph()

    for node in P.get_nodes():
        G.add_node(node.get_name().strip('"'), **node.get_attributes())

    for edge in P.get_edges():
        u = edge.get_source().strip('"')
        v = edge.get_destination().strip('"')
        attr = edge.get_attributes()
        G.add_edge(u, v, **attr)

    return G


def to_pydot(N):
    """Returns a pydot graph from a NetworkX graph N.

    Parameters
    ----------
    N : NetworkX graph
      A graph created with NetworkX

    Examples
    --------
    >>> K5 = nx.complete_graph(5)
    >>> P = nx.nx_pydot.to_pydot(K5)

    Notes
    -----

    """
    import pydot

    # Create a new pydot graph
    if N.is_directed():
        P = pydot.Dot(graph_type='digraph', strict=N.is_directed())
    else:
        P = pydot.Dot(graph_type='graph', strict=N.is_directed())

    # Add nodes to the pydot graph
    for n, nodedata in N.nodes(data=True):
        node = pydot.Node(str(n), **nodedata)
        P.add_node(node)

    # Add edges to the pydot graph
    for u, v, edgedata in N.edges(data=True):
        edge = pydot.Edge(str(u), str(v), **edgedata)
        P.add_edge(edge)

    return P


def graphviz_layout(G, prog='neato', root=None):
    """Create node positions using Pydot and Graphviz.

    Returns a dictionary of positions keyed by node.

    Parameters
    ----------
    G : NetworkX Graph
        The graph for which the layout is computed.
    prog : string (default: 'neato')
        The name of the GraphViz program to use for layout.
        Options depend on GraphViz version but may include:
        'dot', 'twopi', 'fdp', 'sfdp', 'circo'
    root : Node from G or None (default: None)
        The node of G from which to start some layout algorithms.

    Returns
    -------
      Dictionary of (x, y) positions keyed by node.

    Examples
    --------
    >>> G = nx.complete_graph(4)
    >>> pos = nx.nx_pydot.graphviz_layout(G)
    >>> pos = nx.nx_pydot.graphviz_layout(G, prog="dot")

    Notes
    -----
    This is a wrapper for pydot_layout.
    """
    return pydot_layout(G, prog=prog, root=root)


def pydot_layout(G, prog='neato', root=None):
    """Create node positions using :mod:`pydot` and Graphviz.

    Parameters
    ----------
    G : Graph
        NetworkX graph to be laid out.
    prog : string  (default: 'neato')
        Name of the GraphViz command to use for layout.
        Options depend on GraphViz version but may include:
        'dot', 'twopi', 'fdp', 'sfdp', 'circo'
    root : Node from G or None (default: None)
        The node of G from which to start some layout algorithms.

    Returns
    -------
    dict
        Dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.complete_graph(4)
    >>> pos = nx.nx_pydot.pydot_layout(G)
    >>> pos = nx.nx_pydot.pydot_layout(G, prog="dot")

    Notes
    -----
    If you use complex node objects, they may have the same string
    representation and GraphViz could treat them as the same node.
    The layout may assign both nodes a single location. See Issue #1568
    If this occurs in your case, consider relabeling the nodes just
    for the layout computation using something similar to::

        H = nx.convert_node_labels_to_integers(G, label_attribute="node_label")
        H_layout = nx.nx_pydot.pydot_layout(G, prog="dot")
        G_layout = {H.nodes[n]["node_label"]: p for n, p in H_layout.items()}

    """
    import pydot
    P = to_pydot(G)

    if root is not None:
        P.set("root", str(root))

    D = P.create_dot(prog=prog)

    if D == "":  # no data returned
        print(f"Graphviz layout with {prog} failed")
        print()
        print("To debug what happened try:")
        print("P = nx.nx_pydot.to_pydot(G)")
        print("P.write_dot('file.dot')")
        print(f"And then run {prog} on file.dot")
        return

    Q = pydot.graph_from_dot_data(D)

    node_pos = {}
    for n in Q.get_nodes():
        node = n.get_name().strip('"')
        pos = n.get_pos()[1:-1]  # strip leading and trailing double quotes
        if pos != None:
            xx, yy = pos.split(",")
            node_pos[node] = (float(xx), float(yy))

    return node_pos
