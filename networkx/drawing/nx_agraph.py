"""
***************
Graphviz AGraph
***************

Interface to pygraphviz AGraph class.

Examples
--------
>>> G = nx.complete_graph(5)
>>> A = nx.nx_agraph.to_agraph(G)
>>> H = nx.nx_agraph.from_agraph(A)

See Also
--------
 - Pygraphviz: http://pygraphviz.github.io/
 - Graphviz:      https://www.graphviz.org
 - DOT Language:  http://www.graphviz.org/doc/info/lang.html
"""
import os
import tempfile
import networkx as nx
__all__ = ['from_agraph', 'to_agraph', 'write_dot', 'read_dot',
    'graphviz_layout', 'pygraphviz_layout', 'view_pygraphviz']


@nx._dispatchable(graphs=None, returns_graph=True)
def from_agraph(A, create_using=None):
    """Returns a NetworkX Graph or DiGraph from a PyGraphviz graph.

    Parameters
    ----------
    A : PyGraphviz AGraph
      A graph created with PyGraphviz

    create_using : NetworkX graph constructor, optional (default=None)
       Graph type to create. If graph instance, then cleared before populated.
       If `None`, then the appropriate Graph type is inferred from `A`.

    Examples
    --------
    >>> K5 = nx.complete_graph(5)
    >>> A = nx.nx_agraph.to_agraph(K5)
    >>> G = nx.nx_agraph.from_agraph(A)

    Notes
    -----
    The Graph G will have a dictionary G.graph_attr containing
    the default graphviz attributes for graphs, nodes and edges.

    Default node attributes will be in the dictionary G.node_attr
    which is keyed by node.

    Edge attributes will be returned as edge data in G.  With
    edge_attr=False the edge data will be the Graphviz edge weight
    attribute or the value 1 if no edge weight attribute is found.

    """
    if create_using is None:
        if A.is_directed():
            create_using = nx.DiGraph
        else:
            create_using = nx.Graph

    G = nx.empty_graph(0, create_using)
    G.name = A.name

    # Add nodes
    for n in A.nodes():
        G.add_node(n.name, **dict(n.attr))

    # Add edges
    for e in A.edges():
        u, v = e
        attr = dict(e.attr)
        G.add_edge(u.name, v.name, **attr)

    # Add graph attributes
    G.graph.update(A.graph_attr)

    return G


def to_agraph(N):
    """Returns a pygraphviz graph from a NetworkX graph N.

    Parameters
    ----------
    N : NetworkX graph
      A graph created with NetworkX

    Examples
    --------
    >>> K5 = nx.complete_graph(5)
    >>> A = nx.nx_agraph.to_agraph(K5)

    Notes
    -----
    If N has an dict N.graph_attr an attempt will be made first
    to copy properties attached to the graph (see from_agraph)
    and then updated with the calling arguments if any.

    """
    import pygraphviz

    directed = N.is_directed()
    strict = nx.number_of_selfloops(N) == 0 and not N.is_multigraph()
    A = pygraphviz.AGraph(strict=strict, directed=directed)

    # Add nodes
    for n, nodedata in N.nodes(data=True):
        A.add_node(n, **nodedata)

    # Add edges
    if N.is_multigraph():
        for u, v, key, edgedata in N.edges(data=True, keys=True):
            str_edgedata = {k: str(v) for k, v in edgedata.items()}
            A.add_edge(u, v, key=str(key), **str_edgedata)
    else:
        for u, v, edgedata in N.edges(data=True):
            str_edgedata = {k: str(v) for k, v in edgedata.items()}
            A.add_edge(u, v, **str_edgedata)

    # Add graph attributes
    A.graph_attr.update(N.graph.get("graph", {}))
    A.node_attr.update(N.graph.get("node", {}))
    A.edge_attr.update(N.graph.get("edge", {}))

    return A


def write_dot(G, path):
    """Write NetworkX graph G to Graphviz dot format on path.

    Parameters
    ----------
    G : graph
       A networkx graph
    path : filename
       Filename or file handle to write

    Notes
    -----
    To use a specific graph layout, call ``A.layout`` prior to `write_dot`.
    Note that some graphviz layouts are not guaranteed to be deterministic,
    see https://gitlab.com/graphviz/graphviz/-/issues/1767 for more info.
    """
    A = to_agraph(G)
    A.write(path)


@nx._dispatchable(name='agraph_read_dot', graphs=None, returns_graph=True)
def read_dot(path):
    """Returns a NetworkX graph from a dot file on path.

    Parameters
    ----------
    path : file or string
       File name or file handle to read.
    """
    import pygraphviz
    A = pygraphviz.AGraph(file=path)
    return from_agraph(A)


def graphviz_layout(G, prog='neato', root=None, args=''):
    """Create node positions for G using Graphviz.

    Parameters
    ----------
    G : NetworkX graph
      A graph created with NetworkX
    prog : string
      Name of Graphviz layout program
    root : string, optional
      Root node for twopi layout
    args : string, optional
      Extra arguments to Graphviz layout program

    Returns
    -------
    Dictionary of x, y, positions keyed by node.

    Examples
    --------
    >>> G = nx.petersen_graph()
    >>> pos = nx.nx_agraph.graphviz_layout(G)
    >>> pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    Notes
    -----
    This is a wrapper for pygraphviz_layout.

    Note that some graphviz layouts are not guaranteed to be deterministic,
    see https://gitlab.com/graphviz/graphviz/-/issues/1767 for more info.
    """
    return pygraphviz_layout(G, prog=prog, root=root, args=args)


def pygraphviz_layout(G, prog='neato', root=None, args=''):
    """Create node positions for G using Graphviz.

    Parameters
    ----------
    G : NetworkX graph
      A graph created with NetworkX
    prog : string
      Name of Graphviz layout program
    root : string, optional
      Root node for twopi layout
    args : string, optional
      Extra arguments to Graphviz layout program

    Returns
    -------
    node_pos : dict
      Dictionary of x, y, positions keyed by node.

    Examples
    --------
    >>> G = nx.petersen_graph()
    >>> pos = nx.nx_agraph.graphviz_layout(G)
    >>> pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    Notes
    -----
    If you use complex node objects, they may have the same string
    representation and GraphViz could treat them as the same node.
    The layout may assign both nodes a single location. See Issue #1568
    If this occurs in your case, consider relabeling the nodes just
    for the layout computation using something similar to::

        >>> H = nx.convert_node_labels_to_integers(G, label_attribute="node_label")
        >>> H_layout = nx.nx_agraph.pygraphviz_layout(G, prog="dot")
        >>> G_layout = {H.nodes[n]["node_label"]: p for n, p in H_layout.items()}

    Note that some graphviz layouts are not guaranteed to be deterministic,
    see https://gitlab.com/graphviz/graphviz/-/issues/1767 for more info.
    """
    import pygraphviz

    A = to_agraph(G)
    A.layout(prog=prog, args=args)

    node_pos = {}
    for n in G:
        node = A.get_node(n)
        try:
            xx, yy = node.attr["pos"].split(',')
            node_pos[n] = (float(xx), float(yy))
        except:
            print(f"No position for node {n}")

    return node_pos


@nx.utils.open_file(5, 'w+b')
def view_pygraphviz(G, edgelabel=None, prog='dot', args='', suffix='', path=None, show=True):
    """Views the graph G using the specified layout algorithm.

    Parameters
    ----------
    G : NetworkX graph
        The machine to draw.
    edgelabel : str, callable, None
        If a string, then it specifies the edge attribute to be displayed
        on the edge labels. If a callable, then it is called for each
        edge and it should return the string to be displayed on the edges.
        The function signature of `edgelabel` should be edgelabel(data),
        where `data` is the edge attribute dictionary.
    prog : string
        Name of Graphviz layout program.
    args : str
        Additional arguments to pass to the Graphviz layout program.
    suffix : str
        If `filename` is None, we save to a temporary file.  The value of
        `suffix` will appear at the tail end of the temporary filename.
    path : str, None
        The filename used to save the image.  If None, save to a temporary
        file.  File formats are the same as those from pygraphviz.agraph.draw.
    show : bool, default = True
        Whether to display the graph with :mod:`PIL.Image.show`,
        default is `True`. If `False`, the rendered graph is still available
        at `path`.

    Returns
    -------
    path : str
        The filename of the generated image.
    A : PyGraphviz graph
        The PyGraphviz graph instance used to generate the image.

    Notes
    -----
    If this function is called in succession too quickly, sometimes the
    image is not displayed. So you might consider time.sleep(.5) between
    calls if you experience problems.

    Note that some graphviz layouts are not guaranteed to be deterministic,
    see https://gitlab.com/graphviz/graphviz/-/issues/1767 for more info.

    """
    import pygraphviz
    import tempfile
    import os

    if path is None:
        path = tempfile.NamedTemporaryFile(suffix=f'{suffix}.png', delete=False)
        close_file = True
    else:
        close_file = False

    A = to_agraph(G)
    A.layout(prog=prog, args=args)

    if edgelabel is not None:
        if callable(edgelabel):
            for edge in A.edges():
                edge.attr['label'] = str(edgelabel(edge.attr))
        else:
            for edge in A.edges():
                edge.attr['label'] = str(edge.attr.get(edgelabel, ''))

    A.draw(path=path, format='png', prog=prog, args=args)

    if close_file:
        path.close()

    if show:
        from PIL import Image
        Image.open(path.name).show()

    return path.name, A
