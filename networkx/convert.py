"""Functions to convert NetworkX graphs to and from other formats.

The preferred way of converting data to a NetworkX graph is through the
graph constructor.  The constructor calls the to_networkx_graph() function
which attempts to guess the input type and convert it automatically.

Examples
--------
Create a graph with a single edge from a dictionary of dictionaries

>>> d = {0: {1: 1}}  # dict-of-dicts single edge (0,1)
>>> G = nx.Graph(d)

See Also
--------
nx_agraph, nx_pydot
"""
import warnings
from collections.abc import Collection, Generator, Iterator
import networkx as nx
__all__ = ['to_networkx_graph', 'from_dict_of_dicts', 'to_dict_of_dicts',
    'from_dict_of_lists', 'to_dict_of_lists', 'from_edgelist', 'to_edgelist']


def to_networkx_graph(data, create_using=None, multigraph_input=False):
    """Make a NetworkX graph from a known data structure.

    The preferred way to call this is automatically
    from the class constructor

    >>> d = {0: {1: {"weight": 1}}}  # dict-of-dicts single edge (0,1)
    >>> G = nx.Graph(d)

    instead of the equivalent

    >>> G = nx.from_dict_of_dicts(d)

    Parameters
    ----------
    data : object to be converted

        Current known types are:
         any NetworkX graph
         dict-of-dicts
         dict-of-lists
         container (e.g. set, list, tuple) of edges
         iterator (e.g. itertools.chain) that produces edges
         generator of edges
         Pandas DataFrame (row per edge)
         2D numpy array
         scipy sparse array
         pygraphviz agraph

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    multigraph_input : bool (default False)
        If True and  data is a dict_of_dicts,
        try to create a multigraph assuming dict_of_dict_of_lists.
        If data and create_using are both multigraphs then create
        a multigraph from a multigraph.

    """
    if create_using is None:
        create_using = nx.Graph

    if isinstance(data, nx.Graph):
        return nx.convert_to_type(data, create_using)

    G = create_using()
    G.clear()

    if isinstance(data, dict):
        if multigraph_input and all(isinstance(v, dict) for v in data.values()):
            return from_dict_of_dicts(data, create_using=create_using, multigraph_input=True)
        elif all(isinstance(v, dict) for v in data.values()):
            return from_dict_of_dicts(data, create_using=create_using)
        elif all(isinstance(v, (list, set)) for v in data.values()):
            return from_dict_of_lists(data, create_using=create_using)

    if isinstance(data, (list, set, tuple)) or isinstance(data, Iterator) or isinstance(data, Generator):
        return from_edgelist(data, create_using=create_using)

    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return nx.from_pandas_edgelist(data, create_using=create_using)
    except ImportError:
        pass

    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            return nx.from_numpy_array(data, create_using=create_using)
    except ImportError:
        pass

    try:
        from scipy import sparse
        if sparse.issparse(data):
            return nx.from_scipy_sparse_array(data, create_using=create_using)
    except ImportError:
        pass

    try:
        import pygraphviz
        if isinstance(data, pygraphviz.AGraph):
            return nx.from_pygraphviz(data, create_using=create_using)
    except ImportError:
        pass

    raise nx.NetworkXError("Data type not recognized")


@nx._dispatchable
def to_dict_of_lists(G, nodelist=None):
    """Returns adjacency representation of graph as a dictionary of lists.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list
       Use only nodes specified in nodelist

    Notes
    -----
    Completely ignores edge data for MultiGraph and MultiDiGraph.

    """
    if nodelist is None:
        nodelist = G.nodes()

    d = {}
    for n in nodelist:
        d[n] = [nbr for nbr in G.neighbors(n) if nbr in nodelist]
    return d


@nx._dispatchable(graphs=None, returns_graph=True)
def from_dict_of_lists(d, create_using=None):
    """Returns a graph from a dictionary of lists.

    Parameters
    ----------
    d : dictionary of lists
      A dictionary of lists adjacency representation.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    Examples
    --------
    >>> dol = {0: [1]}  # single edge (0,1)
    >>> G = nx.from_dict_of_lists(dol)

    or

    >>> G = nx.Graph(dol)  # use Graph constructor

    """
    G = nx.empty_graph(0, create_using)
    G.add_nodes_from(d)
    if G.is_multigraph():
        G.add_edges_from((n, nbr, k)
            for n, nbrlist in d.items()
            for nbr in nbrlist
            for k in range(G.number_of_edges(n, nbr), len([n, nbr])))
    else:
        G.add_edges_from((n, nbr)
            for n, nbrlist in d.items()
            for nbr in nbrlist)
    return G


def to_dict_of_dicts(G, nodelist=None, edge_data=None):
    """Returns adjacency representation of graph as a dictionary of dictionaries.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list
       Use only nodes specified in nodelist

    edge_data : scalar, optional
       If provided, the value of the dictionary will be set to `edge_data` for
       all edges. Usual values could be `1` or `True`. If `edge_data` is
       `None` (the default), the edgedata in `G` is used, resulting in a
       dict-of-dict-of-dicts. If `G` is a MultiGraph, the result will be a
       dict-of-dict-of-dict-of-dicts. See Notes for an approach to customize
       handling edge data. `edge_data` should *not* be a container.

    Returns
    -------
    dod : dict
       A nested dictionary representation of `G`. Note that the level of
       nesting depends on the type of `G` and the value of `edge_data`
       (see Examples).

    See Also
    --------
    from_dict_of_dicts, to_dict_of_lists

    Notes
    -----
    For a more custom approach to handling edge data, try::

        dod = {
            n: {nbr: custom(n, nbr, dd) for nbr, dd in nbrdict.items()}
            for n, nbrdict in G.adj.items()
        }

    where `custom` returns the desired edge data for each edge between `n` and
    `nbr`, given existing edge data `dd`.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> nx.to_dict_of_dicts(G)
    {0: {1: {}}, 1: {0: {}, 2: {}}, 2: {1: {}}}

    Edge data is preserved by default (``edge_data=None``), resulting
    in dict-of-dict-of-dicts where the innermost dictionary contains the
    edge data:

    >>> G = nx.Graph()
    >>> G.add_edges_from(
    ...     [
    ...         (0, 1, {"weight": 1.0}),
    ...         (1, 2, {"weight": 2.0}),
    ...         (2, 0, {"weight": 1.0}),
    ...     ]
    ... )
    >>> d = nx.to_dict_of_dicts(G)
    >>> d  # doctest: +SKIP
    {0: {1: {'weight': 1.0}, 2: {'weight': 1.0}},
     1: {0: {'weight': 1.0}, 2: {'weight': 2.0}},
     2: {1: {'weight': 2.0}, 0: {'weight': 1.0}}}
    >>> d[1][2]["weight"]
    2.0

    If `edge_data` is not `None`, edge data in the original graph (if any) is
    replaced:

    >>> d = nx.to_dict_of_dicts(G, edge_data=1)
    >>> d
    {0: {1: 1, 2: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 0: 1}}
    >>> d[1][2]
    1

    This also applies to MultiGraphs: edge data is preserved by default:

    >>> G = nx.MultiGraph()
    >>> G.add_edge(0, 1, key="a", weight=1.0)
    'a'
    >>> G.add_edge(0, 1, key="b", weight=5.0)
    'b'
    >>> d = nx.to_dict_of_dicts(G)
    >>> d  # doctest: +SKIP
    {0: {1: {'a': {'weight': 1.0}, 'b': {'weight': 5.0}}},
     1: {0: {'a': {'weight': 1.0}, 'b': {'weight': 5.0}}}}
    >>> d[0][1]["b"]["weight"]
    5.0

    But multi edge data is lost if `edge_data` is not `None`:

    >>> d = nx.to_dict_of_dicts(G, edge_data=10)
    >>> d
    {0: {1: 10}, 1: {0: 10}}
    """
    dod = {}
    if nodelist is None:
        nodelist = G.nodes()
    
    if G.is_multigraph():
        for n in nodelist:
            dod[n] = {}
            for nbr, keydict in G.adj[n].items():
                if nbr in nodelist:
                    dod[n][nbr] = {}
                    for key, data in keydict.items():
                        dod[n][nbr][key] = data if edge_data is None else edge_data
    else:
        for n in nodelist:
            dod[n] = {}
            for nbr, data in G.adj[n].items():
                if nbr in nodelist:
                    dod[n][nbr] = data if edge_data is None else edge_data
    return dod


@nx._dispatchable(graphs=None, returns_graph=True)
def from_dict_of_dicts(d, create_using=None, multigraph_input=False):
    """Returns a graph from a dictionary of dictionaries.

    Parameters
    ----------
    d : dictionary of dictionaries
      A dictionary of dictionaries adjacency representation.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    multigraph_input : bool (default False)
       When True, the dict `d` is assumed
       to be a dict-of-dict-of-dict-of-dict structure keyed by
       node to neighbor to edge keys to edge data for multi-edges.
       Otherwise this routine assumes dict-of-dict-of-dict keyed by
       node to neighbor to edge data.

    Examples
    --------
    >>> dod = {0: {1: {"weight": 1}}}  # single edge (0,1)
    >>> G = nx.from_dict_of_dicts(dod)

    or

    >>> G = nx.Graph(dod)  # use Graph constructor

    """
    G = nx.empty_graph(0, create_using)
    G.add_nodes_from(d)
    
    if multigraph_input:
        if G.is_multigraph():
            G.add_edges_from((u, v, key, data)
                for u, nbrs in d.items()
                for v, keydict in nbrs.items()
                for key, data in keydict.items())
        else:
            G.add_edges_from((u, v, data)
                for u, nbrs in d.items()
                for v, keydict in nbrs.items()
                for data in keydict.values())
    else:
        G.add_edges_from((u, v, data)
            for u, nbrs in d.items()
            for v, data in nbrs.items())
    return G


@nx._dispatchable(preserve_edge_attrs=True)
def to_edgelist(G, nodelist=None):
    """Returns a list of edges in the graph.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list
       Use only nodes specified in nodelist

    """
    if nodelist is None:
        return G.edges(data=True)
    return ((u, v, d) for u, v, d in G.edges(data=True) if u in nodelist and v in nodelist)


@nx._dispatchable(graphs=None, returns_graph=True)
def from_edgelist(edgelist, create_using=None):
    """Returns a graph from a list of edges.

    Parameters
    ----------
    edgelist : list or iterator
      Edge tuples

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    Examples
    --------
    >>> edgelist = [(0, 1)]  # single edge (0,1)
    >>> G = nx.from_edgelist(edgelist)

    or

    >>> G = nx.Graph(edgelist)  # use Graph constructor

    """
    G = nx.empty_graph(0, create_using)
    G.add_edges_from(edgelist)
    return G
