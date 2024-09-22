import networkx as nx
__all__ = ['adjacency_data', 'adjacency_graph']
_attrs = {'id': 'id', 'key': 'key'}


def adjacency_data(G, attrs=_attrs):
    """Returns data in adjacency format that is suitable for JSON serialization
    and use in JavaScript documents.

    Parameters
    ----------
    G : NetworkX graph

    attrs : dict
        A dictionary that contains two keys 'id' and 'key'. The corresponding
        values provide the attribute names for storing NetworkX-internal graph
        data. The values should be unique. Default value:
        :samp:`dict(id='id', key='key')`.

        If some user-defined graph data use these attribute names as data keys,
        they may be silently dropped.

    Returns
    -------
    data : dict
       A dictionary with adjacency formatted data.

    Raises
    ------
    NetworkXError
        If values in attrs are not unique.

    Examples
    --------
    >>> from networkx.readwrite import json_graph
    >>> G = nx.Graph([(1, 2)])
    >>> data = json_graph.adjacency_data(G)

    To serialize with json

    >>> import json
    >>> s = json.dumps(data)

    Notes
    -----
    Graph, node, and link attributes will be written when using this format
    but attribute keys must be strings if you want to serialize the resulting
    data with JSON.

    The default value of attrs will be changed in a future release of NetworkX.

    See Also
    --------
    adjacency_graph, node_link_data, tree_data
    """
    pass


@nx._dispatchable(graphs=None, returns_graph=True)
def adjacency_graph(data, directed=False, multigraph=True, attrs=_attrs):
    """Returns graph from adjacency data format.

    Parameters
    ----------
    data : dict
        Adjacency list formatted graph data

    directed : bool
        If True, and direction not specified in data, return a directed graph.

    multigraph : bool
        If True, and multigraph not specified in data, return a multigraph.

    attrs : dict
        A dictionary that contains two keys 'id' and 'key'. The corresponding
        values provide the attribute names for storing NetworkX-internal graph
        data. The values should be unique. Default value:
        :samp:`dict(id='id', key='key')`.

    Returns
    -------
    G : NetworkX graph
       A NetworkX graph object

    Examples
    --------
    >>> from networkx.readwrite import json_graph
    >>> G = nx.Graph([(1, 2)])
    >>> data = json_graph.adjacency_data(G)
    >>> H = json_graph.adjacency_graph(data)

    Notes
    -----
    The default value of attrs will be changed in a future release of NetworkX.

    See Also
    --------
    adjacency_graph, node_link_data, tree_data
    """
    pass
