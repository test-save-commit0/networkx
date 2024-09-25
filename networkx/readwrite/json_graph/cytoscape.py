import networkx as nx
__all__ = ['cytoscape_data', 'cytoscape_graph']


def cytoscape_data(G, name='name', ident='id'):
    """Returns data in Cytoscape JSON format (cyjs).

    Parameters
    ----------
    G : NetworkX Graph
        The graph to convert to cytoscape format
    name : string
        A string which is mapped to the 'name' node element in cyjs format.
        Must not have the same value as `ident`.
    ident : string
        A string which is mapped to the 'id' node element in cyjs format.
        Must not have the same value as `name`.

    Returns
    -------
    data: dict
        A dictionary with cyjs formatted data.

    Raises
    ------
    NetworkXError
        If the values for `name` and `ident` are identical.

    See Also
    --------
    cytoscape_graph: convert a dictionary in cyjs format to a graph

    References
    ----------
    .. [1] Cytoscape user's manual:
       http://manual.cytoscape.org/en/stable/index.html

    Examples
    --------
    >>> G = nx.path_graph(2)
    >>> nx.cytoscape_data(G)  # doctest: +SKIP
    {'data': [],
     'directed': False,
     'multigraph': False,
     'elements': {'nodes': [{'data': {'id': '0', 'value': 0, 'name': '0'}},
       {'data': {'id': '1', 'value': 1, 'name': '1'}}],
      'edges': [{'data': {'source': 0, 'target': 1}}]}}
    """
    if name == ident:
        raise nx.NetworkXError("name and ident must be different")

    data = {
        "data": [],
        "directed": G.is_directed(),
        "multigraph": G.is_multigraph(),
        "elements": {"nodes": [], "edges": []}
    }

    for node, node_data in G.nodes(data=True):
        node_dict = {"data": {ident: str(node), name: str(node)}}
        node_dict["data"].update((k, v) for k, v in node_data.items() if k != name and k != ident)
        data["elements"]["nodes"].append(node_dict)

    for u, v, edge_data in G.edges(data=True):
        edge_dict = {"data": {"source": str(u), "target": str(v)}}
        edge_dict["data"].update(edge_data)
        data["elements"]["edges"].append(edge_dict)

    return data


@nx._dispatchable(graphs=None, returns_graph=True)
def cytoscape_graph(data, name='name', ident='id'):
    """
    Create a NetworkX graph from a dictionary in cytoscape JSON format.

    Parameters
    ----------
    data : dict
        A dictionary of data conforming to cytoscape JSON format.
    name : string
        A string which is mapped to the 'name' node element in cyjs format.
        Must not have the same value as `ident`.
    ident : string
        A string which is mapped to the 'id' node element in cyjs format.
        Must not have the same value as `name`.

    Returns
    -------
    graph : a NetworkX graph instance
        The `graph` can be an instance of `Graph`, `DiGraph`, `MultiGraph`, or
        `MultiDiGraph` depending on the input data.

    Raises
    ------
    NetworkXError
        If the `name` and `ident` attributes are identical.

    See Also
    --------
    cytoscape_data: convert a NetworkX graph to a dict in cyjs format

    References
    ----------
    .. [1] Cytoscape user's manual:
       http://manual.cytoscape.org/en/stable/index.html

    Examples
    --------
    >>> data_dict = {
    ...     "data": [],
    ...     "directed": False,
    ...     "multigraph": False,
    ...     "elements": {
    ...         "nodes": [
    ...             {"data": {"id": "0", "value": 0, "name": "0"}},
    ...             {"data": {"id": "1", "value": 1, "name": "1"}},
    ...         ],
    ...         "edges": [{"data": {"source": 0, "target": 1}}],
    ...     },
    ... }
    >>> G = nx.cytoscape_graph(data_dict)
    >>> G.name
    ''
    >>> G.nodes()
    NodeView((0, 1))
    >>> G.nodes(data=True)[0]
    {'id': '0', 'value': 0, 'name': '0'}
    >>> G.edges(data=True)
    EdgeDataView([(0, 1, {'source': 0, 'target': 1})])
    """
    if name == ident:
        raise nx.NetworkXError("name and ident must be different")

    if data.get("directed", False):
        graph = nx.DiGraph() if not data.get("multigraph", False) else nx.MultiDiGraph()
    else:
        graph = nx.Graph() if not data.get("multigraph", False) else nx.MultiGraph()

    graph.graph = data.get("data", {})

    for node_data in data["elements"]["nodes"]:
        node_attrs = node_data["data"].copy()
        node_id = node_attrs.pop(ident)
        graph.add_node(node_id, **node_attrs)

    for edge_data in data["elements"]["edges"]:
        edge_attrs = edge_data["data"].copy()
        source = edge_attrs.pop("source")
        target = edge_attrs.pop("target")
        graph.add_edge(source, target, **edge_attrs)

    return graph
