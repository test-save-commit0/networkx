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
    if len(set(attrs.values())) < len(attrs):
        raise nx.NetworkXError("Attribute names are not unique.")

    data = {"directed": G.is_directed(), "multigraph": G.is_multigraph(), "graph": G.graph}
    data["nodes"] = []
    data["adjacency"] = []

    for n, nbrs in G.adjacency():
        node_data = {attrs['id']: n}
        node_data.update(G.nodes[n])
        data["nodes"].append(node_data)

        adj_data = []
        for nbr, edge_data in nbrs.items():
            adj = {attrs['id']: nbr}
            if G.is_multigraph():
                for key, edata in edge_data.items():
                    link = {attrs['key']: key}
                    link.update(edata)
                    adj[attrs['key']] = link
            else:
                adj.update(edge_data)
            adj_data.append(adj)
        data["adjacency"].append(adj_data)

    return data


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
    multigraph = data.get('multigraph', multigraph)
    directed = data.get('directed', directed)
    if multigraph:
        graph = nx.MultiGraph()
    else:
        graph = nx.Graph()
    if directed:
        graph = graph.to_directed()

    graph.graph = data.get('graph', {})
    nodes = data['nodes'] if 'nodes' in data else []
    adjacency = data['adjacency'] if 'adjacency' in data else []

    for node_data, adj_data in zip(nodes, adjacency):
        node = node_data[attrs['id']]
        graph.add_node(node, **{k: v for k, v in node_data.items() if k != attrs['id']})
        for edge in adj_data:
            target = edge[attrs['id']]
            edge_data = {k: v for k, v in edge.items() if k != attrs['id']}
            if multigraph:
                key = edge_data.pop(attrs['key'], None)
                graph.add_edge(node, target, key=key, **edge_data)
            else:
                graph.add_edge(node, target, **edge_data)

    return graph
