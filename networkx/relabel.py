import networkx as nx
__all__ = ['convert_node_labels_to_integers', 'relabel_nodes']


@nx._dispatchable(preserve_all_attrs=True, mutates_input={'not copy': 2},
    returns_graph=True)
def relabel_nodes(G, mapping, copy=True):
    """Relabel the nodes of the graph G according to a given mapping.

    The original node ordering may not be preserved if `copy` is `False` and the
    mapping includes overlap between old and new labels.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    mapping : dictionary
       A dictionary with the old labels as keys and new labels as values.
       A partial mapping is allowed. Mapping 2 nodes to a single node is allowed.
       Any non-node keys in the mapping are ignored.

    copy : bool (optional, default=True)
       If True return a copy, or if False relabel the nodes in place.

    Examples
    --------
    To create a new graph with nodes relabeled according to a given
    dictionary:

    >>> G = nx.path_graph(3)
    >>> sorted(G)
    [0, 1, 2]
    >>> mapping = {0: "a", 1: "b", 2: "c"}
    >>> H = nx.relabel_nodes(G, mapping)
    >>> sorted(H)
    ['a', 'b', 'c']

    Nodes can be relabeled with any hashable object, including numbers
    and strings:

    >>> import string
    >>> G = nx.path_graph(26)  # nodes are integers 0 through 25
    >>> sorted(G)[:3]
    [0, 1, 2]
    >>> mapping = dict(zip(G, string.ascii_lowercase))
    >>> G = nx.relabel_nodes(G, mapping)  # nodes are characters a through z
    >>> sorted(G)[:3]
    ['a', 'b', 'c']
    >>> mapping = dict(zip(G, range(1, 27)))
    >>> G = nx.relabel_nodes(G, mapping)  # nodes are integers 1 through 26
    >>> sorted(G)[:3]
    [1, 2, 3]

    To perform a partial in-place relabeling, provide a dictionary
    mapping only a subset of the nodes, and set the `copy` keyword
    argument to False:

    >>> G = nx.path_graph(3)  # nodes 0-1-2
    >>> mapping = {0: "a", 1: "b"}  # 0->'a' and 1->'b'
    >>> G = nx.relabel_nodes(G, mapping, copy=False)
    >>> sorted(G, key=str)
    [2, 'a', 'b']

    A mapping can also be given as a function:

    >>> G = nx.path_graph(3)
    >>> H = nx.relabel_nodes(G, lambda x: x**2)
    >>> list(H)
    [0, 1, 4]

    In a multigraph, relabeling two or more nodes to the same new node
    will retain all edges, but may change the edge keys in the process:

    >>> G = nx.MultiGraph()
    >>> G.add_edge(0, 1, value="a")  # returns the key for this edge
    0
    >>> G.add_edge(0, 2, value="b")
    0
    >>> G.add_edge(0, 3, value="c")
    0
    >>> mapping = {1: 4, 2: 4, 3: 4}
    >>> H = nx.relabel_nodes(G, mapping, copy=True)
    >>> print(H[0])
    {4: {0: {'value': 'a'}, 1: {'value': 'b'}, 2: {'value': 'c'}}}

    This works for in-place relabeling too:

    >>> G = nx.relabel_nodes(G, mapping, copy=False)
    >>> print(G[0])
    {4: {0: {'value': 'a'}, 1: {'value': 'b'}, 2: {'value': 'c'}}}

    Notes
    -----
    Only the nodes specified in the mapping will be relabeled.
    Any non-node keys in the mapping are ignored.

    The keyword setting copy=False modifies the graph in place.
    Relabel_nodes avoids naming collisions by building a
    directed graph from ``mapping`` which specifies the order of
    relabelings. Naming collisions, such as a->b, b->c, are ordered
    such that "b" gets renamed to "c" before "a" gets renamed "b".
    In cases of circular mappings (e.g. a->b, b->a), modifying the
    graph is not possible in-place and an exception is raised.
    In that case, use copy=True.

    If a relabel operation on a multigraph would cause two or more
    edges to have the same source, target and key, the second edge must
    be assigned a new key to retain all edges. The new key is set
    to the lowest non-negative integer not already used as a key
    for edges between these two nodes. Note that this means non-numeric
    keys may be replaced by numeric keys.

    See Also
    --------
    convert_node_labels_to_integers
    """
    if callable(mapping):
        mapping = {n: mapping(n) for n in G}

    if copy:
        H = G.__class__()
        H.add_nodes_from((mapping.get(n, n), d.copy()) for n, d in G.nodes(data=True))
        H.add_edges_from((mapping.get(u, u), mapping.get(v, v), k, d.copy())
                         for u, v, k, d in G.edges(keys=True, data=True))
        return H
    else:
        for old, new in mapping.items():
            if old in G:
                G._adj[new] = G._adj.pop(old)
                for v in G._adj[new]:
                    G._adj[v][new] = G._adj[v].pop(old)
        G._node = {mapping.get(n, n): d for n, d in G._node.items()}
        return G


@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def convert_node_labels_to_integers(G, first_label=0, ordering='default',
    label_attribute=None):
    """Returns a copy of the graph G with the nodes relabeled using
    consecutive integers.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    first_label : int, optional (default=0)
       An integer specifying the starting offset in numbering nodes.
       The new integer labels are numbered first_label, ..., n-1+first_label.

    ordering : string
       "default" : inherit node ordering from G.nodes()
       "sorted"  : inherit node ordering from sorted(G.nodes())
       "increasing degree" : nodes are sorted by increasing degree
       "decreasing degree" : nodes are sorted by decreasing degree

    label_attribute : string, optional (default=None)
       Name of node attribute to store old label.  If None no attribute
       is created.

    Notes
    -----
    Node and edge attribute data are copied to the new (relabeled) graph.

    There is no guarantee that the relabeling of nodes to integers will
    give the same two integers for two (even identical graphs).
    Use the `ordering` argument to try to preserve the order.

    See Also
    --------
    relabel_nodes
    """
    N = G.number_of_nodes() + first_label
    if ordering == "default":
        mapping = dict(zip(G.nodes(), range(first_label, N)))
    elif ordering == "sorted":
        mapping = dict(zip(sorted(G.nodes()), range(first_label, N)))
    elif ordering == "increasing degree":
        mapping = dict(zip(sorted(G.nodes(), key=G.degree), range(first_label, N)))
    elif ordering == "decreasing degree":
        mapping = dict(zip(sorted(G.nodes(), key=G.degree, reverse=True), range(first_label, N)))
    else:
        raise nx.NetworkXError(f"Unknown ordering: {ordering}")

    H = relabel_nodes(G, mapping)

    # Store old labels as attributes if requested
    if label_attribute is not None:
        nx.set_node_attributes(H, {v: k for k, v in mapping.items()}, label_attribute)

    return H
