"""
Graph products.
"""
from itertools import product
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['tensor_product', 'cartesian_product', 'lexicographic_product',
    'strong_product', 'power', 'rooted_product', 'corona_product',
    'modular_product']
_G_H = {'G': 0, 'H': 1}


@nx._dispatchable(graphs=_G_H, preserve_node_attrs=True, returns_graph=True)
def tensor_product(G, H):
    """Returns the tensor product of G and H.

    The tensor product $P$ of the graphs $G$ and $H$ has a node set that
    is the Cartesian product of the node sets, $V(P)=V(G) \\times V(H)$.
    $P$ has an edge $((u,v), (x,y))$ if and only if $(u,x)$ is an edge in $G$
    and $(v,y)$ is an edge in $H$.

    Tensor product is sometimes also referred to as the categorical product,
    direct product, cardinal product or conjunction.


    Parameters
    ----------
    G, H: graphs
     Networkx graphs.

    Returns
    -------
    P: NetworkX graph
     The tensor product of G and H. P will be a multi-graph if either G
     or H is a multi-graph, will be a directed if G and H are directed,
     and undirected if G and H are undirected.

    Raises
    ------
    NetworkXError
     If G and H are not both directed or both undirected.

    Notes
    -----
    Node attributes in P are two-tuple of the G and H node attributes.
    Missing attributes are assigned None.

    Examples
    --------
    >>> G = nx.Graph()
    >>> H = nx.Graph()
    >>> G.add_node(0, a1=True)
    >>> H.add_node("a", a2="Spam")
    >>> P = nx.tensor_product(G, H)
    >>> list(P)
    [(0, 'a')]

    Edge attributes and edge keys (for multigraphs) are also copied to the
    new product graph
    """
    if G.is_directed() != H.is_directed():
        raise nx.NetworkXError("G and H must be both directed or both undirected.")
    
    GH = nx.Graph()
    if G.is_multigraph() or H.is_multigraph():
        GH = nx.MultiGraph()
    if G.is_directed():
        GH = nx.DiGraph()
        if G.is_multigraph() or H.is_multigraph():
            GH = nx.MultiDiGraph()

    GH.add_nodes_from((n1, n2) for n1 in G for n2 in H)

    for e1 in G.edges(data=True):
        for e2 in H.edges(data=True):
            GH.add_edge((e1[0], e2[0]), (e1[1], e2[1]), **{**e1[2], **e2[2]})

    return GH


@nx._dispatchable(graphs=_G_H, preserve_node_attrs=True, returns_graph=True)
def cartesian_product(G, H):
    """Returns the Cartesian product of G and H.

    The Cartesian product $P$ of the graphs $G$ and $H$ has a node set that
    is the Cartesian product of the node sets, $V(P)=V(G) \\times V(H)$.
    $P$ has an edge $((u,v),(x,y))$ if and only if either $u$ is equal to $x$
    and both $v$ and $y$ are adjacent in $H$ or if $v$ is equal to $y$ and
    both $u$ and $x$ are adjacent in $G$.

    Parameters
    ----------
    G, H: graphs
     Networkx graphs.

    Returns
    -------
    P: NetworkX graph
     The Cartesian product of G and H. P will be a multi-graph if either G
     or H is a multi-graph. Will be a directed if G and H are directed,
     and undirected if G and H are undirected.

    Raises
    ------
    NetworkXError
     If G and H are not both directed or both undirected.

    Notes
    -----
    Node attributes in P are two-tuple of the G and H node attributes.
    Missing attributes are assigned None.

    Examples
    --------
    >>> G = nx.Graph()
    >>> H = nx.Graph()
    >>> G.add_node(0, a1=True)
    >>> H.add_node("a", a2="Spam")
    >>> P = nx.cartesian_product(G, H)
    >>> list(P)
    [(0, 'a')]

    Edge attributes and edge keys (for multigraphs) are also copied to the
    new product graph
    """
    if G.is_directed() != H.is_directed():
        raise nx.NetworkXError("G and H must be both directed or both undirected.")
    
    GH = nx.Graph()
    if G.is_multigraph() or H.is_multigraph():
        GH = nx.MultiGraph()
    if G.is_directed():
        GH = nx.DiGraph()
        if G.is_multigraph() or H.is_multigraph():
            GH = nx.MultiDiGraph()

    GH.add_nodes_from((n1, n2) for n1 in G for n2 in H)

    for n1 in G:
        for e2 in H.edges(data=True):
            GH.add_edge((n1, e2[0]), (n1, e2[1]), **e2[2])

    for e1 in G.edges(data=True):
        for n2 in H:
            GH.add_edge((e1[0], n2), (e1[1], n2), **e1[2])

    return GH


@nx._dispatchable(graphs=_G_H, preserve_node_attrs=True, returns_graph=True)
def lexicographic_product(G, H):
    """Returns the lexicographic product of G and H.

    The lexicographical product $P$ of the graphs $G$ and $H$ has a node set
    that is the Cartesian product of the node sets, $V(P)=V(G) \\times V(H)$.
    $P$ has an edge $((u,v), (x,y))$ if and only if $(u,v)$ is an edge in $G$
    or $u==v$ and $(x,y)$ is an edge in $H$.

    Parameters
    ----------
    G, H: graphs
     Networkx graphs.

    Returns
    -------
    P: NetworkX graph
     The Cartesian product of G and H. P will be a multi-graph if either G
     or H is a multi-graph. Will be a directed if G and H are directed,
     and undirected if G and H are undirected.

    Raises
    ------
    NetworkXError
     If G and H are not both directed or both undirected.

    Notes
    -----
    Node attributes in P are two-tuple of the G and H node attributes.
    Missing attributes are assigned None.

    Examples
    --------
    >>> G = nx.Graph()
    >>> H = nx.Graph()
    >>> G.add_node(0, a1=True)
    >>> H.add_node("a", a2="Spam")
    >>> P = nx.lexicographic_product(G, H)
    >>> list(P)
    [(0, 'a')]

    Edge attributes and edge keys (for multigraphs) are also copied to the
    new product graph
    """
    if G.is_directed() != H.is_directed():
        raise nx.NetworkXError("G and H must be both directed or both undirected.")
    
    GH = nx.Graph()
    if G.is_multigraph() or H.is_multigraph():
        GH = nx.MultiGraph()
    if G.is_directed():
        GH = nx.DiGraph()
        if G.is_multigraph() or H.is_multigraph():
            GH = nx.MultiDiGraph()

    GH.add_nodes_from((n1, n2) for n1 in G for n2 in H)

    for e1 in G.edges(data=True):
        for n2 in H:
            for n2_prime in H:
                GH.add_edge((e1[0], n2), (e1[1], n2_prime), **e1[2])

    for n1 in G:
        for e2 in H.edges(data=True):
            GH.add_edge((n1, e2[0]), (n1, e2[1]), **e2[2])

    return GH


@nx._dispatchable(graphs=_G_H, preserve_node_attrs=True, returns_graph=True)
def strong_product(G, H):
    """Returns the strong product of G and H.

    The strong product $P$ of the graphs $G$ and $H$ has a node set that
    is the Cartesian product of the node sets, $V(P)=V(G) \\times V(H)$.
    $P$ has an edge $((u,v), (x,y))$ if and only if
    $u==v$ and $(x,y)$ is an edge in $H$, or
    $x==y$ and $(u,v)$ is an edge in $G$, or
    $(u,v)$ is an edge in $G$ and $(x,y)$ is an edge in $H$.

    Parameters
    ----------
    G, H: graphs
     Networkx graphs.

    Returns
    -------
    P: NetworkX graph
     The Cartesian product of G and H. P will be a multi-graph if either G
     or H is a multi-graph. Will be a directed if G and H are directed,
     and undirected if G and H are undirected.

    Raises
    ------
    NetworkXError
     If G and H are not both directed or both undirected.

    Notes
    -----
    Node attributes in P are two-tuple of the G and H node attributes.
    Missing attributes are assigned None.

    Examples
    --------
    >>> G = nx.Graph()
    >>> H = nx.Graph()
    >>> G.add_node(0, a1=True)
    >>> H.add_node("a", a2="Spam")
    >>> P = nx.strong_product(G, H)
    >>> list(P)
    [(0, 'a')]

    Edge attributes and edge keys (for multigraphs) are also copied to the
    new product graph
    """
    if G.is_directed() != H.is_directed():
        raise nx.NetworkXError("G and H must be both directed or both undirected.")
    
    GH = nx.Graph()
    if G.is_multigraph() or H.is_multigraph():
        GH = nx.MultiGraph()
    if G.is_directed():
        GH = nx.DiGraph()
        if G.is_multigraph() or H.is_multigraph():
            GH = nx.MultiDiGraph()

    GH.add_nodes_from((n1, n2) for n1 in G for n2 in H)

    # Edges from G
    for e1 in G.edges(data=True):
        for n2 in H:
            GH.add_edge((e1[0], n2), (e1[1], n2), **e1[2])

    # Edges from H
    for n1 in G:
        for e2 in H.edges(data=True):
            GH.add_edge((n1, e2[0]), (n1, e2[1]), **e2[2])

    # Diagonal edges
    for e1 in G.edges(data=True):
        for e2 in H.edges(data=True):
            GH.add_edge((e1[0], e2[0]), (e1[1], e2[1]), **{**e1[2], **e2[2]})

    return GH


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable(returns_graph=True)
def power(G, k):
    """Returns the specified power of a graph.

    The $k$th power of a simple graph $G$, denoted $G^k$, is a
    graph on the same set of nodes in which two distinct nodes $u$ and
    $v$ are adjacent in $G^k$ if and only if the shortest path
    distance between $u$ and $v$ in $G$ is at most $k$.

    Parameters
    ----------
    G : graph
        A NetworkX simple graph object.

    k : positive integer
        The power to which to raise the graph `G`.

    Returns
    -------
    NetworkX simple graph
        `G` to the power `k`.

    Raises
    ------
    ValueError
        If the exponent `k` is not positive.

    NetworkXNotImplemented
        If `G` is not a simple graph.

    Examples
    --------
    The number of edges will never decrease when taking successive
    powers:

    >>> G = nx.path_graph(4)
    >>> list(nx.power(G, 2).edges)
    [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    >>> list(nx.power(G, 3).edges)
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    The `k` th power of a cycle graph on *n* nodes is the complete graph
    on *n* nodes, if `k` is at least ``n // 2``:

    >>> G = nx.cycle_graph(5)
    >>> H = nx.complete_graph(5)
    >>> nx.is_isomorphic(nx.power(G, 2), H)
    True
    >>> G = nx.cycle_graph(8)
    >>> H = nx.complete_graph(8)
    >>> nx.is_isomorphic(nx.power(G, 4), H)
    True

    References
    ----------
    .. [1] J. A. Bondy, U. S. R. Murty, *Graph Theory*. Springer, 2008.

    Notes
    -----
    This definition of "power graph" comes from Exercise 3.1.6 of
    *Graph Theory* by Bondy and Murty [1]_.

    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    
    if not G.is_directed() and not G.is_multigraph():
        H = G.copy()
        
        for _ in range(k - 1):
            edges_to_add = []
            for node in H:
                neighbors = set(H.neighbors(node))
                for neighbor in list(neighbors):
                    neighbors.update(H.neighbors(neighbor))
                neighbors.discard(node)
                edges_to_add.extend((node, v) for v in neighbors)
            
            H.add_edges_from(edges_to_add)
        
        return H
    else:
        raise nx.NetworkXNotImplemented("Graph must be undirected and simple.")


@not_implemented_for('multigraph')
@nx._dispatchable(graphs=_G_H, returns_graph=True)
def rooted_product(G, H, root):
    """Return the rooted product of graphs G and H rooted at root in H.

    A new graph is constructed representing the rooted product of
    the inputted graphs, G and H, with a root in H.
    A rooted product duplicates H for each nodes in G with the root
    of H corresponding to the node in G. Nodes are renamed as the direct
    product of G and H. The result is a subgraph of the cartesian product.

    Parameters
    ----------
    G,H : graph
       A NetworkX graph
    root : node
       A node in H

    Returns
    -------
    R : The rooted product of G and H with a specified root in H

    Notes
    -----
    The nodes of R are the Cartesian Product of the nodes of G and H.
    The nodes of G and H are not relabeled.
    """
    if root not in H:
        raise nx.NetworkXError("root is not a node of H")

    R = nx.Graph()
    
    if G.is_directed() and H.is_directed():
        R = nx.DiGraph()
    elif G.is_multigraph() or H.is_multigraph():
        R = nx.MultiGraph()
        if G.is_directed() and H.is_directed():
            R = nx.MultiDiGraph()

    # Add nodes
    R.add_nodes_from((g, h) for g in G for h in H)

    # Add edges within each copy of H
    for g in G:
        for e in H.edges(data=True):
            R.add_edge((g, e[0]), (g, e[1]), **e[2])

    # Add edges between copies of H
    for e in G.edges(data=True):
        R.add_edge((e[0], root), (e[1], root), **e[2])

    return R


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable(graphs=_G_H, returns_graph=True)
def corona_product(G, H):
    """Returns the Corona product of G and H.

    The corona product of $G$ and $H$ is the graph $C = G \\circ H$ obtained by
    taking one copy of $G$, called the center graph, $|V(G)|$ copies of $H$,
    called the outer graph, and making the $i$-th vertex of $G$ adjacent to
    every vertex of the $i$-th copy of $H$, where $1 ≤ i ≤ |V(G)|$.

    Parameters
    ----------
    G, H: NetworkX graphs
        The graphs to take the carona product of.
        `G` is the center graph and `H` is the outer graph

    Returns
    -------
    C: NetworkX graph
        The Corona product of G and H.

    Raises
    ------
    NetworkXError
        If G and H are not both directed or both undirected.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> H = nx.path_graph(2)
    >>> C = nx.corona_product(G, H)
    >>> list(C)
    [0, 1, 2, 3, (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    >>> print(C)
    Graph with 12 nodes and 16 edges

    References
    ----------
    [1] M. Tavakoli, F. Rahbarnia, and A. R. Ashrafi,
        "Studying the corona product of graphs under some graph invariants,"
        Transactions on Combinatorics, vol. 3, no. 3, pp. 43–49, Sep. 2014,
        doi: 10.22108/toc.2014.5542.
    [2] A. Faraji, "Corona Product in Graph Theory," Ali Faraji, May 11, 2021.
        https://blog.alifaraji.ir/math/graph-theory/corona-product.html (accessed Dec. 07, 2021).
    """
    if G.is_directed() != H.is_directed():
        raise nx.NetworkXError("G and H must be both directed or both undirected.")

    C = G.copy()

    for v in G:
        # Add a copy of H for each node in G
        C.add_nodes_from((v, w) for w in H)
        C.add_edges_from(((v, w1), (v, w2)) for w1, w2 in H.edges())

        # Connect v to every node in its copy of H
        C.add_edges_from((v, (v, w)) for w in H)

    return C


@nx._dispatchable(graphs=_G_H, preserve_edge_attrs=True,
    preserve_node_attrs=True, returns_graph=True)
def modular_product(G, H):
    """Returns the Modular product of G and H.

    The modular product of `G` and `H` is the graph $M = G \\nabla H$,
    consisting of the node set $V(M) = V(G) \\times V(H)$ that is the Cartesian
    product of the node sets of `G` and `H`. Further, M contains an edge ((u, v), (x, y)):

    - if u is adjacent to x in `G` and v is adjacent to y in `H`, or
    - if u is not adjacent to x in `G` and v is not adjacent to y in `H`.

    More formally::

        E(M) = {((u, v), (x, y)) | ((u, x) in E(G) and (v, y) in E(H)) or
                                   ((u, x) not in E(G) and (v, y) not in E(H))}

    Parameters
    ----------
    G, H: NetworkX graphs
        The graphs to take the modular product of.

    Returns
    -------
    M: NetworkX graph
        The Modular product of `G` and `H`.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is not a simple graph.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> H = nx.path_graph(2)
    >>> M = nx.modular_product(G, H)
    >>> list(M)
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    >>> print(M)
    Graph with 8 nodes and 8 edges

    Notes
    -----
    The *modular product* is defined in [1]_ and was first
    introduced as the *weak modular product*.

    The modular product reduces the problem of counting isomorphic subgraphs
    in `G` and `H` to the problem of counting cliques in M. The subgraphs of
    `G` and `H` that are induced by the nodes of a clique in M are
    isomorphic [2]_ [3]_.

    References
    ----------
    .. [1] R. Hammack, W. Imrich, and S. Klavžar,
        "Handbook of Product Graphs", CRC Press, 2011.

    .. [2] H. G. Barrow and R. M. Burstall,
        "Subgraph isomorphism, matching relational structures and maximal
        cliques", Information Processing Letters, vol. 4, issue 4, pp. 83-84,
        1976, https://doi.org/10.1016/0020-0190(76)90049-1.

    .. [3] V. G. Vizing, "Reduction of the problem of isomorphism and isomorphic
        entrance to the task of finding the nondensity of a graph." Proc. Third
        All-Union Conference on Problems of Theoretical Cybernetics. 1974.
    """
    if G.is_multigraph() or H.is_multigraph():
        raise nx.NetworkXNotImplemented("G and H must be simple graphs.")

    M = nx.Graph()

    M.add_nodes_from((g, h) for g in G for h in H)

    for (u, v) in M.nodes():
        for (x, y) in M.nodes():
            if (u, v) != (x, y):
                if ((u == x and v != y) or (u != x and v == y)):
                    continue
                if ((G.has_edge(u, x) and H.has_edge(v, y)) or
                    (not G.has_edge(u, x) and not H.has_edge(v, y))):
                    M.add_edge((u, v), (x, y))

    return M
