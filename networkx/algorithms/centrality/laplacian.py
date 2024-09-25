"""
Laplacian centrality measures.
"""
import networkx as nx
__all__ = ['laplacian_centrality']


@nx._dispatchable(edge_attrs='weight')
def laplacian_centrality(G, normalized=True, nodelist=None, weight='weight',
    walk_type=None, alpha=0.95):
    """Compute the Laplacian centrality for nodes in the graph `G`.

    The Laplacian Centrality of a node ``i`` is measured by the drop in the
    Laplacian Energy after deleting node ``i`` from the graph. The Laplacian Energy
    is the sum of the squared eigenvalues of a graph's Laplacian matrix.

    .. math::

        C_L(u_i,G) = \\frac{(\\Delta E)_i}{E_L (G)} = \\frac{E_L (G)-E_L (G_i)}{E_L (G)}

        E_L (G) = \\sum_{i=0}^n \\lambda_i^2

    Where $E_L (G)$ is the Laplacian energy of graph `G`,
    E_L (G_i) is the Laplacian energy of graph `G` after deleting node ``i``
    and $\\lambda_i$ are the eigenvalues of `G`'s Laplacian matrix.
    This formula shows the normalized value. Without normalization,
    the numerator on the right side is returned.

    Parameters
    ----------
    G : graph
        A networkx graph

    normalized : bool (default = True)
        If True the centrality score is scaled so the sum over all nodes is 1.
        If False the centrality score for each node is the drop in Laplacian
        energy when that node is removed.

    nodelist : list, optional (default = None)
        The rows and columns are ordered according to the nodes in nodelist.
        If nodelist is None, then the ordering is produced by G.nodes().

    weight: string or None, optional (default=`weight`)
        Optional parameter `weight` to compute the Laplacian matrix.
        The edge data key used to compute each value in the matrix.
        If None, then each edge has weight 1.

    walk_type : string or None, optional (default=None)
        Optional parameter `walk_type` used when calling
        :func:`directed_laplacian_matrix <networkx.directed_laplacian_matrix>`.
        One of ``"random"``, ``"lazy"``, or ``"pagerank"``. If ``walk_type=None``
        (the default), then a value is selected according to the properties of `G`:
        - ``walk_type="random"`` if `G` is strongly connected and aperiodic
        - ``walk_type="lazy"`` if `G` is strongly connected but not aperiodic
        - ``walk_type="pagerank"`` for all other cases.

    alpha : real (default = 0.95)
        Optional parameter `alpha` used when calling
        :func:`directed_laplacian_matrix <networkx.directed_laplacian_matrix>`.
        (1 - alpha) is the teleportation probability used with pagerank.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with Laplacian centrality as the value.

    Examples
    --------
    >>> G = nx.Graph()
    >>> edges = [(0, 1, 4), (0, 2, 2), (2, 1, 1), (1, 3, 2), (1, 4, 2), (4, 5, 1)]
    >>> G.add_weighted_edges_from(edges)
    >>> sorted((v, f"{c:0.2f}") for v, c in laplacian_centrality(G).items())
    [(0, '0.70'), (1, '0.90'), (2, '0.28'), (3, '0.22'), (4, '0.26'), (5, '0.04')]

    Notes
    -----
    The algorithm is implemented based on [1]_ with an extension to directed graphs
    using the ``directed_laplacian_matrix`` function.

    Raises
    ------
    NetworkXPointlessConcept
        If the graph `G` is the null graph.
    ZeroDivisionError
        If the graph `G` has no edges (is empty) and normalization is requested.

    References
    ----------
    .. [1] Qi, X., Fuller, E., Wu, Q., Wu, Y., and Zhang, C.-Q. (2012).
       Laplacian centrality: A new centrality measure for weighted networks.
       Information Sciences, 194:240-253.
       https://math.wvu.edu/~cqzhang/Publication-files/my-paper/INS-2012-Laplacian-W.pdf

    See Also
    --------
    :func:`~networkx.linalg.laplacianmatrix.directed_laplacian_matrix`
    :func:`~networkx.linalg.laplacianmatrix.laplacian_matrix`
    """
    import numpy as np

    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Cannot compute centrality for the null graph.")

    if nodelist is None:
        nodelist = list(G)

    if G.is_directed():
        L = nx.directed_laplacian_matrix(G, nodelist=nodelist, weight=weight,
                                         walk_type=walk_type, alpha=alpha)
    else:
        L = nx.laplacian_matrix(G, nodelist=nodelist, weight=weight)

    L = L.astype(float)
    eigenvalues = np.linalg.eigvalsh(L.toarray())
    laplacian_energy = np.sum(eigenvalues ** 2)

    if laplacian_energy == 0:
        raise ZeroDivisionError("Graph has no edges, cannot compute Laplacian centrality.")

    centralities = {}
    for node in nodelist:
        G_minus_node = G.copy()
        G_minus_node.remove_node(node)
        
        if G_minus_node.is_directed():
            L_minus_node = nx.directed_laplacian_matrix(G_minus_node, weight=weight,
                                                        walk_type=walk_type, alpha=alpha)
        else:
            L_minus_node = nx.laplacian_matrix(G_minus_node, weight=weight)
        
        L_minus_node = L_minus_node.astype(float)
        eigenvalues_minus_node = np.linalg.eigvalsh(L_minus_node.toarray())
        laplacian_energy_minus_node = np.sum(eigenvalues_minus_node ** 2)
        
        centralities[node] = laplacian_energy - laplacian_energy_minus_node

    if normalized:
        norm = sum(centralities.values())
        centralities = {node: value / norm for node, value in centralities.items()}

    return centralities
