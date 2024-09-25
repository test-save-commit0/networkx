"""Hubs and authorities analysis of graph structure.
"""
import networkx as nx
__all__ = ['hits']


@nx._dispatchable(preserve_edge_attrs={'G': {'weight': 1}})
def hits(G, max_iter=100, tol=1e-08, nstart=None, normalized=True):
    """Returns HITS hubs and authorities values for nodes.

    The HITS algorithm computes two numbers for a node.
    Authorities estimates the node value based on the incoming links.
    Hubs estimates the node value based on outgoing links.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    max_iter : integer, optional
      Maximum number of iterations in power method.

    tol : float, optional
      Error tolerance used to check convergence in power method iteration.

    nstart : dictionary, optional
      Starting value of each node for power method iteration.

    normalized : bool (default=True)
       Normalize results by the sum of all of the values.

    Returns
    -------
    (hubs,authorities) : two-tuple of dictionaries
       Two dictionaries keyed by node containing the hub and authority
       values.

    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> h, a = nx.hits(G)

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The HITS algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Jon Kleinberg,
       Authoritative sources in a hyperlinked environment
       Journal of the ACM 46 (5): 604-32, 1999.
       doi:10.1145/324133.324140.
       http://www.cs.cornell.edu/home/kleinber/auth.pdf.
    """
    import numpy as np
    from networkx.exception import PowerIterationFailedConvergence

    if len(G) == 0:
        return {}, {}
    
    A = nx.to_numpy_array(G)
    n = A.shape[0]
    
    if nstart is None:
        h = np.ones(n) / n
    else:
        h = np.array(list(nstart.values()))
        h = h / h.sum()
    
    a = np.zeros(n)
    
    for _ in range(max_iter):
        h_last, a_last = h.copy(), a.copy()
        
        a = A.T @ h
        if a.sum() != 0:
            a = a / a.sum()
        
        h = A @ a
        if h.sum() != 0:
            h = h / h.sum()
        
        if np.allclose(h, h_last, atol=tol) and np.allclose(a, a_last, atol=tol):
            break
    else:
        raise PowerIterationFailedConvergence(max_iter)
    
    hubs = dict(zip(G.nodes(), h))
    authorities = dict(zip(G.nodes(), a))
    
    if normalized:
        h_sum, a_sum = sum(hubs.values()), sum(authorities.values())
        hubs = {k: v / h_sum for k, v in hubs.items()}
        authorities = {k: v / a_sum for k, v in authorities.items()}
    
    return hubs, authorities


def _hits_numpy(G, normalized=True):
    """Returns HITS hubs and authorities values for nodes.

    The HITS algorithm computes two numbers for a node.
    Authorities estimates the node value based on the incoming links.
    Hubs estimates the node value based on outgoing links.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    normalized : bool (default=True)
       Normalize results by the sum of all of the values.

    Returns
    -------
    (hubs,authorities) : two-tuple of dictionaries
       Two dictionaries keyed by node containing the hub and authority
       values.

    Examples
    --------
    >>> G = nx.path_graph(4)

    The `hubs` and `authorities` are given by the eigenvectors corresponding to the
    maximum eigenvalues of the hubs_matrix and the authority_matrix, respectively.

    The ``hubs`` and ``authority`` matrices are computed from the adjacency
    matrix:

    >>> adj_ary = nx.to_numpy_array(G)
    >>> hubs_matrix = adj_ary @ adj_ary.T
    >>> authority_matrix = adj_ary.T @ adj_ary

    `_hits_numpy` maps the eigenvector corresponding to the maximum eigenvalue
    of the respective matrices to the nodes in `G`:

    >>> from networkx.algorithms.link_analysis.hits_alg import _hits_numpy
    >>> hubs, authority = _hits_numpy(G)

    Notes
    -----
    The eigenvector calculation uses NumPy's interface to LAPACK.

    The HITS algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Jon Kleinberg,
       Authoritative sources in a hyperlinked environment
       Journal of the ACM 46 (5): 604-32, 1999.
       doi:10.1145/324133.324140.
       http://www.cs.cornell.edu/home/kleinber/auth.pdf.
    """
    import numpy as np
    
    if len(G) == 0:
        return {}, {}
    
    adj_matrix = nx.to_numpy_array(G)
    hubs_matrix = adj_matrix @ adj_matrix.T
    authority_matrix = adj_matrix.T @ adj_matrix
    
    _, hubs_vector = np.linalg.eigh(hubs_matrix, eigvals=(hubs_matrix.shape[0]-1, hubs_matrix.shape[0]-1))
    _, auth_vector = np.linalg.eigh(authority_matrix, eigvals=(authority_matrix.shape[0]-1, authority_matrix.shape[0]-1))
    
    hubs_vector = hubs_vector.flatten().real
    auth_vector = auth_vector.flatten().real
    
    hubs = dict(zip(G.nodes(), hubs_vector))
    authorities = dict(zip(G.nodes(), auth_vector))
    
    if normalized:
        h_sum, a_sum = sum(abs(h) for h in hubs.values()), sum(abs(a) for a in authorities.values())
        hubs = {k: abs(v) / h_sum for k, v in hubs.items()}
        authorities = {k: abs(v) / a_sum for k, v in authorities.items()}
    
    return hubs, authorities


def _hits_scipy(G, max_iter=100, tol=1e-06, nstart=None, normalized=True):
    """Returns HITS hubs and authorities values for nodes.


    The HITS algorithm computes two numbers for a node.
    Authorities estimates the node value based on the incoming links.
    Hubs estimates the node value based on outgoing links.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    max_iter : integer, optional
      Maximum number of iterations in power method.

    tol : float, optional
      Error tolerance used to check convergence in power method iteration.

    nstart : dictionary, optional
      Starting value of each node for power method iteration.

    normalized : bool (default=True)
       Normalize results by the sum of all of the values.

    Returns
    -------
    (hubs,authorities) : two-tuple of dictionaries
       Two dictionaries keyed by node containing the hub and authority
       values.

    Examples
    --------
    >>> from networkx.algorithms.link_analysis.hits_alg import _hits_scipy
    >>> G = nx.path_graph(4)
    >>> h, a = _hits_scipy(G)

    Notes
    -----
    This implementation uses SciPy sparse matrices.

    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The HITS algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs.

    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Jon Kleinberg,
       Authoritative sources in a hyperlinked environment
       Journal of the ACM 46 (5): 604-632, 1999.
       doi:10.1145/324133.324140.
       http://www.cs.cornell.edu/home/kleinber/auth.pdf.
    """
    import numpy as np
    from scipy import sparse
    from networkx.exception import PowerIterationFailedConvergence
    
    if len(G) == 0:
        return {}, {}
    
    A = nx.to_scipy_sparse_array(G, dtype=float)
    n = A.shape[0]
    
    if nstart is None:
        h = np.ones(n) / n
    else:
        h = np.array(list(nstart.values()))
        h = h / h.sum()
    
    a = np.zeros(n)
    
    for _ in range(max_iter):
        h_last, a_last = h.copy(), a.copy()
        
        a = A.T @ h
        if a.sum() != 0:
            a = a / a.sum()
        
        h = A @ a
        if h.sum() != 0:
            h = h / h.sum()
        
        if np.allclose(h, h_last, atol=tol) and np.allclose(a, a_last, atol=tol):
            break
    else:
        raise PowerIterationFailedConvergence(max_iter)
    
    hubs = dict(zip(G.nodes(), h))
    authorities = dict(zip(G.nodes(), a))
    
    if normalized:
        h_sum, a_sum = sum(hubs.values()), sum(authorities.values())
        hubs = {k: v / h_sum for k, v in hubs.items()}
        authorities = {k: v / a_sum for k, v in authorities.items()}
    
    return hubs, authorities
