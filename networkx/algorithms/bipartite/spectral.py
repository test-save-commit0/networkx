"""
Spectral bipartivity measure.
"""
import networkx as nx
__all__ = ['spectral_bipartivity']


@nx._dispatchable(edge_attrs='weight')
def spectral_bipartivity(G, nodes=None, weight='weight'):
    """Returns the spectral bipartivity.

    Parameters
    ----------
    G : NetworkX graph

    nodes : list or container  optional(default is all nodes)
      Nodes to return value of spectral bipartivity contribution.

    weight : string or None  optional (default = 'weight')
      Edge data key to use for edge weights. If None, weights set to 1.

    Returns
    -------
    sb : float or dict
       A single number if the keyword nodes is not specified, or
       a dictionary keyed by node with the spectral bipartivity contribution
       of that node as the value.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> bipartite.spectral_bipartivity(G)
    1.0

    Notes
    -----
    This implementation uses Numpy (dense) matrices which are not efficient
    for storing large sparse graphs.

    See Also
    --------
    color

    References
    ----------
    .. [1] E. Estrada and J. A. Rodríguez-Velázquez, "Spectral measures of
       bipartivity in complex networks", PhysRev E 72, 046105 (2005)
    """
    import numpy as np
    from scipy import linalg

    if G.number_of_nodes() == 0:
        if nodes is None:
            return 0
        return {}.fromkeys(nodes, 0)

    nodelist = list(G)
    A = nx.to_numpy_array(G, nodelist=nodelist, weight=weight)
    expA = linalg.expm(A)
    expmA = linalg.expm(-A)
    coshA = (expA + expmA) / 2

    if nodes is None:
        # Compute the spectral bipartivity for the entire graph
        sb = np.sum(expmA) / np.sum(coshA)
    else:
        # Compute the spectral bipartivity contribution for specified nodes
        nodes = set(nodes) & set(nodelist)
        indices = [nodelist.index(n) for n in nodes]
        sb = {n: expmA[i, i] / coshA[i, i] for i, n in zip(indices, nodes)}

    return sb
