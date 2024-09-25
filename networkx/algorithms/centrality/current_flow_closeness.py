"""Current-flow closeness centrality measures."""
import networkx as nx
from networkx.algorithms.centrality.flow_matrix import CGInverseLaplacian, FullInverseLaplacian, SuperLUInverseLaplacian
from networkx.utils import not_implemented_for, reverse_cuthill_mckee_ordering
__all__ = ['current_flow_closeness_centrality', 'information_centrality']


@not_implemented_for('directed')
@nx._dispatchable(edge_attrs='weight')
def current_flow_closeness_centrality(G, weight=None, dtype=float, solver='lu'):
    """Compute current-flow closeness centrality for nodes.

    Current-flow closeness centrality is variant of closeness
    centrality based on effective resistance between nodes in
    a network. This metric is also known as information centrality.

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      The weight reflects the capacity or the strength of the
      edge.

    dtype: data type (default=float)
      Default data type for internal matrices.
      Set to np.float32 for lower memory consumption.

    solver: string (default='lu')
       Type of linear solver to use for computing the flow matrix.
       Options are "full" (uses most memory), "lu" (recommended), and
       "cg" (uses least memory).

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with current flow closeness centrality as the value.

    See Also
    --------
    closeness_centrality

    Notes
    -----
    The algorithm is from Brandes [1]_.

    See also [2]_ for the original definition of information centrality.

    References
    ----------
    .. [1] Ulrik Brandes and Daniel Fleischer,
       Centrality Measures Based on Current Flow.
       Proc. 22nd Symp. Theoretical Aspects of Computer Science (STACS '05).
       LNCS 3404, pp. 533-544. Springer-Verlag, 2005.
       https://doi.org/10.1007/978-3-540-31856-9_44

    .. [2] Karen Stephenson and Marvin Zelen:
       Rethinking centrality: Methods and examples.
       Social Networks 11(1):1-37, 1989.
       https://doi.org/10.1016/0378-8733(89)90016-6
    """
    import numpy as np

    if G.is_directed():
        raise nx.NetworkXError("Current flow closeness centrality not defined for directed graphs.")

    if solver == 'full':
        solver = FullInverseLaplacian(G, weight=weight, dtype=dtype)
    elif solver == 'lu':
        solver = SuperLUInverseLaplacian(G, weight=weight, dtype=dtype)
    elif solver == 'cg':
        solver = CGInverseLaplacian(G, weight=weight, dtype=dtype)
    else:
        raise nx.NetworkXError("Unknown solver: %s" % solver)

    n = G.number_of_nodes()
    ordering = list(reverse_cuthill_mckee_ordering(G))
    L = solver.L.tocsr()

    centrality = {}
    for node in G:
        row = ordering.index(node)
        T = sum(L[row, ordering.index(other)] for other in G)
        centrality[node] = (n - 1) / T if T != 0 else 0.0

    return centrality


information_centrality = current_flow_closeness_centrality
