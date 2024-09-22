"""
====================
Biadjacency matrices
====================
"""
import itertools
import networkx as nx
from networkx.convert_matrix import _generate_weighted_edges
__all__ = ['biadjacency_matrix', 'from_biadjacency_matrix']


@nx._dispatchable(edge_attrs='weight')
def biadjacency_matrix(G, row_order, column_order=None, dtype=None, weight=
    'weight', format='csr'):
    """Returns the biadjacency matrix of the bipartite graph G.

    Let `G = (U, V, E)` be a bipartite graph with node sets
    `U = u_{1},...,u_{r}` and `V = v_{1},...,v_{s}`. The biadjacency
    matrix [1]_ is the `r` x `s` matrix `B` in which `b_{i,j} = 1`
    if, and only if, `(u_i, v_j) \\in E`. If the parameter `weight` is
    not `None` and matches the name of an edge attribute, its value is
    used instead of 1.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    row_order : list of nodes
       The rows of the matrix are ordered according to the list of nodes.

    column_order : list, optional
       The columns of the matrix are ordered according to the list of nodes.
       If column_order is None, then the ordering of columns is arbitrary.

    dtype : NumPy data-type, optional
        A valid NumPy dtype used to initialize the array. If None, then the
        NumPy default is used.

    weight : string or None, optional (default='weight')
       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.

    format : str in {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'}
        The type of the matrix to be returned (default 'csr').  For
        some algorithms different implementations of sparse matrices
        can perform better.  See [2]_ for details.

    Returns
    -------
    M : SciPy sparse array
        Biadjacency matrix representation of the bipartite graph G.

    Notes
    -----
    No attempt is made to check that the input graph is bipartite.

    For directed bipartite graphs only successors are considered as neighbors.
    To obtain an adjacency matrix with ones (or weight values) for both
    predecessors and successors you have to generate two biadjacency matrices
    where the rows of one of them are the columns of the other, and then add
    one to the transpose of the other.

    See Also
    --------
    adjacency_matrix
    from_biadjacency_matrix

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Adjacency_matrix#Adjacency_matrix_of_a_bipartite_graph
    .. [2] Scipy Dev. References, "Sparse Matrices",
       https://docs.scipy.org/doc/scipy/reference/sparse.html
    """
    pass


@nx._dispatchable(graphs=None, returns_graph=True)
def from_biadjacency_matrix(A, create_using=None, edge_attribute='weight'):
    """Creates a new bipartite graph from a biadjacency matrix given as a
    SciPy sparse array.

    Parameters
    ----------
    A: scipy sparse array
      A biadjacency matrix representation of a graph

    create_using: NetworkX graph
       Use specified graph for result.  The default is Graph()

    edge_attribute: string
       Name of edge attribute to store matrix numeric value. The data will
       have the same type as the matrix entry (int, float, (real,imag)).

    Notes
    -----
    The nodes are labeled with the attribute `bipartite` set to an integer
    0 or 1 representing membership in part 0 or part 1 of the bipartite graph.

    If `create_using` is an instance of :class:`networkx.MultiGraph` or
    :class:`networkx.MultiDiGraph` and the entries of `A` are of
    type :class:`int`, then this function returns a multigraph (of the same
    type as `create_using`) with parallel edges. In this case, `edge_attribute`
    will be ignored.

    See Also
    --------
    biadjacency_matrix
    from_numpy_array

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Adjacency_matrix#Adjacency_matrix_of_a_bipartite_graph
    """
    pass
