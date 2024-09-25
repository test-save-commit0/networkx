"""Generator for Sudoku graphs

This module gives a generator for n-Sudoku graphs. It can be used to develop
algorithms for solving or generating Sudoku puzzles.

A completed Sudoku grid is a 9x9 array of integers between 1 and 9, with no
number appearing twice in the same row, column, or 3x3 box.

+---------+---------+---------+
| | 8 6 4 | | 3 7 1 | | 2 5 9 |
| | 3 2 5 | | 8 4 9 | | 7 6 1 |
| | 9 7 1 | | 2 6 5 | | 8 4 3 |
+---------+---------+---------+
| | 4 3 6 | | 1 9 2 | | 5 8 7 |
| | 1 9 8 | | 6 5 7 | | 4 3 2 |
| | 2 5 7 | | 4 8 3 | | 9 1 6 |
+---------+---------+---------+
| | 6 8 9 | | 7 3 4 | | 1 2 5 |
| | 7 1 3 | | 5 2 8 | | 6 9 4 |
| | 5 4 2 | | 9 1 6 | | 3 7 8 |
+---------+---------+---------+


The Sudoku graph is an undirected graph with 81 vertices, corresponding to
the cells of a Sudoku grid. It is a regular graph of degree 20. Two distinct
vertices are adjacent if and only if the corresponding cells belong to the
same row, column, or box. A completed Sudoku grid corresponds to a vertex
coloring of the Sudoku graph with nine colors.

More generally, the n-Sudoku graph is a graph with n^4 vertices, corresponding
to the cells of an n^2 by n^2 grid. Two distinct vertices are adjacent if and
only if they belong to the same row, column, or n by n box.

References
----------
.. [1] Herzberg, A. M., & Murty, M. R. (2007). Sudoku squares and chromatic
    polynomials. Notices of the AMS, 54(6), 708-717.
.. [2] Sander, Torsten (2009), "Sudoku graphs are integral",
    Electronic Journal of Combinatorics, 16 (1): Note 25, 7pp, MR 2529816
.. [3] Wikipedia contributors. "Glossary of Sudoku." Wikipedia, The Free
    Encyclopedia, 3 Dec. 2019. Web. 22 Dec. 2019.
"""
import networkx as nx
from networkx.exception import NetworkXError
__all__ = ['sudoku_graph']


@nx._dispatchable(graphs=None, returns_graph=True)
def sudoku_graph(n=3):
    """Returns the n-Sudoku graph. The default value of n is 3.

    The n-Sudoku graph is a graph with n^4 vertices, corresponding to the
    cells of an n^2 by n^2 grid. Two distinct vertices are adjacent if and
    only if they belong to the same row, column, or n-by-n box.

    Parameters
    ----------
    n: integer
       The order of the Sudoku graph, equal to the square root of the
       number of rows. The default is 3.

    Returns
    -------
    NetworkX graph
        The n-Sudoku graph Sud(n).

    Examples
    --------
    >>> G = nx.sudoku_graph()
    >>> G.number_of_nodes()
    81
    >>> G.number_of_edges()
    810
    >>> sorted(G.neighbors(42))
    [6, 15, 24, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 51, 52, 53, 60, 69, 78]
    >>> G = nx.sudoku_graph(2)
    >>> G.number_of_nodes()
    16
    >>> G.number_of_edges()
    56

    References
    ----------
    .. [1] Herzberg, A. M., & Murty, M. R. (2007). Sudoku squares and chromatic
       polynomials. Notices of the AMS, 54(6), 708-717.
    .. [2] Sander, Torsten (2009), "Sudoku graphs are integral",
       Electronic Journal of Combinatorics, 16 (1): Note 25, 7pp, MR 2529816
    .. [3] Wikipedia contributors. "Glossary of Sudoku." Wikipedia, The Free
       Encyclopedia, 3 Dec. 2019. Web. 22 Dec. 2019.
    """
    if not isinstance(n, int) or n < 1:
        raise NetworkXError("n must be a positive integer")

    G = nx.Graph()
    n_squared = n * n

    # Add nodes
    for i in range(n_squared * n_squared):
        G.add_node(i)

    # Add edges
    for i in range(n_squared * n_squared):
        row, col = divmod(i, n_squared)
        box_row, box_col = divmod(row, n), divmod(col, n)

        # Same row
        for j in range(n_squared):
            if j != col:
                G.add_edge(i, row * n_squared + j)

        # Same column
        for j in range(n_squared):
            if j != row:
                G.add_edge(i, j * n_squared + col)

        # Same box
        for r in range(box_row[0] * n, (box_row[0] + 1) * n):
            for c in range(box_col[0] * n, (box_col[0] + 1) * n):
                if r != row or c != col:
                    G.add_edge(i, r * n_squared + c)

    return G
