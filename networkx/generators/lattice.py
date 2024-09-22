"""Functions for generating grid graphs and lattices

The :func:`grid_2d_graph`, :func:`triangular_lattice_graph`, and
:func:`hexagonal_lattice_graph` functions correspond to the three
`regular tilings of the plane`_, the square, triangular, and hexagonal
tilings, respectively. :func:`grid_graph` and :func:`hypercube_graph`
are similar for arbitrary dimensions. Useful relevant discussion can
be found about `Triangular Tiling`_, and `Square, Hex and Triangle Grids`_

.. _regular tilings of the plane: https://en.wikipedia.org/wiki/List_of_regular_polytopes_and_compounds#Euclidean_tilings
.. _Square, Hex and Triangle Grids: http://www-cs-students.stanford.edu/~amitp/game-programming/grids/
.. _Triangular Tiling: https://en.wikipedia.org/wiki/Triangular_tiling

"""
from itertools import repeat
from math import sqrt
import networkx as nx
from networkx.classes import set_node_attributes
from networkx.exception import NetworkXError
from networkx.generators.classic import cycle_graph, empty_graph, path_graph
from networkx.relabel import relabel_nodes
from networkx.utils import flatten, nodes_or_number, pairwise
__all__ = ['grid_2d_graph', 'grid_graph', 'hypercube_graph',
    'triangular_lattice_graph', 'hexagonal_lattice_graph']


@nx._dispatchable(graphs=None, returns_graph=True)
@nodes_or_number([0, 1])
def grid_2d_graph(m, n, periodic=False, create_using=None):
    """Returns the two-dimensional grid graph.

    The grid graph has each node connected to its four nearest neighbors.

    Parameters
    ----------
    m, n : int or iterable container of nodes
        If an integer, nodes are from `range(n)`.
        If a container, elements become the coordinate of the nodes.

    periodic : bool or iterable
        If `periodic` is True, both dimensions are periodic. If False, none
        are periodic.  If `periodic` is iterable, it should yield 2 bool
        values indicating whether the 1st and 2nd axes, respectively, are
        periodic.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    NetworkX graph
        The (possibly periodic) grid graph of the specified dimensions.

    """
    pass


@nx._dispatchable(graphs=None, returns_graph=True)
def grid_graph(dim, periodic=False):
    """Returns the *n*-dimensional grid graph.

    The dimension *n* is the length of the list `dim` and the size in
    each dimension is the value of the corresponding list element.

    Parameters
    ----------
    dim : list or tuple of numbers or iterables of nodes
        'dim' is a tuple or list with, for each dimension, either a number
        that is the size of that dimension or an iterable of nodes for
        that dimension. The dimension of the grid_graph is the length
        of `dim`.

    periodic : bool or iterable
        If `periodic` is True, all dimensions are periodic. If False all
        dimensions are not periodic. If `periodic` is iterable, it should
        yield `dim` bool values each of which indicates whether the
        corresponding axis is periodic.

    Returns
    -------
    NetworkX graph
        The (possibly periodic) grid graph of the specified dimensions.

    Examples
    --------
    To produce a 2 by 3 by 4 grid graph, a graph on 24 nodes:

    >>> from networkx import grid_graph
    >>> G = grid_graph(dim=(2, 3, 4))
    >>> len(G)
    24
    >>> G = grid_graph(dim=(range(7, 9), range(3, 6)))
    >>> len(G)
    6
    """
    pass


@nx._dispatchable(graphs=None, returns_graph=True)
def hypercube_graph(n):
    """Returns the *n*-dimensional hypercube graph.

    The nodes are the integers between 0 and ``2 ** n - 1``, inclusive.

    For more information on the hypercube graph, see the Wikipedia
    article `Hypercube graph`_.

    .. _Hypercube graph: https://en.wikipedia.org/wiki/Hypercube_graph

    Parameters
    ----------
    n : int
        The dimension of the hypercube.
        The number of nodes in the graph will be ``2 ** n``.

    Returns
    -------
    NetworkX graph
        The hypercube graph of dimension *n*.
    """
    pass


@nx._dispatchable(graphs=None, returns_graph=True)
def triangular_lattice_graph(m, n, periodic=False, with_positions=True,
    create_using=None):
    """Returns the $m$ by $n$ triangular lattice graph.

    The `triangular lattice graph`_ is a two-dimensional `grid graph`_ in
    which each square unit has a diagonal edge (each grid unit has a chord).

    The returned graph has $m$ rows and $n$ columns of triangles. Rows and
    columns include both triangles pointing up and down. Rows form a strip
    of constant height. Columns form a series of diamond shapes, staggered
    with the columns on either side. Another way to state the size is that
    the nodes form a grid of `m+1` rows and `(n + 1) // 2` columns.
    The odd row nodes are shifted horizontally relative to the even rows.

    Directed graph types have edges pointed up or right.

    Positions of nodes are computed by default or `with_positions is True`.
    The position of each node (embedded in a euclidean plane) is stored in
    the graph using equilateral triangles with sidelength 1.
    The height between rows of nodes is thus $\\sqrt(3)/2$.
    Nodes lie in the first quadrant with the node $(0, 0)$ at the origin.

    .. _triangular lattice graph: http://mathworld.wolfram.com/TriangularGrid.html
    .. _grid graph: http://www-cs-students.stanford.edu/~amitp/game-programming/grids/
    .. _Triangular Tiling: https://en.wikipedia.org/wiki/Triangular_tiling

    Parameters
    ----------
    m : int
        The number of rows in the lattice.

    n : int
        The number of columns in the lattice.

    periodic : bool (default: False)
        If True, join the boundary vertices of the grid using periodic
        boundary conditions. The join between boundaries is the final row
        and column of triangles. This means there is one row and one column
        fewer nodes for the periodic lattice. Periodic lattices require
        `m >= 3`, `n >= 5` and are allowed but misaligned if `m` or `n` are odd

    with_positions : bool (default: True)
        Store the coordinates of each node in the graph node attribute 'pos'.
        The coordinates provide a lattice with equilateral triangles.
        Periodic positions shift the nodes vertically in a nonlinear way so
        the edges don't overlap so much.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    NetworkX graph
        The *m* by *n* triangular lattice graph.
    """
    pass


@nx._dispatchable(graphs=None, returns_graph=True)
def hexagonal_lattice_graph(m, n, periodic=False, with_positions=True,
    create_using=None):
    """Returns an `m` by `n` hexagonal lattice graph.

    The *hexagonal lattice graph* is a graph whose nodes and edges are
    the `hexagonal tiling`_ of the plane.

    The returned graph will have `m` rows and `n` columns of hexagons.
    `Odd numbered columns`_ are shifted up relative to even numbered columns.

    Positions of nodes are computed by default or `with_positions is True`.
    Node positions creating the standard embedding in the plane
    with sidelength 1 and are stored in the node attribute 'pos'.
    `pos = nx.get_node_attributes(G, 'pos')` creates a dict ready for drawing.

    .. _hexagonal tiling: https://en.wikipedia.org/wiki/Hexagonal_tiling
    .. _Odd numbered columns: http://www-cs-students.stanford.edu/~amitp/game-programming/grids/

    Parameters
    ----------
    m : int
        The number of rows of hexagons in the lattice.

    n : int
        The number of columns of hexagons in the lattice.

    periodic : bool
        Whether to make a periodic grid by joining the boundary vertices.
        For this to work `n` must be even and both `n > 1` and `m > 1`.
        The periodic connections create another row and column of hexagons
        so these graphs have fewer nodes as boundary nodes are identified.

    with_positions : bool (default: True)
        Store the coordinates of each node in the graph node attribute 'pos'.
        The coordinates provide a lattice with vertical columns of hexagons
        offset to interleave and cover the plane.
        Periodic positions shift the nodes vertically in a nonlinear way so
        the edges don't overlap so much.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        If graph is directed, edges will point up or right.

    Returns
    -------
    NetworkX graph
        The *m* by *n* hexagonal lattice graph.
    """
    pass
