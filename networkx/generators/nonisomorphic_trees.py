"""
Implementation of the Wright, Richmond, Odlyzko and McKay (WROM)
algorithm for the enumeration of all non-isomorphic free trees of a
given order.  Rooted trees are represented by level sequences, i.e.,
lists in which the i-th element specifies the distance of vertex i to
the root.

"""
__all__ = ['nonisomorphic_trees', 'number_of_nonisomorphic_trees']
import networkx as nx


@nx._dispatchable(graphs=None, returns_graph=True)
def nonisomorphic_trees(order, create='graph'):
    """Generates lists of nonisomorphic trees

    Parameters
    ----------
    order : int
       order of the desired tree(s)

    create : one of {"graph", "matrix"} (default="graph")
       If ``"graph"`` is selected a list of ``Graph`` instances will be returned,
       if matrix is selected a list of adjacency matrices will be returned.

       .. deprecated:: 3.3

          The `create` argument is deprecated and will be removed in NetworkX
          version 3.5. In the future, `nonisomorphic_trees` will yield graph
          instances by default. To generate adjacency matrices, call
          ``nx.to_numpy_array`` on the output, e.g.::

             [nx.to_numpy_array(G) for G in nx.nonisomorphic_trees(N)]

    Yields
    ------
    list
       A list of nonisomorphic trees, in one of two formats depending on the
       value of the `create` parameter:
       - ``create="graph"``: yields a list of `networkx.Graph` instances
       - ``create="matrix"``: yields a list of list-of-lists representing adjacency matrices
    """
    if order < 1:
        return

    layout = [0] * order
    while layout is not None:
        if create == 'matrix':
            yield _layout_to_matrix(layout)
        else:
            yield _layout_to_graph(layout)
        layout = _next_tree(layout)


@nx._dispatchable(graphs=None)
def number_of_nonisomorphic_trees(order):
    """Returns the number of nonisomorphic trees

    Parameters
    ----------
    order : int
      order of the desired tree(s)

    Returns
    -------
    length : Number of nonisomorphic graphs for the given order

    References
    ----------
    .. [1] Otter, Richard. "The number of trees." Annals of Mathematics (1948): 583-599.
    """
    if order < 1:
        return 0
    return sum(1 for _ in nonisomorphic_trees(order))


def _next_rooted_tree(predecessor, p=None):
    """One iteration of the Beyer-Hedetniemi algorithm."""
    if p is None:
        p = len(predecessor) - 1
    if p == 0:
        return None
    if predecessor[p - 1] < predecessor[p]:
        successor = predecessor[:]
        successor[p] = predecessor[p - 1] + 1
        return successor
    return _next_rooted_tree(predecessor, p - 1)


def _next_tree(candidate):
    """One iteration of the Wright, Richmond, Odlyzko and McKay
    algorithm."""
    left, right = _split_tree(candidate)
    if not right:
        return None
    next_right = _next_rooted_tree(right)
    if next_right is None:
        return _next_tree(left + [0])
    return left + next_right


def _split_tree(layout):
    """Returns a tuple of two layouts, one containing the left
    subtree of the root vertex, and one containing the original tree
    with the left subtree removed."""
    if len(layout) <= 1:
        return [], []
    for i, level in enumerate(layout[1:], 1):
        if level == 1:
            return layout[:i], layout[i:]
    return layout, []


def _layout_to_matrix(layout):
    """Create the adjacency matrix for the tree specified by the
    given layout (level sequence)."""
    n = len(layout)
    matrix = [[0] * n for _ in range(n)]
    for child, parent in enumerate(layout[1:], 1):
        parent = next(i for i in range(child - 1, -1, -1) if layout[i] == layout[child] - 1)
        matrix[parent][child] = matrix[child][parent] = 1
    return matrix


def _layout_to_graph(layout):
    """Create a NetworkX Graph for the tree specified by the
    given layout(level sequence)"""
    G = nx.Graph()
    G.add_nodes_from(range(len(layout)))
    for child, parent in enumerate(layout[1:], 1):
        parent = next(i for i in range(child - 1, -1, -1) if layout[i] == layout[child] - 1)
        G.add_edge(parent, child)
    return G
