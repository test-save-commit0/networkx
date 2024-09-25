"""Functions for encoding and decoding trees.

Since a tree is a highly restricted form of graph, it can be represented
concisely in several ways. This module includes functions for encoding
and decoding trees in the form of nested tuples and Prüfer
sequences. The former requires a rooted tree, whereas the latter can be
applied to unrooted trees. Furthermore, there is a bijection from Prüfer
sequences to labeled trees.

"""
from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['from_nested_tuple', 'from_prufer_sequence', 'NotATree',
    'to_nested_tuple', 'to_prufer_sequence']


class NotATree(nx.NetworkXException):
    """Raised when a function expects a tree (that is, a connected
    undirected graph with no cycles) but gets a non-tree graph as input
    instead.

    """


@not_implemented_for('directed')
@nx._dispatchable(graphs='T')
def to_nested_tuple(T, root, canonical_form=False):
    """Returns a nested tuple representation of the given tree.

    The nested tuple representation of a tree is defined
    recursively. The tree with one node and no edges is represented by
    the empty tuple, ``()``. A tree with ``k`` subtrees is represented
    by a tuple of length ``k`` in which each element is the nested tuple
    representation of a subtree.

    Parameters
    ----------
    T : NetworkX graph
        An undirected graph object representing a tree.

    root : node
        The node in ``T`` to interpret as the root of the tree.

    canonical_form : bool
        If ``True``, each tuple is sorted so that the function returns
        a canonical form for rooted trees. This means "lighter" subtrees
        will appear as nested tuples before "heavier" subtrees. In this
        way, each isomorphic rooted tree has the same nested tuple
        representation.

    Returns
    -------
    tuple
        A nested tuple representation of the tree.

    Notes
    -----
    This function is *not* the inverse of :func:`from_nested_tuple`; the
    only guarantee is that the rooted trees are isomorphic.

    See also
    --------
    from_nested_tuple
    to_prufer_sequence

    Examples
    --------
    The tree need not be a balanced binary tree::

        >>> T = nx.Graph()
        >>> T.add_edges_from([(0, 1), (0, 2), (0, 3)])
        >>> T.add_edges_from([(1, 4), (1, 5)])
        >>> T.add_edges_from([(3, 6), (3, 7)])
        >>> root = 0
        >>> nx.to_nested_tuple(T, root)
        (((), ()), (), ((), ()))

    Continuing the above example, if ``canonical_form`` is ``True``, the
    nested tuples will be sorted::

        >>> nx.to_nested_tuple(T, root, canonical_form=True)
        ((), ((), ()), ((), ()))

    Even the path graph can be interpreted as a tree::

        >>> T = nx.path_graph(4)
        >>> root = 0
        >>> nx.to_nested_tuple(T, root)
        ((((),),),)

    """
    def _to_tuple(node):
        children = [child for child in T.neighbors(node) if child != parent.get(node)]
        if not children:
            return ()
        subtrees = [_to_tuple(child) for child in children]
        if canonical_form:
            subtrees.sort(key=lambda x: (len(x), x))
        return tuple(subtrees)

    if not nx.is_tree(T):
        raise NotATree("The graph is not a tree.")

    parent = {root: None}
    stack = [root]
    while stack:
        node = stack.pop()
        for child in T.neighbors(node):
            if child not in parent:
                parent[child] = node
                stack.append(child)

    return _to_tuple(root)


@nx._dispatchable(graphs=None, returns_graph=True)
def from_nested_tuple(sequence, sensible_relabeling=False):
    """Returns the rooted tree corresponding to the given nested tuple.

    The nested tuple representation of a tree is defined
    recursively. The tree with one node and no edges is represented by
    the empty tuple, ``()``. A tree with ``k`` subtrees is represented
    by a tuple of length ``k`` in which each element is the nested tuple
    representation of a subtree.

    Parameters
    ----------
    sequence : tuple
        A nested tuple representing a rooted tree.

    sensible_relabeling : bool
        Whether to relabel the nodes of the tree so that nodes are
        labeled in increasing order according to their breadth-first
        search order from the root node.

    Returns
    -------
    NetworkX graph
        The tree corresponding to the given nested tuple, whose root
        node is node 0. If ``sensible_labeling`` is ``True``, nodes will
        be labeled in breadth-first search order starting from the root
        node.

    Notes
    -----
    This function is *not* the inverse of :func:`to_nested_tuple`; the
    only guarantee is that the rooted trees are isomorphic.

    See also
    --------
    to_nested_tuple
    from_prufer_sequence

    Examples
    --------
    Sensible relabeling ensures that the nodes are labeled from the root
    starting at 0::

        >>> balanced = (((), ()), ((), ()))
        >>> T = nx.from_nested_tuple(balanced, sensible_relabeling=True)
        >>> edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        >>> all((u, v) in T.edges() or (v, u) in T.edges() for (u, v) in edges)
        True

    """
    def _from_tuple(tup, parent=None):
        node = next(counter)
        T.add_node(node)
        if parent is not None:
            T.add_edge(parent, node)
        for subtree in tup:
            _from_tuple(subtree, node)

    T = nx.Graph()
    counter = iter(range(len(sequence) + 1))
    _from_tuple(sequence)

    if sensible_relabeling:
        mapping = {old: new for new, old in enumerate(nx.bfs_tree(T, 0))}
        T = nx.relabel_nodes(T, mapping)

    return T


@not_implemented_for('directed')
@nx._dispatchable(graphs='T')
def to_prufer_sequence(T):
    """Returns the Prüfer sequence of the given tree.

    A *Prüfer sequence* is a list of *n* - 2 numbers between 0 and
    *n* - 1, inclusive. The tree corresponding to a given Prüfer
    sequence can be recovered by repeatedly joining a node in the
    sequence with a node with the smallest potential degree according to
    the sequence.

    Parameters
    ----------
    T : NetworkX graph
        An undirected graph object representing a tree.

    Returns
    -------
    list
        The Prüfer sequence of the given tree.

    Raises
    ------
    NetworkXPointlessConcept
        If the number of nodes in `T` is less than two.

    NotATree
        If `T` is not a tree.

    KeyError
        If the set of nodes in `T` is not {0, …, *n* - 1}.

    Notes
    -----
    There is a bijection from labeled trees to Prüfer sequences. This
    function is the inverse of the :func:`from_prufer_sequence`
    function.

    Sometimes Prüfer sequences use nodes labeled from 1 to *n* instead
    of from 0 to *n* - 1. This function requires nodes to be labeled in
    the latter form. You can use :func:`~networkx.relabel_nodes` to
    relabel the nodes of your tree to the appropriate format.

    This implementation is from [1]_ and has a running time of
    $O(n)$.

    See also
    --------
    to_nested_tuple
    from_prufer_sequence

    References
    ----------
    .. [1] Wang, Xiaodong, Lei Wang, and Yingjie Wu.
           "An optimal algorithm for Prufer codes."
           *Journal of Software Engineering and Applications* 2.02 (2009): 111.
           <https://doi.org/10.4236/jsea.2009.22016>

    Examples
    --------
    There is a bijection between Prüfer sequences and labeled trees, so
    this function is the inverse of the :func:`from_prufer_sequence`
    function:

    >>> edges = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)]
    >>> tree = nx.Graph(edges)
    >>> sequence = nx.to_prufer_sequence(tree)
    >>> sequence
    [3, 3, 3, 4]
    >>> tree2 = nx.from_prufer_sequence(sequence)
    >>> list(tree2.edges()) == edges
    True

    """
    if not nx.is_tree(T):
        raise NotATree("The graph is not a tree.")

    n = T.number_of_nodes()
    if n < 2:
        raise nx.NetworkXPointlessConcept("Prüfer sequence undefined for trees with fewer than two nodes.")

    if set(T.nodes()) != set(range(n)):
        raise KeyError("The nodes must be labeled 0, ..., n-1")

    degree = dict(T.degree())
    leaves = [node for node in T.nodes() if degree[node] == 1]
    sequence = []

    for _ in range(n - 2):
        leaf = min(leaves)
        neighbor = next(iter(T.neighbors(leaf)))
        sequence.append(neighbor)
        degree[neighbor] -= 1
        if degree[neighbor] == 1:
            leaves.append(neighbor)
        leaves.remove(leaf)

    return sequence


@nx._dispatchable(graphs=None, returns_graph=True)
def from_prufer_sequence(sequence):
    """Returns the tree corresponding to the given Prüfer sequence.

    A *Prüfer sequence* is a list of *n* - 2 numbers between 0 and
    *n* - 1, inclusive. The tree corresponding to a given Prüfer
    sequence can be recovered by repeatedly joining a node in the
    sequence with a node with the smallest potential degree according to
    the sequence.

    Parameters
    ----------
    sequence : list
        A Prüfer sequence, which is a list of *n* - 2 integers between
        zero and *n* - 1, inclusive.

    Returns
    -------
    NetworkX graph
        The tree corresponding to the given Prüfer sequence.

    Raises
    ------
    NetworkXError
        If the Prüfer sequence is not valid.

    Notes
    -----
    There is a bijection from labeled trees to Prüfer sequences. This
    function is the inverse of the :func:`from_prufer_sequence` function.

    Sometimes Prüfer sequences use nodes labeled from 1 to *n* instead
    of from 0 to *n* - 1. This function requires nodes to be labeled in
    the latter form. You can use :func:`networkx.relabel_nodes` to
    relabel the nodes of your tree to the appropriate format.

    This implementation is from [1]_ and has a running time of
    $O(n)$.

    References
    ----------
    .. [1] Wang, Xiaodong, Lei Wang, and Yingjie Wu.
           "An optimal algorithm for Prufer codes."
           *Journal of Software Engineering and Applications* 2.02 (2009): 111.
           <https://doi.org/10.4236/jsea.2009.22016>

    See also
    --------
    from_nested_tuple
    to_prufer_sequence

    Examples
    --------
    There is a bijection between Prüfer sequences and labeled trees, so
    this function is the inverse of the :func:`to_prufer_sequence`
    function:

    >>> edges = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)]
    >>> tree = nx.Graph(edges)
    >>> sequence = nx.to_prufer_sequence(tree)
    >>> sequence
    [3, 3, 3, 4]
    >>> tree2 = nx.from_prufer_sequence(sequence)
    >>> list(tree2.edges()) == edges
    True

    """
    n = len(sequence) + 2
    if not all(0 <= x < n for x in sequence):
        raise nx.NetworkXError("Prüfer sequence is not valid.")

    T = nx.Graph()
    T.add_nodes_from(range(n))
    
    degree = [1] * n
    for i in sequence:
        degree[i] += 1

    for u in sequence:
        for v in range(n):
            if degree[v] == 1:
                T.add_edge(u, v)
                degree[u] -= 1
                degree[v] -= 1
                break

    last_two = [i for i in range(n) if degree[i] == 1]
    T.add_edge(last_two[0], last_two[1])

    return T
