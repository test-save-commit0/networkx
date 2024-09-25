"""
An algorithm for finding if two undirected trees are isomorphic,
and if so returns an isomorphism between the two sets of nodes.

This algorithm uses a routine to tell if two rooted trees (trees with a
specified root node) are isomorphic, which may be independently useful.

This implements an algorithm from:
The Design and Analysis of Computer Algorithms
by Aho, Hopcroft, and Ullman
Addison-Wesley Publishing 1974
Example 3.2 pp. 84-86.

A more understandable version of this algorithm is described in:
Homework Assignment 5
McGill University SOCS 308-250B, Winter 2002
by Matthew Suderman
http://crypto.cs.mcgill.ca/~crepeau/CS250/2004/HW5+.pdf
"""
import networkx as nx
from networkx.utils.decorators import not_implemented_for
__all__ = ['rooted_tree_isomorphism', 'tree_isomorphism']


@nx._dispatchable(graphs={'t1': 0, 't2': 2}, returns_graph=True)
def root_trees(t1, root1, t2, root2):
    """Create a single digraph dT of free trees t1 and t2
    #   with roots root1 and root2 respectively
    # rename the nodes with consecutive integers
    # so that all nodes get a unique name between both trees

    # our new "fake" root node is 0
    # t1 is numbers from 1 ... n
    # t2 is numbered from n+1 to 2n
    """
    dT = nx.DiGraph()
    dT.add_node(0)  # Add the fake root node

    def add_tree(T, root, start):
        mapping = {root: start}
        stack = [(root, start)]
        next_id = start + 1

        while stack:
            parent, parent_id = stack.pop()
            for child in T.neighbors(parent):
                if child not in mapping:
                    mapping[child] = next_id
                    dT.add_edge(parent_id, next_id)
                    stack.append((child, next_id))
                    next_id += 1

        return mapping

    t1_mapping = add_tree(t1, root1, 1)
    t2_mapping = add_tree(t2, root2, len(t1) + 1)

    dT.add_edge(0, 1)  # Connect fake root to t1's root
    dT.add_edge(0, len(t1) + 1)  # Connect fake root to t2's root

    nx.set_node_attributes(dT, {0: {"tree": "root"}})
    nx.set_node_attributes(dT, {v: {"tree": "t1", "original": k} for k, v in t1_mapping.items()})
    nx.set_node_attributes(dT, {v: {"tree": "t2", "original": k} for k, v in t2_mapping.items()})

    return dT


@nx._dispatchable(graphs={'t1': 0, 't2': 2})
def rooted_tree_isomorphism(t1, root1, t2, root2):
    """
    Given two rooted trees `t1` and `t2`,
    with roots `root1` and `root2` respectively
    this routine will determine if they are isomorphic.

    These trees may be either directed or undirected,
    but if they are directed, all edges should flow from the root.

    It returns the isomorphism, a mapping of the nodes of `t1` onto the nodes
    of `t2`, such that two trees are then identical.

    Note that two trees may have more than one isomorphism, and this
    routine just returns one valid mapping.

    Parameters
    ----------
    `t1` :  NetworkX graph
        One of the trees being compared

    `root1` : a node of `t1` which is the root of the tree

    `t2` : undirected NetworkX graph
        The other tree being compared

    `root2` : a node of `t2` which is the root of the tree

    This is a subroutine used to implement `tree_isomorphism`, but will
    be somewhat faster if you already have rooted trees.

    Returns
    -------
    isomorphism : list
        A list of pairs in which the left element is a node in `t1`
        and the right element is a node in `t2`.  The pairs are in
        arbitrary order.  If the nodes in one tree is mapped to the names in
        the other, then trees will be identical. Note that an isomorphism
        will not necessarily be unique.

        If `t1` and `t2` are not isomorphic, then it returns the empty list.
    """
    def tree_hash(T, root):
        labels = {}
        stack = [(root, None)]
        while stack:
            node, parent = stack.pop()
            children = [child for child in T.neighbors(node) if child != parent]
            if not children:
                labels[node] = '()'
            else:
                stack.extend((child, node) for child in children)
        
        while len(labels) < len(T):
            for node in T:
                if node not in labels:
                    children = [child for child in T.neighbors(node) if child != parent]
                    if all(child in labels for child in children):
                        labels[node] = '(' + ','.join(sorted(labels[child] for child in children)) + ')'
        
        return labels[root]

    if tree_hash(t1, root1) != tree_hash(t2, root2):
        return []

    isomorphism = []
    stack = [(root1, root2)]
    while stack:
        n1, n2 = stack.pop()
        isomorphism.append((n1, n2))
        children1 = [c for c in t1.neighbors(n1) if c not in dict(isomorphism)]
        children2 = [c for c in t2.neighbors(n2) if c not in dict(isomorphism).values()]
        
        if len(children1) != len(children2):
            return []
        
        children1.sort(key=lambda x: tree_hash(t1, x))
        children2.sort(key=lambda x: tree_hash(t2, x))
        stack.extend(zip(children1, children2))

    return isomorphism


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable(graphs={'t1': 0, 't2': 1})
def tree_isomorphism(t1, t2):
    """
    Given two undirected (or free) trees `t1` and `t2`,
    this routine will determine if they are isomorphic.
    It returns the isomorphism, a mapping of the nodes of `t1` onto the nodes
    of `t2`, such that two trees are then identical.

    Note that two trees may have more than one isomorphism, and this
    routine just returns one valid mapping.

    Parameters
    ----------
    t1 : undirected NetworkX graph
        One of the trees being compared

    t2 : undirected NetworkX graph
        The other tree being compared

    Returns
    -------
    isomorphism : list
        A list of pairs in which the left element is a node in `t1`
        and the right element is a node in `t2`.  The pairs are in
        arbitrary order.  If the nodes in one tree is mapped to the names in
        the other, then trees will be identical. Note that an isomorphism
        will not necessarily be unique.

        If `t1` and `t2` are not isomorphic, then it returns the empty list.

    Notes
    -----
    This runs in O(n*log(n)) time for trees with n nodes.
    """
    if len(t1) != len(t2):
        return []

    def find_center(T):
        if len(T) <= 2:
            return list(T.nodes())[0]
        leaves = [n for n in T.nodes() if T.degree(n) == 1]
        while len(T) > 2:
            new_leaves = []
            for leaf in leaves:
                neighbor = list(T.neighbors(leaf))[0]
                T.remove_node(leaf)
                if T.degree(neighbor) == 1:
                    new_leaves.append(neighbor)
            leaves = new_leaves
        return leaves[0]

    center1 = find_center(t1.copy())
    center2 = find_center(t2.copy())

    return rooted_tree_isomorphism(t1, center1, t2, center2)
