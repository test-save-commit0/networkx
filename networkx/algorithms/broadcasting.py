"""Routines to calculate the broadcast time of certain graphs.

Broadcasting is an information dissemination problem in which a node in a graph,
called the originator, must distribute a message to all other nodes by placing
a series of calls along the edges of the graph. Once informed, other nodes aid
the originator in distributing the message.

The broadcasting must be completed as quickly as possible subject to the
following constraints:
- Each call requires one unit of time.
- A node can only participate in one call per unit of time.
- Each call only involves two adjacent nodes: a sender and a receiver.
"""
import networkx as nx
from networkx import NetworkXError
from networkx.utils import not_implemented_for
__all__ = ['tree_broadcast_center', 'tree_broadcast_time']


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def tree_broadcast_center(G):
    """Return the Broadcast Center of the tree `G`.

    The broadcast center of a graph G denotes the set of nodes having
    minimum broadcast time [1]_. This is a linear algorithm for determining
    the broadcast center of a tree with ``N`` nodes, as a by-product it also
    determines the broadcast time from the broadcast center.

    Parameters
    ----------
    G : undirected graph
        The graph should be an undirected tree

    Returns
    -------
    BC : (int, set) tuple
        minimum broadcast number of the tree, set of broadcast centers

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    References
    ----------
    .. [1] Slater, P.J., Cockayne, E.J., Hedetniemi, S.T,
       Information dissemination in trees. SIAM J.Comput. 10(4), 692–701 (1981)
    """
    if not nx.is_tree(G):
        raise NetworkXError("The graph G must be a tree.")

    def dfs(node, parent):
        max_subtree_height = 0
        for neighbor in G[node]:
            if neighbor != parent:
                subtree_height = dfs(neighbor, node)
                max_subtree_height = max(max_subtree_height, subtree_height)
        return max_subtree_height + 1

    # First DFS to find the height of each subtree
    root = next(iter(G))  # Choose an arbitrary root
    heights = {node: dfs(node, None) for node in G}

    # Second DFS to find the broadcast centers
    def find_centers(node, parent, depth):
        nonlocal min_broadcast_time, broadcast_centers
        max_distance = max(depth, heights[node] - 1)
        
        if max_distance < min_broadcast_time:
            min_broadcast_time = max_distance
            broadcast_centers = {node}
        elif max_distance == min_broadcast_time:
            broadcast_centers.add(node)

        for neighbor in G[node]:
            if neighbor != parent:
                find_centers(neighbor, node, max(depth + 1, heights[node] - 1))

    min_broadcast_time = float('inf')
    broadcast_centers = set()
    find_centers(root, None, 0)

    return min_broadcast_time, broadcast_centers


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def tree_broadcast_time(G, node=None):
    """Return the Broadcast Time of the tree `G`.

    The minimum broadcast time of a node is defined as the minimum amount
    of time required to complete broadcasting starting from the
    originator. The broadcast time of a graph is the maximum over
    all nodes of the minimum broadcast time from that node [1]_.
    This function returns the minimum broadcast time of `node`.
    If `node` is None the broadcast time for the graph is returned.

    Parameters
    ----------
    G : undirected graph
        The graph should be an undirected tree
    node: int, optional
        index of starting node. If `None`, the algorithm returns the broadcast
        time of the tree.

    Returns
    -------
    BT : int
        Broadcast Time of a node in a tree

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    References
    ----------
    .. [1] Harutyunyan, H. A. and Li, Z.
        "A Simple Construction of Broadcast Graphs."
        In Computing and Combinatorics. COCOON 2019
        (Ed. D. Z. Du and C. Tian.) Springer, pp. 240-253, 2019.
    """
    if not nx.is_tree(G):
        raise NetworkXError("The graph G must be a tree.")

    def dfs(current, parent):
        max_depth = 0
        for neighbor in G[current]:
            if neighbor != parent:
                depth = dfs(neighbor, current)
                max_depth = max(max_depth, depth + 1)
        return max_depth

    if node is None:
        # If no node is specified, return the broadcast time of the tree
        return max(dfs(n, None) for n in G)
    else:
        # If a node is specified, return its broadcast time
        if node not in G:
            raise NetworkXError(f"Node {node} is not in the graph.")
        return dfs(node, None)
