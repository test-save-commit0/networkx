"""Lukes Algorithm for exact optimal weighted tree partitioning."""
from copy import deepcopy
from functools import lru_cache
from random import choice
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['lukes_partitioning']
D_EDGE_W = 'weight'
D_EDGE_VALUE = 1.0
D_NODE_W = 'weight'
D_NODE_VALUE = 1
PKEY = 'partitions'
CLUSTER_EVAL_CACHE_SIZE = 2048


@nx._dispatchable(node_attrs='node_weight', edge_attrs='edge_weight')
def lukes_partitioning(G, max_size, node_weight=None, edge_weight=None):
    """Optimal partitioning of a weighted tree using the Lukes algorithm.

    This algorithm partitions a connected, acyclic graph featuring integer
    node weights and float edge weights. The resulting clusters are such
    that the total weight of the nodes in each cluster does not exceed
    max_size and that the weight of the edges that are cut by the partition
    is minimum. The algorithm is based on [1]_.

    Parameters
    ----------
    G : NetworkX graph

    max_size : int
        Maximum weight a partition can have in terms of sum of
        node_weight for all nodes in the partition

    edge_weight : key
        Edge data key to use as weight. If None, the weights are all
        set to one.

    node_weight : key
        Node data key to use as weight. If None, the weights are all
        set to one. The data must be int.

    Returns
    -------
    partition : list
        A list of sets of nodes representing the clusters of the
        partition.

    Raises
    ------
    NotATree
        If G is not a tree.
    TypeError
        If any of the values of node_weight is not int.

    References
    ----------
    .. [1] Lukes, J. A. (1974).
       "Efficient Algorithm for the Partitioning of Trees."
       IBM Journal of Research and Development, 18(3), 217–224.

    """
    if not nx.is_tree(G):
        raise nx.NotATree("Input graph is not a tree.")

    # Set default weights if not provided
    if edge_weight is None:
        edge_weight = D_EDGE_W
        nx.set_edge_attributes(G, D_EDGE_VALUE, D_EDGE_W)
    if node_weight is None:
        node_weight = D_NODE_W
        nx.set_node_attributes(G, D_NODE_VALUE, D_NODE_W)

    # Check if node weights are integers
    for node, data in G.nodes(data=True):
        if not isinstance(data.get(node_weight, D_NODE_VALUE), int):
            raise TypeError(f"Node weight for node {node} is not an integer.")

    # Choose an arbitrary root
    root = next(iter(G.nodes()))

    # Initialize memoization cache
    memo = {}

    def dp(node, parent, remaining_size):
        if (node, remaining_size) in memo:
            return memo[(node, remaining_size)]

        node_w = G.nodes[node].get(node_weight, D_NODE_VALUE)
        if node_w > remaining_size:
            return float('inf'), []

        children = [child for child in G.neighbors(node) if child != parent]
        if not children:
            return 0, [{node}]

        best_cut = float('inf')
        best_partition = []

        for size in range(node_w, remaining_size + 1):
            cut, partition = 0, [{node}]
            for child in children:
                child_cut, child_partition = dp(child, node, size - node_w)
                cut += child_cut
                cut += G[node][child].get(edge_weight, D_EDGE_VALUE)
                partition.extend(child_partition)

            remaining_cut, remaining_partition = dp(children[0], node, max_size)
            for child in children[1:]:
                child_cut, child_partition = dp(child, node, max_size)
                remaining_cut += child_cut
                remaining_partition.extend(child_partition)

            total_cut = cut + remaining_cut
            if total_cut < best_cut:
                best_cut = total_cut
                best_partition = partition + remaining_partition

        memo[(node, remaining_size)] = (best_cut, best_partition)
        return best_cut, best_partition

    _, partition = dp(root, None, max_size)
    return partition
