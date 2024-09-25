"""Functions for computing an approximate minimum weight vertex cover.

A |vertex cover|_ is a subset of nodes such that each edge in the graph
is incident to at least one node in the subset.

.. _vertex cover: https://en.wikipedia.org/wiki/Vertex_cover
.. |vertex cover| replace:: *vertex cover*

"""
import networkx as nx
__all__ = ['min_weighted_vertex_cover']


@nx._dispatchable(node_attrs='weight')
def min_weighted_vertex_cover(G, weight=None):
    """Returns an approximate minimum weighted vertex cover.

    The set of nodes returned by this function is guaranteed to be a
    vertex cover, and the total weight of the set is guaranteed to be at
    most twice the total weight of the minimum weight vertex cover. In
    other words,

    .. math::

       w(S) \\leq 2 * w(S^*),

    where $S$ is the vertex cover returned by this function,
    $S^*$ is the vertex cover of minimum weight out of all vertex
    covers of the graph, and $w$ is the function that computes the
    sum of the weights of each node in that given set.

    Parameters
    ----------
    G : NetworkX graph

    weight : string, optional (default = None)
        If None, every node has weight 1. If a string, use this node
        attribute as the node weight. A node without this attribute is
        assumed to have weight 1.

    Returns
    -------
    min_weighted_cover : set
        Returns a set of nodes whose weight sum is no more than twice
        the weight sum of the minimum weight vertex cover.

    Notes
    -----
    For a directed graph, a vertex cover has the same definition: a set
    of nodes such that each edge in the graph is incident to at least
    one node in the set. Whether the node is the head or tail of the
    directed edge is ignored.

    This is the local-ratio algorithm for computing an approximate
    vertex cover. The algorithm greedily reduces the costs over edges,
    iteratively building a cover. The worst-case runtime of this
    implementation is $O(m \\log n)$, where $n$ is the number
    of nodes and $m$ the number of edges in the graph.

    References
    ----------
    .. [1] Bar-Yehuda, R., and Even, S. (1985). "A local-ratio theorem for
       approximating the weighted vertex cover problem."
       *Annals of Discrete Mathematics*, 25, 27–46
       <http://www.cs.technion.ac.il/~reuven/PDF/vc_lr.pdf>

    """
    import heapq

    # If no weight is provided, use unit weights
    if weight is None:
        weight_func = lambda node: 1
    else:
        weight_func = lambda node: G.nodes[node].get(weight, 1)

    # Create a max heap of edges based on the sum of node weights
    edge_heap = [
        (-weight_func(u) - weight_func(v), (u, v))
        for u, v in G.edges()
    ]
    heapq.heapify(edge_heap)

    # Initialize the vertex cover and node costs
    vertex_cover = set()
    node_costs = {node: weight_func(node) for node in G.nodes()}

    while edge_heap and any(node_costs.values()):
        # Get the edge with the maximum weight
        _, (u, v) = heapq.heappop(edge_heap)

        # If either node has zero cost, skip this edge
        if node_costs[u] == 0 or node_costs[v] == 0:
            continue

        # Add both nodes to the vertex cover
        vertex_cover.update([u, v])

        # Reduce the costs of adjacent nodes
        cost = min(node_costs[u], node_costs[v])
        for node in [u, v]:
            for neighbor in G.neighbors(node):
                if neighbor in node_costs:
                    node_costs[neighbor] = max(0, node_costs[neighbor] - cost)

    return vertex_cover
