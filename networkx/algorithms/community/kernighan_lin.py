"""Functions for computing the Kernighan–Lin bipartition algorithm."""
from itertools import count
import networkx as nx
from networkx.algorithms.community.community_utils import is_partition
from networkx.utils import BinaryHeap, not_implemented_for, py_random_state
__all__ = ['kernighan_lin_bisection']


def _kernighan_lin_sweep(edges, side):
    """
    This is a modified form of Kernighan-Lin, which moves single nodes at a
    time, alternating between sides to keep the bisection balanced.  We keep
    two min-heaps of swap costs to make optimal-next-move selection fast.
    """
    n = len(side)
    heap_left = BinaryHeap()
    heap_right = BinaryHeap()
    side_cost = [0] * n
    
    # Initialize the heaps and side_cost
    for u, neighbors in enumerate(edges):
        cost = sum(w for v, w in neighbors.items() if side[v] != side[u])
        side_cost[u] = cost
        if side[u]:
            heap_left.insert(u, -cost)
        else:
            heap_right.insert(u, -cost)
    
    swaps = []
    gains = []
    total_gain = 0
    
    for _ in range(n):
        if len(heap_left) == 0 or len(heap_right) == 0:
            break
        
        left_node, left_cost = heap_left.pop()
        right_node, right_cost = heap_right.pop()
        
        left_cost = -left_cost
        right_cost = -right_cost
        
        gain = left_cost + right_cost
        
        for v, w in edges[left_node].items():
            if side[v] != side[left_node]:
                side_cost[v] -= w
            else:
                side_cost[v] += w
        
        for v, w in edges[right_node].items():
            if side[v] != side[right_node]:
                side_cost[v] -= w
            else:
                side_cost[v] += w
        
        side[left_node] = not side[left_node]
        side[right_node] = not side[right_node]
        
        # Update heaps
        for v in range(n):
            if v != left_node and v != right_node:
                if side[v]:
                    heap_left.insert(v, -side_cost[v])
                else:
                    heap_right.insert(v, -side_cost[v])
        
        swaps.append((left_node, right_node))
        gains.append(gain)
        total_gain += gain
    
    if total_gain > 0:
        max_gain_index = gains.index(max(gains))
        return swaps[:max_gain_index + 1]
    return []


@not_implemented_for('directed')
@py_random_state(4)
@nx._dispatchable(edge_attrs='weight')
def kernighan_lin_bisection(G, partition=None, max_iter=10, weight='weight',
    seed=None):
    """Partition a graph into two blocks using the Kernighan–Lin
    algorithm.

    This algorithm partitions a network into two sets by iteratively
    swapping pairs of nodes to reduce the edge cut between the two sets.  The
    pairs are chosen according to a modified form of Kernighan-Lin [1]_, which
    moves node individually, alternating between sides to keep the bisection
    balanced.

    Parameters
    ----------
    G : NetworkX graph
        Graph must be undirected.

    partition : tuple
        Pair of iterables containing an initial partition. If not
        specified, a random balanced partition is used.

    max_iter : int
        Maximum number of times to attempt swaps to find an
        improvement before giving up.

    weight : key
        Edge data key to use as weight. If None, the weights are all
        set to one.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Only used if partition is None

    Returns
    -------
    partition : tuple
        A pair of sets of nodes representing the bipartition.

    Raises
    ------
    NetworkXError
        If partition is not a valid partition of the nodes of the graph.

    References
    ----------
    .. [1] Kernighan, B. W.; Lin, Shen (1970).
       "An efficient heuristic procedure for partitioning graphs."
       *Bell Systems Technical Journal* 49: 291--307.
       Oxford University Press 2011.

    """
    # Check if the graph is directed
    if G.is_directed():
        raise nx.NetworkXError("Kernighan-Lin algorithm not defined for directed graphs.")

    # Create initial partition if not provided
    if partition is None:
        nodes = list(G.nodes())
        random_state = seed if seed is not None else nx.utils.create_random_state()
        random_state.shuffle(nodes)
        half = len(nodes) // 2
        partition = (set(nodes[:half]), set(nodes[half:]))
    else:
        partition = (set(partition[0]), set(partition[1]))

    # Validate the partition
    if not is_partition(G, partition):
        raise nx.NetworkXError("partition is not a valid partition of the graph")

    # Create a mapping of nodes to their partition (True for left, False for right)
    side = {node: True for node in partition[0]}
    side.update({node: False for node in partition[1]})

    # Create a list of weighted edge dictionaries
    edges = [
        {v: G[u][v].get(weight, 1) for v in G[u]} for u in G.nodes()
    ]

    for _ in range(max_iter):
        swaps = _kernighan_lin_sweep(edges, side)
        if not swaps:
            break

        for u, v in swaps:
            side[u], side[v] = side[v], side[u]

    # Create the final partition based on the side mapping
    final_partition = (set(u for u, s in side.items() if s),
                       set(u for u, s in side.items() if not s))

    return final_partition
