"""
Stoer-Wagner minimum cut algorithm.
"""
from itertools import islice
import networkx as nx
from ...utils import BinaryHeap, arbitrary_element, not_implemented_for
__all__ = ['stoer_wagner']


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable(edge_attrs='weight')
def stoer_wagner(G, weight='weight', heap=BinaryHeap):
    """Returns the weighted minimum edge cut using the Stoer-Wagner algorithm.

    Determine the minimum edge cut of a connected graph using the
    Stoer-Wagner algorithm. In weighted cases, all weights must be
    nonnegative.

    The running time of the algorithm depends on the type of heaps used:

    ============== =============================================
    Type of heap   Running time
    ============== =============================================
    Binary heap    $O(n (m + n) \\log n)$
    Fibonacci heap $O(nm + n^2 \\log n)$
    Pairing heap   $O(2^{2 \\sqrt{\\log \\log n}} nm + n^2 \\log n)$
    ============== =============================================

    Parameters
    ----------
    G : NetworkX graph
        Edges of the graph are expected to have an attribute named by the
        weight parameter below. If this attribute is not present, the edge is
        considered to have unit weight.

    weight : string
        Name of the weight attribute of the edges. If the attribute is not
        present, unit weight is assumed. Default value: 'weight'.

    heap : class
        Type of heap to be used in the algorithm. It should be a subclass of
        :class:`MinHeap` or implement a compatible interface.

        If a stock heap implementation is to be used, :class:`BinaryHeap` is
        recommended over :class:`PairingHeap` for Python implementations without
        optimized attribute accesses (e.g., CPython) despite a slower
        asymptotic running time. For Python implementations with optimized
        attribute accesses (e.g., PyPy), :class:`PairingHeap` provides better
        performance. Default value: :class:`BinaryHeap`.

    Returns
    -------
    cut_value : integer or float
        The sum of weights of edges in a minimum cut.

    partition : pair of node lists
        A partitioning of the nodes that defines a minimum cut.

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or a multigraph.

    NetworkXError
        If the graph has less than two nodes, is not connected or has a
        negative-weighted edge.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge("x", "a", weight=3)
    >>> G.add_edge("x", "b", weight=1)
    >>> G.add_edge("a", "c", weight=3)
    >>> G.add_edge("b", "c", weight=5)
    >>> G.add_edge("b", "d", weight=4)
    >>> G.add_edge("d", "e", weight=2)
    >>> G.add_edge("c", "y", weight=2)
    >>> G.add_edge("e", "y", weight=3)
    >>> cut_value, partition = nx.stoer_wagner(G)
    >>> cut_value
    4
    """
    if len(G) < 2:
        raise nx.NetworkXError("Graph has less than two nodes.")
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph is not connected.")

    # Check for negative weights
    if any(d.get(weight, 1) < 0 for u, v, d in G.edges(data=True)):
        raise nx.NetworkXError("Graph has a negative-weighted edge.")

    # Initialize the algorithm
    A = {arbitrary_element(G)}
    G_copy = G.copy()
    best_cut_value = float('inf')
    best_partition = None

    while len(G_copy) > 1:
        # Find the most tightly connected node
        cut_value, s, t = minimum_cut_phase(G_copy, A, weight, heap)
        
        # Update the best cut if necessary
        if cut_value < best_cut_value:
            best_cut_value = cut_value
            best_partition = (list(A), list(set(G) - A))

        # Merge the two nodes
        if s != t:
            G_copy = nx.contracted_nodes(G_copy, s, t, self_loops=False)
        A.add(s)

    return best_cut_value, best_partition

def minimum_cut_phase(G, A, weight, heap):
    """Performs a minimum cut phase of the Stoer-Wagner algorithm."""
    n = len(G)
    h = heap()
    seen = set()
    
    # Initialize the heap with the first node from A
    start = next(iter(A))
    h.insert(start, 0)
    
    for _ in range(n - 1):
        # Extract the node with the highest connection to A
        v = h.extract_min()
        seen.add(v)
        
        # Update the connection values for the neighbors
        for u, d in G[v].items():
            if u not in seen:
                w = d.get(weight, 1)
                if u in h:
                    h.decrease_key(u, h[u] - w)
                else:
                    h.insert(u, -w)
    
    # The last two nodes are s and t
    t = v
    s = h.extract_min()
    cut_value = -h[s]
    
    return cut_value, s, t
