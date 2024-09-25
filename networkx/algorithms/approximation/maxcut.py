import networkx as nx
from networkx.utils.decorators import not_implemented_for, py_random_state
__all__ = ['randomized_partitioning', 'one_exchange']


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(1)
@nx._dispatchable(edge_attrs='weight')
def randomized_partitioning(G, seed=None, p=0.5, weight=None):
    """Compute a random partitioning of the graph nodes and its cut value.

    A partitioning is calculated by observing each node
    and deciding to add it to the partition with probability `p`,
    returning a random cut and its corresponding value (the
    sum of weights of edges connecting different partitions).

    Parameters
    ----------
    G : NetworkX graph

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    p : scalar
        Probability for each node to be part of the first partition.
        Should be in [0,1]

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    cut_size : scalar
        Value of the minimum cut.

    partition : pair of node sets
        A partitioning of the nodes that defines a minimum cut.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> cut_size, partition = nx.approximation.randomized_partitioning(G, seed=1)
    >>> cut_size
    6
    >>> partition
    ({0, 3, 4}, {1, 2})

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.
    """
    if not 0 <= p <= 1:
        raise ValueError("p must be in the range [0, 1]")

    nodes = list(G.nodes())
    partition_1 = set()
    partition_2 = set()

    # Randomly assign nodes to partitions
    for node in nodes:
        if seed.random() < p:
            partition_1.add(node)
        else:
            partition_2.add(node)

    # Calculate the cut size
    cut_size = 0
    for u, v, edge_data in G.edges(data=True):
        if weight is None:
            edge_weight = 1
        else:
            edge_weight = edge_data.get(weight, 1)

        if (u in partition_1 and v in partition_2) or (u in partition_2 and v in partition_1):
            cut_size += edge_weight

    return cut_size, (partition_1, partition_2)


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(2)
@nx._dispatchable(edge_attrs='weight')
def one_exchange(G, initial_cut=None, seed=None, weight=None):
    """Compute a partitioning of the graphs nodes and the corresponding cut value.

    Use a greedy one exchange strategy to find a locally maximal cut
    and its value, it works by finding the best node (one that gives
    the highest gain to the cut value) to add to the current cut
    and repeats this process until no improvement can be made.

    Parameters
    ----------
    G : networkx Graph
        Graph to find a maximum cut for.

    initial_cut : set
        Cut to use as a starting point. If not supplied the algorithm
        starts with an empty cut.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    cut_value : scalar
        Value of the maximum cut.

    partition : pair of node sets
        A partitioning of the nodes that defines a maximum cut.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> curr_cut_size, partition = nx.approximation.one_exchange(G, seed=1)
    >>> curr_cut_size
    6
    >>> partition
    ({0, 2}, {1, 3, 4})

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.
    """
    nodes = list(G.nodes())
    if initial_cut is None:
        initial_cut = set()
    else:
        initial_cut = set(initial_cut)

    complement = set(nodes) - initial_cut

    def calculate_cut_value(cut):
        cut_value = 0
        for u, v, edge_data in G.edges(data=True):
            if weight is None:
                edge_weight = 1
            else:
                edge_weight = edge_data.get(weight, 1)

            if (u in cut and v not in cut) or (u not in cut and v in cut):
                cut_value += edge_weight
        return cut_value

    current_cut = initial_cut.copy()
    current_cut_value = calculate_cut_value(current_cut)

    improved = True
    while improved:
        improved = False
        for node in nodes:
            # Try moving the node to the other partition
            if node in current_cut:
                new_cut = current_cut - {node}
            else:
                new_cut = current_cut | {node}

            new_cut_value = calculate_cut_value(new_cut)

            if new_cut_value > current_cut_value:
                current_cut = new_cut
                current_cut_value = new_cut_value
                improved = True
                break

    return current_cut_value, (current_cut, set(nodes) - current_cut)
