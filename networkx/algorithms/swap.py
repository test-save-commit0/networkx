"""Swap edges in a graph.
"""
import math
import networkx as nx
from networkx.utils import py_random_state
__all__ = ['double_edge_swap', 'connected_double_edge_swap',
    'directed_edge_swap']


@nx.utils.not_implemented_for('undirected')
@py_random_state(3)
@nx._dispatchable(mutates_input=True, returns_graph=True)
def directed_edge_swap(G, *, nswap=1, max_tries=100, seed=None):
    """Swap three edges in a directed graph while keeping the node degrees fixed.

    A directed edge swap swaps three edges such that a -> b -> c -> d becomes
    a -> c -> b -> d. This pattern of swapping allows all possible states with the
    same in- and out-degree distribution in a directed graph to be reached.

    If the swap would create parallel edges (e.g. if a -> c already existed in the
    previous example), another attempt is made to find a suitable trio of edges.

    Parameters
    ----------
    G : DiGraph
       A directed graph

    nswap : integer (optional, default=1)
       Number of three-edge (directed) swaps to perform

    max_tries : integer (optional, default=100)
       Maximum number of attempts to swap edges

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : DiGraph
       The graph after the edges are swapped.

    Raises
    ------
    NetworkXError
        If `G` is not directed, or
        If nswap > max_tries, or
        If there are fewer than 4 nodes or 3 edges in `G`.
    NetworkXAlgorithmError
        If the number of swap attempts exceeds `max_tries` before `nswap` swaps are made

    Notes
    -----
    Does not enforce any connectivity constraints.

    The graph G is modified in place.

    A later swap is allowed to undo a previous swap.

    References
    ----------
    .. [1] Erdős, Péter L., et al. “A Simple Havel-Hakimi Type Algorithm to Realize
           Graphical Degree Sequences of Directed Graphs.” ArXiv:0905.4913 [Math],
           Jan. 2010. https://doi.org/10.48550/arXiv.0905.4913.
           Published  2010 in Elec. J. Combinatorics (17(1)). R66.
           http://www.combinatorics.org/Volume_17/PDF/v17i1r66.pdf
    .. [2] “Combinatorics - Reaching All Possible Simple Directed Graphs with a given
           Degree Sequence with 2-Edge Swaps.” Mathematics Stack Exchange,
           https://math.stackexchange.com/questions/22272/. Accessed 30 May 2022.
    """
    pass


@py_random_state(3)
@nx._dispatchable(mutates_input=True, returns_graph=True)
def double_edge_swap(G, nswap=1, max_tries=100, seed=None):
    """Swap two edges in the graph while keeping the node degrees fixed.

    A double-edge swap removes two randomly chosen edges u-v and x-y
    and creates the new edges u-x and v-y::

     u--v            u  v
            becomes  |  |
     x--y            x  y

    If either the edge u-x or v-y already exist no swap is performed
    and another attempt is made to find a suitable edge pair.

    Parameters
    ----------
    G : graph
       An undirected graph

    nswap : integer (optional, default=1)
       Number of double-edge swaps to perform

    max_tries : integer (optional)
       Maximum number of attempts to swap edges

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : graph
       The graph after double edge swaps.

    Raises
    ------
    NetworkXError
        If `G` is directed, or
        If `nswap` > `max_tries`, or
        If there are fewer than 4 nodes or 2 edges in `G`.
    NetworkXAlgorithmError
        If the number of swap attempts exceeds `max_tries` before `nswap` swaps are made

    Notes
    -----
    Does not enforce any connectivity constraints.

    The graph G is modified in place.
    """
    if G.is_directed():
        raise nx.NetworkXError("Graph must be undirected.")
    
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    
    if len(G) < 4 or G.number_of_edges() < 2:
        raise nx.NetworkXError("Graph has fewer than four nodes or fewer than two edges.")
    
    swapcount = 0
    tries = 0
    
    while swapcount < nswap and tries < max_tries:
        tries += 1
        u, v = seed.choice(list(G.edges()))
        x, y = seed.choice(list(G.edges()))
        
        # Ensure we have four distinct nodes
        if len({u, v, x, y}) < 4:
            continue
        
        # Check if the swap would create parallel edges
        if (u, x) not in G.edges() and (v, y) not in G.edges():
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            G.add_edge(u, x)
            G.add_edge(v, y)
            swapcount += 1
    
    if tries >= max_tries:
        raise nx.NetworkXAlgorithmError(f"Maximum number of swap attempts ({max_tries}) exceeded before desired swaps achieved ({nswap}).")
    
    return G


@py_random_state(3)
@nx._dispatchable(mutates_input=True)
def connected_double_edge_swap(G, nswap=1, _window_threshold=3, seed=None):
    """Attempts the specified number of double-edge swaps in the graph `G`.

    A double-edge swap removes two randomly chosen edges `(u, v)` and `(x,
    y)` and creates the new edges `(u, x)` and `(v, y)`::

     u--v            u  v
            becomes  |  |
     x--y            x  y

    If either `(u, x)` or `(v, y)` already exist, then no swap is performed
    so the actual number of swapped edges is always *at most* `nswap`.

    Parameters
    ----------
    G : graph
       An undirected graph

    nswap : integer (optional, default=1)
       Number of double-edge swaps to perform

    _window_threshold : integer

       The window size below which connectedness of the graph will be checked
       after each swap.

       The "window" in this function is a dynamically updated integer that
       represents the number of swap attempts to make before checking if the
       graph remains connected. It is an optimization used to decrease the
       running time of the algorithm in exchange for increased complexity of
       implementation.

       If the window size is below this threshold, then the algorithm checks
       after each swap if the graph remains connected by checking if there is a
       path joining the two nodes whose edge was just removed. If the window
       size is above this threshold, then the algorithm performs do all the
       swaps in the window and only then check if the graph is still connected.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    int
       The number of successful swaps

    Raises
    ------

    NetworkXError

       If the input graph is not connected, or if the graph has fewer than four
       nodes.

    Notes
    -----

    The initial graph `G` must be connected, and the resulting graph is
    connected. The graph `G` is modified in place.

    References
    ----------
    .. [1] C. Gkantsidis and M. Mihail and E. Zegura,
           The Markov chain simulation method for generating connected
           power law random graphs, 2003.
           http://citeseer.ist.psu.edu/gkantsidis03markov.html
    """
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph not connected")
    
    if len(G) < 4:
        raise nx.NetworkXError("Graph has fewer than four nodes.")
    
    window = 1
    swapcount = 0
    edges = list(G.edges())
    nodes = list(G.nodes())
    
    for i in range(nswap):
        for j in range(window):
            u, v = seed.choice(edges)
            x, y = seed.choice(edges)
            
            # Ensure we have four distinct nodes
            if len({u, v, x, y}) < 4:
                continue
            
            # Check if the swap would create parallel edges
            if (u, x) not in G.edges() and (v, y) not in G.edges():
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                G.add_edge(u, x)
                G.add_edge(v, y)
                edges.remove((u, v))
                edges.remove((x, y))
                edges.append((u, x))
                edges.append((v, y))
                swapcount += 1
        
        if window < _window_threshold:
            # Check if the graph is still connected
            if not nx.has_path(G, u, v) or not nx.has_path(G, x, y):
                # If not, undo the last swap
                G.remove_edge(u, x)
                G.remove_edge(v, y)
                G.add_edge(u, v)
                G.add_edge(x, y)
                edges.remove((u, x))
                edges.remove((v, y))
                edges.append((u, v))
                edges.append((x, y))
                swapcount -= 1
        elif window == _window_threshold:
            # Check if the graph is still connected
            if not nx.is_connected(G):
                # If not, undo all swaps in the window
                for _ in range(window):
                    u, x = edges.pop()
                    v, y = edges.pop()
                    G.remove_edge(u, x)
                    G.remove_edge(v, y)
                    G.add_edge(u, v)
                    G.add_edge(x, y)
                    edges.append((u, v))
                    edges.append((x, y))
                    swapcount -= 1
        
        # Update window size
        if window < len(G):
            window += 1
    
    return swapcount
