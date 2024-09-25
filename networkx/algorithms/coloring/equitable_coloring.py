"""
Equitable coloring of graphs with bounded degree.
"""
from collections import defaultdict
import networkx as nx
__all__ = ['equitable_color']


@nx._dispatchable
def is_coloring(G, coloring):
    """Determine if the coloring is a valid coloring for the graph G."""
    for node, color in coloring.items():
        for neighbor in G[node]:
            if coloring.get(neighbor) == color:
                return False
    return True


@nx._dispatchable
def is_equitable(G, coloring, num_colors=None):
    """Determines if the coloring is valid and equitable for the graph G."""
    if not is_coloring(G, coloring):
        return False
    
    color_counts = defaultdict(int)
    for color in coloring.values():
        color_counts[color] += 1
    
    if num_colors is None:
        num_colors = len(set(coloring.values()))
    
    min_count = min(color_counts.values())
    max_count = max(color_counts.values())
    
    return max_count - min_count <= 1 and len(color_counts) == num_colors


def change_color(u, X, Y, N, H, F, C, L):
    """Change the color of 'u' from X to Y and update N, H, F, C."""
    C[u] = Y
    L[X].remove(u)
    L[Y].append(u)
    for v in N[u]:
        H[v][X] -= 1
        H[v][Y] += 1
        if H[v][X] == 0:
            F[v].remove(X)
        if H[v][Y] == 1:
            F[v].append(Y)


def move_witnesses(src_color, dst_color, N, H, F, C, T_cal, L):
    """Move witness along a path from src_color to dst_color."""
    while src_color != dst_color:
        w = T_cal[src_color]
        next_color = C[w]
        change_color(w, src_color, next_color, N, H, F, C, L)
        src_color = next_color


@nx._dispatchable(mutates_input=True)
def pad_graph(G, num_colors):
    """Add a disconnected complete clique K_p such that the number of nodes in
    the graph becomes a multiple of `num_colors`.

    Assumes that the graph's nodes are labelled using integers.

    Returns the number of nodes with each color.
    """
    n = G.number_of_nodes()
    remainder = n % num_colors
    if remainder == 0:
        return n // num_colors
    
    p = num_colors - remainder
    max_node = max(G.nodes())
    new_nodes = range(max_node + 1, max_node + p + 1)
    G.add_nodes_from(new_nodes)
    
    for i in new_nodes:
        for j in new_nodes:
            if i != j:
                G.add_edge(i, j)
    
    return (n + p) // num_colors


def procedure_P(V_minus, V_plus, N, H, F, C, L, excluded_colors=None):
    """Procedure P as described in the paper."""
    if excluded_colors is None:
        excluded_colors = set()
    
    T_cal = {}
    for X in V_minus:
        if X in excluded_colors:
            continue
        for u in L[X]:
            Y = min(F[u] - excluded_colors - set(T_cal.keys()), default=None)
            if Y is not None and Y in V_plus:
                T_cal[X] = u
                break
    
    if len(T_cal) == min(len(V_minus), len(V_plus)):
        for X in T_cal:
            u = T_cal[X]
            Y = min(F[u] - excluded_colors - set(T_cal.keys()))
            change_color(u, X, Y, N, H, F, C, L)
        return True
    return False


@nx._dispatchable
def equitable_color(G, num_colors):
    """Provides an equitable coloring for nodes of `G`.

    Attempts to color a graph using `num_colors` colors, where no neighbors of
    a node can have same color as the node itself and the number of nodes with
    each color differ by at most 1. `num_colors` must be greater than the
    maximum degree of `G`. The algorithm is described in [1]_ and has
    complexity O(num_colors * n**2).

    Parameters
    ----------
    G : networkX graph
       The nodes of this graph will be colored.

    num_colors : number of colors to use
       This number must be at least one more than the maximum degree of nodes
       in the graph.

    Returns
    -------
    A dictionary with keys representing nodes and values representing
    corresponding coloring.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> nx.coloring.equitable_color(G, num_colors=3)  # doctest: +SKIP
    {0: 2, 1: 1, 2: 2, 3: 0}

    Raises
    ------
    NetworkXAlgorithmError
        If `num_colors` is not at least the maximum degree of the graph `G`

    References
    ----------
    .. [1] Kierstead, H. A., Kostochka, A. V., Mydlarz, M., & Szemerédi, E.
        (2010). A fast algorithm for equitable coloring. Combinatorica, 30(2),
        217-224.
    """
    if num_colors <= max(G.degree())[1]:
        raise nx.NetworkXAlgorithmError(
            f"num_colors must be greater than the maximum degree of G ({max(G.degree())[1]})"
        )

    n = G.number_of_nodes()
    q = pad_graph(G, num_colors)

    N = {u: set(G[u]) for u in G}
    H = {u: defaultdict(int) for u in G}
    F = {u: [] for u in G}
    C = {}
    L = defaultdict(list)

    for u in G:
        C[u] = 0
        L[0].append(u)
        for v in N[u]:
            H[v][0] += 1
        for i in range(1, num_colors):
            F[u].append(i)

    for i in range(num_colors):
        V_minus = [j for j in range(num_colors) if len(L[j]) < q]
        V_plus = [j for j in range(num_colors) if len(L[j]) > q]
        while V_minus and V_plus:
            if not procedure_P(V_minus, V_plus, N, H, F, C, L):
                X = V_minus.pop(0)
                Y = V_plus.pop(0)
                u = L[Y].pop()
                change_color(u, Y, X, N, H, F, C, L)
            V_minus = [j for j in range(num_colors) if len(L[j]) < q]
            V_plus = [j for j in range(num_colors) if len(L[j]) > q]

    return {node: C[node] for node in G.nodes() if node in C}
