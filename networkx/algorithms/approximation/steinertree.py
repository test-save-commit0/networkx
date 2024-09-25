from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
__all__ = ['metric_closure', 'steiner_tree']


@not_implemented_for('directed')
@nx._dispatchable(edge_attrs='weight', returns_graph=True)
def metric_closure(G, weight='weight'):
    """Return the metric closure of a graph.

    The metric closure of a graph *G* is the complete graph in which each edge
    is weighted by the shortest path distance between the nodes in *G* .

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    NetworkX graph
        Metric closure of the graph `G`.

    """
    M = nx.Graph()
    M.add_nodes_from(G)
    for u in G:
        for v in G:
            if u != v:
                d = nx.shortest_path_length(G, u, v, weight=weight)
                M.add_edge(u, v, weight=d)
    return M


def _kou_steiner_tree(G, terminal_nodes, weight='weight'):
    # Step 1: Compute the complete distance graph of the terminal nodes
    M = metric_closure(G, weight=weight)
    H = M.subgraph(terminal_nodes)

    # Step 2: Compute the minimum spanning tree of H
    mst = nx.minimum_spanning_tree(H, weight=weight)

    # Step 3: Compute the subgraph of G by replacing each edge in mst
    # with the corresponding shortest path in G
    steiner_tree = nx.Graph()
    for u, v in mst.edges():
        path = nx.shortest_path(G, u, v, weight=weight)
        nx.add_path(steiner_tree, path, weight=G[path[0]][path[1]][weight])

    # Step 4: Compute the minimum spanning tree of steiner_tree
    return nx.minimum_spanning_tree(steiner_tree, weight=weight)

def _mehlhorn_steiner_tree(G, terminal_nodes, weight='weight'):
    # Step 1: For each non-terminal node, find the closest terminal node
    closest_terminal = {}
    for v in G:
        if v not in terminal_nodes:
            distances = [(t, nx.shortest_path_length(G, v, t, weight=weight)) for t in terminal_nodes]
            closest_terminal[v] = min(distances, key=lambda x: x[1])[0]

    # Step 2: Construct the complete graph on terminal nodes
    H = nx.Graph()
    for u in terminal_nodes:
        for v in terminal_nodes:
            if u != v:
                path = nx.shortest_path(G, u, v, weight=weight)
                distance = sum(G[path[i]][path[i+1]][weight] for i in range(len(path)-1))
                H.add_edge(u, v, weight=distance)

    # Step 3: Find the minimum spanning tree of H
    mst = nx.minimum_spanning_tree(H, weight=weight)

    # Step 4: Expand the tree to include the shortest paths in G
    steiner_tree = nx.Graph()
    for u, v in mst.edges():
        path = nx.shortest_path(G, u, v, weight=weight)
        nx.add_path(steiner_tree, path, weight=G[path[0]][path[1]][weight])

    # Step 5: Remove non-terminal leaves
    while True:
        leaves = [node for node in steiner_tree if steiner_tree.degree(node) == 1]
        non_terminal_leaves = [leaf for leaf in leaves if leaf not in terminal_nodes]
        if not non_terminal_leaves:
            break
        for leaf in non_terminal_leaves:
            steiner_tree.remove_node(leaf)

    return steiner_tree

ALGORITHMS = {'kou': _kou_steiner_tree, 'mehlhorn': _mehlhorn_steiner_tree}


@not_implemented_for('directed')
@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def steiner_tree(G, terminal_nodes, weight='weight', method=None):
    """Return an approximation to the minimum Steiner tree of a graph.

    The minimum Steiner tree of `G` w.r.t a set of `terminal_nodes` (also *S*)
    is a tree within `G` that spans those nodes and has minimum size (sum of
    edge weights) among all such trees.

    The approximation algorithm is specified with the `method` keyword
    argument. All three available algorithms produce a tree whose weight is
    within a ``(2 - (2 / l))`` factor of the weight of the optimal Steiner tree,
    where ``l`` is the minimum number of leaf nodes across all possible Steiner
    trees.

    * ``"kou"`` [2]_ (runtime $O(|S| |V|^2)$) computes the minimum spanning tree of
      the subgraph of the metric closure of *G* induced by the terminal nodes,
      where the metric closure of *G* is the complete graph in which each edge is
      weighted by the shortest path distance between the nodes in *G*.

    * ``"mehlhorn"`` [3]_ (runtime $O(|E|+|V|\\log|V|)$) modifies Kou et al.'s
      algorithm, beginning by finding the closest terminal node for each
      non-terminal. This data is used to create a complete graph containing only
      the terminal nodes, in which edge is weighted with the shortest path
      distance between them. The algorithm then proceeds in the same way as Kou
      et al..

    Parameters
    ----------
    G : NetworkX graph

    terminal_nodes : list
         A list of terminal nodes for which minimum steiner tree is
         to be found.

    weight : string (default = 'weight')
        Use the edge attribute specified by this string as the edge weight.
        Any edge attribute not present defaults to 1.

    method : string, optional (default = 'mehlhorn')
        The algorithm to use to approximate the Steiner tree.
        Supported options: 'kou', 'mehlhorn'.
        Other inputs produce a ValueError.

    Returns
    -------
    NetworkX graph
        Approximation to the minimum steiner tree of `G` induced by
        `terminal_nodes` .

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed.

    ValueError
        If the specified `method` is not supported.

    Notes
    -----
    For multigraphs, the edge between two nodes with minimum weight is the
    edge put into the Steiner tree.


    References
    ----------
    .. [1] Steiner_tree_problem on Wikipedia.
           https://en.wikipedia.org/wiki/Steiner_tree_problem
    .. [2] Kou, L., G. Markowsky, and L. Berman. 1981.
           ‘A Fast Algorithm for Steiner Trees’.
           Acta Informatica 15 (2): 141–45.
           https://doi.org/10.1007/BF00288961.
    .. [3] Mehlhorn, Kurt. 1988.
           ‘A Faster Approximation Algorithm for the Steiner Problem in Graphs’.
           Information Processing Letters 27 (3): 125–28.
           https://doi.org/10.1016/0020-0190(88)90066-X.
    """
    pass
