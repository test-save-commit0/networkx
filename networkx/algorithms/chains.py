"""Functions for finding chains in a graph."""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['chain_decomposition']


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def chain_decomposition(G, root=None):
    """Returns the chain decomposition of a graph.

    The *chain decomposition* of a graph with respect a depth-first
    search tree is a set of cycles or paths derived from the set of
    fundamental cycles of the tree in the following manner. Consider
    each fundamental cycle with respect to the given tree, represented
    as a list of edges beginning with the nontree edge oriented away
    from the root of the tree. For each fundamental cycle, if it
    overlaps with any previous fundamental cycle, just take the initial
    non-overlapping segment, which is a path instead of a cycle. Each
    cycle or path is called a *chain*. For more information, see [1]_.

    Parameters
    ----------
    G : undirected graph

    root : node (optional)
       A node in the graph `G`. If specified, only the chain
       decomposition for the connected component containing this node
       will be returned. This node indicates the root of the depth-first
       search tree.

    Yields
    ------
    chain : list
       A list of edges representing a chain. There is no guarantee on
       the orientation of the edges in each chain (for example, if a
       chain includes the edge joining nodes 1 and 2, the chain may
       include either (1, 2) or (2, 1)).

    Raises
    ------
    NodeNotFound
       If `root` is not in the graph `G`.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 4), (3, 4), (3, 5), (4, 5)])
    >>> list(nx.chain_decomposition(G))
    [[(4, 5), (5, 3), (3, 4)]]

    Notes
    -----
    The worst-case running time of this implementation is linear in the
    number of nodes and number of edges [1]_.

    References
    ----------
    .. [1] Jens M. Schmidt (2013). "A simple test on 2-vertex-
       and 2-edge-connectivity." *Information Processing Letters*,
       113, 241–244. Elsevier. <https://doi.org/10.1016/j.ipl.2013.01.016>

    """
    if root is not None and root not in G:
        raise nx.NodeNotFound(f"Node {root} is not in the graph.")

    def dfs_edges(node, parent=None):
        visited.add(node)
        for neighbor in G[node]:
            if neighbor not in visited:
                yield node, neighbor
                yield from dfs_edges(neighbor, node)
            elif neighbor != parent and neighbor in visited:
                yield node, neighbor

    def find_chain(start, end):
        chain = [start]
        while chain[-1] != end:
            for edge in tree_edges:
                if edge[0] == chain[-1]:
                    chain.append(edge[1])
                    break
                elif edge[1] == chain[-1]:
                    chain.append(edge[0])
                    break
        return list(zip(chain[:-1], chain[1:]))

    if root is None:
        components = nx.connected_components(G)
    else:
        components = [nx.node_connected_component(G, root)]

    for component in components:
        subgraph = G.subgraph(component)
        start = next(iter(subgraph))
        visited = set()
        tree_edges = list(dfs_edges(start))
        non_tree_edges = set(subgraph.edges()) - set(tree_edges)

        for u, v in non_tree_edges:
            chain = find_chain(u, v)
            chain.append((u, v))
            yield chain
