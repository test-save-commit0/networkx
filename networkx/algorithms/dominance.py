"""
Dominance algorithms.
"""
from functools import reduce
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['immediate_dominators', 'dominance_frontiers']


@not_implemented_for('undirected')
@nx._dispatchable
def immediate_dominators(G, start):
    """Returns the immediate dominators of all nodes of a directed graph.

    Parameters
    ----------
    G : a DiGraph or MultiDiGraph
        The graph where dominance is to be computed.

    start : node
        The start node of dominance computation.

    Returns
    -------
    idom : dict keyed by nodes
        A dict containing the immediate dominators of each node reachable from
        `start`.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is undirected.

    NetworkXError
        If `start` is not in `G`.

    Notes
    -----
    Except for `start`, the immediate dominators are the parents of their
    corresponding nodes in the dominator tree.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 5), (3, 4), (4, 5)])
    >>> sorted(nx.immediate_dominators(G, 1).items())
    [(1, 1), (2, 1), (3, 1), (4, 3), (5, 1)]

    References
    ----------
    .. [1] K. D. Cooper, T. J. Harvey, and K. Kennedy.
           A simple, fast dominance algorithm.
           Software Practice & Experience, 4:110, 2001.
    """
    if start not in G:
        raise nx.NetworkXError(f"Start node {start} is not in G")

    idom = {start: start}
    order = list(nx.dfs_preorder_nodes(G, start))
    dfn = {u: i for i, u in enumerate(order)}
    vertex = {i: v for v, i in dfn.items()}
    semi = dfn.copy()
    parent = dfn.copy()
    pred = {u: set() for u in order}
    ancestor = {}
    label = {}
    dom = {}

    def compress(v):
        if ancestor[v] != ancestor[ancestor[v]]:
            compress(ancestor[v])
            if semi[label[ancestor[v]]] < semi[label[v]]:
                label[v] = label[ancestor[v]]
            ancestor[v] = ancestor[ancestor[v]]

    def eval(v):
        if v not in ancestor:
            return v
        compress(v)
        return label[v]

    for v in reversed(order[1:]):
        for u in G.predecessors(v):
            if u in dfn:
                pred[v].add(u)
                if dfn[u] < dfn[v]:
                    semi[v] = min(semi[v], dfn[u])
                else:
                    semi[v] = min(semi[v], semi[eval(u)])
        ancestor[v] = parent[v]
        dom[semi[v]] = v
        w = dom[semi[v]]
        while w != v:
            if semi[w] >= semi[v]:
                idom[w] = v
            else:
                idom[w] = idom[v]
            w = dom[semi[w]]

    for v in order[1:]:
        if idom[v] != vertex[semi[v]]:
            idom[v] = idom[idom[v]]

    return {v: idom[dfn[v]] for v in G if v in dfn}


@nx._dispatchable
def dominance_frontiers(G, start):
    """Returns the dominance frontiers of all nodes of a directed graph.

    Parameters
    ----------
    G : a DiGraph or MultiDiGraph
        The graph where dominance is to be computed.

    start : node
        The start node of dominance computation.

    Returns
    -------
    df : dict keyed by nodes
        A dict containing the dominance frontiers of each node reachable from
        `start` as lists.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is undirected.

    NetworkXError
        If `start` is not in `G`.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 5), (3, 4), (4, 5)])
    >>> sorted((u, sorted(df)) for u, df in nx.dominance_frontiers(G, 1).items())
    [(1, []), (2, [5]), (3, [5]), (4, [5]), (5, [])]

    References
    ----------
    .. [1] K. D. Cooper, T. J. Harvey, and K. Kennedy.
           A simple, fast dominance algorithm.
           Software Practice & Experience, 4:110, 2001.
    """
    if start not in G:
        raise nx.NetworkXError(f"Start node {start} is not in G")

    idom = immediate_dominators(G, start)
    df = {u: set() for u in G}
    
    # Compute children in the dominator tree
    dom_children = {u: set() for u in G}
    for v, u in idom.items():
        if u != v:
            dom_children[u].add(v)

    def dfs(u):
        for v in G.successors(u):
            if idom[v] != u:
                df[u].add(v)
        for child in dom_children[u]:
            dfs(child)
            df[u].update(v for v in df[child] if idom[v] != u)

    dfs(start)
    return {u: list(frontiers) for u, frontiers in df.items()}
