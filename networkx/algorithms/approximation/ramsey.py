"""
Ramsey numbers.
"""
import networkx as nx
from networkx.utils import not_implemented_for
from ...utils import arbitrary_element
__all__ = ['ramsey_R2']


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def ramsey_R2(G):
    """Compute the largest clique and largest independent set in `G`.

    This can be used to estimate bounds for the 2-color
    Ramsey number `R(2;s,t)` for `G`.

    This is a recursive implementation which could run into trouble
    for large recursions. Note that self-loop edges are ignored.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    max_pair : (set, set) tuple
        Maximum clique, Maximum independent set.

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.
    """
    if len(G) == 0:
        return set(), set()

    v = arbitrary_element(G)
    G_v = G.subgraph(set(G) - {v} - set(G[v]))
    clique_G_v, indep_G_v = ramsey_R2(G_v)
    clique_N_v, indep_N_v = ramsey_R2(G.subgraph(G[v]))

    if len(clique_N_v) > len(indep_G_v):
        clique = {v} | clique_N_v
        indep = indep_G_v
    else:
        clique = clique_G_v
        indep = {v} | indep_G_v

    return clique, indep
