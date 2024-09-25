"""Attracting components."""
import networkx as nx
from networkx.utils.decorators import not_implemented_for
__all__ = ['number_attracting_components', 'attracting_components',
    'is_attracting_component']


@not_implemented_for('undirected')
@nx._dispatchable
def attracting_components(G):
    """Generates the attracting components in `G`.

    An attracting component in a directed graph `G` is a strongly connected
    component with the property that a random walker on the graph will never
    leave the component, once it enters the component.

    The nodes in attracting components can also be thought of as recurrent
    nodes.  If a random walker enters the attractor containing the node, then
    the node will be visited infinitely often.

    To obtain induced subgraphs on each component use:
    ``(G.subgraph(c).copy() for c in attracting_components(G))``

    Parameters
    ----------
    G : DiGraph, MultiDiGraph
        The graph to be analyzed.

    Returns
    -------
    attractors : generator of sets
        A generator of sets of nodes, one for each attracting component of G.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is undirected.

    See Also
    --------
    number_attracting_components
    is_attracting_component

    """
    scc = nx.strongly_connected_components(G)
    for component in scc:
        if all(G.out_degree(node) == 0 or G.out_degree(node) == sum(1 for _ in G.neighbors(node) if _ in component) for node in component):
            yield component


@not_implemented_for('undirected')
@nx._dispatchable
def number_attracting_components(G):
    """Returns the number of attracting components in `G`.

    Parameters
    ----------
    G : DiGraph, MultiDiGraph
        The graph to be analyzed.

    Returns
    -------
    n : int
        The number of attracting components in G.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is undirected.

    See Also
    --------
    attracting_components
    is_attracting_component

    """
    return sum(1 for _ in attracting_components(G))


@not_implemented_for('undirected')
@nx._dispatchable
def is_attracting_component(G):
    """Returns True if `G` consists of a single attracting component.

    Parameters
    ----------
    G : DiGraph, MultiDiGraph
        The graph to be analyzed.

    Returns
    -------
    attracting : bool
        True if `G` has a single attracting component. Otherwise, False.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is undirected.

    See Also
    --------
    attracting_components
    number_attracting_components

    """
    return number_attracting_components(G) == 1 and len(G) == len(next(attracting_components(G)))
