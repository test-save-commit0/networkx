"""Trophic levels"""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['trophic_levels', 'trophic_differences',
    'trophic_incoherence_parameter']


@not_implemented_for('undirected')
@nx._dispatchable(edge_attrs='weight')
def trophic_levels(G, weight='weight'):
    """Compute the trophic levels of nodes.

    The trophic level of a node $i$ is

    .. math::

        s_i = 1 + \\frac{1}{k^{in}_i} \\sum_{j} a_{ij} s_j

    where $k^{in}_i$ is the in-degree of i

    .. math::

        k^{in}_i = \\sum_{j} a_{ij}

    and nodes with $k^{in}_i = 0$ have $s_i = 1$ by convention.

    These are calculated using the method outlined in Levine [1]_.

    Parameters
    ----------
    G : DiGraph
        A directed networkx graph

    Returns
    -------
    nodes : dict
        Dictionary of nodes with trophic level as the value.

    References
    ----------
    .. [1] Stephen Levine (1980) J. theor. Biol. 83, 195-207
    """
    trophic_levels = {}
    in_degree = dict(G.in_degree(weight=weight))
    
    # Initialize trophic levels
    for node in G.nodes():
        if in_degree[node] == 0:
            trophic_levels[node] = 1
        else:
            trophic_levels[node] = 0
    
    # Iteratively update trophic levels until convergence
    converged = False
    while not converged:
        old_levels = trophic_levels.copy()
        for node in G.nodes():
            if in_degree[node] > 0:
                trophic_levels[node] = 1 + sum(trophic_levels[pred] * G[pred][node].get(weight, 1) 
                                               for pred in G.predecessors(node)) / in_degree[node]
        
        # Check for convergence
        converged = all(abs(trophic_levels[node] - old_levels[node]) < 1e-6 for node in G.nodes())
    
    return trophic_levels


@not_implemented_for('undirected')
@nx._dispatchable(edge_attrs='weight')
def trophic_differences(G, weight='weight'):
    """Compute the trophic differences of the edges of a directed graph.

    The trophic difference $x_ij$ for each edge is defined in Johnson et al.
    [1]_ as:

    .. math::
        x_ij = s_j - s_i

    Where $s_i$ is the trophic level of node $i$.

    Parameters
    ----------
    G : DiGraph
        A directed networkx graph

    Returns
    -------
    diffs : dict
        Dictionary of edges with trophic differences as the value.

    References
    ----------
    .. [1] Samuel Johnson, Virginia Dominguez-Garcia, Luca Donetti, Miguel A.
        Munoz (2014) PNAS "Trophic coherence determines food-web stability"
    """
    trophic_levels = trophic_levels(G, weight=weight)
    diffs = {}
    
    for u, v in G.edges():
        diffs[(u, v)] = trophic_levels[v] - trophic_levels[u]
    
    return diffs


@not_implemented_for('undirected')
@nx._dispatchable(edge_attrs='weight')
def trophic_incoherence_parameter(G, weight='weight', cannibalism=False):
    """Compute the trophic incoherence parameter of a graph.

    Trophic coherence is defined as the homogeneity of the distribution of
    trophic distances: the more similar, the more coherent. This is measured by
    the standard deviation of the trophic differences and referred to as the
    trophic incoherence parameter $q$ by [1].

    Parameters
    ----------
    G : DiGraph
        A directed networkx graph

    cannibalism: Boolean
        If set to False, self edges are not considered in the calculation

    Returns
    -------
    trophic_incoherence_parameter : float
        The trophic coherence of a graph

    References
    ----------
    .. [1] Samuel Johnson, Virginia Dominguez-Garcia, Luca Donetti, Miguel A.
        Munoz (2014) PNAS "Trophic coherence determines food-web stability"
    """
    import math
    
    diffs = trophic_differences(G, weight=weight)
    
    if not cannibalism:
        diffs = {edge: diff for edge, diff in diffs.items() if edge[0] != edge[1]}
    
    if not diffs:
        return 0.0
    
    mean_diff = sum(diffs.values()) / len(diffs)
    variance = sum((diff - mean_diff) ** 2 for diff in diffs.values()) / len(diffs)
    
    return math.sqrt(variance)
