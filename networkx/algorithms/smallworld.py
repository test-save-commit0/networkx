"""Functions for estimating the small-world-ness of graphs.

A small world network is characterized by a small average shortest path length,
and a large clustering coefficient.

Small-worldness is commonly measured with the coefficient sigma or omega.

Both coefficients compare the average clustering coefficient and shortest path
length of a given graph against the same quantities for an equivalent random
or lattice graph.

For more information, see the Wikipedia article on small-world network [1]_.

.. [1] Small-world network:: https://en.wikipedia.org/wiki/Small-world_network

"""
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
__all__ = ['random_reference', 'lattice_reference', 'sigma', 'omega']


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(3)
@nx._dispatchable(returns_graph=True)
def random_reference(G, niter=1, connectivity=True, seed=None):
    """Compute a random graph by swapping edges of a given graph.

    Parameters
    ----------
    G : graph
        An undirected graph with 4 or more nodes.

    niter : integer (optional, default=1)
        An edge is rewired approximately `niter` times.

    connectivity : boolean (optional, default=True)
        When True, ensure connectivity for the randomized graph.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : graph
        The randomized graph.

    Raises
    ------
    NetworkXError
        If there are fewer than 4 nodes or 2 edges in `G`

    Notes
    -----
    The implementation is adapted from the algorithm by Maslov and Sneppen
    (2002) [1]_.

    References
    ----------
    .. [1] Maslov, Sergei, and Kim Sneppen.
           "Specificity and stability in topology of protein networks."
           Science 296.5569 (2002): 910-913.
    """
    import random

    if len(G) < 4:
        raise nx.NetworkXError("Graph has fewer than four nodes.")
    if G.number_of_edges() < 2:
        raise nx.NetworkXError("Graph has fewer than two edges.")

    G = G.copy()
    edges = list(G.edges())
    nodes = list(G.nodes())
    nswap = int(round(len(edges) * niter))

    for _ in range(nswap):
        (u1, v1), (u2, v2) = random.sample(edges, 2)
        
        if len(set([u1, v1, u2, v2])) < 4:
            continue
        
        if not connectivity or (nx.has_path(G, u1, u2) and nx.has_path(G, v1, v2)):
            G.remove_edge(u1, v1)
            G.remove_edge(u2, v2)
            G.add_edge(u1, v2)
            G.add_edge(u2, v1)
            edges.remove((u1, v1))
            edges.remove((u2, v2))
            edges.append((u1, v2))
            edges.append((u2, v1))

    return G


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(4)
@nx._dispatchable(returns_graph=True)
def lattice_reference(G, niter=5, D=None, connectivity=True, seed=None):
    """Latticize the given graph by swapping edges.

    Parameters
    ----------
    G : graph
        An undirected graph.

    niter : integer (optional, default=1)
        An edge is rewired approximately niter times.

    D : numpy.array (optional, default=None)
        Distance to the diagonal matrix.

    connectivity : boolean (optional, default=True)
        Ensure connectivity for the latticized graph when set to True.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : graph
        The latticized graph.

    Raises
    ------
    NetworkXError
        If there are fewer than 4 nodes or 2 edges in `G`

    Notes
    -----
    The implementation is adapted from the algorithm by Sporns et al. [1]_.
    which is inspired from the original work by Maslov and Sneppen(2002) [2]_.

    References
    ----------
    .. [1] Sporns, Olaf, and Jonathan D. Zwi.
       "The small world of the cerebral cortex."
       Neuroinformatics 2.2 (2004): 145-162.
    .. [2] Maslov, Sergei, and Kim Sneppen.
       "Specificity and stability in topology of protein networks."
       Science 296.5569 (2002): 910-913.
    """
    import random
    import numpy as np

    if len(G) < 4:
        raise nx.NetworkXError("Graph has fewer than four nodes.")
    if G.number_of_edges() < 2:
        raise nx.NetworkXError("Graph has fewer than two edges.")

    G = G.copy()
    n = len(G)
    edges = list(G.edges())
    num_edges = len(edges)

    if D is None:
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = abs(i - j)

    nswap = int(round(num_edges * niter))

    for _ in range(nswap):
        (u1, v1), (u2, v2) = random.sample(edges, 2)
        
        if len(set([u1, v1, u2, v2])) < 4:
            continue
        
        if D[u1, v2] + D[u2, v1] < D[u1, v1] + D[u2, v2]:
            if not connectivity or (nx.has_path(G, u1, u2) and nx.has_path(G, v1, v2)):
                G.remove_edge(u1, v1)
                G.remove_edge(u2, v2)
                G.add_edge(u1, v2)
                G.add_edge(u2, v1)
                edges.remove((u1, v1))
                edges.remove((u2, v2))
                edges.append((u1, v2))
                edges.append((u2, v1))

    return G


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(3)
@nx._dispatchable
def sigma(G, niter=100, nrand=10, seed=None):
    """Returns the small-world coefficient (sigma) of the given graph.

    The small-world coefficient is defined as:
    sigma = C/Cr / L/Lr
    where C and L are respectively the average clustering coefficient and
    average shortest path length of G. Cr and Lr are respectively the average
    clustering coefficient and average shortest path length of an equivalent
    random graph.

    A graph is commonly classified as small-world if sigma>1.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    niter : integer (optional, default=100)
        Approximate number of rewiring per edge to compute the equivalent
        random graph.
    nrand : integer (optional, default=10)
        Number of random graphs generated to compute the average clustering
        coefficient (Cr) and average shortest path length (Lr).
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    sigma : float
        The small-world coefficient of G.

    Notes
    -----
    The implementation is adapted from Humphries et al. [1]_ [2]_.

    References
    ----------
    .. [1] The brainstem reticular formation is a small-world, not scale-free,
           network M. D. Humphries, K. Gurney and T. J. Prescott,
           Proc. Roy. Soc. B 2006 273, 503-511, doi:10.1098/rspb.2005.3354.
    .. [2] Humphries and Gurney (2008).
           "Network 'Small-World-Ness': A Quantitative Method for Determining
           Canonical Network Equivalence".
           PLoS One. 3 (4). PMID 18446219. doi:10.1371/journal.pone.0002051.
    """
    import numpy as np

    # Compute clustering coefficient and average shortest path length for G
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)

    # Generate random graphs and compute their properties
    Cr_list = []
    Lr_list = []
    for _ in range(nrand):
        G_rand = random_reference(G, niter=niter, seed=seed)
        Cr_list.append(nx.average_clustering(G_rand))
        Lr_list.append(nx.average_shortest_path_length(G_rand))

    # Compute average Cr and Lr
    Cr = np.mean(Cr_list)
    Lr = np.mean(Lr_list)

    # Compute sigma
    sigma = (C / Cr) / (L / Lr)

    return sigma


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(3)
@nx._dispatchable
def omega(G, niter=5, nrand=10, seed=None):
    """Returns the small-world coefficient (omega) of a graph

    The small-world coefficient of a graph G is:

    omega = Lr/L - C/Cl

    where C and L are respectively the average clustering coefficient and
    average shortest path length of G. Lr is the average shortest path length
    of an equivalent random graph and Cl is the average clustering coefficient
    of an equivalent lattice graph.

    The small-world coefficient (omega) measures how much G is like a lattice
    or a random graph. Negative values mean G is similar to a lattice whereas
    positive values mean G is a random graph.
    Values close to 0 mean that G has small-world characteristics.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    niter: integer (optional, default=5)
        Approximate number of rewiring per edge to compute the equivalent
        random graph.

    nrand: integer (optional, default=10)
        Number of random graphs generated to compute the maximal clustering
        coefficient (Cr) and average shortest path length (Lr).

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.


    Returns
    -------
    omega : float
        The small-world coefficient (omega)

    Notes
    -----
    The implementation is adapted from the algorithm by Telesford et al. [1]_.

    References
    ----------
    .. [1] Telesford, Joyce, Hayasaka, Burdette, and Laurienti (2011).
           "The Ubiquity of Small-World Networks".
           Brain Connectivity. 1 (0038): 367-75.  PMC 3604768. PMID 22432451.
           doi:10.1089/brain.2011.0038.
    """
    import numpy as np

    # Compute clustering coefficient and average shortest path length for G
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)

    # Generate random graphs and compute their properties
    Lr_list = []
    for _ in range(nrand):
        G_rand = random_reference(G, niter=niter, seed=seed)
        Lr_list.append(nx.average_shortest_path_length(G_rand))

    # Compute average Lr
    Lr = np.mean(Lr_list)

    # Generate lattice graph and compute its clustering coefficient
    G_lat = lattice_reference(G, niter=niter, seed=seed)
    Cl = nx.average_clustering(G_lat)

    # Compute omega
    omega = Lr / L - C / Cl

    return omega
