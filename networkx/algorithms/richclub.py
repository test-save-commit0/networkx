"""Functions for computing rich-club coefficients."""
from itertools import accumulate
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['rich_club_coefficient']


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def rich_club_coefficient(G, normalized=True, Q=100, seed=None):
    """Returns the rich-club coefficient of the graph `G`.

    For each degree *k*, the *rich-club coefficient* is the ratio of the
    number of actual to the number of potential edges for nodes with
    degree greater than *k*:

    .. math::

        \\phi(k) = \\frac{2 E_k}{N_k (N_k - 1)}

    where `N_k` is the number of nodes with degree larger than *k*, and
    `E_k` is the number of edges among those nodes.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph with neither parallel edges nor self-loops.
    normalized : bool (optional)
        Normalize using randomized network as in [1]_
    Q : float (optional, default=100)
        If `normalized` is True, perform `Q * m` double-edge
        swaps, where `m` is the number of edges in `G`, to use as a
        null-model for normalization.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    rc : dictionary
       A dictionary, keyed by degree, with rich-club coefficient values.

    Raises
    ------
    NetworkXError
        If `G` has fewer than four nodes and ``normalized=True``.
        A randomly sampled graph for normalization cannot be generated in this case.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (4, 5)])
    >>> rc = nx.rich_club_coefficient(G, normalized=False, seed=42)
    >>> rc[0]
    0.4

    Notes
    -----
    The rich club definition and algorithm are found in [1]_.  This
    algorithm ignores any edge weights and is not defined for directed
    graphs or graphs with parallel edges or self loops.

    Normalization is done by computing the rich club coefficient for a randomly
    sampled graph with the same degree distribution as `G` by
    repeatedly swapping the endpoints of existing edges. For graphs with fewer than 4
    nodes, it is not possible to generate a random graph with a prescribed
    degree distribution, as the degree distribution fully determines the graph
    (hence making the coefficients trivially normalized to 1).
    This function raises an exception in this case.

    Estimates for appropriate values of `Q` are found in [2]_.

    References
    ----------
    .. [1] Julian J. McAuley, Luciano da Fontoura Costa,
       and Tibério S. Caetano,
       "The rich-club phenomenon across complex network hierarchies",
       Applied Physics Letters Vol 91 Issue 8, August 2007.
       https://arxiv.org/abs/physics/0701290
    .. [2] R. Milo, N. Kashtan, S. Itzkovitz, M. E. J. Newman, U. Alon,
       "Uniform generation of random graphs with arbitrary degree
       sequences", 2006. https://arxiv.org/abs/cond-mat/0312028
    """
    if len(G) < 4 and normalized:
        raise nx.NetworkXError("Graph has fewer than four nodes.")
    
    rc = _compute_rc(G)
    
    if normalized:
        # Create a random graph with the same degree sequence for normalization
        R = nx.configuration_model(list(d for n, d in G.degree()), seed=seed)
        R = nx.Graph(R)  # Remove parallel edges
        R.remove_edges_from(nx.selfloop_edges(R))  # Remove self-loops
        
        rc_R = _compute_rc(R)
        
        rc = {k: v / rc_R[k] if rc_R[k] > 0 else 0 for k, v in rc.items()}
    
    return rc


def _compute_rc(G):
    """Returns the rich-club coefficient for each degree in the graph
    `G`.

    `G` is an undirected graph without multiedges.

    Returns a dictionary mapping degree to rich-club coefficient for
    that degree.

    """
    degrees = [d for n, d in G.degree()]
    max_degree = max(degrees)
    nodes = G.number_of_nodes()
    
    # Count how many nodes have degree greater than k
    Nk = [nodes - i for i, _ in enumerate(accumulate(degrees.count(d) for d in range(max_degree + 1)))]
    
    # Count number of edges for nodes with degree greater than k
    Ek = [G.number_of_edges()]
    for k in range(1, max_degree + 1):
        Ek.append(sum(1 for u, v in G.edges() if G.degree(u) > k and G.degree(v) > k))
    
    # Compute rich-club coefficient for each degree
    rc = {}
    for k in range(max_degree + 1):
        if Nk[k] > 1:
            rc[k] = (2 * Ek[k]) / (Nk[k] * (Nk[k] - 1))
        else:
            rc[k] = 0
    
    return rc
