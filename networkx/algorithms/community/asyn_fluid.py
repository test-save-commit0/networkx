"""Asynchronous Fluid Communities algorithm for community detection."""
from collections import Counter
import networkx as nx
from networkx.algorithms.components import is_connected
from networkx.exception import NetworkXError
from networkx.utils import groups, not_implemented_for, py_random_state
__all__ = ['asyn_fluidc']


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(3)
@nx._dispatchable
def asyn_fluidc(G, k, max_iter=100, seed=None):
    """Returns communities in `G` as detected by Fluid Communities algorithm.

    The asynchronous fluid communities algorithm is described in
    [1]_. The algorithm is based on the simple idea of fluids interacting
    in an environment, expanding and pushing each other. Its initialization is
    random, so found communities may vary on different executions.

    The algorithm proceeds as follows. First each of the initial k communities
    is initialized in a random vertex in the graph. Then the algorithm iterates
    over all vertices in a random order, updating the community of each vertex
    based on its own community and the communities of its neighbors. This
    process is performed several times until convergence.
    At all times, each community has a total density of 1, which is equally
    distributed among the vertices it contains. If a vertex changes of
    community, vertex densities of affected communities are adjusted
    immediately. When a complete iteration over all vertices is done, such that
    no vertex changes the community it belongs to, the algorithm has converged
    and returns.

    This is the original version of the algorithm described in [1]_.
    Unfortunately, it does not support weighted graphs yet.

    Parameters
    ----------
    G : NetworkX graph
        Graph must be simple and undirected.

    k : integer
        The number of communities to be found.

    max_iter : integer
        The number of maximum iterations allowed. By default 100.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    communities : iterable
        Iterable of communities given as sets of nodes.

    Notes
    -----
    k variable is not an optional argument.

    References
    ----------
    .. [1] Parés F., Garcia-Gasulla D. et al. "Fluid Communities: A
       Competitive and Highly Scalable Community Detection Algorithm".
       [https://arxiv.org/pdf/1703.09307.pdf].
    """
    if not is_connected(G):
        raise NetworkXError("Graph must be connected.")

    # Initialize communities
    vertices = list(G)
    seed.shuffle(vertices)
    communities = {i: {vertices[i]} for i in range(k)}
    vertex_comm = {v: c for c, vs in communities.items() for v in vs}

    # Initialize densities
    density = {i: 1.0 / len(comm) for i, comm in communities.items()}

    for _ in range(max_iter):
        changes = False
        seed.shuffle(vertices)

        for v in vertices:
            old_comm = vertex_comm[v]
            comm_counter = Counter()

            # Count communities of neighbors
            for neighbor in G[v]:
                neighbor_comm = vertex_comm[neighbor]
                comm_counter[neighbor_comm] += density[neighbor_comm]

            # Find the community with maximum density
            new_comm = max(comm_counter, key=comm_counter.get, default=old_comm)

            if new_comm != old_comm:
                # Update communities
                communities[old_comm].remove(v)
                communities[new_comm].add(v)
                vertex_comm[v] = new_comm

                # Update densities
                old_size, new_size = len(communities[old_comm]), len(communities[new_comm])
                density[old_comm] = 1.0 / old_size if old_size > 0 else 0
                density[new_comm] = 1.0 / new_size

                changes = True

        if not changes:
            break

    return [comm for comm in communities.values() if comm]
