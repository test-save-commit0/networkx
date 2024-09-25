"""Generate graphs with a given joint degree and directed joint degree"""
import networkx as nx
from networkx.utils import py_random_state
__all__ = ['is_valid_joint_degree', 'is_valid_directed_joint_degree',
    'joint_degree_graph', 'directed_joint_degree_graph']


@nx._dispatchable(graphs=None)
def is_valid_joint_degree(joint_degrees):
    """Checks whether the given joint degree dictionary is realizable.

    A *joint degree dictionary* is a dictionary of dictionaries, in
    which entry ``joint_degrees[k][l]`` is an integer representing the
    number of edges joining nodes of degree *k* with nodes of degree
    *l*. Such a dictionary is realizable as a simple graph if and only
    if the following conditions are satisfied.

    - each entry must be an integer,
    - the total number of nodes of degree *k*, computed by
      ``sum(joint_degrees[k].values()) / k``, must be an integer,
    - the total number of edges joining nodes of degree *k* with
      nodes of degree *l* cannot exceed the total number of possible edges,
    - each diagonal entry ``joint_degrees[k][k]`` must be even (this is
      a convention assumed by the :func:`joint_degree_graph` function).


    Parameters
    ----------
    joint_degrees :  dictionary of dictionary of integers
        A joint degree dictionary in which entry ``joint_degrees[k][l]``
        is the number of edges joining nodes of degree *k* with nodes of
        degree *l*.

    Returns
    -------
    bool
        Whether the given joint degree dictionary is realizable as a
        simple graph.

    References
    ----------
    .. [1] M. Gjoka, M. Kurant, A. Markopoulou, "2.5K Graphs: from Sampling
       to Generation", IEEE Infocom, 2013.
    .. [2] I. Stanton, A. Pinar, "Constructing and sampling graphs with a
       prescribed joint degree distribution", Journal of Experimental
       Algorithmics, 2012.
    """
    for k, v in joint_degrees.items():
        for l, count in v.items():
            # Check if each entry is an integer
            if not isinstance(count, int):
                return False
            
            # Check if diagonal entries are even
            if k == l and count % 2 != 0:
                return False
    
    for k in joint_degrees:
        # Check if the total number of nodes of degree k is an integer
        if sum(joint_degrees[k].values()) % k != 0:
            return False
        
        for l in joint_degrees[k]:
            # Check if the number of edges doesn't exceed the possible maximum
            max_edges = (sum(joint_degrees[k].values()) // k) * (sum(joint_degrees[l].values()) // l)
            if k == l:
                max_edges = (max_edges - sum(joint_degrees[k].values()) // k) // 2
            if joint_degrees[k][l] > max_edges:
                return False
    
    return True


def _neighbor_switch(G, w, unsat, h_node_residual, avoid_node_id=None):
    """Releases one free stub for ``w``, while preserving joint degree in G.

    Parameters
    ----------
    G : NetworkX graph
        Graph in which the neighbor switch will take place.
    w : integer
        Node id for which we will execute this neighbor switch.
    unsat : set of integers
        Set of unsaturated node ids that have the same degree as w.
    h_node_residual: dictionary of integers
        Keeps track of the remaining stubs  for a given node.
    avoid_node_id: integer
        Node id to avoid when selecting w_prime.

    Notes
    -----
    First, it selects *w_prime*, an  unsaturated node that has the same degree
    as ``w``. Second, it selects *switch_node*, a neighbor node of ``w`` that
    is not  connected to *w_prime*. Then it executes an edge swap i.e. removes
    (``w``,*switch_node*) and adds (*w_prime*,*switch_node*). Gjoka et. al. [1]
    prove that such an edge swap is always possible.

    References
    ----------
    .. [1] M. Gjoka, B. Tillman, A. Markopoulou, "Construction of Simple
       Graphs with a Target Joint Degree Matrix and Beyond", IEEE Infocom, '15
    """
    w_prime = next((node for node in unsat if node != avoid_node_id), None)
    if w_prime is None:
        return None

    for switch_node in G.neighbors(w):
        if not G.has_edge(w_prime, switch_node):
            G.remove_edge(w, switch_node)
            G.add_edge(w_prime, switch_node)
            h_node_residual[w] += 1
            h_node_residual[w_prime] -= 1
            unsat.remove(w_prime)
            if h_node_residual[w_prime] == 0:
                unsat.add(w)
            return w_prime

    return None


@py_random_state(1)
@nx._dispatchable(graphs=None, returns_graph=True)
def joint_degree_graph(joint_degrees, seed=None):
    """Generates a random simple graph with the given joint degree dictionary.

    Parameters
    ----------
    joint_degrees :  dictionary of dictionary of integers
        A joint degree dictionary in which entry ``joint_degrees[k][l]`` is the
        number of edges joining nodes of degree *k* with nodes of degree *l*.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : Graph
        A graph with the specified joint degree dictionary.

    Raises
    ------
    NetworkXError
        If *joint_degrees* dictionary is not realizable.

    Notes
    -----
    In each iteration of the "while loop" the algorithm picks two disconnected
    nodes *v* and *w*, of degree *k* and *l* correspondingly,  for which
    ``joint_degrees[k][l]`` has not reached its target yet. It then adds
    edge (*v*, *w*) and increases the number of edges in graph G by one.

    The intelligence of the algorithm lies in the fact that  it is always
    possible to add an edge between such disconnected nodes *v* and *w*,
    even if one or both nodes do not have free stubs. That is made possible by
    executing a "neighbor switch", an edge rewiring move that releases
    a free stub while keeping the joint degree of G the same.

    The algorithm continues for E (number of edges) iterations of
    the "while loop", at the which point all entries of the given
    ``joint_degrees[k][l]`` have reached their target values and the
    construction is complete.

    References
    ----------
    ..  [1] M. Gjoka, B. Tillman, A. Markopoulou, "Construction of Simple
        Graphs with a Target Joint Degree Matrix and Beyond", IEEE Infocom, '15

    Examples
    --------
    >>> joint_degrees = {
    ...     1: {4: 1},
    ...     2: {2: 2, 3: 2, 4: 2},
    ...     3: {2: 2, 4: 1},
    ...     4: {1: 1, 2: 2, 3: 1},
    ... }
    >>> G = nx.joint_degree_graph(joint_degrees)
    >>>
    """
    if not is_valid_joint_degree(joint_degrees):
        raise nx.NetworkXError("Joint degree dictionary is not realizable.")

    # Create empty graph
    G = nx.Graph()

    # Create nodes with given degrees
    node_id = 0
    degree_to_nodes = {}
    for k in joint_degrees:
        num_nodes = sum(joint_degrees[k].values()) // k
        degree_to_nodes[k] = set(range(node_id, node_id + num_nodes))
        G.add_nodes_from(degree_to_nodes[k])
        node_id += num_nodes

    # Initialize residual stubs and unsaturated nodes
    h_node_residual = {node: k for k, nodes in degree_to_nodes.items() for node in nodes}
    unsat = set(G.nodes())

    # Add edges
    target_edges = {(k, l): joint_degrees[k][l] for k in joint_degrees for l in joint_degrees[k]}
    current_edges = {(k, l): 0 for k, l in target_edges}

    while unsat:
        k, l = seed.choice(list(target_edges.keys()))
        if current_edges[k, l] < target_edges[k, l]:
            v = seed.choice(list(degree_to_nodes[k] & unsat))
            w = seed.choice(list(degree_to_nodes[l] & unsat - {v}))

            if not G.has_edge(v, w):
                G.add_edge(v, w)
                current_edges[k, l] += 1
                current_edges[l, k] += 1
                h_node_residual[v] -= 1
                h_node_residual[w] -= 1

                if h_node_residual[v] == 0:
                    unsat.remove(v)
                if h_node_residual[w] == 0:
                    unsat.remove(w)
            else:
                w_prime = _neighbor_switch(G, w, unsat, h_node_residual, avoid_node_id=v)
                if w_prime is not None:
                    G.add_edge(v, w_prime)
                    current_edges[k, l] += 1
                    current_edges[l, k] += 1
                    h_node_residual[v] -= 1
                    h_node_residual[w_prime] -= 1

                    if h_node_residual[v] == 0:
                        unsat.remove(v)
                    if h_node_residual[w_prime] == 0:
                        unsat.remove(w_prime)

    return G


@nx._dispatchable(graphs=None)
def is_valid_directed_joint_degree(in_degrees, out_degrees, nkk):
    """Checks whether the given directed joint degree input is realizable

    Parameters
    ----------
    in_degrees :  list of integers
        in degree sequence contains the in degrees of nodes.
    out_degrees : list of integers
        out degree sequence contains the out degrees of nodes.
    nkk  :  dictionary of dictionary of integers
        directed joint degree dictionary. for nodes of out degree k (first
        level of dict) and nodes of in degree l (second level of dict)
        describes the number of edges.

    Returns
    -------
    boolean
        returns true if given input is realizable, else returns false.

    Notes
    -----
    Here is the list of conditions that the inputs (in/out degree sequences,
    nkk) need to satisfy for simple directed graph realizability:

    - Condition 0: in_degrees and out_degrees have the same length
    - Condition 1: nkk[k][l]  is integer for all k,l
    - Condition 2: sum(nkk[k])/k = number of nodes with partition id k, is an
                   integer and matching degree sequence
    - Condition 3: number of edges and non-chords between k and l cannot exceed
                   maximum possible number of edges


    References
    ----------
    [1] B. Tillman, A. Markopoulou, C. T. Butts & M. Gjoka,
        "Construction of Directed 2K Graphs". In Proc. of KDD 2017.
    """
    # Condition 0: in_degrees and out_degrees have the same length
    if len(in_degrees) != len(out_degrees):
        return False

    # Condition 1: nkk[k][l] is integer for all k,l
    for k in nkk:
        for l in nkk[k]:
            if not isinstance(nkk[k][l], int):
                return False

    # Condition 2: sum(nkk[k])/k = number of nodes with partition id k, is an integer and matching degree sequence
    in_degree_counts = {}
    out_degree_counts = {}
    for k in nkk:
        out_degree_counts[k] = sum(nkk[k].values()) // k
        if sum(nkk[k].values()) % k != 0:
            return False
        for l in nkk[k]:
            in_degree_counts[l] = in_degree_counts.get(l, 0) + nkk[k][l] // l
            if nkk[k][l] % l != 0:
                return False

    if set(in_degree_counts.keys()) != set(out_degree_counts.keys()):
        return False

    if (sorted(in_degree_counts.values()) != sorted(in_degrees.count(i) for i in set(in_degrees)) or
        sorted(out_degree_counts.values()) != sorted(out_degrees.count(i) for i in set(out_degrees))):
        return False

    # Condition 3: number of edges and non-chords between k and l cannot exceed maximum possible number of edges
    for k in nkk:
        for l in nkk[k]:
            max_edges = out_degree_counts[k] * in_degree_counts[l]
            if k == l:
                max_edges -= min(out_degree_counts[k], in_degree_counts[l])
            if nkk[k][l] > max_edges:
                return False

    return True


def _directed_neighbor_switch(G, w, unsat, h_node_residual_out, chords,
    h_partition_in, partition):
    """Releases one free stub for node w, while preserving joint degree in G.

    Parameters
    ----------
    G : networkx directed graph
        graph within which the edge swap will take place.
    w : integer
        node id for which we need to perform a neighbor switch.
    unsat: set of integers
        set of node ids that have the same degree as w and are unsaturated.
    h_node_residual_out: dict of integers
        for a given node, keeps track of the remaining stubs to be added.
    chords: set of tuples
        keeps track of available positions to add edges.
    h_partition_in: dict of integers
        for a given node, keeps track of its partition id (in degree).
    partition: integer
        partition id to check if chords have to be updated.

    Notes
    -----
    First, it selects node w_prime that (1) has the same degree as w and
    (2) is unsaturated. Then, it selects node v, a neighbor of w, that is
    not connected to w_prime and does an edge swap i.e. removes (w,v) and
    adds (w_prime,v). If neighbor switch is not possible for w using
    w_prime and v, then return w_prime; in [1] it's proven that
    such unsaturated nodes can be used.

    References
    ----------
    [1] B. Tillman, A. Markopoulou, C. T. Butts & M. Gjoka,
        "Construction of Directed 2K Graphs". In Proc. of KDD 2017.
    """
    w_prime = next(iter(unsat))
    for v in G.successors(w):
        if not G.has_edge(w_prime, v):
            G.remove_edge(w, v)
            G.add_edge(w_prime, v)
            h_node_residual_out[w] += 1
            h_node_residual_out[w_prime] -= 1
            unsat.remove(w_prime)
            if h_node_residual_out[w_prime] == 0:
                unsat.add(w)
            if partition == h_partition_in[v]:
                chords.add((w_prime, v))
                chords.remove((w, v))
            return w_prime
    return w_prime


def _directed_neighbor_switch_rev(G, w, unsat, h_node_residual_in, chords,
    h_partition_out, partition):
    """The reverse of directed_neighbor_switch.

    Parameters
    ----------
    G : networkx directed graph
        graph within which the edge swap will take place.
    w : integer
        node id for which we need to perform a neighbor switch.
    unsat: set of integers
        set of node ids that have the same degree as w and are unsaturated.
    h_node_residual_in: dict of integers
        for a given node, keeps track of the remaining stubs to be added.
    chords: set of tuples
        keeps track of available positions to add edges.
    h_partition_out: dict of integers
        for a given node, keeps track of its partition id (out degree).
    partition: integer
        partition id to check if chords have to be updated.

    Notes
    -----
    Same operation as directed_neighbor_switch except it handles this operation
    for incoming edges instead of outgoing.
    """
    w_prime = next(iter(unsat))
    for v in G.predecessors(w):
        if not G.has_edge(v, w_prime):
            G.remove_edge(v, w)
            G.add_edge(v, w_prime)
            h_node_residual_in[w] += 1
            h_node_residual_in[w_prime] -= 1
            unsat.remove(w_prime)
            if h_node_residual_in[w_prime] == 0:
                unsat.add(w)
            if partition == h_partition_out[v]:
                chords.add((v, w_prime))
                chords.remove((v, w))
            return w_prime
    return w_prime


@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def directed_joint_degree_graph(in_degrees, out_degrees, nkk, seed=None):
    """Generates a random simple directed graph with the joint degree.

    Parameters
    ----------
    degree_seq :  list of tuples (of size 3)
        degree sequence contains tuples of nodes with node id, in degree and
        out degree.
    nkk  :  dictionary of dictionary of integers
        directed joint degree dictionary, for nodes of out degree k (first
        level of dict) and nodes of in degree l (second level of dict)
        describes the number of edges.
    seed : hashable object, optional
        Seed for random number generator.

    Returns
    -------
    G : Graph
        A directed graph with the specified inputs.

    Raises
    ------
    NetworkXError
        If degree_seq and nkk are not realizable as a simple directed graph.


    Notes
    -----
    Similarly to the undirected version:
    In each iteration of the "while loop" the algorithm picks two disconnected
    nodes v and w, of degree k and l correspondingly,  for which nkk[k][l] has
    not reached its target yet i.e. (for given k,l): n_edges_add < nkk[k][l].
    It then adds edge (v,w) and always increases the number of edges in graph G
    by one.

    The intelligence of the algorithm lies in the fact that  it is always
    possible to add an edge between disconnected nodes v and w, for which
    nkk[degree(v)][degree(w)] has not reached its target, even if one or both
    nodes do not have free stubs. If either node v or w does not have a free
    stub, we perform a "neighbor switch", an edge rewiring move that releases a
    free stub while keeping nkk the same.

    The difference for the directed version lies in the fact that neighbor
    switches might not be able to rewire, but in these cases unsaturated nodes
    can be reassigned to use instead, see [1] for detailed description and
    proofs.

    The algorithm continues for E (number of edges in the graph) iterations of
    the "while loop", at which point all entries of the given nkk[k][l] have
    reached their target values and the construction is complete.

    References
    ----------
    [1] B. Tillman, A. Markopoulou, C. T. Butts & M. Gjoka,
        "Construction of Directed 2K Graphs". In Proc. of KDD 2017.

    Examples
    --------
    >>> in_degrees = [0, 1, 1, 2]
    >>> out_degrees = [1, 1, 1, 1]
    >>> nkk = {1: {1: 2, 2: 2}}
    >>> G = nx.directed_joint_degree_graph(in_degrees, out_degrees, nkk)
    >>>
    """
    if not is_valid_directed_joint_degree(in_degrees, out_degrees, nkk):
        raise nx.NetworkXError("Invalid directed joint degree input")

    G = nx.DiGraph()
    node_id = 0
    in_degree_to_nodes = {}
    out_degree_to_nodes = {}

    for in_deg, out_deg in zip(in_degrees, out_degrees):
        G.add_node(node_id, in_degree=in_deg, out_degree=out_deg)
        in_degree_to_nodes.setdefault(in_deg, set()).add(node_id)
        out_degree_to_nodes.setdefault(out_deg, set()).add(node_id)
        node_id += 1

    h_node_residual_in = {node: G.nodes[node]['in_degree'] for node in G}
    h_node_residual_out = {node: G.nodes[node]['out_degree'] for node in G}
    unsat_in = set(G.nodes())
    unsat_out = set(G.nodes())
    chords = set()

    target_edges = {(k, l): nkk[k][l] for k in nkk for l in nkk[k]}
    current_edges = {(k, l): 0 for k, l in target_edges}

    while unsat_out and unsat_in:
        k, l = seed.choice(list(target_edges.keys()))
        if current_edges[k, l] < target_edges[k, l]:
            v = seed.choice(list(out_degree_to_nodes[k] & unsat_out))
            w = seed.choice(list(in_degree_to_nodes[l] & unsat_in - {v}))

            if not G.has_edge(v, w):
                G.add_edge(v, w)
                current_edges[k, l] += 1
                h_node_residual_out[v] -= 1
                h_node_residual_in[w] -= 1

                if h_node_residual_out[v] == 0:
                    unsat_out.remove(v)
                if h_node_residual_in[w] == 0:
                    unsat_in.remove(w)
            else:
                w_prime = _directed_neighbor_switch(G, w, unsat_in, h_node_residual_in, chords, out_degree_to_nodes, k)
                if w_prime is not None:
                    G.add_edge(v, w_prime)
                    current_edges[k, l] += 1
                    h_node_residual_out[v] -= 1
                    h_node_residual_in[w_prime] -= 1

                    if h_node_residual_out[v] == 0:
                        unsat_out.remove(v)
                    if h_node_residual_in[w_prime] == 0:
                        unsat_in.remove(w_prime)
                else:
                    v_prime = _directed_neighbor_switch_rev(G, v, unsat_out, h_node_residual_out, chords, in_degree_to_nodes, l)
                    if v_prime is not None:
                        G.add_edge(v_prime, w)
                        current_edges[k, l] += 1
                        h_node_residual_out[v_prime] -= 1
                        h_node_residual_in[w] -= 1

                        if h_node_residual_out[v_prime] == 0:
                            unsat_out.remove(v_prime)
                        if h_node_residual_in[w] == 0:
                            unsat_in.remove(w)

    return G
