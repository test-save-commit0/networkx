"""
***************
VF2++ Algorithm
***************

An implementation of the VF2++ algorithm [1]_ for Graph Isomorphism testing.

The simplest interface to use this module is to call:

`vf2pp_is_isomorphic`: to check whether two graphs are isomorphic.
`vf2pp_isomorphism`: to obtain the node mapping between two graphs,
in case they are isomorphic.
`vf2pp_all_isomorphisms`: to generate all possible mappings between two graphs,
if isomorphic.

Introduction
------------
The VF2++ algorithm, follows a similar logic to that of VF2, while also
introducing new easy-to-check cutting rules and determining the optimal access
order of nodes. It is also implemented in a non-recursive manner, which saves
both time and space, when compared to its previous counterpart.

The optimal node ordering is obtained after taking into consideration both the
degree but also the label rarity of each node.
This way we place the nodes that are more likely to match, first in the order,
thus examining the most promising branches in the beginning.
The rules also consider node labels, making it easier to prune unfruitful
branches early in the process.

Examples
--------

Suppose G1 and G2 are Isomorphic Graphs. Verification is as follows:

Without node labels:

>>> import networkx as nx
>>> G1 = nx.path_graph(4)
>>> G2 = nx.path_graph(4)
>>> nx.vf2pp_is_isomorphic(G1, G2, node_label=None)
True
>>> nx.vf2pp_isomorphism(G1, G2, node_label=None)
{1: 1, 2: 2, 0: 0, 3: 3}

With node labels:

>>> G1 = nx.path_graph(4)
>>> G2 = nx.path_graph(4)
>>> mapped = {1: 1, 2: 2, 3: 3, 0: 0}
>>> nx.set_node_attributes(G1, dict(zip(G1, ["blue", "red", "green", "yellow"])), "label")
>>> nx.set_node_attributes(G2, dict(zip([mapped[u] for u in G1], ["blue", "red", "green", "yellow"])), "label")
>>> nx.vf2pp_is_isomorphic(G1, G2, node_label="label")
True
>>> nx.vf2pp_isomorphism(G1, G2, node_label="label")
{1: 1, 2: 2, 0: 0, 3: 3}

References
----------
.. [1] Jüttner, Alpár & Madarasi, Péter. (2018). "VF2++—An improved subgraph
   isomorphism algorithm". Discrete Applied Mathematics. 242.
   https://doi.org/10.1016/j.dam.2018.02.018

"""
import collections
import networkx as nx
__all__ = ['vf2pp_isomorphism', 'vf2pp_is_isomorphic', 'vf2pp_all_isomorphisms'
    ]
_GraphParameters = collections.namedtuple('_GraphParameters', ['G1', 'G2',
    'G1_labels', 'G2_labels', 'nodes_of_G1Labels', 'nodes_of_G2Labels',
    'G2_nodes_of_degree'])
_StateParameters = collections.namedtuple('_StateParameters', ['mapping',
    'reverse_mapping', 'T1', 'T1_in', 'T1_tilde', 'T1_tilde_in', 'T2',
    'T2_in', 'T2_tilde', 'T2_tilde_in'])


@nx._dispatchable(graphs={'G1': 0, 'G2': 1}, node_attrs={'node_label':
    'default_label'})
def vf2pp_isomorphism(G1, G2, node_label=None, default_label=None):
    """Return an isomorphic mapping between `G1` and `G2` if it exists.

    Parameters
    ----------
    G1, G2 : NetworkX Graph or MultiGraph instances.
        The two graphs to check for isomorphism.

    node_label : str, optional
        The name of the node attribute to be used when comparing nodes.
        The default is `None`, meaning node attributes are not considered
        in the comparison. Any node that doesn't have the `node_label`
        attribute uses `default_label` instead.

    default_label : scalar
        Default value to use when a node doesn't have an attribute
        named `node_label`. Default is `None`.

    Returns
    -------
    dict or None
        Node mapping if the two graphs are isomorphic. None otherwise.
    """
    if len(G1) != len(G2):
        return None

    G1_degree = {n: G1.degree(n) for n in G1}
    G2_degree = {n: G2.degree(n) for n in G2}

    graph_params = _initialize_parameters(G1, G2, G2_degree, node_label, default_label)
    node_order = _matching_order(graph_params)

    state_params = _StateParameters({}, {}, set(), set(), set(), set(), set(), set(), set(), set())
    stack = []

    for u in node_order:
        candidates = _find_candidates(u, graph_params, state_params, G1_degree)
        if not candidates:
            if not stack:
                return None
            u, _ = stack.pop()
            _restore_Tinout(u, state_params.mapping[u], graph_params, state_params)
            del state_params.reverse_mapping[state_params.mapping[u]]
            del state_params.mapping[u]
        else:
            v = candidates.pop()
            stack.append((u, candidates))
            state_params.mapping[u] = v
            state_params.reverse_mapping[v] = u
            _update_Tinout(u, v, graph_params, state_params)

        if len(state_params.mapping) == len(G1):
            return state_params.mapping

    return None


@nx._dispatchable(graphs={'G1': 0, 'G2': 1}, node_attrs={'node_label':
    'default_label'})
def vf2pp_is_isomorphic(G1, G2, node_label=None, default_label=None):
    """Examines whether G1 and G2 are isomorphic.

    Parameters
    ----------
    G1, G2 : NetworkX Graph or MultiGraph instances.
        The two graphs to check for isomorphism.

    node_label : str, optional
        The name of the node attribute to be used when comparing nodes.
        The default is `None`, meaning node attributes are not considered
        in the comparison. Any node that doesn't have the `node_label`
        attribute uses `default_label` instead.

    default_label : scalar
        Default value to use when a node doesn't have an attribute
        named `node_label`. Default is `None`.

    Returns
    -------
    bool
        True if the two graphs are isomorphic, False otherwise.
    """
    return vf2pp_isomorphism(G1, G2, node_label, default_label) is not None


@nx._dispatchable(graphs={'G1': 0, 'G2': 1}, node_attrs={'node_label':
    'default_label'})
def vf2pp_all_isomorphisms(G1, G2, node_label=None, default_label=None):
    """Yields all the possible mappings between G1 and G2.

    Parameters
    ----------
    G1, G2 : NetworkX Graph or MultiGraph instances.
        The two graphs to check for isomorphism.

    node_label : str, optional
        The name of the node attribute to be used when comparing nodes.
        The default is `None`, meaning node attributes are not considered
        in the comparison. Any node that doesn't have the `node_label`
        attribute uses `default_label` instead.

    default_label : scalar
        Default value to use when a node doesn't have an attribute
        named `node_label`. Default is `None`.

    Yields
    ------
    dict
        Isomorphic mapping between the nodes in `G1` and `G2`.
    """
    if len(G1) != len(G2):
        return

    G1_degree = {n: G1.degree(n) for n in G1}
    G2_degree = {n: G2.degree(n) for n in G2}

    graph_params = _initialize_parameters(G1, G2, G2_degree, node_label, default_label)
    node_order = _matching_order(graph_params)

    state_params = _StateParameters({}, {}, set(), set(), set(), set(), set(), set(), set(), set())
    stack = []

    for u in node_order:
        candidates = _find_candidates(u, graph_params, state_params, G1_degree)
        while candidates:
            v = candidates.pop()
            stack.append((u, candidates))
            state_params.mapping[u] = v
            state_params.reverse_mapping[v] = u
            _update_Tinout(u, v, graph_params, state_params)

            if len(state_params.mapping) == len(G1):
                yield state_params.mapping.copy()
                u, candidates = stack.pop()
                _restore_Tinout(u, state_params.mapping[u], graph_params, state_params)
                del state_params.reverse_mapping[state_params.mapping[u]]
                del state_params.mapping[u]
            else:
                break
        else:
            if not stack:
                return
            u, candidates = stack.pop()
            _restore_Tinout(u, state_params.mapping[u], graph_params, state_params)
            del state_params.reverse_mapping[state_params.mapping[u]]
            del state_params.mapping[u]


def _initialize_parameters(G1, G2, G2_degree, node_label=None, default_label=-1):
    """Initializes all the necessary parameters for VF2++

    Parameters
    ----------
    G1,G2: NetworkX Graph or MultiGraph instances.
        The two graphs to check for isomorphism or monomorphism

    G1_labels,G2_labels: dict
        The label of every node in G1 and G2 respectively

    Returns
    -------
    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2
        G1_labels,G2_labels: dict

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_out, T2_out: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti
    """
    G1_labels = {node: G1.nodes[node].get(node_label, default_label) for node in G1}
    G2_labels = {node: G2.nodes[node].get(node_label, default_label) for node in G2}

    nodes_of_G1Labels = collections.defaultdict(set)
    for node, label in G1_labels.items():
        nodes_of_G1Labels[label].add(node)

    nodes_of_G2Labels = collections.defaultdict(set)
    for node, label in G2_labels.items():
        nodes_of_G2Labels[label].add(node)

    G2_nodes_of_degree = collections.defaultdict(set)
    for node, degree in G2_degree.items():
        G2_nodes_of_degree[degree].add(node)

    graph_params = _GraphParameters(G1, G2, G1_labels, G2_labels, nodes_of_G1Labels, nodes_of_G2Labels, G2_nodes_of_degree)

    return graph_params


def _matching_order(graph_params):
    """The node ordering as introduced in VF2++.

    Notes
    -----
    Taking into account the structure of the Graph and the node labeling, the nodes are placed in an order such that,
    most of the unfruitful/infeasible branches of the search space can be pruned on high levels, significantly
    decreasing the number of visited states. The premise is that, the algorithm will be able to recognize
    inconsistencies early, proceeding to go deep into the search tree only if it's needed.

    Parameters
    ----------
    graph_params: namedtuple
        Contains:

            G1,G2: NetworkX Graph or MultiGraph instances.
                The two graphs to check for isomorphism or monomorphism.

            G1_labels,G2_labels: dict
                The label of every node in G1 and G2 respectively.

    Returns
    -------
    node_order: list
        The ordering of the nodes.
    """
    G1, G2, G1_labels, G2_labels, nodes_of_G1Labels, nodes_of_G2Labels, _ = graph_params

    label_frequency = {label: len(nodes) for label, nodes in nodes_of_G2Labels.items()}
    node_order = sorted(G1.nodes(), key=lambda n: (label_frequency[G1_labels[n]], -G1.degree(n)))

    return node_order


def _find_candidates(u, graph_params, state_params, G1_degree):
    """Given node u of G1, finds the candidates of u from G2.

    Parameters
    ----------
    u: Graph node
        The node from G1 for which to find the candidates from G2.

    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2: NetworkX Graph or MultiGraph instances.
            The two graphs to check for isomorphism or monomorphism

        G1_labels,G2_labels: dict
            The label of every node in G1 and G2 respectively

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_tilde, T2_tilde: set
            Ti_tilde contains all the nodes from Gi, that are neither in the mapping nor in Ti

    Returns
    -------
    candidates: set
        The nodes from G2 which are candidates for u.
    """
    G1, G2, G1_labels, G2_labels, nodes_of_G1Labels, nodes_of_G2Labels, G2_nodes_of_degree = graph_params
    mapping, reverse_mapping, T1, T1_in, T1_tilde, T1_tilde_in, T2, T2_in, T2_tilde, T2_tilde_in = state_params

    candidates = set()

    if u in T1:
        candidates = T2
    elif u in T1_tilde:
        candidates = T2_tilde
    else:
        label = G1_labels[u]
        degree = G1_degree[u]
        candidates = nodes_of_G2Labels[label] & G2_nodes_of_degree[degree]

    return candidates - set(reverse_mapping.keys())


def _feasibility(node1, node2, graph_params, state_params):
    """Given a candidate pair of nodes u and v from G1 and G2 respectively, checks if it's feasible to extend the
    mapping, i.e. if u and v can be matched.

    Notes
    -----
    This function performs all the necessary checking by applying both consistency and cutting rules.

    Parameters
    ----------
    node1, node2: Graph node
        The candidate pair of nodes being checked for matching

    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2: NetworkX Graph or MultiGraph instances.
            The two graphs to check for isomorphism or monomorphism

        G1_labels,G2_labels: dict
            The label of every node in G1 and G2 respectively

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_out, T2_out: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti

    Returns
    -------
    True if all checks are successful, False otherwise.
    """
    return _consistent_PT(node1, node2, graph_params, state_params) and not _cut_PT(node1, node2, graph_params, state_params)


def _cut_PT(u, v, graph_params, state_params):
    """Implements the cutting rules for the ISO problem.

    Parameters
    ----------
    u, v: Graph node
        The two candidate nodes being examined.

    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2: NetworkX Graph or MultiGraph instances.
            The two graphs to check for isomorphism or monomorphism

        G1_labels,G2_labels: dict
            The label of every node in G1 and G2 respectively

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_tilde, T2_tilde: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti

    Returns
    -------
    True if we should prune this branch, i.e. the node pair failed the cutting checks. False otherwise.
    """
    G1, G2, G1_labels, G2_labels, nodes_of_G1Labels, nodes_of_G2Labels, _ = graph_params
    mapping, reverse_mapping, T1, T1_in, T1_tilde, T1_tilde_in, T2, T2_in, T2_tilde, T2_tilde_in = state_params

    # Check label compatibility
    if G1_labels[u] != G2_labels[v]:
        return True

    # Check degree compatibility
    if G1.degree(u) != G2.degree(v):
        return True

    # Check neighbor label compatibility
    u_neighbor_labels = {G1_labels[n] for n in G1.neighbors(u)}
    v_neighbor_labels = {G2_labels[n] for n in G2.neighbors(v)}
    if u_neighbor_labels != v_neighbor_labels:
        return True

    return False


def _consistent_PT(u, v, graph_params, state_params):
    """Checks the consistency of extending the mapping using the current node pair.

    Parameters
    ----------
    u, v: Graph node
        The two candidate nodes being examined.

    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2: NetworkX Graph or MultiGraph instances.
            The two graphs to check for isomorphism or monomorphism

        G1_labels,G2_labels: dict
            The label of every node in G1 and G2 respectively

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_out, T2_out: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti

    Returns
    -------
    True if the pair passes all the consistency checks successfully. False otherwise.
    """
    G1, G2, G1_labels, G2_labels, nodes_of_G1Labels, nodes_of_G2Labels, _ = graph_params
    mapping, reverse_mapping, T1, T1_in, T1_tilde, T1_tilde_in, T2, T2_in, T2_tilde, T2_tilde_in = state_params

    # Check if the nodes are already mapped
    if u in mapping or v in reverse_mapping:
        return False

    # Check connectivity consistency
    for n1, n2 in mapping.items():
        if (G1.has_edge(u, n1) != G2.has_edge(v, n2)):
            return False

    return True


def _update_Tinout(new_node1, new_node2, graph_params, state_params):
    """Updates the Ti/Ti_out (i=1,2) when a new node pair u-v is added to the mapping.

    Notes
    -----
    This function should be called right after the feasibility checks are passed, and node1 is mapped to node2. The
    purpose of this function is to avoid brute force computing of Ti/Ti_out by iterating over all nodes of the graph
    and checking which nodes satisfy the necessary conditions. Instead, in every step of the algorithm we focus
    exclusively on the two nodes that are being added to the mapping, incrementally updating Ti/Ti_out.

    Parameters
    ----------
    new_node1, new_node2: Graph node
        The two new nodes, added to the mapping.

    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2: NetworkX Graph or MultiGraph instances.
            The two graphs to check for isomorphism or monomorphism

        G1_labels,G2_labels: dict
            The label of every node in G1 and G2 respectively

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_tilde, T2_tilde: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti
    """
    pass


def _restore_Tinout(popped_node1, popped_node2, graph_params, state_params):
    """Restores the previous version of Ti/Ti_out when a node pair is deleted from the mapping.

    Parameters
    ----------
    popped_node1, popped_node2: Graph node
        The two nodes deleted from the mapping.

    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2: NetworkX Graph or MultiGraph instances.
            The two graphs to check for isomorphism or monomorphism

        G1_labels,G2_labels: dict
            The label of every node in G1 and G2 respectively

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_tilde, T2_tilde: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti
    """
    G1, G2 = graph_params.G1, graph_params.G2
    mapping, reverse_mapping, T1, T1_in, T1_tilde, T1_tilde_in, T2, T2_in, T2_tilde, T2_tilde_in = state_params

    # Restore T1 and T1_tilde
    T1_tilde.add(popped_node1)
    for neighbor in G1.neighbors(popped_node1):
        if neighbor not in mapping:
            if all(mapped_neighbor not in G1.neighbors(neighbor) for mapped_neighbor in mapping):
                T1.discard(neighbor)
                T1_tilde.add(neighbor)

    # Restore T2 and T2_tilde
    T2_tilde.add(popped_node2)
    for neighbor in G2.neighbors(popped_node2):
        if neighbor not in reverse_mapping:
            if all(mapped_neighbor not in G2.neighbors(neighbor) for mapped_neighbor in reverse_mapping):
                T2.discard(neighbor)
                T2_tilde.add(neighbor)
