"""
Label propagation community detection algorithms.
"""
from collections import Counter, defaultdict, deque
import networkx as nx
from networkx.utils import groups, not_implemented_for, py_random_state
__all__ = ['label_propagation_communities', 'asyn_lpa_communities',
    'fast_label_propagation_communities']


@py_random_state('seed')
@nx._dispatchable(edge_attrs='weight')
def fast_label_propagation_communities(G, *, weight=None, seed=None):
    """Returns communities in `G` as detected by fast label propagation.

    The fast label propagation algorithm is described in [1]_. The algorithm is
    probabilistic and the found communities may vary in different executions.

    The algorithm operates as follows. First, the community label of each node is
    set to a unique label. The algorithm then repeatedly updates the labels of
    the nodes to the most frequent label in their neighborhood. In case of ties,
    a random label is chosen from the most frequent labels.

    The algorithm maintains a queue of nodes that still need to be processed.
    Initially, all nodes are added to the queue in a random order. Then the nodes
    are removed from the queue one by one and processed. If a node updates its label,
    all its neighbors that have a different label are added to the queue (if not
    already in the queue). The algorithm stops when the queue is empty.

    Parameters
    ----------
    G : Graph, DiGraph, MultiGraph, or MultiDiGraph
        Any NetworkX graph.

    weight : string, or None (default)
        The edge attribute representing a non-negative weight of an edge. If None,
        each edge is assumed to have weight one. The weight of an edge is used in
        determining the frequency with which a label appears among the neighbors of
        a node (edge with weight `w` is equivalent to `w` unweighted edges).

    seed : integer, random_state, or None (default)
        Indicator of random number generation state. See :ref:`Randomness<randomness>`.

    Returns
    -------
    communities : iterable
        Iterable of communities given as sets of nodes.

    Notes
    -----
    Edge directions are ignored for directed graphs.
    Edge weights must be non-negative numbers.

    References
    ----------
    .. [1] Vincent A. Traag & Lovro Šubelj. "Large network community detection by
       fast label propagation." Scientific Reports 13 (2023): 2701.
       https://doi.org/10.1038/s41598-023-29610-z
    """
    pass


def _fast_label_count(G, comms, node, weight=None):
    """Computes the frequency of labels in the neighborhood of a node.

    Returns a dictionary keyed by label to the frequency of that label.
    """
    pass


@py_random_state(2)
@nx._dispatchable(edge_attrs='weight')
def asyn_lpa_communities(G, weight=None, seed=None):
    """Returns communities in `G` as detected by asynchronous label
    propagation.

    The asynchronous label propagation algorithm is described in
    [1]_. The algorithm is probabilistic and the found communities may
    vary on different executions.

    The algorithm proceeds as follows. After initializing each node with
    a unique label, the algorithm repeatedly sets the label of a node to
    be the label that appears most frequently among that nodes
    neighbors. The algorithm halts when each node has the label that
    appears most frequently among its neighbors. The algorithm is
    asynchronous because each node is updated without waiting for
    updates on the remaining nodes.

    This generalized version of the algorithm in [1]_ accepts edge
    weights.

    Parameters
    ----------
    G : Graph

    weight : string
        The edge attribute representing the weight of an edge.
        If None, each edge is assumed to have weight one. In this
        algorithm, the weight of an edge is used in determining the
        frequency with which a label appears among the neighbors of a
        node: a higher weight means the label appears more often.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    communities : iterable
        Iterable of communities given as sets of nodes.

    Notes
    -----
    Edge weight attributes must be numerical.

    References
    ----------
    .. [1] Raghavan, Usha Nandini, Réka Albert, and Soundar Kumara. "Near
           linear time algorithm to detect community structures in large-scale
           networks." Physical Review E 76.3 (2007): 036106.
    """
    pass


@not_implemented_for('directed')
@nx._dispatchable
def label_propagation_communities(G):
    """Generates community sets determined by label propagation

    Finds communities in `G` using a semi-synchronous label propagation
    method [1]_. This method combines the advantages of both the synchronous
    and asynchronous models. Not implemented for directed graphs.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph.

    Returns
    -------
    communities : iterable
        A dict_values object that contains a set of nodes for each community.

    Raises
    ------
    NetworkXNotImplemented
       If the graph is directed

    References
    ----------
    .. [1] Cordasco, G., & Gargano, L. (2010, December). Community detection
       via semi-synchronous label propagation algorithms. In Business
       Applications of Social Network Analysis (BASNA), 2010 IEEE International
       Workshop on (pp. 1-8). IEEE.
    """
    pass


def _color_network(G):
    """Colors the network so that neighboring nodes all have distinct colors.

    Returns a dict keyed by color to a set of nodes with that color.
    """
    pass


def _labeling_complete(labeling, G):
    """Determines whether or not LPA is done.

    Label propagation is complete when all nodes have a label that is
    in the set of highest frequency labels amongst its neighbors.

    Nodes with no neighbors are considered complete.
    """
    pass


def _most_frequent_labels(node, labeling, G):
    """Returns a set of all labels with maximum frequency in `labeling`.

    Input `labeling` should be a dict keyed by node to labels.
    """
    pass


def _update_label(node, labeling, G):
    """Updates the label of a node using the Prec-Max tie breaking algorithm

    The algorithm is explained in: 'Community Detection via Semi-Synchronous
    Label Propagation Algorithms' Cordasco and Gargano, 2011
    """
    pass
