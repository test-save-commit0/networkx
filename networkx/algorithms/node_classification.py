""" This module provides the functions for node classification problem.

The functions in this module are not imported
into the top level `networkx` namespace.
You can access these functions by importing
the `networkx.algorithms.node_classification` modules,
then accessing the functions as attributes of `node_classification`.
For example:

  >>> from networkx.algorithms import node_classification
  >>> G = nx.path_graph(4)
  >>> G.edges()
  EdgeView([(0, 1), (1, 2), (2, 3)])
  >>> G.nodes[0]["label"] = "A"
  >>> G.nodes[3]["label"] = "B"
  >>> node_classification.harmonic_function(G)
  ['A', 'A', 'B', 'B']

References
----------
Zhu, X., Ghahramani, Z., & Lafferty, J. (2003, August).
Semi-supervised learning using gaussian fields and harmonic functions.
In ICML (Vol. 3, pp. 912-919).
"""
import networkx as nx
__all__ = ['harmonic_function', 'local_and_global_consistency']


@nx.utils.not_implemented_for('directed')
@nx._dispatchable(node_attrs='label_name')
def harmonic_function(G, max_iter=30, label_name='label'):
    """Node classification by Harmonic function

    Function for computing Harmonic function algorithm by Zhu et al.

    Parameters
    ----------
    G : NetworkX Graph
    max_iter : int
        maximum number of iterations allowed
    label_name : string
        name of target labels to predict

    Returns
    -------
    predicted : list
        List of length ``len(G)`` with the predicted labels for each node.

    Raises
    ------
    NetworkXError
        If no nodes in `G` have attribute `label_name`.

    Examples
    --------
    >>> from networkx.algorithms import node_classification
    >>> G = nx.path_graph(4)
    >>> G.nodes[0]["label"] = "A"
    >>> G.nodes[3]["label"] = "B"
    >>> G.nodes(data=True)
    NodeDataView({0: {'label': 'A'}, 1: {}, 2: {}, 3: {'label': 'B'}})
    >>> G.edges()
    EdgeView([(0, 1), (1, 2), (2, 3)])
    >>> predicted = node_classification.harmonic_function(G)
    >>> predicted
    ['A', 'A', 'B', 'B']

    References
    ----------
    Zhu, X., Ghahramani, Z., & Lafferty, J. (2003, August).
    Semi-supervised learning using gaussian fields and harmonic functions.
    In ICML (Vol. 3, pp. 912-919).
    """
    import numpy as np
    from scipy import sparse

    # Get label information
    labels, label_dict = _get_label_info(G, label_name)
    if len(labels) == 0:
        raise nx.NetworkXError(f"No nodes in G have the attribute {label_name}")

    n_total = len(G)
    n_labeled = len(labels)
    n_classes = len(label_dict)

    # Create adjacency matrix
    adj_matrix = nx.adjacency_matrix(G)

    # Create diagonal degree matrix
    degrees = sparse.diags([dict(G.degree()).get(i, 0) for i in range(n_total)])

    # Compute graph Laplacian
    laplacian = degrees - adj_matrix

    # Partition Laplacian matrix
    lap_uu = laplacian[n_labeled:, n_labeled:]
    lap_ul = laplacian[n_labeled:, :n_labeled]

    # Create label matrix
    F = np.zeros((n_total, n_classes))
    for idx, label in labels:
        F[idx, label] = 1

    # Iterative solution
    Fu = np.zeros((n_total - n_labeled, n_classes))
    for _ in range(max_iter):
        Fu_new = -sparse.linalg.spsolve(lap_uu, lap_ul.dot(F[:n_labeled]))
        if np.allclose(Fu, Fu_new):
            break
        Fu = Fu_new

    F[n_labeled:] = Fu

    # Get predicted labels
    predicted = [label_dict[i] for i in F.argmax(axis=1)]

    return predicted


@nx.utils.not_implemented_for('directed')
@nx._dispatchable(node_attrs='label_name')
def local_and_global_consistency(G, alpha=0.99, max_iter=30, label_name='label'):
    """Node classification by Local and Global Consistency

    Function for computing Local and global consistency algorithm by Zhou et al.

    Parameters
    ----------
    G : NetworkX Graph
    alpha : float
        Clamping factor
    max_iter : int
        Maximum number of iterations allowed
    label_name : string
        Name of target labels to predict

    Returns
    -------
    predicted : list
        List of length ``len(G)`` with the predicted labels for each node.

    Raises
    ------
    NetworkXError
        If no nodes in `G` have attribute `label_name`.

    Examples
    --------
    >>> from networkx.algorithms import node_classification
    >>> G = nx.path_graph(4)
    >>> G.nodes[0]["label"] = "A"
    >>> G.nodes[3]["label"] = "B"
    >>> G.nodes(data=True)
    NodeDataView({0: {'label': 'A'}, 1: {}, 2: {}, 3: {'label': 'B'}})
    >>> G.edges()
    EdgeView([(0, 1), (1, 2), (2, 3)])
    >>> predicted = node_classification.local_and_global_consistency(G)
    >>> predicted
    ['A', 'A', 'B', 'B']

    References
    ----------
    Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004).
    Learning with local and global consistency.
    Advances in neural information processing systems, 16(16), 321-328.
    """
    import numpy as np
    from scipy import sparse

    # Get label information
    labels, label_dict = _get_label_info(G, label_name)
    if len(labels) == 0:
        raise nx.NetworkXError(f"No nodes in G have the attribute {label_name}")

    n_total = len(G)
    n_classes = len(label_dict)

    # Create adjacency matrix
    adj_matrix = nx.adjacency_matrix(G)

    # Create diagonal degree matrix
    degrees = sparse.diags([dict(G.degree()).get(i, 0) for i in range(n_total)])

    # Compute normalized graph Laplacian
    laplacian = sparse.eye(n_total) - alpha * sparse.linalg.inv(degrees) @ adj_matrix

    # Create initial label matrix
    F = np.zeros((n_total, n_classes))
    for idx, label in labels:
        F[idx, label] = 1

    # Iterative solution
    for _ in range(max_iter):
        F_new = sparse.linalg.spsolve(laplacian, F)
        if np.allclose(F, F_new):
            break
        F = F_new

    # Get predicted labels
    predicted = [label_dict[i] for i in F.argmax(axis=1)]

    return predicted


def _get_label_info(G, label_name):
    """Get and return information of labels from the input graph

    Parameters
    ----------
    G : Network X graph
    label_name : string
        Name of the target label

    Returns
    -------
    labels : numpy array, shape = [n_labeled_samples, 2]
        Array of pairs of labeled node ID and label ID
    label_dict : numpy array, shape = [n_classes]
        Array of labels
        i-th element contains the label corresponding label ID `i`
    """
    import numpy as np

    labels = []
    label_set = set()

    for node, data in G.nodes(data=True):
        if label_name in data:
            label = data[label_name]
            label_set.add(label)
            labels.append((node, label))

    label_dict = np.array(sorted(label_set))
    label_to_id = {label: i for i, label in enumerate(label_dict)}

    labels = np.array([(node, label_to_id[label]) for node, label in labels])

    return labels, label_dict
