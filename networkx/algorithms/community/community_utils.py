"""Helper functions for community-finding algorithms."""
import networkx as nx
__all__ = ['is_partition']


@nx._dispatchable
def is_partition(G, communities):
    """Returns *True* if `communities` is a partition of the nodes of `G`.

    A partition of a universe set is a family of pairwise disjoint sets
    whose union is the entire universe set.

    Parameters
    ----------
    G : NetworkX graph.

    communities : list or iterable of sets of nodes
        If not a list, the iterable is converted internally to a list.
        If it is an iterator it is exhausted.

    """
    pass
