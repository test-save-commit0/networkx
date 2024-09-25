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
    # Convert communities to a list if it's not already
    communities = list(communities)
    
    # Get all nodes in the graph
    all_nodes = set(G.nodes())
    
    # Get all nodes in the communities
    community_nodes = set().union(*communities)
    
    # Check if all nodes in the graph are in the communities
    if all_nodes != community_nodes:
        return False
    
    # Check if communities are pairwise disjoint
    seen_nodes = set()
    for community in communities:
        if seen_nodes.intersection(community):
            return False
        seen_nodes.update(community)
    
    return True
