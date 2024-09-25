"""
Generators for interval graph.
"""
from collections.abc import Sequence
import networkx as nx
__all__ = ['interval_graph']


@nx._dispatchable(graphs=None, returns_graph=True)
def interval_graph(intervals):
    """Generates an interval graph for a list of intervals given.

    In graph theory, an interval graph is an undirected graph formed from a set
    of closed intervals on the real line, with a vertex for each interval
    and an edge between vertices whose intervals intersect.
    It is the intersection graph of the intervals.

    More information can be found at:
    https://en.wikipedia.org/wiki/Interval_graph

    Parameters
    ----------
    intervals : a sequence of intervals, say (l, r) where l is the left end,
    and r is the right end of the closed interval.

    Returns
    -------
    G : networkx graph

    Examples
    --------
    >>> intervals = [(-2, 3), [1, 4], (2, 3), (4, 6)]
    >>> G = nx.interval_graph(intervals)
    >>> sorted(G.edges)
    [((-2, 3), (1, 4)), ((-2, 3), (2, 3)), ((1, 4), (2, 3)), ((1, 4), (4, 6))]

    Raises
    ------
    :exc:`TypeError`
        if `intervals` contains None or an element which is not
        collections.abc.Sequence or not a length of 2.
    :exc:`ValueError`
        if `intervals` contains an interval such that min1 > max1
        where min1,max1 = interval
    """
    if not isinstance(intervals, Sequence):
        raise TypeError("intervals must be a sequence")
    
    G = nx.Graph()
    
    for i, interval in enumerate(intervals):
        if not isinstance(interval, Sequence) or len(interval) != 2:
            raise TypeError(f"Interval {i} must be a sequence of length 2")
        
        min1, max1 = interval
        if min1 > max1:
            raise ValueError(f"Invalid interval {i}: min1 > max1")
        
        G.add_node(interval)
    
    for i, interval1 in enumerate(intervals):
        min1, max1 = interval1
        for interval2 in intervals[i+1:]:
            min2, max2 = interval2
            if (min1 <= min2 <= max1) or (min2 <= min1 <= max2):
                G.add_edge(interval1, interval2)
    
    return G
