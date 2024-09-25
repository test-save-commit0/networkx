"""
*****************************
Time-respecting VF2 Algorithm
*****************************

An extension of the VF2 algorithm for time-respecting graph isomorphism
testing in temporal graphs.

A temporal graph is one in which edges contain a datetime attribute,
denoting when interaction occurred between the incident nodes. A
time-respecting subgraph of a temporal graph is a subgraph such that
all interactions incident to a node occurred within a time threshold,
delta, of each other. A directed time-respecting subgraph has the
added constraint that incoming interactions to a node must precede
outgoing interactions from the same node - this enforces a sense of
directed flow.

Introduction
------------

The TimeRespectingGraphMatcher and TimeRespectingDiGraphMatcher
extend the GraphMatcher and DiGraphMatcher classes, respectively,
to include temporal constraints on matches. This is achieved through
a semantic check, via the semantic_feasibility() function.

As well as including G1 (the graph in which to seek embeddings) and
G2 (the subgraph structure of interest), the name of the temporal
attribute on the edges and the time threshold, delta, must be supplied
as arguments to the matching constructors.

A delta of zero is the strictest temporal constraint on the match -
only embeddings in which all interactions occur at the same time will
be returned. A delta of one day will allow embeddings in which
adjacent interactions occur up to a day apart.

Examples
--------

Examples will be provided when the datetime type has been incorporated.


Temporal Subgraph Isomorphism
-----------------------------

A brief discussion of the somewhat diverse current literature will be
included here.

References
----------

[1] Redmond, U. and Cunningham, P. Temporal subgraph isomorphism. In:
The 2013 IEEE/ACM International Conference on Advances in Social
Networks Analysis and Mining (ASONAM). Niagara Falls, Canada; 2013:
pages 1451 - 1452. [65]

For a discussion of the literature on temporal networks:

[3] P. Holme and J. Saramaki. Temporal networks. Physics Reports,
519(3):97–125, 2012.

Notes
-----

Handles directed and undirected graphs and graphs with parallel edges.

"""
import networkx as nx
from .isomorphvf2 import DiGraphMatcher, GraphMatcher
__all__ = ['TimeRespectingGraphMatcher', 'TimeRespectingDiGraphMatcher']


class TimeRespectingGraphMatcher(GraphMatcher):

    def __init__(self, G1, G2, temporal_attribute_name, delta):
        """Initialize TimeRespectingGraphMatcher.

        G1 and G2 should be nx.Graph or nx.MultiGraph instances.

        Examples
        --------
        To create a TimeRespectingGraphMatcher which checks for
        syntactic and semantic feasibility:

        >>> from networkx.algorithms import isomorphism
        >>> from datetime import timedelta
        >>> G1 = nx.Graph(nx.path_graph(4, create_using=nx.Graph()))

        >>> G2 = nx.Graph(nx.path_graph(4, create_using=nx.Graph()))

        >>> GM = isomorphism.TimeRespectingGraphMatcher(G1, G2, "date", timedelta(days=1))
        """
        self.temporal_attribute_name = temporal_attribute_name
        self.delta = delta
        super().__init__(G1, G2)

    def one_hop(self, Gx, Gx_node, neighbors):
        """
        Edges one hop out from a node in the mapping should be
        time-respecting with respect to each other.
        """
        dates = []
        for neighbor in neighbors:
            edge_data = Gx.get_edge_data(Gx_node, neighbor)
            if isinstance(edge_data, dict):
                dates.append(edge_data.get(self.temporal_attribute_name))
            elif isinstance(edge_data, list):
                dates.extend(e.get(self.temporal_attribute_name) for e in edge_data if isinstance(e, dict))
        
        return all(abs(d1 - d2) <= self.delta for d1 in dates for d2 in dates if d1 != d2)

    def two_hop(self, Gx, core_x, Gx_node, neighbors):
        """
        Paths of length 2 from Gx_node should be time-respecting.
        """
        for n1 in neighbors:
            if n1 in core_x:
                for n2 in Gx.neighbors(n1):
                    if n2 in core_x and n2 != Gx_node:
                        e1 = Gx.get_edge_data(Gx_node, n1)
                        e2 = Gx.get_edge_data(n1, n2)
                        t1 = e1.get(self.temporal_attribute_name) if isinstance(e1, dict) else None
                        t2 = e2.get(self.temporal_attribute_name) if isinstance(e2, dict) else None
                        if t1 and t2 and abs(t1 - t2) > self.delta:
                            return False
        return True

    def semantic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is semantically
        feasible.

        Any subclass which redefines semantic_feasibility() must
        maintain the self.tests if needed, to keep the match() method
        functional. Implementations should consider multigraphs.
        """
        G1_nbrs = set(self.G1.neighbors(G1_node)) - set(self.core_1.keys())
        G2_nbrs = set(self.G2.neighbors(G2_node)) - set(self.core_2.keys())
        
        # Check one-hop time-respecting property
        if not self.one_hop(self.G1, G1_node, G1_nbrs) or not self.one_hop(self.G2, G2_node, G2_nbrs):
            return False
        
        # Check two-hop time-respecting property
        if not self.two_hop(self.G1, self.core_1, G1_node, G1_nbrs) or not self.two_hop(self.G2, self.core_2, G2_node, G2_nbrs):
            return False
        
        return True


class TimeRespectingDiGraphMatcher(DiGraphMatcher):

    def __init__(self, G1, G2, temporal_attribute_name, delta):
        """Initialize TimeRespectingDiGraphMatcher.

        G1 and G2 should be nx.DiGraph or nx.MultiDiGraph instances.

        Examples
        --------
        To create a TimeRespectingDiGraphMatcher which checks for
        syntactic and semantic feasibility:

        >>> from networkx.algorithms import isomorphism
        >>> from datetime import timedelta
        >>> G1 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))

        >>> G2 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))

        >>> GM = isomorphism.TimeRespectingDiGraphMatcher(G1, G2, "date", timedelta(days=1))
        """
        self.temporal_attribute_name = temporal_attribute_name
        self.delta = delta
        super().__init__(G1, G2)

    def get_pred_dates(self, Gx, Gx_node, core_x, pred):
        """
        Get the dates of edges from predecessors.
        """
        dates = []
        for p in pred:
            if p in core_x:
                edge_data = Gx.get_edge_data(p, Gx_node)
                if isinstance(edge_data, dict):
                    dates.append(edge_data.get(self.temporal_attribute_name))
                elif isinstance(edge_data, list):
                    dates.extend(e.get(self.temporal_attribute_name) for e in edge_data if isinstance(e, dict))
        return dates

    def get_succ_dates(self, Gx, Gx_node, core_x, succ):
        """
        Get the dates of edges to successors.
        """
        dates = []
        for s in succ:
            if s in core_x:
                edge_data = Gx.get_edge_data(Gx_node, s)
                if isinstance(edge_data, dict):
                    dates.append(edge_data.get(self.temporal_attribute_name))
                elif isinstance(edge_data, list):
                    dates.extend(e.get(self.temporal_attribute_name) for e in edge_data if isinstance(e, dict))
        return dates

    def one_hop(self, Gx, Gx_node, core_x, pred, succ):
        """
        The ego node.
        """
        pred_dates = self.get_pred_dates(Gx, Gx_node, core_x, pred)
        succ_dates = self.get_succ_dates(Gx, Gx_node, core_x, succ)
        return self.test_one(pred_dates, succ_dates) and self.test_two(pred_dates, succ_dates)

    def two_hop_pred(self, Gx, Gx_node, core_x, pred):
        """
        The predecessors of the ego node.
        """
        for p in pred:
            if p in core_x:
                p_pred = set(Gx.predecessors(p)) - {Gx_node}
                p_succ = set(Gx.successors(p)) - {Gx_node}
                if not self.one_hop(Gx, p, core_x, p_pred, p_succ):
                    return False
        return True

    def two_hop_succ(self, Gx, Gx_node, core_x, succ):
        """
        The successors of the ego node.
        """
        for s in succ:
            if s in core_x:
                s_pred = set(Gx.predecessors(s)) - {Gx_node}
                s_succ = set(Gx.successors(s)) - {Gx_node}
                if not self.one_hop(Gx, s, core_x, s_pred, s_succ):
                    return False
        return True

    def test_one(self, pred_dates, succ_dates):
        """
        Edges one hop out from Gx_node in the mapping should be
        time-respecting with respect to each other, regardless of
        direction.
        """
        all_dates = pred_dates + succ_dates
        return all(abs(d1 - d2) <= self.delta for d1 in all_dates for d2 in all_dates if d1 != d2)

    def test_two(self, pred_dates, succ_dates):
        """
        Edges from a dual Gx_node in the mapping should be ordered in
        a time-respecting manner.
        """
        return all(p <= s for p in pred_dates for s in succ_dates)

    def semantic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is semantically
        feasible.

        Any subclass which redefines semantic_feasibility() must
        maintain the self.tests if needed, to keep the match() method
        functional. Implementations should consider multigraphs.
        """
        G1_pred = set(self.G1.predecessors(G1_node)) - set(self.core_1.keys())
        G2_pred = set(self.G2.predecessors(G2_node)) - set(self.core_2.keys())
        G1_succ = set(self.G1.successors(G1_node)) - set(self.core_1.keys())
        G2_succ = set(self.G2.successors(G2_node)) - set(self.core_2.keys())

        # Check one-hop time-respecting property
        if not (self.one_hop(self.G1, G1_node, self.core_1, G1_pred, G1_succ) and
                self.one_hop(self.G2, G2_node, self.core_2, G2_pred, G2_succ)):
            return False

        # Check two-hop time-respecting property
        if not (self.two_hop_pred(self.G1, G1_node, self.core_1, G1_pred) and
                self.two_hop_pred(self.G2, G2_node, self.core_2, G2_pred) and
                self.two_hop_succ(self.G1, G1_node, self.core_1, G1_succ) and
                self.two_hop_succ(self.G2, G2_node, self.core_2, G2_succ)):
            return False

        return True
