"""
Famous social networks.
"""
import networkx as nx
__all__ = ['karate_club_graph', 'davis_southern_women_graph',
    'florentine_families_graph', 'les_miserables_graph']


@nx._dispatchable(graphs=None, returns_graph=True)
def karate_club_graph():
    """Returns Zachary's Karate Club graph.

    Each node in the returned graph has a node attribute 'club' that
    indicates the name of the club to which the member represented by that node
    belongs, either 'Mr. Hi' or 'Officer'. Each edge has a weight based on the
    number of contexts in which that edge's incident node members interacted.

    Examples
    --------
    To get the name of the club to which a node belongs::

        >>> G = nx.karate_club_graph()
        >>> G.nodes[5]["club"]
        'Mr. Hi'
        >>> G.nodes[9]["club"]
        'Officer'

    References
    ----------
    .. [1] Zachary, Wayne W.
       "An Information Flow Model for Conflict and Fission in Small Groups."
       *Journal of Anthropological Research*, 33, 452--473, (1977).
    """
    pass


@nx._dispatchable(graphs=None, returns_graph=True)
def davis_southern_women_graph():
    """Returns Davis Southern women social network.

    This is a bipartite graph.

    References
    ----------
    .. [1] A. Davis, Gardner, B. B., Gardner, M. R., 1941. Deep South.
        University of Chicago Press, Chicago, IL.
    """
    pass


@nx._dispatchable(graphs=None, returns_graph=True)
def florentine_families_graph():
    """Returns Florentine families graph.

    References
    ----------
    .. [1] Ronald L. Breiger and Philippa E. Pattison
       Cumulated social roles: The duality of persons and their algebras,1
       Social Networks, Volume 8, Issue 3, September 1986, Pages 215-256
    """
    pass


@nx._dispatchable(graphs=None, returns_graph=True)
def les_miserables_graph():
    """Returns coappearance network of characters in the novel Les Miserables.

    References
    ----------
    .. [1] D. E. Knuth, 1993.
       The Stanford GraphBase: a platform for combinatorial computing,
       pp. 74-87. New York: AcM Press.
    """
    pass
