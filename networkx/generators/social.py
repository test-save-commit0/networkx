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
    G = nx.Graph()
    G.add_nodes_from(range(34))

    club1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21]
    club2 = [9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

    for node in club1:
        G.nodes[node]['club'] = 'Mr. Hi'
    for node in club2:
        G.nodes[node]['club'] = 'Officer'

    edges = [(0, 1, 4), (0, 2, 5), (0, 3, 3), (0, 4, 3), (0, 5, 3), (0, 6, 3), (0, 7, 2), (0, 8, 2), (0, 10, 2),
             (0, 11, 3), (0, 12, 1), (0, 13, 3), (0, 17, 2), (0, 19, 2), (0, 21, 2), (0, 31, 2), (1, 2, 6), (1, 3, 3),
             (1, 7, 4), (1, 13, 5), (1, 17, 2), (1, 19, 1), (1, 21, 2), (1, 30, 2), (2, 3, 3), (2, 7, 4), (2, 8, 5),
             (2, 9, 1), (2, 13, 3), (2, 27, 2), (2, 28, 2), (2, 32, 2), (3, 7, 3), (3, 12, 3), (3, 13, 3), (4, 6, 2),
             (4, 10, 3), (5, 6, 5), (5, 10, 3), (5, 16, 3), (6, 16, 3), (8, 30, 3), (8, 32, 3), (8, 33, 4), (9, 33, 2),
             (13, 33, 3), (14, 32, 3), (14, 33, 2), (15, 32, 3), (15, 33, 2), (18, 32, 1), (18, 33, 2), (19, 33, 2),
             (20, 32, 2), (20, 33, 2), (22, 32, 2), (22, 33, 2), (23, 25, 5), (23, 27, 4), (23, 29, 4), (23, 32, 2),
             (23, 33, 4), (24, 25, 2), (24, 27, 3), (24, 31, 2), (25, 31, 4), (26, 29, 3), (26, 33, 2), (27, 33, 4),
             (28, 31, 2), (28, 33, 2), (29, 32, 2), (29, 33, 2), (30, 32, 3), (30, 33, 3), (31, 32, 3), (31, 33, 3),
             (32, 33, 4)]

    G.add_weighted_edges_from(edges)
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def davis_southern_women_graph():
    """Returns Davis Southern women social network.

    This is a bipartite graph.

    References
    ----------
    .. [1] A. Davis, Gardner, B. B., Gardner, M. R., 1941. Deep South.
        University of Chicago Press, Chicago, IL.
    """
    G = nx.Graph()
    women = ['Evelyn', 'Laura', 'Theresa', 'Brenda', 'Charlotte', 'Frances', 'Eleanor',
             'Pearl', 'Ruth', 'Verne', 'Myrna', 'Katherine', 'Sylvia', 'Nora', 'Helen',
             'Dorothy', 'Olivia', 'Flora']
    events = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14']

    G.add_nodes_from(women, bipartite=0)
    G.add_nodes_from(events, bipartite=1)

    G.add_edges_from([('Evelyn', 'E1'), ('Evelyn', 'E2'), ('Evelyn', 'E3'), ('Evelyn', 'E4'), ('Evelyn', 'E5'),
                      ('Evelyn', 'E6'), ('Evelyn', 'E8'), ('Evelyn', 'E9'), ('Laura', 'E1'), ('Laura', 'E2'),
                      ('Laura', 'E3'), ('Laura', 'E5'), ('Laura', 'E6'), ('Laura', 'E7'), ('Laura', 'E8'),
                      ('Theresa', 'E2'), ('Theresa', 'E3'), ('Theresa', 'E4'), ('Theresa', 'E5'), ('Theresa', 'E6'),
                      ('Theresa', 'E7'), ('Theresa', 'E8'), ('Theresa', 'E9'), ('Brenda', 'E1'), ('Brenda', 'E3'),
                      ('Brenda', 'E4'), ('Brenda', 'E5'), ('Brenda', 'E6'), ('Brenda', 'E7'), ('Brenda', 'E8'),
                      ('Charlotte', 'E3'), ('Charlotte', 'E4'), ('Charlotte', 'E5'), ('Charlotte', 'E7'),
                      ('Frances', 'E3'), ('Frances', 'E5'), ('Frances', 'E6'), ('Frances', 'E8'),
                      ('Eleanor', 'E5'), ('Eleanor', 'E6'), ('Eleanor', 'E7'), ('Eleanor', 'E8'),
                      ('Pearl', 'E6'), ('Pearl', 'E8'), ('Pearl', 'E9'),
                      ('Ruth', 'E5'), ('Ruth', 'E7'), ('Ruth', 'E8'), ('Ruth', 'E9'),
                      ('Verne', 'E7'), ('Verne', 'E8'), ('Verne', 'E9'), ('Verne', 'E10'),
                      ('Myrna', 'E8'), ('Myrna', 'E9'), ('Myrna', 'E10'), ('Myrna', 'E12'),
                      ('Katherine', 'E8'), ('Katherine', 'E9'), ('Katherine', 'E10'), ('Katherine', 'E12'),
                      ('Sylvia', 'E7'), ('Sylvia', 'E8'), ('Sylvia', 'E9'), ('Sylvia', 'E10'), ('Sylvia', 'E12'),
                      ('Nora', 'E6'), ('Nora', 'E7'), ('Nora', 'E9'), ('Nora', 'E10'), ('Nora', 'E11'),
                      ('Helen', 'E7'), ('Helen', 'E8'), ('Helen', 'E10'), ('Helen', 'E11'), ('Helen', 'E12'),
                      ('Dorothy', 'E8'), ('Dorothy', 'E9'), ('Dorothy', 'E10'), ('Dorothy', 'E11'), ('Dorothy', 'E12'),
                      ('Olivia', 'E9'), ('Olivia', 'E11'),
                      ('Flora', 'E9'), ('Flora', 'E11')])

    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def florentine_families_graph():
    """Returns Florentine families graph.

    References
    ----------
    .. [1] Ronald L. Breiger and Philippa E. Pattison
       Cumulated social roles: The duality of persons and their algebras,1
       Social Networks, Volume 8, Issue 3, September 1986, Pages 215-256
    """
    G = nx.Graph()
    G.add_edges_from([
        ('Acciaiuoli', 'Medici'),
        ('Castellani', 'Peruzzi'),
        ('Castellani', 'Strozzi'),
        ('Castellani', 'Barbadori'),
        ('Medici', 'Barbadori'),
        ('Medici', 'Ridolfi'),
        ('Medici', 'Tornabuoni'),
        ('Medici', 'Albizzi'),
        ('Medici', 'Salviati'),
        ('Salviati', 'Pazzi'),
        ('Peruzzi', 'Strozzi'),
        ('Peruzzi', 'Bischeri'),
        ('Strozzi', 'Ridolfi'),
        ('Strozzi', 'Bischeri'),
        ('Ridolfi', 'Tornabuoni'),
        ('Tornabuoni', 'Guadagni'),
        ('Albizzi', 'Ginori'),
        ('Albizzi', 'Guadagni'),
        ('Bischeri', 'Guadagni'),
        ('Guadagni', 'Lamberteschi')
    ])
    return G


@nx._dispatchable(graphs=None, returns_graph=True)
def les_miserables_graph():
    """Returns coappearance network of characters in the novel Les Miserables.

    References
    ----------
    .. [1] D. E. Knuth, 1993.
       The Stanford GraphBase: a platform for combinatorial computing,
       pp. 74-87. New York: AcM Press.
    """
    G = nx.Graph()
    characters = [
        "Myriel", "Napoleon", "MlleBaptistine", "MmeMagloire", "CountessDeLo",
        "Geborand", "Champtercier", "Cravatte", "Count", "OldMan", "Labarre",
        "Valjean", "Marguerite", "MmeDeR", "Isabeau", "Gervais", "Tholomyes",
        "Listolier", "Fameuil", "Blacheville", "Favourite", "Dahlia", "Zephine",
        "Fantine", "MmeThenardier", "Thenardier", "Cosette", "Javert", "Fauchelevent",
        "Bamatabois", "Perpetue", "Simplice", "Scaufflaire", "Woman1", "Judge",
        "Champmathieu", "Brevet", "Chenildieu", "Cochepaille", "Pontmercy",
        "Boulatruelle", "Eponine", "Anzelma", "Woman2", "MotherInnocent", "Gribier",
        "Jondrette", "MmeBurgon", "Gavroche", "Gillenormand", "Magnon", "MlleGillenormand",
        "MmePontmercy", "MlleVaubois", "LtGillenormand", "Marius", "BaronessT",
        "Mabeuf", "Enjolras", "Combeferre", "Prouvaire", "Feuilly", "Courfeyrac",
        "Bahorel", "Bossuet", "Joly", "Grantaire", "MotherPlutarch", "Gueulemer",
        "Babet", "Claquesous", "Montparnasse", "Toussaint", "Child1", "Child2",
        "Brujon", "MmeHucheloup"
    ]
    G.add_nodes_from(characters)

    coappearances = [
        ("Napoleon", "Myriel", 1), ("Mlle.Baptistine", "Myriel", 8),
        ("Mme.Magloire", "Myriel", 10), ("Mme.Magloire", "Mlle.Baptistine", 6),
        ("CountessDeLo", "Myriel", 1), ("Geborand", "Myriel", 1),
        ("Champtercier", "Myriel", 1), ("Cravatte", "Myriel", 1),
        ("Count", "Myriel", 2), ("OldMan", "Myriel", 1), ("Valjean", "Labarre", 1),
        ("Valjean", "Mme.Magloire", 3), ("Valjean", "Mlle.Baptistine", 3),
        ("Valjean", "Myriel", 11), ("Marguerite", "Valjean", 3),
        ("Mme.deR", "Valjean", 1), ("Isabeau", "Valjean", 1),
        ("Gervais", "Valjean", 1), ("Listolier", "Tholomyes", 4),
        ("Fameuil", "Tholomyes", 4), ("Fameuil", "Listolier", 4),
        ("Blacheville", "Tholomyes", 4), ("Blacheville", "Listolier", 4),
        ("Blacheville", "Fameuil", 4), ("Favourite", "Tholomyes", 3),
        ("Favourite", "Listolier", 3), ("Favourite", "Fameuil", 3),
        ("Favourite", "Blacheville", 4), ("Dahlia", "Tholomyes", 3),
        ("Dahlia", "Listolier", 3), ("Dahlia", "Fameuil", 3),
        ("Dahlia", "Blacheville", 3), ("Dahlia", "Favourite", 5),
        ("Zephine", "Tholomyes", 3), ("Zephine", "Listolier", 3),
        ("Zephine", "Fameuil", 3), ("Zephine", "Blacheville", 3),
        ("Zephine", "Favourite", 4), ("Zephine", "Dahlia", 4),
        ("Fantine", "Tholomyes", 3), ("Fantine", "Listolier", 3),
        ("Fantine", "Fameuil", 3), ("Fantine", "Blacheville", 3),
        ("Fantine", "Favourite", 4), ("Fantine", "Dahlia", 4),
        ("Fantine", "Zephine", 4), ("Fantine", "Marguerite", 2),
        ("Fantine", "Valjean", 9), ("Mme.Thenardier", "Fantine", 2),
        ("Mme.Thenardier", "Valjean", 7), ("Thenardier", "Mme.Thenardier", 13),
        ("Thenardier", "Fantine", 1), ("Thenardier", "Valjean", 12),
        ("Cosette", "Mme.Thenardier", 4), ("Cosette", "Valjean", 31),
        ("Cosette", "Tholomyes", 1), ("Cosette", "Thenardier", 1),
        ("Javert", "Valjean", 17), ("Javert", "Fantine", 5),
        ("Javert", "Thenardier", 5), ("Javert", "Mme.Thenardier", 1),
        ("Javert", "Cosette", 1), ("Fauchelevent", "Valjean", 8),
        ("Fauchelevent", "Javert", 1), ("Bamatabois", "Fantine", 1),
        ("Bamatabois", "Javert", 1), ("Bamatabois", "Valjean", 2),
        ("Perpetue", "Fantine", 1), ("Simplice", "Perpetue", 2),
        ("Simplice", "Valjean", 3), ("Simplice", "Fantine", 2),
        ("Simplice", "Javert", 1), ("Scaufflaire", "Valjean", 1),
        ("Woman1", "Valjean", 2), ("Woman1", "Javert", 1),
        ("Judge", "Valjean", 3), ("Judge", "Bamatabois", 2),
        ("Champmathieu", "Valjean", 3), ("Champmathieu", "Judge", 3),
        ("Champmathieu", "Bamatabois", 2), ("Brevet", "Judge", 2),
        ("Brevet", "Champmathieu", 2), ("Brevet", "Valjean", 2),
        ("Brevet", "Bamatabois", 1), ("Chenildieu", "Judge", 2),
        ("Chenildieu", "Champmathieu", 2), ("Chenildieu", "Brevet", 2),
        ("Chenildieu", "Valjean", 2), ("Chenildieu", "Bamatabois", 1),
        ("Cochepaille", "Judge", 2), ("Cochepaille", "Champmathieu", 2),
        ("Cochepaille", "Brevet", 2), ("Cochepaille", "Chenildieu", 2),
        ("Cochepaille", "Valjean", 2), ("Cochepaille", "Bamatabois", 1),
        ("Pontmercy", "Thenardier", 1), ("Boulatruelle", "Thenardier", 1),
        ("Eponine", "Mme.Thenardier", 2), ("Eponine", "Thenardier", 3),
        ("Anzelma", "Eponine", 2), ("Anzelma", "Thenardier", 2),
        ("Anzelma", "Mme.Thenardier", 1), ("Woman2", "Valjean", 3),
        ("Woman2", "Cosette", 1), ("Woman2", "Javert", 1),
        ("MotherInnocent", "Fauchelevent", 3), ("MotherInnocent", "Valjean", 1),
        ("Gribier", "Fauchelevent", 2), ("Mme.Burgon", "Jondrette", 1),
        ("Gavroche", "Mme.Burgon", 2), ("Gavroche", "Thenardier", 1),
        ("Gavroche", "Javert", 1), ("Gavroche", "Valjean", 1),
        ("Gillenormand", "Cosette", 3), ("Gillenormand", "Valjean", 2),
        ("Magnon", "Gillenormand", 1), ("Magnon", "Mme.Thenardier", 1),
        ("Mlle.Gillenormand", "Gillenormand", 9), ("Mlle.Gillenormand", "Cosette", 2),
        ("Mlle.Gillenormand", "Valjean", 2), ("Mme.Pontmercy", "Mlle.Gillenormand", 1),
        ("Mme.Pontmercy", "Pontmercy", 1), ("Mlle.Vaubois", "Mlle.Gillenormand", 1),
        ("Lt.Gillenormand", "Mlle.Gillenormand", 2), ("Lt.Gillenormand", "Gillenormand", 1),
        ("Lt.Gillenormand", "Cosette", 1), ("Marius", "Mlle.Gillenormand", 6),
        ("Marius", "Gillenormand", 12), ("Marius", "Pontmercy", 1),
        ("Marius", "Lt.Gillenormand", 1), ("Marius", "Cosette", 21),
        ("Marius", "Valjean", 19), ("Marius", "Tholomyes", 1),
        ("Marius", "Thenardier", 2), ("Marius", "Eponine", 5),
        ("Marius", "Gavroche", 4), ("BaronessT", "Gillenormand", 1),
        ("BaronessT", "Marius", 1), ("Mabeuf", "Marius", 1),
        ("Mabeuf", "Eponine", 1), ("Mabeuf", "Gavroche", 1),
        ("Enjolras", "Marius", 7), ("Enjolras", "Gavroche", 7),
        ("Enjolras", "Javert", 6), ("Enjolras", "Mabeuf", 1),
        ("Enjolras", "Valjean", 4), ("Combeferre", "Enjolras", 15),
        ("Combeferre", "Marius", 5), ("Combeferre", "Gavroche", 6),
        ("Combeferre", "Mabeuf", 2), ("Prouvaire", "Gavroche", 1),
        ("Prouvaire", "Enjolras", 4), ("Prouvaire", "Combeferre", 2),
        ("Feuilly", "Gavroche", 2), ("Feuilly", "Enjolras", 6),
        ("Feuilly", "Prouvaire", 2), ("Feuilly", "Combeferre", 5),
        ("Feuilly", "Mabeuf", 1), ("Feuilly", "Marius", 1),
        ("Courfeyrac", "Marius", 9), ("Courfeyrac", "Enjolras", 17),
        ("Courfeyrac", "Combeferre", 13), ("Courfeyrac", "Gavroche", 7),
        ("Courfeyrac", "Mabeuf", 2), ("Courfeyrac", "Eponine", 1),
        ("Courfeyrac", "Feuilly", 6), ("Courfeyrac", "Prouvaire", 3),
        ("Bahorel", "Combeferre", 5), ("Bahorel", "Gavroche", 5),
        ("Bahorel", "Courfeyrac", 6), ("Bahorel", "Mabeuf", 2),
        ("Bahorel", "Enjolras", 4), ("Bahorel", "Feuilly", 3),
        ("Bahorel", "Prouvaire", 2), ("Bahorel", "Marius", 1),
        ("Bossuet", "Marius", 5), ("Bossuet", "Courfeyrac", 12),
        ("Bossuet", "Gavroche", 5), ("Bossuet", "Bahorel", 4),
        ("Bossuet", "Enjolras", 10), ("Bossuet", "Feuilly", 6),
        ("Bossuet", "Prouvaire", 2), ("Bossuet", "Combeferre", 9),
        ("Bossuet", "Mabeuf", 1), ("Bossuet", "Valjean", 1),
        ("Joly", "Bahorel", 5), ("Joly", "Bossuet", 7),
        ("Joly", "Gavroche", 3), ("Joly", "Courfeyrac", 5),
        ("Joly", "Enjolras", 5), ("Joly", "Feuilly", 5),
        ("Joly", "Prouvaire", 2), ("Joly", "Combeferre", 5),
        ("Joly", "Mabeuf", 1), ("Joly", "Marius", 2),
        ("Grantaire", "Bossuet", 3), ("Grantaire", "Enjolras", 3),
        ("Grantaire", "Combeferre", 1), ("Grantaire", "Courfeyrac", 2),
        ("Grantaire", "Joly", 2), ("Grantaire", "Gavroche", 1),
        ("Grantaire", "Bahorel", 1), ("Grantaire", "Feuilly", 1),
        ("Grantaire", "Prouvaire", 1), ("MotherPlutarch", "Mabeuf", 3),
        ("Gueulemer", "Thenardier", 5), ("Gueulemer", "Valjean", 1),
        ("Gueulemer", "Mme.Thenardier", 1), ("Gueulemer", "Javert", 1),
        ("Gueulemer", "Gavroche", 1), ("Gueulemer", "Eponine", 1),
        ("Babet", "Thenardier", 6), ("Babet", "Gueulemer", 6),
        ("Babet", "Valjean", 1), ("Babet", "Mme.Thenardier", 1),
        ("Babet", "Javert", 2), ("Babet", "Gavroche", 1),
        ("Babet", "Eponine", 1), ("Claquesous", "Thenardier", 4),
        ("Claquesous", "Babet", 4), ("Claquesous", "Gueulemer", 4),
        ("Claquesous", "Valjean", 1), ("Claquesous", "Mme.Thenardier", 1),
        ("Claquesous", "Javert", 1), ("Claquesous", "Eponine", 1),
        ("Claquesous", "Enjolras", 1), ("Montparnasse", "Javert", 1),
        ("Montparnasse", "Babet", 2), ("Montparnasse", "Gueulemer", 2),
        ("Montparnasse", "Claquesous", 2), ("Montparnasse", "Valjean", 1),
        ("Montparnasse", "Gavroche", 1), ("Montparnasse", "Eponine", 1),
        ("Montparnasse", "Thenardier", 1), ("Toussaint", "Cosette", 2),
        ("Toussaint", "Javert", 1), ("Toussaint", "Valjean", 1),
        ("Child1", "Gavroche", 2), ("Child2", "Gavroche", 2),
        ("Child2", "Child1", 3), ("Brujon", "Babet", 3),
        ("Brujon", "Gueulemer", 3), ("Brujon", "Thenardier", 3),
        ("Brujon", "Gavroche", 1), ("Brujon", "Eponine", 1),
        ("Brujon", "Claquesous", 1), ("Brujon", "Montparnasse", 1),
        ("Mme.Hucheloup", "Bossuet", 1), ("Mme.Hucheloup", "Joly", 1),
        ("Mme.Hucheloup", "Grantaire", 1), ("Mme.Hucheloup", "Bahorel", 1),
        ("Mme.Hucheloup", "Courfeyrac", 1), ("Mme.Hucheloup", "Gavroche", 1),
        ("Mme.Hucheloup", "Enjolras", 1)
    ]

    G.add_weighted_edges_from(coappearances)
    return G
