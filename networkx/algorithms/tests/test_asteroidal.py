import networkx as nx
from networkx.algorithms.asteroidal import find_asteroidal_triple, is_at_free, create_component_structure


def test_is_at_free():
    is_at_free = nx.asteroidal.is_at_free

    cycle = nx.cycle_graph(6)
    assert not is_at_free(cycle)

    path = nx.path_graph(6)
    assert is_at_free(path)

    small_graph = nx.complete_graph(2)
    assert is_at_free(small_graph)

    petersen = nx.petersen_graph()
    assert not is_at_free(petersen)

    clique = nx.complete_graph(6)
    assert is_at_free(clique)

    line_clique = nx.line_graph(clique)
    assert not is_at_free(line_clique)


def test_find_asteroidal_triple():
    # Test with a cycle graph (should find an asteroidal triple)
    cycle = nx.cycle_graph(6)
    at = find_asteroidal_triple(cycle)
    assert at is not None
    assert len(at) == 3
    assert all(not cycle.has_edge(at[i], at[j]) for i in range(3) for j in range(i+1, 3))

    # Test with a path graph (should not find an asteroidal triple)
    path = nx.path_graph(6)
    assert find_asteroidal_triple(path) is None

    # Test with a complete graph (should not find an asteroidal triple)
    clique = nx.complete_graph(6)
    assert find_asteroidal_triple(clique) is None

    # Test with Petersen graph (should find an asteroidal triple)
    petersen = nx.petersen_graph()
    at = find_asteroidal_triple(petersen)
    assert at is not None
    assert len(at) == 3
    assert all(not petersen.has_edge(at[i], at[j]) for i in range(3) for j in range(i+1, 3))


def test_create_component_structure():
    # Test with a simple graph
    G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    cs = create_component_structure(G)
    
    # Check that the structure is correct for each node
    assert cs[0][2] == cs[0][3]  # 2 and 3 should be in the same component when removing N[0]
    assert cs[0][1] == 0  # 1 should be in N[0]
    assert cs[1][3] == cs[1][4]  # 3 and 4 should be in the same component when removing N[1]
    assert cs[1][0] == 0  # 0 should be in N[1]

    # Test with a disconnected graph
    G = nx.Graph([(0, 1), (2, 3)])
    cs = create_component_structure(G)
    
    # Check that disconnected components are identified correctly
    assert cs[0][2] != cs[0][3]  # 2 and 3 should be in different components when removing N[0]
    assert cs[1][2] != cs[1][3]  # 2 and 3 should be in different components when removing N[1]
