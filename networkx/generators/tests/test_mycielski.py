"""Unit tests for the :mod:`networkx.generators.mycielski` module."""

import pytest

import networkx as nx


class TestMycielski:
    def test_construction(self):
        G = nx.path_graph(2)
        M = nx.mycielskian(G)
        assert nx.is_isomorphic(M, nx.cycle_graph(5))

    def test_size(self):
        G = nx.path_graph(2)
        M = nx.mycielskian(G, 2)
        assert len(M) == 11
        assert M.size() == 20

    def test_mycielski_graph_generator(self):
        G = nx.mycielski_graph(1)
        assert nx.is_isomorphic(G, nx.empty_graph(1))
        G = nx.mycielski_graph(2)
        assert nx.is_isomorphic(G, nx.path_graph(2))
        G = nx.mycielski_graph(3)
        assert nx.is_isomorphic(G, nx.cycle_graph(5))
        G = nx.mycielski_graph(4)
        assert nx.is_isomorphic(G, nx.mycielskian(nx.cycle_graph(5)))
        with pytest.raises(ValueError, match="n must be a positive integer"):
            nx.mycielski_graph(0)

    def test_mycielskian_raises(self):
        G = nx.Graph()
        with pytest.raises(ValueError, match="Number of iterations must be non-negative"):
            nx.mycielskian(G, -1)

    def test_mycielskian_empty_graph(self):
        G = nx.Graph()
        M = nx.mycielskian(G)
        assert nx.is_isomorphic(M, nx.path_graph(2))

    def test_mycielskian_multiple_iterations(self):
        G = nx.path_graph(2)
        M = nx.mycielskian(G, iterations=2)
        assert M.number_of_nodes() == 11
        assert M.number_of_edges() == 20

    def test_mycielski_graph_properties(self):
        for i in range(1, 5):
            G = nx.mycielski_graph(i)
            assert nx.number_of_nodes(G) == 3 * 2**(i-2) - 1
            assert nx.is_connected(G)
            assert nx.is_triangle_free(G)
            assert nx.chromatic_number(G) == i

    def test_mycielskian_preserves_triangle_free(self):
        G = nx.cycle_graph(5)
        M = nx.mycielskian(G)
        assert nx.is_triangle_free(G)
        assert nx.is_triangle_free(M)
