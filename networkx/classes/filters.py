"""Filter factories to hide or show sets of nodes and edges.

These filters return the function used when creating `SubGraph`.
"""
__all__ = ['no_filter', 'hide_nodes', 'hide_edges', 'hide_multiedges',
    'hide_diedges', 'hide_multidiedges', 'show_nodes', 'show_edges',
    'show_multiedges', 'show_diedges', 'show_multidiedges']


def no_filter(*items):
    """Returns a filter function that always evaluates to True."""
    return lambda *args: True


def hide_nodes(nodes):
    """Returns a filter function that hides specific nodes."""
    nodes_set = set(nodes)
    return lambda node: node not in nodes_set


def hide_diedges(edges):
    """Returns a filter function that hides specific directed edges."""
    edges_set = set((u, v) for u, v in edges)
    return lambda u, v, k: (u, v) not in edges_set


def hide_edges(edges):
    """Returns a filter function that hides specific undirected edges."""
    edges_set = set(frozenset((u, v)) for u, v in edges)
    return lambda u, v, k: frozenset((u, v)) not in edges_set


def hide_multidiedges(edges):
    """Returns a filter function that hides specific multi-directed edges."""
    edges_set = set((u, v, k) for u, v, k in edges)
    return lambda u, v, k: (u, v, k) not in edges_set


def hide_multiedges(edges):
    """Returns a filter function that hides specific multi-undirected edges."""
    edges_set = set(frozenset((u, v, k)) for u, v, k in edges)
    return lambda u, v, k: frozenset((u, v, k)) not in edges_set


class show_nodes:
    """Filter class to show specific nodes."""

    def __init__(self, nodes):
        self.nodes = set(nodes)

    def __call__(self, node):
        return node in self.nodes


def show_diedges(edges):
    """Returns a filter function that shows specific directed edges."""
    edges_set = set((u, v) for u, v in edges)
    return lambda u, v, k: (u, v) in edges_set


def show_edges(edges):
    """Returns a filter function that shows specific undirected edges."""
    edges_set = set(frozenset((u, v)) for u, v in edges)
    return lambda u, v, k: frozenset((u, v)) in edges_set


def show_multidiedges(edges):
    """Returns a filter function that shows specific multi-directed edges."""
    edges_set = set((u, v, k) for u, v, k in edges)
    return lambda u, v, k: (u, v, k) in edges_set


def show_multiedges(edges):
    """Returns a filter function that shows specific multi-undirected edges."""
    edges_set = set(frozenset((u, v, k)) for u, v, k in edges)
    return lambda u, v, k: frozenset((u, v, k)) in edges_set
