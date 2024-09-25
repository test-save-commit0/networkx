"""
Text-based visual representations of graphs
"""
import sys
import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
__all__ = ['forest_str', 'generate_network_text', 'write_network_text']


class BaseGlyphs:
    pass


class AsciiBaseGlyphs(BaseGlyphs):
    empty: str = '+'
    newtree_last: str = '+-- '
    newtree_mid: str = '+-- '
    endof_forest: str = '    '
    within_forest: str = ':   '
    within_tree: str = '|   '


class AsciiDirectedGlyphs(AsciiBaseGlyphs):
    last: str = 'L-> '
    mid: str = '|-> '
    backedge: str = '<-'
    vertical_edge: str = '!'


class AsciiUndirectedGlyphs(AsciiBaseGlyphs):
    last: str = 'L-- '
    mid: str = '|-- '
    backedge: str = '-'
    vertical_edge: str = '|'


class UtfBaseGlyphs(BaseGlyphs):
    empty: str = '╙'
    newtree_last: str = '╙── '
    newtree_mid: str = '╟── '
    endof_forest: str = '    '
    within_forest: str = '╎   '
    within_tree: str = '│   '


class UtfDirectedGlyphs(UtfBaseGlyphs):
    last: str = '└─╼ '
    mid: str = '├─╼ '
    backedge: str = '╾'
    vertical_edge: str = '╽'


class UtfUndirectedGlyphs(UtfBaseGlyphs):
    last: str = '└── '
    mid: str = '├── '
    backedge: str = '─'
    vertical_edge: str = '│'


def generate_network_text(graph, with_labels=True, sources=None, max_depth=
    None, ascii_only=False, vertical_chains=False):
    """Generate lines in the "network text" format

    This works via a depth-first traversal of the graph and writing a line for
    each unique node encountered. Non-tree edges are written to the right of
    each node, and connection to a non-tree edge is indicated with an ellipsis.
    This representation works best when the input graph is a forest, but any
    graph can be represented.

    This notation is original to networkx, although it is simple enough that it
    may be known in existing literature. See #5602 for details. The procedure
    is summarized as follows:

    1. Given a set of source nodes (which can be specified, or automatically
    discovered via finding the (strongly) connected components and choosing one
    node with minimum degree from each), we traverse the graph in depth first
    order.

    2. Each reachable node will be printed exactly once on it's own line.

    3. Edges are indicated in one of four ways:

        a. a parent "L-style" connection on the upper left. This corresponds to
        a traversal in the directed DFS tree.

        b. a backref "<-style" connection shown directly on the right. For
        directed graphs, these are drawn for any incoming edges to a node that
        is not a parent edge. For undirected graphs, these are drawn for only
        the non-parent edges that have already been represented (The edges that
        have not been represented will be handled in the recursive case).

        c. a child "L-style" connection on the lower right. Drawing of the
        children are handled recursively.

        d. if ``vertical_chains`` is true, and a parent node only has one child
        a "vertical-style" edge is drawn between them.

    4. The children of each node (wrt the directed DFS tree) are drawn
    underneath and to the right of it. In the case that a child node has already
    been drawn the connection is replaced with an ellipsis ("...") to indicate
    that there is one or more connections represented elsewhere.

    5. If a maximum depth is specified, an edge to nodes past this maximum
    depth will be represented by an ellipsis.

    6. If a a node has a truthy "collapse" value, then we do not traverse past
    that node.

    Parameters
    ----------
    graph : nx.DiGraph | nx.Graph
        Graph to represent

    with_labels : bool | str
        If True will use the "label" attribute of a node to display if it
        exists otherwise it will use the node value itself. If given as a
        string, then that attribute name will be used instead of "label".
        Defaults to True.

    sources : List
        Specifies which nodes to start traversal from. Note: nodes that are not
        reachable from one of these sources may not be shown. If unspecified,
        the minimal set of nodes needed to reach all others will be used.

    max_depth : int | None
        The maximum depth to traverse before stopping. Defaults to None.

    ascii_only : Boolean
        If True only ASCII characters are used to construct the visualization

    vertical_chains : Boolean
        If True, chains of nodes will be drawn vertically when possible.

    Yields
    ------
    str : a line of generated text

    Examples
    --------
    >>> graph = nx.path_graph(10)
    >>> graph.add_node("A")
    >>> graph.add_node("B")
    >>> graph.add_node("C")
    >>> graph.add_node("D")
    >>> graph.add_edge(9, "A")
    >>> graph.add_edge(9, "B")
    >>> graph.add_edge(9, "C")
    >>> graph.add_edge("C", "D")
    >>> graph.add_edge("C", "E")
    >>> graph.add_edge("C", "F")
    >>> nx.write_network_text(graph)
    ╙── 0
        └── 1
            └── 2
                └── 3
                    └── 4
                        └── 5
                            └── 6
                                └── 7
                                    └── 8
                                        └── 9
                                            ├── A
                                            ├── B
                                            └── C
                                                ├── D
                                                ├── E
                                                └── F
    >>> nx.write_network_text(graph, vertical_chains=True)
    ╙── 0
        │
        1
        │
        2
        │
        3
        │
        4
        │
        5
        │
        6
        │
        7
        │
        8
        │
        9
        ├── A
        ├── B
        └── C
            ├── D
            ├── E
            └── F
    """
    if sources is None:
        sources = _find_sources(graph)

    glyphs = (AsciiDirectedGlyphs() if ascii_only else UtfDirectedGlyphs()) if graph.is_directed() else (AsciiUndirectedGlyphs() if ascii_only else UtfUndirectedGlyphs())

    def _generate_lines(node, prefix='', depth=0, parent=None, seen=None):
        if seen is None:
            seen = set()

        if node in seen:
            yield f'{prefix}{glyphs.mid} ...'
            return

        seen.add(node)

        if max_depth is not None and depth > max_depth:
            yield f'{prefix}{glyphs.mid} ...'
            return

        label = node
        if with_labels:
            label = graph.nodes[node].get('label', node) if isinstance(with_labels, bool) else graph.nodes[node].get(with_labels, node)

        backedges = []
        if graph.is_directed():
            backedges = [pred for pred in graph.predecessors(node) if pred != parent and pred in seen]
        else:
            backedges = [neigh for neigh in graph.neighbors(node) if neigh != parent and neigh in seen]

        if backedges:
            backedge_str = f' {glyphs.backedge} {", ".join(map(str, backedges))}'
        else:
            backedge_str = ''

        yield f'{prefix}{glyphs.mid} {label}{backedge_str}'

        children = [child for child in graph.neighbors(node) if child != parent and child not in seen]

        for i, child in enumerate(children):
            is_last = (i == len(children) - 1)
            new_prefix = prefix + (glyphs.endof_forest if is_last else glyphs.within_forest)

            if vertical_chains and len(children) == 1:
                yield f'{new_prefix}{glyphs.vertical_edge}'
                yield from _generate_lines(child, new_prefix, depth + 1, node, seen)
            else:
                yield from _generate_lines(child, new_prefix, depth + 1, node, seen)

    for source in sources:
        yield f'{glyphs.empty}{glyphs.newtree_last} {source}'
        yield from _generate_lines(source)


@open_file(1, 'w')
def write_network_text(graph, path=None, with_labels=True, sources=None,
    max_depth=None, ascii_only=False, end='\n', vertical_chains=False):
    """Creates a nice text representation of a graph

    This works via a depth-first traversal of the graph and writing a line for
    each unique node encountered. Non-tree edges are written to the right of
    each node, and connection to a non-tree edge is indicated with an ellipsis.
    This representation works best when the input graph is a forest, but any
    graph can be represented.

    Parameters
    ----------
    graph : nx.DiGraph | nx.Graph
        Graph to represent

    path : string or file or callable or None
       Filename or file handle for data output.
       if a function, then it will be called for each generated line.
       if None, this will default to "sys.stdout.write"

    with_labels : bool | str
        If True will use the "label" attribute of a node to display if it
        exists otherwise it will use the node value itself. If given as a
        string, then that attribute name will be used instead of "label".
        Defaults to True.

    sources : List
        Specifies which nodes to start traversal from. Note: nodes that are not
        reachable from one of these sources may not be shown. If unspecified,
        the minimal set of nodes needed to reach all others will be used.

    max_depth : int | None
        The maximum depth to traverse before stopping. Defaults to None.

    ascii_only : Boolean
        If True only ASCII characters are used to construct the visualization

    end : string
        The line ending character

    vertical_chains : Boolean
        If True, chains of nodes will be drawn vertically when possible.

    Examples
    --------
    >>> graph = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    >>> nx.write_network_text(graph)
    ╙── 0
        ├─╼ 1
        │   ├─╼ 3
        │   └─╼ 4
        └─╼ 2
            ├─╼ 5
            └─╼ 6

    >>> # A near tree with one non-tree edge
    >>> graph.add_edge(5, 1)
    >>> nx.write_network_text(graph)
    ╙── 0
        ├─╼ 1 ╾ 5
        │   ├─╼ 3
        │   └─╼ 4
        └─╼ 2
            ├─╼ 5
            │   └─╼  ...
            └─╼ 6

    >>> graph = nx.cycle_graph(5)
    >>> nx.write_network_text(graph)
    ╙── 0
        ├── 1
        │   └── 2
        │       └── 3
        │           └── 4 ─ 0
        └──  ...

    >>> graph = nx.cycle_graph(5, nx.DiGraph)
    >>> nx.write_network_text(graph, vertical_chains=True)
    ╙── 0 ╾ 4
        ╽
        1
        ╽
        2
        ╽
        3
        ╽
        4
        └─╼  ...

    >>> nx.write_network_text(graph, vertical_chains=True, ascii_only=True)
    +-- 0 <- 4
        !
        1
        !
        2
        !
        3
        !
        4
        L->  ...

    >>> graph = nx.generators.barbell_graph(4, 2)
    >>> nx.write_network_text(graph, vertical_chains=False)
    ╙── 4
        ├── 5
        │   └── 6
        │       ├── 7
        │       │   ├── 8 ─ 6
        │       │   │   └── 9 ─ 6, 7
        │       │   └──  ...
        │       └──  ...
        └── 3
            ├── 0
            │   ├── 1 ─ 3
            │   │   └── 2 ─ 0, 3
            │   └──  ...
            └──  ...
    >>> nx.write_network_text(graph, vertical_chains=True)
    ╙── 4
        ├── 5
        │   │
        │   6
        │   ├── 7
        │   │   ├── 8 ─ 6
        │   │   │   │
        │   │   │   9 ─ 6, 7
        │   │   └──  ...
        │   └──  ...
        └── 3
            ├── 0
            │   ├── 1 ─ 3
            │   │   │
            │   │   2 ─ 0, 3
            │   └──  ...
            └──  ...

    >>> graph = nx.complete_graph(5, create_using=nx.Graph)
    >>> nx.write_network_text(graph)
    ╙── 0
        ├── 1
        │   ├── 2 ─ 0
        │   │   ├── 3 ─ 0, 1
        │   │   │   └── 4 ─ 0, 1, 2
        │   │   └──  ...
        │   └──  ...
        └──  ...

    >>> graph = nx.complete_graph(3, create_using=nx.DiGraph)
    >>> nx.write_network_text(graph)
    ╙── 0 ╾ 1, 2
        ├─╼ 1 ╾ 2
        │   ├─╼ 2 ╾ 0
        │   │   └─╼  ...
        │   └─╼  ...
        └─╼  ...
    """
    pass


def _find_sources(graph):
    """
    Determine a minimal set of nodes such that the entire graph is reachable
    """
    if graph.is_directed():
        sccs = list(nx.strongly_connected_components(graph))
        return [min(scc, key=lambda n: graph.in_degree(n)) for scc in sccs]
    else:
        return [min(cc, key=graph.degree) for cc in nx.connected_components(graph)]


def forest_str(graph, with_labels=True, sources=None, write=None,
    ascii_only=False):
    """Creates a nice utf8 representation of a forest

    This function has been superseded by
    :func:`nx.readwrite.text.generate_network_text`, which should be used
    instead.

    Parameters
    ----------
    graph : nx.DiGraph | nx.Graph
        Graph to represent (must be a tree, forest, or the empty graph)

    with_labels : bool
        If True will use the "label" attribute of a node to display if it
        exists otherwise it will use the node value itself. Defaults to True.

    sources : List
        Mainly relevant for undirected forests, specifies which nodes to list
        first. If unspecified the root nodes of each tree will be used for
        directed forests; for undirected forests this defaults to the nodes
        with the smallest degree.

    write : callable
        Function to use to write to, if None new lines are appended to
        a list and returned. If set to the `print` function, lines will
        be written to stdout as they are generated. If specified,
        this function will return None. Defaults to None.

    ascii_only : Boolean
        If True only ASCII characters are used to construct the visualization

    Returns
    -------
    str | None :
        utf8 representation of the tree / forest

    Examples
    --------
    >>> graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
    >>> print(nx.forest_str(graph))
    ╙── 0
        ├─╼ 1
        │   ├─╼ 3
        │   │   ├─╼ 7
        │   │   └─╼ 8
        │   └─╼ 4
        │       ├─╼ 9
        │       └─╼ 10
        └─╼ 2
            ├─╼ 5
            │   ├─╼ 11
            │   └─╼ 12
            └─╼ 6
                ├─╼ 13
                └─╼ 14


    >>> graph = nx.balanced_tree(r=1, h=2, create_using=nx.Graph)
    >>> print(nx.forest_str(graph))
    ╙── 0
        └── 1
            └── 2

    >>> print(nx.forest_str(graph, ascii_only=True))
    +-- 0
        L-- 1
            L-- 2
    """
    lines = list(generate_network_text(graph, with_labels=with_labels, sources=sources, ascii_only=ascii_only))
    
    if write is None:
        return '\n'.join(lines)
    else:
        for line in lines:
            write(line + '\n')


def _parse_network_text(lines):
    """Reconstructs a graph from a network text representation.

    This is mainly used for testing.  Network text is for display, not
    serialization, as such this cannot parse all network text representations
    because node labels can be ambiguous with the glyphs and indentation used
    to represent edge structure. Additionally, there is no way to determine if
    disconnected graphs were originally directed or undirected.

    Parameters
    ----------
    lines : list or iterator of strings
        Input data in network text format

    Returns
    -------
    G: NetworkX graph
        The graph corresponding to the lines in network text format.
    """
    G = nx.DiGraph()
    stack = []
    current_depth = -1

    for line in lines:
        depth = (len(line) - len(line.lstrip())) // 4
        content = line.strip()

        if not content:
            continue

        if depth <= current_depth:
            for _ in range(current_depth - depth + 1):
                stack.pop()

        current_depth = depth

        if '─' in content or '--' in content:
            node = content.split('─')[-1].split('--')[-1].strip()
            if stack:
                G.add_edge(stack[-1], node)
            stack.append(node)
        elif '╾' in content or '<-' in content:
            node, backedges = content.split('╾' if '╾' in content else '<-')
            node = node.strip()
            backedges = [edge.strip() for edge in backedges.split(',')]
            if stack:
                G.add_edge(stack[-1], node)
            for backedge in backedges:
                G.add_edge(backedge, node)
            stack.append(node)
        elif '...' in content:
            continue
        else:
            node = content
            if stack:
                G.add_edge(stack[-1], node)
            stack.append(node)

    return G
