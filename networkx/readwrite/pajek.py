"""
*****
Pajek
*****
Read graphs in Pajek format.

This implementation handles directed and undirected graphs including
those with self loops and parallel edges.

Format
------
See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
for format information.

"""
import warnings
import networkx as nx
from networkx.utils import open_file
__all__ = ['read_pajek', 'parse_pajek', 'generate_pajek', 'write_pajek']


def generate_pajek(G):
    """Generate lines in Pajek graph format.

    Parameters
    ----------
    G : graph
       A Networkx graph

    References
    ----------
    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
    for format information.
    """
    if G.name == '':
        name = 'NetworkX'
    else:
        name = G.name
    yield f'*Network {name}\n'
    
    # Write nodes
    yield f'*Vertices {G.number_of_nodes()}\n'
    for i, node in enumerate(G.nodes(), start=1):
        yield f'{i} {make_qstr(node)}\n'
    
    # Write edges
    if G.is_directed():
        yield '*Arcs\n'
    else:
        yield '*Edges\n'
    
    for u, v, data in G.edges(data=True):
        edge = ' '.join(map(make_qstr, (G.nodes().index(u) + 1, G.nodes().index(v) + 1)))
        if data:
            edge += f' {make_qstr(data)}'
        yield edge + '\n'


@open_file(1, mode='wb')
def write_pajek(G, path, encoding='UTF-8'):
    """Write graph in Pajek format to path.

    Parameters
    ----------
    G : graph
       A Networkx graph
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be compressed.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_pajek(G, "test.net")

    Warnings
    --------
    Optional node attributes and edge attributes must be non-empty strings.
    Otherwise it will not be written into the file. You will need to
    convert those attributes to strings if you want to keep them.

    References
    ----------
    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
    for format information.
    """
    for line in generate_pajek(G):
        line = line.encode(encoding)
        path.write(line)


@open_file(0, mode='rb')
@nx._dispatchable(graphs=None, returns_graph=True)
def read_pajek(path, encoding='UTF-8'):
    """Read graph in Pajek format from path.

    Parameters
    ----------
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be uncompressed.

    Returns
    -------
    G : NetworkX MultiGraph or MultiDiGraph.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_pajek(G, "test.net")
    >>> G = nx.read_pajek("test.net")

    To create a Graph instead of a MultiGraph use

    >>> G1 = nx.Graph(G)

    References
    ----------
    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
    for format information.
    """
    lines = (line.decode(encoding) for line in path)
    return parse_pajek(lines)


@nx._dispatchable(graphs=None, returns_graph=True)
def parse_pajek(lines):
    """Parse Pajek format graph from string or iterable.

    Parameters
    ----------
    lines : string or iterable
       Data in Pajek format.

    Returns
    -------
    G : NetworkX graph

    See Also
    --------
    read_pajek

    """
    import shlex
    
    lines = iter(lines)
    G = nx.MultiDiGraph()
    
    # Skip comments and empty lines
    for line in lines:
        line = line.strip()
        if line.startswith('*'):
            break
    
    # Process vertices
    if line.lower().startswith('*vertices'):
        for line in lines:
            if line.lower().startswith('*arcs') or line.lower().startswith('*edges'):
                break
            split = shlex.split(line)
            if len(split) < 2:
                continue
            v = split[1]
            G.add_node(v)
    
    # Process edges
    if line.lower().startswith('*arcs'):
        G = nx.MultiDiGraph(G)
    elif line.lower().startswith('*edges'):
        G = nx.MultiGraph(G)
    
    for line in lines:
        split = shlex.split(line)
        if len(split) < 2:
            continue
        u, v = split[:2]
        data = split[2:] if len(split) > 2 else {}
        G.add_edge(u, v, **data)
    
    return G


def make_qstr(t):
    """Returns the string representation of t.
    Add outer double-quotes if the string has a space.
    """
    s = str(t)
    if ' ' in s:
        return f'"{s}"'
    return s
