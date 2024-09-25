"""
Read graphs in LEDA format.

LEDA is a C++ class library for efficient data types and algorithms.

Format
------
See http://www.algorithmic-solutions.info/leda_guide/graphs/leda_native_graph_fileformat.html

"""
__all__ = ['read_leda', 'parse_leda']
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file


@open_file(0, mode='rb')
@nx._dispatchable(graphs=None, returns_graph=True)
def read_leda(path, encoding='UTF-8'):
    """Read graph in LEDA format from path.

    Parameters
    ----------
    path : file or string
       File or filename to read.  Filenames ending in .gz or .bz2  will be
       uncompressed.

    Returns
    -------
    G : NetworkX graph

    Examples
    --------
    G=nx.read_leda('file.leda')

    References
    ----------
    .. [1] http://www.algorithmic-solutions.info/leda_guide/graphs/leda_native_graph_fileformat.html
    """
    lines = path.read().decode(encoding)
    return parse_leda(lines)


@nx._dispatchable(graphs=None, returns_graph=True)
def parse_leda(lines):
    """Read graph in LEDA format from string or iterable.

    Parameters
    ----------
    lines : string or iterable
       Data in LEDA format.

    Returns
    -------
    G : NetworkX graph

    Examples
    --------
    G=nx.parse_leda(string)

    References
    ----------
    .. [1] http://www.algorithmic-solutions.info/leda_guide/graphs/leda_native_graph_fileformat.html
    """
    if isinstance(lines, str):
        lines = iter(lines.split('\n'))
    lines = iter([line.rstrip('\n') for line in lines])

    try:
        header = next(lines)
        if header != 'LEDA.GRAPH':
            raise NetworkXError('LEDA file must start with LEDA.GRAPH')

        directed = next(lines)
        if directed not in ['0', '1']:
            raise NetworkXError('Second line must be 0 or 1')

        G = nx.DiGraph() if directed == '1' else nx.Graph()

        num_nodes = int(next(lines))
        for i in range(num_nodes):
            node_data = next(lines).split('|')
            if len(node_data) < 2:
                raise NetworkXError(f'Invalid node data: {node_data}')
            G.add_node(i + 1, label=node_data[0], data=node_data[1])

        num_edges = int(next(lines))
        for i in range(num_edges):
            edge_data = next(lines).split('|')
            if len(edge_data) < 4:
                raise NetworkXError(f'Invalid edge data: {edge_data}')
            source, target, label, data = edge_data[:4]
            G.add_edge(int(source), int(target), label=label, data=data)

    except StopIteration:
        raise NetworkXError('Incomplete LEDA.GRAPH data')

    return G
