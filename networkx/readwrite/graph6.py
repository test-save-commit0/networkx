"""Functions for reading and writing graphs in the *graph6* format.

The *graph6* file format is suitable for small graphs or large dense
graphs. For large sparse graphs, use the *sparse6* format.

For more information, see the `graph6`_ homepage.

.. _graph6: http://users.cecs.anu.edu.au/~bdm/data/formats.html

"""
from itertools import islice
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for, open_file
__all__ = ['from_graph6_bytes', 'read_graph6', 'to_graph6_bytes',
    'write_graph6']


def _generate_graph6_bytes(G, nodes, header):
    """Yield bytes in the graph6 encoding of a graph.

    `G` is an undirected simple graph. `nodes` is the list of nodes for
    which the node-induced subgraph will be encoded; if `nodes` is the
    list of all nodes in the graph, the entire graph will be
    encoded. `header` is a Boolean that specifies whether to generate
    the header ``b'>>graph6<<'`` before the remaining data.

    This function generates `bytes` objects in the following order:

    1. the header (if requested),
    2. the encoding of the number of nodes,
    3. each character, one-at-a-time, in the encoding of the requested
       node-induced subgraph,
    4. a newline character.

    This function raises :exc:`ValueError` if the graph is too large for
    the graph6 format (that is, greater than ``2 ** 36`` nodes).

    """
    if len(G) >= 2**36:
        raise ValueError("graph6 format supports only graphs with less than 2^36 nodes")
    
    if header:
        yield b'>>graph6<<'

    n = len(nodes)
    yield from n_to_data(n)

    edges = G.subgraph(nodes).edges()
    bits = ((i < j and (i, j) in edges) for j in range(n) for i in range(j))
    char = 0
    for i, bit in enumerate(bits):
        char = (char << 1) | bit
        if (i + 1) % 6 == 0:
            yield bytes([char + 63])
            char = 0
    if i % 6 != 5:
        char <<= 5 - (i % 6)
        yield bytes([char + 63])
    
    yield b'\n'


@nx._dispatchable(graphs=None, returns_graph=True)
def from_graph6_bytes(bytes_in):
    """Read a simple undirected graph in graph6 format from bytes.

    Parameters
    ----------
    bytes_in : bytes
       Data in graph6 format, without a trailing newline.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If bytes_in is unable to be parsed in graph6 format

    ValueError
        If any character ``c`` in bytes_in does not satisfy
        ``63 <= ord(c) < 127``.

    Examples
    --------
    >>> G = nx.from_graph6_bytes(b"A_")
    >>> sorted(G.edges())
    [(0, 1)]

    See Also
    --------
    read_graph6, write_graph6

    References
    ----------
    .. [1] Graph6 specification
           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    if bytes_in.startswith(b'>>graph6<<'):
        bytes_in = bytes_in[10:]

    if not all(63 <= c < 127 for c in bytes_in if c != ord('\n')):
        raise ValueError("Invalid character in graph6 data")

    n, data = data_to_n(bytes_in)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    bits = iter((ord(c) - 63) & 0b111111 for c in data)
    for j in range(1, n):
        for i in range(j):
            if next(bits, None):
                G.add_edge(i, j)

    return G


@not_implemented_for('directed')
@not_implemented_for('multigraph')
def to_graph6_bytes(G, nodes=None, header=True):
    """Convert a simple undirected graph to bytes in graph6 format.

    Parameters
    ----------
    G : Graph (undirected)

    nodes: list or iterable
       Nodes are labeled 0...n-1 in the order provided.  If None the ordering
       given by ``G.nodes()`` is used.

    header: bool
       If True add '>>graph6<<' bytes to head of data.

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    ValueError
        If the graph has at least ``2 ** 36`` nodes; the graph6 format
        is only defined for graphs of order less than ``2 ** 36``.

    Examples
    --------
    >>> nx.to_graph6_bytes(nx.path_graph(2))
    b'>>graph6<<A_\\n'

    See Also
    --------
    from_graph6_bytes, read_graph6, write_graph6_bytes

    Notes
    -----
    The returned bytes end with a newline character.

    The format does not support edge or node labels, parallel edges or
    self loops. If self loops are present they are silently ignored.

    References
    ----------
    .. [1] Graph6 specification
           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    if nodes is None:
        nodes = list(G.nodes())
    else:
        nodes = list(nodes)

    return b''.join(_generate_graph6_bytes(G, nodes, header))


@open_file(0, mode='rb')
@nx._dispatchable(graphs=None, returns_graph=True)
def read_graph6(path):
    """Read simple undirected graphs in graph6 format from path.

    Parameters
    ----------
    path : file or string
       File or filename to write.

    Returns
    -------
    G : Graph or list of Graphs
       If the file contains multiple lines then a list of graphs is returned

    Raises
    ------
    NetworkXError
        If the string is unable to be parsed in graph6 format

    Examples
    --------
    You can read a graph6 file by giving the path to the file::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False) as f:
        ...     _ = f.write(b">>graph6<<A_\\n")
        ...     _ = f.seek(0)
        ...     G = nx.read_graph6(f.name)
        >>> list(G.edges())
        [(0, 1)]

    You can also read a graph6 file by giving an open file-like object::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     _ = f.write(b">>graph6<<A_\\n")
        ...     _ = f.seek(0)
        ...     G = nx.read_graph6(f)
        >>> list(G.edges())
        [(0, 1)]

    See Also
    --------
    from_graph6_bytes, write_graph6

    References
    ----------
    .. [1] Graph6 specification
           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    with open(path, 'rb') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if len(lines) == 1:
        return from_graph6_bytes(lines[0])
    else:
        return [from_graph6_bytes(line) for line in lines]


@not_implemented_for('directed')
@not_implemented_for('multigraph')
@open_file(1, mode='wb')
def write_graph6(G, path, nodes=None, header=True):
    """Write a simple undirected graph to a path in graph6 format.

    Parameters
    ----------
    G : Graph (undirected)

    path : str
       The path naming the file to which to write the graph.

    nodes: list or iterable
       Nodes are labeled 0...n-1 in the order provided.  If None the ordering
       given by ``G.nodes()`` is used.

    header: bool
       If True add '>>graph6<<' string to head of data

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    ValueError
        If the graph has at least ``2 ** 36`` nodes; the graph6 format
        is only defined for graphs of order less than ``2 ** 36``.

    Examples
    --------
    You can write a graph6 file by giving the path to a file::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False) as f:
        ...     nx.write_graph6(nx.path_graph(2), f.name)
        ...     _ = f.seek(0)
        ...     print(f.read())
        b'>>graph6<<A_\\n'

    See Also
    --------
    from_graph6_bytes, read_graph6

    Notes
    -----
    The function writes a newline character after writing the encoding
    of the graph.

    The format does not support edge or node labels, parallel edges or
    self loops.  If self loops are present they are silently ignored.

    References
    ----------
    .. [1] Graph6 specification
           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    with open(path, 'wb') as f:
        f.write(to_graph6_bytes(G, nodes=nodes, header=header))


@not_implemented_for('directed')
@not_implemented_for('multigraph')
def write_graph6_file(G, f, nodes=None, header=True):
    """Write a simple undirected graph to a file-like object in graph6 format.

    Parameters
    ----------
    G : Graph (undirected)

    f : file-like object
       The file to write.

    nodes: list or iterable
       Nodes are labeled 0...n-1 in the order provided.  If None the ordering
       given by ``G.nodes()`` is used.

    header: bool
       If True add '>>graph6<<' string to head of data

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    ValueError
        If the graph has at least ``2 ** 36`` nodes; the graph6 format
        is only defined for graphs of order less than ``2 ** 36``.

    Examples
    --------
    You can write a graph6 file by giving an open file-like object::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     nx.write_graph6(nx.path_graph(2), f)
        ...     _ = f.seek(0)
        ...     print(f.read())
        b'>>graph6<<A_\\n'

    See Also
    --------
    from_graph6_bytes, read_graph6

    Notes
    -----
    The function writes a newline character after writing the encoding
    of the graph.

    The format does not support edge or node labels, parallel edges or
    self loops.  If self loops are present they are silently ignored.

    References
    ----------
    .. [1] Graph6 specification
           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    f.write(to_graph6_bytes(G, nodes=nodes, header=header))


def data_to_n(data):
    """Read initial one-, four- or eight-unit value from graph6
    integer sequence.

    Return (value, rest of seq.)"""
    if data[0] <= 62:
        return data[0], data[1:]
    if data[1] <= 62:
        return (data[0] - 63) * 64 + data[1], data[2:]
    return (data[0] - 63) * 64 * 64 + (data[1] - 63) * 64 + data[2], data[3:]


def n_to_data(n):
    """Convert an integer to one-, four- or eight-unit graph6 sequence.

    This function is undefined if `n` is not in ``range(2 ** 36)``.

    """
    if n <= 62:
        return bytes([n + 63])
    if n <= 258047:
        return bytes([126, (n >> 6) + 63, (n & 63) + 63])
    return bytes([126, 126, (n >> 12) + 63, ((n >> 6) & 63) + 63, (n & 63) + 63])
