"""
This module provides the following: read and write of p2g format
used in metabolic pathway studies.

See https://web.archive.org/web/20080626113807/http://www.cs.purdue.edu/homes/koyuturk/pathway/ for a description.

The summary is included here:

A file that describes a uniquely labeled graph (with extension ".gr")
format looks like the following:


name
3 4
a
1 2
b

c
0 2

"name" is simply a description of what the graph corresponds to. The
second line displays the number of nodes and number of edges,
respectively. This sample graph contains three nodes labeled "a", "b",
and "c". The rest of the graph contains two lines for each node. The
first line for a node contains the node label. After the declaration
of the node label, the out-edges of that node in the graph are
provided. For instance, "a" is linked to nodes 1 and 2, which are
labeled "b" and "c", while the node labeled "b" has no outgoing
edges. Observe that node labeled "c" has an outgoing edge to
itself. Indeed, self-loops are allowed. Node index starts from 0.

"""
import networkx as nx
from networkx.utils import open_file


@open_file(1, mode='w')
def write_p2g(G, path, encoding='utf-8'):
    """Write NetworkX graph in p2g format.

    Notes
    -----
    This format is meant to be used with directed graphs with
    possible self loops.
    """
    path.write(f"{G.name}\n".encode(encoding))
    path.write(f"{G.number_of_nodes()} {G.number_of_edges()}\n".encode(encoding))
    
    # Create a mapping of nodes to their indices
    node_to_index = {node: i for i, node in enumerate(G.nodes())}
    
    for node in G.nodes():
        path.write(f"{node}\n".encode(encoding))
        out_edges = " ".join(str(node_to_index[neighbor]) for neighbor in G.successors(node))
        path.write(f"{out_edges}\n".encode(encoding))


@open_file(0, mode='r')
@nx._dispatchable(graphs=None, returns_graph=True)
def read_p2g(path, encoding='utf-8'):
    """Read graph in p2g format from path.

    Returns
    -------
    MultiDiGraph

    Notes
    -----
    If you want a DiGraph (with no self loops allowed and no edge data)
    use D=nx.DiGraph(read_p2g(path))
    """
    return parse_p2g(path)


@nx._dispatchable(graphs=None, returns_graph=True)
def parse_p2g(lines):
    """Parse p2g format graph from string or iterable.

    Returns
    -------
    MultiDiGraph
    """
    G = nx.MultiDiGraph()
    lines = iter(lines)
    
    # Read graph name
    G.name = next(lines).strip()
    
    # Read number of nodes and edges
    num_nodes, num_edges = map(int, next(lines).strip().split())
    
    # Read nodes and edges
    node_list = []
    for _ in range(num_nodes):
        node = next(lines).strip()
        node_list.append(node)
        G.add_node(node)
        
        out_edges = next(lines).strip().split()
        for target in out_edges:
            G.add_edge(node, node_list[int(target)])
    
    return G
