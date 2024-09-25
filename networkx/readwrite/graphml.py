"""
*******
GraphML
*******
Read and write graphs in GraphML format.

.. warning::

    This parser uses the standard xml library present in Python, which is
    insecure - see :external+python:mod:`xml` for additional information.
    Only parse GraphML files you trust.

This implementation does not support mixed graphs (directed and unidirected
edges together), hyperedges, nested graphs, or ports.

"GraphML is a comprehensive and easy-to-use file format for graphs. It
consists of a language core to describe the structural properties of a
graph and a flexible extension mechanism to add application-specific
data. Its main features include support of

    * directed, undirected, and mixed graphs,
    * hypergraphs,
    * hierarchical graphs,
    * graphical representations,
    * references to external data,
    * application-specific attribute data, and
    * light-weight parsers.

Unlike many other file formats for graphs, GraphML does not use a
custom syntax. Instead, it is based on XML and hence ideally suited as
a common denominator for all kinds of services generating, archiving,
or processing graphs."

http://graphml.graphdrawing.org/

Format
------
GraphML is an XML format.  See
http://graphml.graphdrawing.org/specification.html for the specification and
http://graphml.graphdrawing.org/primer/graphml-primer.html
for examples.
"""
import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
__all__ = ['write_graphml', 'read_graphml', 'generate_graphml',
    'write_graphml_xml', 'write_graphml_lxml', 'parse_graphml',
    'GraphMLWriter', 'GraphMLReader']


@open_file(1, mode='wb')
def write_graphml_xml(G, path, encoding='utf-8', prettyprint=True,
    infer_numeric_types=False, named_key_ids=False, edge_id_from_attribute=None
    ):
    """Write G in GraphML XML format to path

    Parameters
    ----------
    G : graph
       A networkx graph
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be compressed.
    encoding : string (optional)
       Encoding for text data.
    prettyprint : bool (optional)
       If True use line breaks and indenting in output XML.
    infer_numeric_types : boolean
       Determine if numeric types should be generalized.
       For example, if edges have both int and float 'weight' attributes,
       we infer in GraphML that both are floats.
    named_key_ids : bool (optional)
       If True use attr.name as value for key elements' id attribute.
    edge_id_from_attribute : dict key (optional)
        If provided, the graphml edge id is set by looking up the corresponding
        edge data attribute keyed by this parameter. If `None` or the key does not exist in edge data,
        the edge id is set by the edge key if `G` is a MultiGraph, else the edge id is left unset.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_graphml(G, "test.graphml")

    Notes
    -----
    This implementation does not support mixed graphs (directed
    and unidirected edges together) hyperedges, nested graphs, or ports.
    """
    pass


@open_file(1, mode='wb')
def write_graphml_lxml(G, path, encoding='utf-8', prettyprint=True,
    infer_numeric_types=False, named_key_ids=False, edge_id_from_attribute=None
    ):
    """Write G in GraphML XML format to path

    This function uses the LXML framework and should be faster than
    the version using the xml library.

    Parameters
    ----------
    G : graph
       A networkx graph
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be compressed.
    encoding : string (optional)
       Encoding for text data.
    prettyprint : bool (optional)
       If True use line breaks and indenting in output XML.
    infer_numeric_types : boolean
       Determine if numeric types should be generalized.
       For example, if edges have both int and float 'weight' attributes,
       we infer in GraphML that both are floats.
    named_key_ids : bool (optional)
       If True use attr.name as value for key elements' id attribute.
    edge_id_from_attribute : dict key (optional)
        If provided, the graphml edge id is set by looking up the corresponding
        edge data attribute keyed by this parameter. If `None` or the key does not exist in edge data,
        the edge id is set by the edge key if `G` is a MultiGraph, else the edge id is left unset.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_graphml_lxml(G, "fourpath.graphml")

    Notes
    -----
    This implementation does not support mixed graphs (directed
    and unidirected edges together) hyperedges, nested graphs, or ports.
    """
    pass


def generate_graphml(G, encoding='utf-8', prettyprint=True, named_key_ids=
    False, edge_id_from_attribute=None):
    """Generate GraphML lines for G

    Parameters
    ----------
    G : graph
       A networkx graph
    encoding : string (optional)
       Encoding for text data.
    prettyprint : bool (optional)
       If True use line breaks and indenting in output XML.
    named_key_ids : bool (optional)
       If True use attr.name as value for key elements' id attribute.
    edge_id_from_attribute : dict key (optional)
        If provided, the graphml edge id is set by looking up the corresponding
        edge data attribute keyed by this parameter. If `None` or the key does not exist in edge data,
        the edge id is set by the edge key if `G` is a MultiGraph, else the edge id is left unset.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> linefeed = chr(10)  # linefeed = 

    >>> s = linefeed.join(nx.generate_graphml(G))
    >>> for line in nx.generate_graphml(G):  # doctest: +SKIP
    ...     print(line)

    Notes
    -----
    This implementation does not support mixed graphs (directed and unidirected
    edges together) hyperedges, nested graphs, or ports.
    """
    pass


@open_file(0, mode='rb')
@nx._dispatchable(graphs=None, returns_graph=True)
def read_graphml(path, node_type=str, edge_key_type=int, force_multigraph=False
    ):
    """Read graph in GraphML format from path.

    Parameters
    ----------
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be compressed.

    node_type: Python type (default: str)
       Convert node ids to this type

    edge_key_type: Python type (default: int)
       Convert graphml edge ids to this type. Multigraphs use id as edge key.
       Non-multigraphs add to edge attribute dict with name "id".

    force_multigraph : bool (default: False)
       If True, return a multigraph with edge keys. If False (the default)
       return a multigraph when multiedges are in the graph.

    Returns
    -------
    graph: NetworkX graph
        If parallel edges are present or `force_multigraph=True` then
        a MultiGraph or MultiDiGraph is returned. Otherwise a Graph/DiGraph.
        The returned graph is directed if the file indicates it should be.

    Notes
    -----
    Default node and edge attributes are not propagated to each node and edge.
    They can be obtained from `G.graph` and applied to node and edge attributes
    if desired using something like this:

    >>> default_color = G.graph["node_default"]["color"]  # doctest: +SKIP
    >>> for node, data in G.nodes(data=True):  # doctest: +SKIP
    ...     if "color" not in data:
    ...         data["color"] = default_color
    >>> default_color = G.graph["edge_default"]["color"]  # doctest: +SKIP
    >>> for u, v, data in G.edges(data=True):  # doctest: +SKIP
    ...     if "color" not in data:
    ...         data["color"] = default_color

    This implementation does not support mixed graphs (directed and unidirected
    edges together), hypergraphs, nested graphs, or ports.

    For multigraphs the GraphML edge "id" will be used as the edge
    key.  If not specified then they "key" attribute will be used.  If
    there is no "key" attribute a default NetworkX multigraph edge key
    will be provided.

    Files with the yEd "yfiles" extension can be read. The type of the node's
    shape is preserved in the `shape_type` node attribute.

    yEd compressed files ("file.graphmlz" extension) can be read by renaming
    the file to "file.graphml.gz".

    """
    pass


@nx._dispatchable(graphs=None, returns_graph=True)
def parse_graphml(graphml_string, node_type=str, edge_key_type=int,
    force_multigraph=False):
    """Read graph in GraphML format from string.

    Parameters
    ----------
    graphml_string : string
       String containing graphml information
       (e.g., contents of a graphml file).

    node_type: Python type (default: str)
       Convert node ids to this type

    edge_key_type: Python type (default: int)
       Convert graphml edge ids to this type. Multigraphs use id as edge key.
       Non-multigraphs add to edge attribute dict with name "id".

    force_multigraph : bool (default: False)
       If True, return a multigraph with edge keys. If False (the default)
       return a multigraph when multiedges are in the graph.


    Returns
    -------
    graph: NetworkX graph
        If no parallel edges are found a Graph or DiGraph is returned.
        Otherwise a MultiGraph or MultiDiGraph is returned.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> linefeed = chr(10)  # linefeed = 

    >>> s = linefeed.join(nx.generate_graphml(G))
    >>> H = nx.parse_graphml(s)

    Notes
    -----
    Default node and edge attributes are not propagated to each node and edge.
    They can be obtained from `G.graph` and applied to node and edge attributes
    if desired using something like this:

    >>> default_color = G.graph["node_default"]["color"]  # doctest: +SKIP
    >>> for node, data in G.nodes(data=True):  # doctest: +SKIP
    ...     if "color" not in data:
    ...         data["color"] = default_color
    >>> default_color = G.graph["edge_default"]["color"]  # doctest: +SKIP
    >>> for u, v, data in G.edges(data=True):  # doctest: +SKIP
    ...     if "color" not in data:
    ...         data["color"] = default_color

    This implementation does not support mixed graphs (directed and unidirected
    edges together), hypergraphs, nested graphs, or ports.

    For multigraphs the GraphML edge "id" will be used as the edge
    key.  If not specified then they "key" attribute will be used.  If
    there is no "key" attribute a default NetworkX multigraph edge key
    will be provided.

    """
    pass


class GraphML:
    NS_GRAPHML = 'http://graphml.graphdrawing.org/xmlns'
    NS_XSI = 'http://www.w3.org/2001/XMLSchema-instance'
    NS_Y = 'http://www.yworks.com/xml/graphml'
    SCHEMALOCATION = ' '.join(['http://graphml.graphdrawing.org/xmlns',
        'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd'])
    convert_bool = {'true': True, 'false': False, '0': False, (0): False,
        '1': True, (1): True}

    def get_xml_type(self, key):
        """Wrapper around the xml_type dict that raises a more informative
        exception message when a user attempts to use data of a type not
        supported by GraphML."""
        try:
            return self.xml_type[key]
        except KeyError:
            raise nx.NetworkXError(f"GraphML writer does not support {key} as data values.")


class GraphMLWriter(GraphML):

    def __init__(self, graph=None, encoding='utf-8', prettyprint=True,
        infer_numeric_types=False, named_key_ids=False,
        edge_id_from_attribute=None):
        self.construct_types()
        from xml.etree.ElementTree import Element
        self.myElement = Element
        self.infer_numeric_types = infer_numeric_types
        self.prettyprint = prettyprint
        self.named_key_ids = named_key_ids
        self.edge_id_from_attribute = edge_id_from_attribute
        self.encoding = encoding
        self.xml = self.myElement('graphml', {'xmlns': self.NS_GRAPHML,
            'xmlns:xsi': self.NS_XSI, 'xsi:schemaLocation': self.
            SCHEMALOCATION})
        self.keys = {}
        self.attributes = defaultdict(list)
        self.attribute_types = defaultdict(set)
        if graph is not None:
            self.add_graph_element(graph)

    def __str__(self):
        from xml.etree.ElementTree import tostring
        if self.prettyprint:
            self.indent(self.xml)
        s = tostring(self.xml).decode(self.encoding)
        return s

    def attr_type(self, name, scope, value):
        """Infer the attribute type of data named name. Currently this only
        supports inference of numeric types.

        If self.infer_numeric_types is false, type is used. Otherwise, pick the
        most general of types found across all values with name and scope. This
        means edges with data named 'weight' are treated separately from nodes
        with data named 'weight'.
        """
        if not self.infer_numeric_types:
            return self.get_xml_type(type(value))
        
        types = self.attribute_types[(name, scope)]
        types.add(type(value))
        
        if str in types:
            return self.get_xml_type(str)
        elif float in types:
            return self.get_xml_type(float)
        elif int in types:
            return self.get_xml_type(int)
        else:
            return self.get_xml_type(list(types)[0])

    def add_data(self, name, element_type, value, scope='all', default=None):
        """
        Make a data element for an edge or a node. Keep a log of the
        type in the keys table.
        """
        key_id = f"{scope}_{name}"
        if key_id not in self.keys:
            attr_type = self.attr_type(name, scope, value)
            key_kwargs = {"id": key_id, "for": scope, "attr.name": name, "attr.type": attr_type}
            key_element = self.myElement("key", **key_kwargs)
            if default is not None:
                default_element = self.myElement("default")
                default_element.text = str(default)
                key_element.append(default_element)
            self.xml.insert(0, key_element)
            self.keys[key_id] = key_element
        
        data_element = self.myElement("data", key=key_id)
        data_element.text = str(value)
        return data_element

    def add_attributes(self, scope, xml_obj, data, default):
        """Appends attribute data to edges or nodes, and stores type information
        to be added later. See add_graph_element.
        """
        for k, v in data.items():
            self.attribute_types[(k, scope)].add(type(v))
            self.attributes[scope].append((k, v))
            xml_obj.append(self.add_data(k, self.get_xml_type(type(v)), v, scope, default.get(k)))

    def add_graph_element(self, G):
        """
        Serialize graph G in GraphML to the stream.
        """
        default_edge_data = G.graph.get('edge_default', {})
        default_node_data = G.graph.get('node_default', {})

        graph_element = self.myElement("graph", id=G.name,
                                       edgedefault="directed" if G.is_directed() else "undirected")
        
        for node, data in G.nodes(data=True):
            node_element = self.myElement("node", id=str(node))
            self.add_attributes("node", node_element, data, default_node_data)
            graph_element.append(node_element)
        
        for u, v, data in G.edges(data=True):
            edge_element = self.myElement("edge", source=str(u), target=str(v))
            if self.edge_id_from_attribute:
                edge_id = data.get(self.edge_id_from_attribute)
                if edge_id is not None:
                    edge_element.set("id", str(edge_id))
            self.add_attributes("edge", edge_element, data, default_edge_data)
            graph_element.append(edge_element)
        
        self.xml.append(graph_element)

    def add_graphs(self, graph_list):
        """Add many graphs to this GraphML document."""
        for G in graph_list:
            self.add_graph_element(G)


class IncrementalElement:
    """Wrapper for _IncrementalWriter providing an Element like interface.

    This wrapper does not intend to be a complete implementation but rather to
    deal with those calls used in GraphMLWriter.
    """

    def __init__(self, xml, prettyprint):
        self.xml = xml
        self.prettyprint = prettyprint


class GraphMLWriterLxml(GraphMLWriter):

    def __init__(self, path, graph=None, encoding='utf-8', prettyprint=True,
        infer_numeric_types=False, named_key_ids=False,
        edge_id_from_attribute=None):
        self.construct_types()
        import lxml.etree as lxmletree
        self.myElement = lxmletree.Element
        self._encoding = encoding
        self._prettyprint = prettyprint
        self.named_key_ids = named_key_ids
        self.edge_id_from_attribute = edge_id_from_attribute
        self.infer_numeric_types = infer_numeric_types
        self._xml_base = lxmletree.xmlfile(path, encoding=encoding)
        self._xml = self._xml_base.__enter__()
        self._xml.write_declaration()
        self.xml = []
        self._keys = self.xml
        self._graphml = self._xml.element('graphml', {'xmlns': self.
            NS_GRAPHML, 'xmlns:xsi': self.NS_XSI, 'xsi:schemaLocation':
            self.SCHEMALOCATION})
        self._graphml.__enter__()
        self.keys = {}
        self.attribute_types = defaultdict(set)
        if graph is not None:
            self.add_graph_element(graph)

    def add_graph_element(self, G):
        """
        Serialize graph G in GraphML to the stream.
        """
        pass

    def add_attributes(self, scope, xml_obj, data, default):
        """Appends attribute data."""
        pass

    def __str__(self):
        return object.__str__(self)


write_graphml = write_graphml_lxml


class GraphMLReader(GraphML):
    """Read a GraphML document.  Produces NetworkX graph objects."""

    def __init__(self, node_type=str, edge_key_type=int, force_multigraph=False
        ):
        self.construct_types()
        self.node_type = node_type
        self.edge_key_type = edge_key_type
        self.multigraph = force_multigraph
        self.edge_ids = {}

    def __call__(self, path=None, string=None):
        from xml.etree.ElementTree import ElementTree, fromstring
        if path is not None:
            self.xml = ElementTree(file=path)
        elif string is not None:
            self.xml = fromstring(string)
        else:
            raise ValueError("Must specify either 'path' or 'string' as kwarg")
        keys, defaults = self.find_graphml_keys(self.xml)
        for g in self.xml.findall(f'{{{self.NS_GRAPHML}}}graph'):
            yield self.make_graph(g, keys, defaults)

    def add_node(self, G, node_xml, graphml_keys, defaults):
        """Add a node to the graph."""
        node_id = self.node_type(node_xml.get("id"))
        data = self.decode_data_elements(graphml_keys, node_xml)
        G.add_node(node_id, **data)

    def add_edge(self, G, edge_element, graphml_keys):
        """Add an edge to the graph."""
        source = self.node_type(edge_element.get("source"))
        target = self.node_type(edge_element.get("target"))
        data = self.decode_data_elements(graphml_keys, edge_element)
        edge_id = edge_element.get("id")
        if edge_id is not None:
            data["id"] = edge_id
        if G.is_multigraph():
            key = self.edge_key_type(edge_element.get("id")) if edge_id else None
            G.add_edge(source, target, key, **data)
        else:
            G.add_edge(source, target, **data)

    def decode_data_elements(self, graphml_keys, obj_xml):
        """Use the key information to decode the data XML if present."""
        data = {}
        for data_element in obj_xml.findall(f"{{{self.NS_GRAPHML}}}data"):
            key = data_element.get("key")
            if key in graphml_keys:
                data_type = graphml_keys[key]["type"]
                value = data_element.text
                if data_type == "boolean":
                    value = self.convert_bool[value.lower()]
                elif data_type == "int":
                    value = int(value)
                elif data_type == "float" or data_type == "double":
                    value = float(value)
                data[graphml_keys[key]["name"]] = value
        return data

    def find_graphml_keys(self, graph_element):
        """Extracts all the keys and key defaults from the xml."""
        keys = {}
        defaults = {}
        for k in graph_element.findall(f"{{{self.NS_GRAPHML}}}key"):
            key_id = k.get("id")
            key_for = k.get("for")
            key_name = k.get("attr.name")
            key_type = k.get("attr.type")
            if key_type is None:
                key_type = "string"
            keys[key_id] = {"name": key_name, "type": key_type, "for": key_for}
            default = k.find(f"{{{self.NS_GRAPHML}}}default")
            if default is not None:
                defaults[key_name] = default.text
        return keys, defaults
