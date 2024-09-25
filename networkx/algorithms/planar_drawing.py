from collections import defaultdict
import networkx as nx
__all__ = ['combinatorial_embedding_to_pos']


def combinatorial_embedding_to_pos(embedding, fully_triangulate=False):
    """Assigns every node a (x, y) position based on the given embedding

    The algorithm iteratively inserts nodes of the input graph in a certain
    order and rearranges previously inserted nodes so that the planar drawing
    stays valid. This is done efficiently by only maintaining relative
    positions during the node placements and calculating the absolute positions
    at the end. For more information see [1]_.

    Parameters
    ----------
    embedding : nx.PlanarEmbedding
        This defines the order of the edges

    fully_triangulate : bool
        If set to True the algorithm adds edges to a copy of the input
        embedding and makes it chordal.

    Returns
    -------
    pos : dict
        Maps each node to a tuple that defines the (x, y) position

    References
    ----------
    .. [1] M. Chrobak and T.H. Payne:
        A Linear-time Algorithm for Drawing a Planar Graph on a Grid 1989
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.6677

    """
    # Triangulate the embedding if required
    triangulated_embedding, outer_face = triangulate_embedding(embedding, fully_triangulate)
    
    # Get the canonical ordering of nodes
    canonical_ordering = get_canonical_ordering(triangulated_embedding, outer_face)
    
    # Initialize the position dictionary
    pos = {}
    
    # Place the first three nodes
    v1, v2, v3 = canonical_ordering[:3]
    pos[v1[0]] = (0, 0)
    pos[v2[0]] = (1, 0)
    pos[v3[0]] = (0, 1)
    
    # Initialize the tree structure for relative positions
    tree = {v1[0]: [], v2[0]: [], v3[0]: []}
    
    # Place the remaining nodes
    for k in range(3, len(canonical_ordering)):
        vk, wp_wq = canonical_ordering[k]
        
        # Find the leftmost and rightmost neighbors
        left_neighbor = wp_wq[0]
        right_neighbor = wp_wq[-1]
        
        # Calculate the relative x-coordinate
        delta_x = pos[right_neighbor][0] - pos[left_neighbor][0] + 1
        
        # Set the relative position
        pos[vk] = (delta_x, k)
        
        # Update the tree structure
        tree[vk] = []
        for neighbor in wp_wq:
            if neighbor != left_neighbor and neighbor != right_neighbor:
                tree[vk].append(neighbor)
                tree[neighbor] = [child for child in tree[neighbor] if child not in wp_wq]
    
    # Calculate absolute positions
    set_position(None, tree, list(pos.keys()), 0, 0, pos)
    
    return pos


def set_position(parent, tree, remaining_nodes, delta_x, y_coordinate, pos):
    """Helper method to calculate the absolute position of nodes."""
    if not remaining_nodes:
        return

    node = remaining_nodes.pop(0)
    if parent is not None:
        pos[node] = (pos[parent][0] + delta_x, y_coordinate)
    
    for child in tree[node]:
        set_position(node, tree, remaining_nodes, pos[node][0] - pos[child][0], y_coordinate + 1, pos)


def get_canonical_ordering(embedding, outer_face):
    """Returns a canonical ordering of the nodes

    The canonical ordering of nodes (v1, ..., vn) must fulfill the following
    conditions:
    (See Lemma 1 in [2]_)

    - For the subgraph G_k of the input graph induced by v1, ..., vk it holds:
        - 2-connected
        - internally triangulated
        - the edge (v1, v2) is part of the outer face
    - For a node v(k+1) the following holds:
        - The node v(k+1) is part of the outer face of G_k
        - It has at least two neighbors in G_k
        - All neighbors of v(k+1) in G_k lie consecutively on the outer face of
          G_k (excluding the edge (v1, v2)).

    The algorithm used here starts with G_n (containing all nodes). It first
    selects the nodes v1 and v2. And then tries to find the order of the other
    nodes by checking which node can be removed in order to fulfill the
    conditions mentioned above. This is done by calculating the number of
    chords of nodes on the outer face. For more information see [1]_.

    Parameters
    ----------
    embedding : nx.PlanarEmbedding
        The embedding must be triangulated
    outer_face : list
        The nodes on the outer face of the graph

    Returns
    -------
    ordering : list
        A list of tuples `(vk, wp_wq)`. Here `vk` is the node at this position
        in the canonical ordering. The element `wp_wq` is a list of nodes that
        make up the outer face of G_k.

    References
    ----------
    .. [1] Steven Chaplick.
        Canonical Orders of Planar Graphs and (some of) Their Applications 2015
        https://wuecampus2.uni-wuerzburg.de/moodle/pluginfile.php/545727/mod_resource/content/0/vg-ss15-vl03-canonical-orders-druckversion.pdf
    .. [2] M. Chrobak and T.H. Payne:
        A Linear-time Algorithm for Drawing a Planar Graph on a Grid 1989
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.6677

    """
    n = len(embedding)
    ordering = []
    remaining_nodes = set(embedding.nodes())
    current_face = outer_face.copy()

    # Select v1 and v2
    v1, v2 = current_face[:2]
    ordering.append((v1, [v2]))
    ordering.append((v2, [v1]))
    remaining_nodes.remove(v1)
    remaining_nodes.remove(v2)

    while len(ordering) < n:
        for v in current_face[2:]:
            if v in remaining_nodes:
                neighbors = set(embedding[v])
                face_neighbors = [u for u in current_face if u in neighbors]
                if len(face_neighbors) >= 2:
                    wp_wq = face_neighbors
                    ordering.append((v, wp_wq))
                    remaining_nodes.remove(v)
                    current_face = [u for u in current_face if u != v] + wp_wq[1:-1]
                    break

    return ordering


def triangulate_face(embedding, v1, v2):
    """Triangulates the face given by half edge (v, w)

    Parameters
    ----------
    embedding : nx.PlanarEmbedding
    v1 : node
        The half-edge (v1, v2) belongs to the face that gets triangulated
    v2 : node
    """
    face = [v1, v2]
    current = embedding[v2].next_face(v1)
    while current != v1:
        face.append(current)
        current = embedding[current].next_face(face[-2])

    if len(face) <= 3:
        return  # Face is already triangulated

    # Add edges to triangulate the face
    for i in range(2, len(face) - 1):
        embedding.add_edge(v1, face[i])


def triangulate_embedding(embedding, fully_triangulate=True):
    """Triangulates the embedding.

    Traverses faces of the embedding and adds edges to a copy of the
    embedding to triangulate it.
    The method also ensures that the resulting graph is 2-connected by adding
    edges if the same vertex is contained twice on a path around a face.

    Parameters
    ----------
    embedding : nx.PlanarEmbedding
        The input graph must contain at least 3 nodes.

    fully_triangulate : bool
        If set to False the face with the most nodes is chooses as outer face.
        This outer face does not get triangulated.

    Returns
    -------
    (embedding, outer_face) : (nx.PlanarEmbedding, list) tuple
        The element `embedding` is a new embedding containing all edges from
        the input embedding and the additional edges to triangulate the graph.
        The element `outer_face` is a list of nodes that lie on the outer face.
        If the graph is fully triangulated these are three arbitrary connected
        nodes.

    """
    if len(embedding) < 3:
        raise nx.NetworkXException("Input graph must have at least 3 nodes.")

    # Create a copy of the embedding
    triangulated_embedding = embedding.copy()

    # Find all faces
    faces = list(triangulated_embedding.traverse_faces())

    # Choose the outer face if not fully triangulating
    if not fully_triangulate:
        outer_face = max(faces, key=len)
        faces.remove(outer_face)
    else:
        outer_face = None

    # Triangulate each face
    for face in faces:
        make_bi_connected(triangulated_embedding, face[0], face[1], set())
        for i in range(len(face) - 2):
            triangulate_face(triangulated_embedding, face[0], face[i + 2])

    if outer_face is None:
        # If fully triangulated, choose any three connected nodes as outer face
        outer_face = list(triangulated_embedding.nodes())[:3]

    return triangulated_embedding, outer_face


def make_bi_connected(embedding, starting_node, outgoing_node, edges_counted):
    """Triangulate a face and make it 2-connected

    This method also adds all edges on the face to `edges_counted`.

    Parameters
    ----------
    embedding: nx.PlanarEmbedding
        The embedding that defines the faces
    starting_node : node
        A node on the face
    outgoing_node : node
        A node such that the half edge (starting_node, outgoing_node) belongs
        to the face
    edges_counted: set
        Set of all half-edges that belong to a face that have been visited

    Returns
    -------
    face_nodes: list
        A list of all nodes at the border of this face
    """
    face_nodes = [starting_node]
    current_node = outgoing_node
    previous_node = starting_node

    while current_node != starting_node:
        face_nodes.append(current_node)
        edges_counted.add((previous_node, current_node))
        edges_counted.add((current_node, previous_node))

        next_node = embedding[current_node].next_face(previous_node)
        previous_node = current_node
        current_node = next_node

    # Make the face 2-connected
    for i in range(len(face_nodes)):
        for j in range(i + 2, len(face_nodes)):
            if face_nodes[i] != face_nodes[j] and not embedding.has_edge(face_nodes[i], face_nodes[j]):
                embedding.add_edge(face_nodes[i], face_nodes[j])

    return face_nodes
