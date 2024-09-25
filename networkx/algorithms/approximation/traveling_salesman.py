"""
=================================
Travelling Salesman Problem (TSP)
=================================

Implementation of approximate algorithms
for solving and approximating the TSP problem.

Categories of algorithms which are implemented:

- Christofides (provides a 3/2-approximation of TSP)
- Greedy
- Simulated Annealing (SA)
- Threshold Accepting (TA)
- Asadpour Asymmetric Traveling Salesman Algorithm

The Travelling Salesman Problem tries to find, given the weight
(distance) between all points where a salesman has to visit, the
route so that:

- The total distance (cost) which the salesman travels is minimized.
- The salesman returns to the starting point.
- Note that for a complete graph, the salesman visits each point once.

The function `travelling_salesman_problem` allows for incomplete
graphs by finding all-pairs shortest paths, effectively converting
the problem to a complete graph problem. It calls one of the
approximate methods on that problem and then converts the result
back to the original graph using the previously found shortest paths.

TSP is an NP-hard problem in combinatorial optimization,
important in operations research and theoretical computer science.

http://en.wikipedia.org/wiki/Travelling_salesman_problem
"""
import math
import networkx as nx
from networkx.algorithms.tree.mst import random_spanning_tree
from networkx.utils import not_implemented_for, pairwise, py_random_state
__all__ = ['traveling_salesman_problem', 'christofides', 'asadpour_atsp',
    'greedy_tsp', 'simulated_annealing_tsp', 'threshold_accepting_tsp']


def swap_two_nodes(soln, seed):
    """Swap two nodes in `soln` to give a neighbor solution.

    Parameters
    ----------
    soln : list of nodes
        Current cycle of nodes

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    list
        The solution after move is applied. (A neighbor solution.)

    Notes
    -----
        This function assumes that the incoming list `soln` is a cycle
        (that the first and last element are the same) and also that
        we don't want any move to change the first node in the list
        (and thus not the last node either).

        The input list is changed as well as returned. Make a copy if needed.

    See Also
    --------
        move_one_node
    """
    rng = np.random.default_rng(seed)
    n = len(soln)
    i, j = rng.choice(range(1, n - 1), size=2, replace=False)
    soln[i], soln[j] = soln[j], soln[i]
    return soln


def move_one_node(soln, seed):
    """Move one node to another position to give a neighbor solution.

    The node to move and the position to move to are chosen randomly.
    The first and last nodes are left untouched as soln must be a cycle
    starting at that node.

    Parameters
    ----------
    soln : list of nodes
        Current cycle of nodes

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    list
        The solution after move is applied. (A neighbor solution.)

    Notes
    -----
        This function assumes that the incoming list `soln` is a cycle
        (that the first and last element are the same) and also that
        we don't want any move to change the first node in the list
        (and thus not the last node either).

        The input list is changed as well as returned. Make a copy if needed.

    See Also
    --------
        swap_two_nodes
    """
    rng = np.random.default_rng(seed)
    n = len(soln)
    i = rng.integers(1, n - 1)
    j = rng.integers(1, n - 1)
    while i == j:
        j = rng.integers(1, n - 1)
    node = soln.pop(i)
    soln.insert(j, node)
    return soln


@not_implemented_for('directed')
@nx._dispatchable(edge_attrs='weight')
def christofides(G, weight='weight', tree=None):
    """Approximate a solution of the traveling salesman problem

    Compute a 3/2-approximation of the traveling salesman problem
    in a complete undirected graph using Christofides [1]_ algorithm.

    Parameters
    ----------
    G : Graph
        `G` should be a complete weighted undirected graph.
        The distance between all pairs of nodes should be included.

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    tree : NetworkX graph or None (default: None)
        A minimum spanning tree of G. Or, if None, the minimum spanning
        tree is computed using :func:`networkx.minimum_spanning_tree`

    Returns
    -------
    list
        List of nodes in `G` along a cycle with a 3/2-approximation of
        the minimal Hamiltonian cycle.

    References
    ----------
    .. [1] Christofides, Nicos. "Worst-case analysis of a new heuristic for
       the travelling salesman problem." No. RR-388. Carnegie-Mellon Univ
       Pittsburgh Pa Management Sciences Research Group, 1976.
    """
    if tree is None:
        tree = nx.minimum_spanning_tree(G, weight=weight)
    
    # Find odd degree vertices
    odd_degree_vertices = [v for v, d in tree.degree() if d % 2 == 1]
    
    # Compute minimum weight perfect matching
    subgraph = G.subgraph(odd_degree_vertices)
    matching = nx.min_weight_matching(subgraph, weight=weight)
    
    # Combine matching and MST
    multigraph = nx.MultiGraph(tree)
    multigraph.add_edges_from(matching)
    
    # Find Eulerian circuit
    eulerian_circuit = list(nx.eulerian_circuit(multigraph))
    
    # Extract Hamiltonian cycle
    hamiltonian_cycle = []
    visited = set()
    for u, v in eulerian_circuit:
        if u not in visited:
            hamiltonian_cycle.append(u)
            visited.add(u)
    hamiltonian_cycle.append(hamiltonian_cycle[0])
    
    return hamiltonian_cycle


def _shortcutting(circuit):
    """Remove duplicate nodes in the path"""
    return list(dict.fromkeys(circuit))


@nx._dispatchable(edge_attrs='weight')
def traveling_salesman_problem(G, weight='weight', nodes=None, cycle=True,
    method=None, **kwargs):
    """Find the shortest path in `G` connecting specified nodes

    This function allows approximate solution to the traveling salesman
    problem on networks that are not complete graphs and/or where the
    salesman does not need to visit all nodes.

    This function proceeds in two steps. First, it creates a complete
    graph using the all-pairs shortest_paths between nodes in `nodes`.
    Edge weights in the new graph are the lengths of the paths
    between each pair of nodes in the original graph.
    Second, an algorithm (default: `christofides` for undirected and
    `asadpour_atsp` for directed) is used to approximate the minimal Hamiltonian
    cycle on this new graph. The available algorithms are:

     - christofides
     - greedy_tsp
     - simulated_annealing_tsp
     - threshold_accepting_tsp
     - asadpour_atsp

    Once the Hamiltonian Cycle is found, this function post-processes to
    accommodate the structure of the original graph. If `cycle` is ``False``,
    the biggest weight edge is removed to make a Hamiltonian path.
    Then each edge on the new complete graph used for that analysis is
    replaced by the shortest_path between those nodes on the original graph.
    If the input graph `G` includes edges with weights that do not adhere to
    the triangle inequality, such as when `G` is not a complete graph (i.e
    length of non-existent edges is infinity), then the returned path may
    contain some repeating nodes (other than the starting node).

    Parameters
    ----------
    G : NetworkX graph
        A possibly weighted graph

    nodes : collection of nodes (default=G.nodes)
        collection (list, set, etc.) of nodes to visit

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    cycle : bool (default: True)
        Indicates whether a cycle should be returned, or a path.
        Note: the cycle is the approximate minimal cycle.
        The path simply removes the biggest edge in that cycle.

    method : function (default: None)
        A function that returns a cycle on all nodes and approximates
        the solution to the traveling salesman problem on a complete
        graph. The returned cycle is then used to find a corresponding
        solution on `G`. `method` should be callable; take inputs
        `G`, and `weight`; and return a list of nodes along the cycle.

        Provided options include :func:`christofides`, :func:`greedy_tsp`,
        :func:`simulated_annealing_tsp` and :func:`threshold_accepting_tsp`.

        If `method is None`: use :func:`christofides` for undirected `G` and
        :func:`asadpour_atsp` for directed `G`.

    **kwargs : dict
        Other keyword arguments to be passed to the `method` function passed in.

    Returns
    -------
    list
        List of nodes in `G` along a path with an approximation of the minimal
        path through `nodes`.

    Raises
    ------
    NetworkXError
        If `G` is a directed graph it has to be strongly connected or the
        complete version cannot be generated.

    Examples
    --------
    >>> tsp = nx.approximation.traveling_salesman_problem
    >>> G = nx.cycle_graph(9)
    >>> G[4][5]["weight"] = 5  # all other weights are 1
    >>> tsp(G, nodes=[3, 6])
    [3, 2, 1, 0, 8, 7, 6, 7, 8, 0, 1, 2, 3]
    >>> path = tsp(G, cycle=False)
    >>> path in ([4, 3, 2, 1, 0, 8, 7, 6, 5], [5, 6, 7, 8, 0, 1, 2, 3, 4])
    True

    While no longer required, you can still build (curry) your own function
    to provide parameter values to the methods.

    >>> SA_tsp = nx.approximation.simulated_annealing_tsp
    >>> method = lambda G, weight: SA_tsp(G, "greedy", weight=weight, temp=500)
    >>> path = tsp(G, cycle=False, method=method)
    >>> path in ([4, 3, 2, 1, 0, 8, 7, 6, 5], [5, 6, 7, 8, 0, 1, 2, 3, 4])
    True

    Otherwise, pass other keyword arguments directly into the tsp function.

    >>> path = tsp(
    ...     G,
    ...     cycle=False,
    ...     method=nx.approximation.simulated_annealing_tsp,
    ...     init_cycle="greedy",
    ...     temp=500,
    ... )
    >>> path in ([4, 3, 2, 1, 0, 8, 7, 6, 5], [5, 6, 7, 8, 0, 1, 2, 3, 4])
    True
    """
    if nodes is None:
        nodes = list(G.nodes())
    
    # Create a complete graph
    H = nx.Graph()
    for u in nodes:
        for v in nodes:
            if u != v:
                if G.is_directed():
                    if not nx.has_path(G, u, v):
                        raise nx.NetworkXError("G is not strongly connected.")
                    path = nx.shortest_path(G, u, v, weight=weight)
                    path_weight = sum(G[path[i]][path[i+1]].get(weight, 1) for i in range(len(path)-1))
                else:
                    path = nx.shortest_path(G, u, v, weight=weight)
                    path_weight = sum(G[path[i]][path[i+1]].get(weight, 1) for i in range(len(path)-1))
                H.add_edge(u, v, weight=path_weight)
    
    # Choose the TSP method
    if method is None:
        method = christofides if not G.is_directed() else asadpour_atsp
    
    # Solve TSP on the complete graph
    tsp_cycle = method(H, weight=weight, **kwargs)
    
    # Post-process the solution
    if not cycle:
        # Remove the heaviest edge to create a path
        heaviest_edge = max(((u, v) for u, v in nx.utils.pairwise(tsp_cycle)), key=lambda e: H[e[0]][e[1]][weight])
        tsp_cycle.remove(heaviest_edge[1])
    
    # Replace edges with shortest paths in the original graph
    final_path = []
    for u, v in nx.utils.pairwise(tsp_cycle):
        if G.is_directed():
            path = nx.shortest_path(G, u, v, weight=weight)
        else:
            path = nx.shortest_path(G, u, v, weight=weight)
        final_path.extend(path[:-1])
    
    if cycle:
        final_path.append(final_path[0])
    
    return final_path


@not_implemented_for('undirected')
@py_random_state(2)
@nx._dispatchable(edge_attrs='weight', mutates_input=True)
def asadpour_atsp(G, weight='weight', seed=None, source=None):
    """
    Returns an approximate solution to the traveling salesman problem.

    This approximate solution is one of the best known approximations for the
    asymmetric traveling salesman problem developed by Asadpour et al,
    [1]_. The algorithm first solves the Held-Karp relaxation to find a lower
    bound for the weight of the cycle. Next, it constructs an exponential
    distribution of undirected spanning trees where the probability of an
    edge being in the tree corresponds to the weight of that edge using a
    maximum entropy rounding scheme. Next we sample that distribution
    $2 \\lceil \\ln n \\rceil$ times and save the minimum sampled tree once the
    direction of the arcs is added back to the edges. Finally, we augment
    then short circuit that graph to find the approximate tour for the
    salesman.

    Parameters
    ----------
    G : nx.DiGraph
        The graph should be a complete weighted directed graph. The
        distance between all paris of nodes should be included and the triangle
        inequality should hold. That is, the direct edge between any two nodes
        should be the path of least cost.

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    source : node label (default=`None`)
        If given, return the cycle starting and ending at the given node.

    Returns
    -------
    cycle : list of nodes
        Returns the cycle (list of nodes) that a salesman can follow to minimize
        the total weight of the trip.

    Raises
    ------
    NetworkXError
        If `G` is not complete or has less than two nodes, the algorithm raises
        an exception.

    NetworkXError
        If `source` is not `None` and is not a node in `G`, the algorithm raises
        an exception.

    NetworkXNotImplemented
        If `G` is an undirected graph.

    References
    ----------
    .. [1] A. Asadpour, M. X. Goemans, A. Madry, S. O. Gharan, and A. Saberi,
       An o(log n/log log n)-approximation algorithm for the asymmetric
       traveling salesman problem, Operations research, 65 (2017),
       pp. 1043–1061

    Examples
    --------
    >>> import networkx as nx
    >>> import networkx.algorithms.approximation as approx
    >>> G = nx.complete_graph(3, create_using=nx.DiGraph)
    >>> nx.set_edge_attributes(
    ...     G, {(0, 1): 2, (1, 2): 2, (2, 0): 2, (0, 2): 1, (2, 1): 1, (1, 0): 1}, "weight"
    ... )
    >>> tour = approx.asadpour_atsp(G, source=0)
    >>> tour
    [0, 2, 1, 0]
    """
    if not isinstance(G, nx.DiGraph):
        raise nx.NetworkXNotImplemented("asadpour_atsp works only for directed graphs.")
    
    if len(G) < 2:
        raise nx.NetworkXError("Graph must have at least two nodes.")
    
    if not nx.is_strongly_connected(G):
        raise nx.NetworkXError("Graph must be strongly connected.")
    
    if source is not None and source not in G:
        raise nx.NetworkXError("The source node is not in G")
    
    # Step 1: Solve Held-Karp relaxation
    hk_solution = held_karp_ascent(G, weight=weight)
    
    # Step 2: Construct exponential distribution of spanning trees
    tree_distribution = spanning_tree_distribution(G, hk_solution)
    
    # Step 3: Sample from the distribution and find minimum weight tree
    rng = np.random.default_rng(seed)
    n = len(G)
    num_samples = 2 * math.ceil(math.log(n))
    
    min_tree = None
    min_weight = float('inf')
    
    for _ in range(num_samples):
        tree = random_spanning_tree(G, weight=tree_distribution)
        tree_weight = sum(G[u][v][weight] for u, v in tree.edges())
        if tree_weight < min_weight:
            min_tree = tree
            min_weight = tree_weight
    
    # Step 4: Augment and short-circuit the minimum weight tree
    cycle = nx.eulerian_circuit(min_tree)
    tour = list(dict.fromkeys(u for u, v in cycle))
    
    if source is not None:
        start_index = tour.index(source)
        tour = tour[start_index:] + tour[:start_index]
    
    tour.append(tour[0])
    
    return tour


@nx._dispatchable(edge_attrs='weight', mutates_input=True, returns_graph=True)
def held_karp_ascent(G, weight='weight'):
    """
    Minimizes the Held-Karp relaxation of the TSP for `G`

    Solves the Held-Karp relaxation of the input complete digraph and scales
    the output solution for use in the Asadpour [1]_ ASTP algorithm.

    The Held-Karp relaxation defines the lower bound for solutions to the
    ATSP, although it does return a fractional solution. This is used in the
    Asadpour algorithm as an initial solution which is later rounded to a
    integral tree within the spanning tree polytopes. This function solves
    the relaxation with the branch and bound method in [2]_.

    Parameters
    ----------
    G : nx.DiGraph
        The graph should be a complete weighted directed graph.
        The distance between all paris of nodes should be included.

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    Returns
    -------
    OPT : float
        The cost for the optimal solution to the Held-Karp relaxation
    z : dict or nx.Graph
        A symmetrized and scaled version of the optimal solution to the
        Held-Karp relaxation for use in the Asadpour algorithm.

        If an integral solution is found, then that is an optimal solution for
        the ATSP problem and that is returned instead.

    References
    ----------
    .. [1] A. Asadpour, M. X. Goemans, A. Madry, S. O. Gharan, and A. Saberi,
       An o(log n/log log n)-approximation algorithm for the asymmetric
       traveling salesman problem, Operations research, 65 (2017),
       pp. 1043–1061

    .. [2] M. Held, R. M. Karp, The traveling-salesman problem and minimum
           spanning trees, Operations Research, 1970-11-01, Vol. 18 (6),
           pp.1138-1162
    """
    n = len(G)
    nodes = list(G.nodes())
    
    # Initialize dual variables
    pi = {i: 0 for i in nodes}
    
    # Initialize the lower bound
    lower_bound = 0
    
    # Main loop
    for _ in range(100):  # You may need to adjust the number of iterations
        # Compute reduced costs
        reduced_costs = {(i, j): G[i][j][weight] - pi[i] + pi[j] for i in nodes for j in nodes if i != j}
        
        # Find minimum 1-tree
        T = nx.minimum_spanning_tree(nx.Graph(reduced_costs))
        T_cost = sum(reduced_costs[e] for e in T.edges())
        
        # Update lower bound
        current_lower_bound = T_cost + sum(pi.values()) * 2
        if current_lower_bound > lower_bound:
            lower_bound = current_lower_bound
        
        # Check if we have found an optimal tour
        if nx.is_hamiltonian_path(T):
            return lower_bound, T
        
        # Update dual variables
        degrees = dict(T.degree())
        step_size = 1 / math.sqrt(_ + 1)  # Decreasing step size
        for i in nodes:
            pi[i] += step_size * (degrees.get(i, 0) - 2)
    
    # Construct the symmetrized solution
    z = nx.Graph()
    for i in nodes:
        for j in nodes:
            if i != j:
                weight_ij = G[i][j][weight] - pi[i] + pi[j]
                weight_ji = G[j][i][weight] - pi[j] + pi[i]
                z.add_edge(i, j, weight=(weight_ij + weight_ji) / 2)
    
    return lower_bound, z


@nx._dispatchable
def spanning_tree_distribution(G, z):
    """
    Find the asadpour exponential distribution of spanning trees.

    Solves the Maximum Entropy Convex Program in the Asadpour algorithm [1]_
    using the approach in section 7 to build an exponential distribution of
    undirected spanning trees.

    This algorithm ensures that the probability of any edge in a spanning
    tree is proportional to the sum of the probabilities of the tress
    containing that edge over the sum of the probabilities of all spanning
    trees of the graph.

    Parameters
    ----------
    G : nx.MultiGraph
        The undirected support graph for the Held Karp relaxation

    z : dict
        The output of `held_karp_ascent()`, a scaled version of the Held-Karp
        solution.

    Returns
    -------
    gamma : dict
        The probability distribution which approximately preserves the marginal
        probabilities of `z`.
    """
    n = len(G)
    epsilon = 1 / (8 * n)
    
    # Initialize gamma
    gamma = {e: z[e]['weight'] for e in G.edges()}
    
    # Main loop
    for _ in range(100):  # You may need to adjust the number of iterations
        # Compute the current distribution
        T = nx.minimum_spanning_tree(G, weight=gamma)
        p = {e: 1 if e in T.edges() else 0 for e in G.edges()}
        
        # Check if we're close enough to z
        if all(abs(p[e] - z[e]['weight']) <= epsilon for e in G.edges()):
            break
        
        # Update gamma
        for e in G.edges():
            if p[e] < z[e]['weight'] - epsilon:
                gamma[e] *= (1 + epsilon)
            elif p[e] > z[e]['weight'] + epsilon:
                gamma[e] *= (1 - epsilon)
    
    return gamma


@nx._dispatchable(edge_attrs='weight')
def greedy_tsp(G, weight='weight', source=None):
    """Return a low cost cycle starting at `source` and its cost.

    This approximates a solution to the traveling salesman problem.
    It finds a cycle of all the nodes that a salesman can visit in order
    to visit many nodes while minimizing total distance.
    It uses a simple greedy algorithm.
    In essence, this function returns a large cycle given a source point
    for which the total cost of the cycle is minimized.

    Parameters
    ----------
    G : Graph
        The Graph should be a complete weighted undirected graph.
        The distance between all pairs of nodes should be included.

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    source : node, optional (default: first node in list(G))
        Starting node.  If None, defaults to ``next(iter(G))``

    Returns
    -------
    cycle : list of nodes
        Returns the cycle (list of nodes) that a salesman
        can follow to minimize total weight of the trip.

    Raises
    ------
    NetworkXError
        If `G` is not complete, the algorithm raises an exception.

    Examples
    --------
    >>> from networkx.algorithms import approximation as approx
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from(
    ...     {
    ...         ("A", "B", 3),
    ...         ("A", "C", 17),
    ...         ("A", "D", 14),
    ...         ("B", "A", 3),
    ...         ("B", "C", 12),
    ...         ("B", "D", 16),
    ...         ("C", "A", 13),
    ...         ("C", "B", 12),
    ...         ("C", "D", 4),
    ...         ("D", "A", 14),
    ...         ("D", "B", 15),
    ...         ("D", "C", 2),
    ...     }
    ... )
    >>> cycle = approx.greedy_tsp(G, source="D")
    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    >>> cycle
    ['D', 'C', 'B', 'A', 'D']
    >>> cost
    31

    Notes
    -----
    This implementation of a greedy algorithm is based on the following:

    - The algorithm adds a node to the solution at every iteration.
    - The algorithm selects a node not already in the cycle whose connection
      to the previous node adds the least cost to the cycle.

    A greedy algorithm does not always give the best solution.
    However, it can construct a first feasible solution which can
    be passed as a parameter to an iterative improvement algorithm such
    as Simulated Annealing, or Threshold Accepting.

    Time complexity: It has a running time $O(|V|^2)$
    """
    if source is None:
        source = next(iter(G))
    
    if source not in G:
        raise nx.NetworkXError("Starting node not in graph")
    
    if len(G) == 1:
        return [source]
    
    nodeset = set(G)
    nodeset.remove(source)
    cycle = [source]
    next_node = source
    
    while nodeset:
        edges = ((next_node, v, G[next_node][v].get(weight, 1)) for v in nodeset)
        (_, next_node, min_weight) = min(edges, key=lambda x: x[2])
        cycle.append(next_node)
        nodeset.remove(next_node)
    
    cycle.append(cycle[0])
    
    return cycle


@py_random_state(9)
@nx._dispatchable(edge_attrs='weight')
def simulated_annealing_tsp(G, init_cycle, weight='weight', source=None,
    temp=100, move='1-1', max_iterations=10, N_inner=100, alpha=0.01, seed=None
    ):
    """Returns an approximate solution to the traveling salesman problem.

    This function uses simulated annealing to approximate the minimal cost
    cycle through the nodes. Starting from a suboptimal solution, simulated
    annealing perturbs that solution, occasionally accepting changes that make
    the solution worse to escape from a locally optimal solution. The chance
    of accepting such changes decreases over the iterations to encourage
    an optimal result.  In summary, the function returns a cycle starting
    at `source` for which the total cost is minimized. It also returns the cost.

    The chance of accepting a proposed change is related to a parameter called
    the temperature (annealing has a physical analogue of steel hardening
    as it cools). As the temperature is reduced, the chance of moves that
    increase cost goes down.

    Parameters
    ----------
    G : Graph
        `G` should be a complete weighted graph.
        The distance between all pairs of nodes should be included.

    init_cycle : list of all nodes or "greedy"
        The initial solution (a cycle through all nodes returning to the start).
        This argument has no default to make you think about it.
        If "greedy", use `greedy_tsp(G, weight)`.
        Other common starting cycles are `list(G) + [next(iter(G))]` or the final
        result of `simulated_annealing_tsp` when doing `threshold_accepting_tsp`.

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    source : node, optional (default: first node in list(G))
        Starting node.  If None, defaults to ``next(iter(G))``

    temp : int, optional (default=100)
        The algorithm's temperature parameter. It represents the initial
        value of temperature

    move : "1-1" or "1-0" or function, optional (default="1-1")
        Indicator of what move to use when finding new trial solutions.
        Strings indicate two special built-in moves:

        - "1-1": 1-1 exchange which transposes the position
          of two elements of the current solution.
          The function called is :func:`swap_two_nodes`.
          For example if we apply 1-1 exchange in the solution
          ``A = [3, 2, 1, 4, 3]``
          we can get the following by the transposition of 1 and 4 elements:
          ``A' = [3, 2, 4, 1, 3]``
        - "1-0": 1-0 exchange which moves an node in the solution
          to a new position.
          The function called is :func:`move_one_node`.
          For example if we apply 1-0 exchange in the solution
          ``A = [3, 2, 1, 4, 3]``
          we can transfer the fourth element to the second position:
          ``A' = [3, 4, 2, 1, 3]``

        You may provide your own functions to enact a move from
        one solution to a neighbor solution. The function must take
        the solution as input along with a `seed` input to control
        random number generation (see the `seed` input here).
        Your function should maintain the solution as a cycle with
        equal first and last node and all others appearing once.
        Your function should return the new solution.

    max_iterations : int, optional (default=10)
        Declared done when this number of consecutive iterations of
        the outer loop occurs without any change in the best cost solution.

    N_inner : int, optional (default=100)
        The number of iterations of the inner loop.

    alpha : float between (0, 1), optional (default=0.01)
        Percentage of temperature decrease in each iteration
        of outer loop

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    cycle : list of nodes
        Returns the cycle (list of nodes) that a salesman
        can follow to minimize total weight of the trip.

    Raises
    ------
    NetworkXError
        If `G` is not complete the algorithm raises an exception.

    Examples
    --------
    >>> from networkx.algorithms import approximation as approx
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from(
    ...     {
    ...         ("A", "B", 3),
    ...         ("A", "C", 17),
    ...         ("A", "D", 14),
    ...         ("B", "A", 3),
    ...         ("B", "C", 12),
    ...         ("B", "D", 16),
    ...         ("C", "A", 13),
    ...         ("C", "B", 12),
    ...         ("C", "D", 4),
    ...         ("D", "A", 14),
    ...         ("D", "B", 15),
    ...         ("D", "C", 2),
    ...     }
    ... )
    >>> cycle = approx.simulated_annealing_tsp(G, "greedy", source="D")
    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    >>> cycle
    ['D', 'C', 'B', 'A', 'D']
    >>> cost
    31
    >>> incycle = ["D", "B", "A", "C", "D"]
    >>> cycle = approx.simulated_annealing_tsp(G, incycle, source="D")
    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    >>> cycle
    ['D', 'C', 'B', 'A', 'D']
    >>> cost
    31

    Notes
    -----
    Simulated Annealing is a metaheuristic local search algorithm.
    The main characteristic of this algorithm is that it accepts
    even solutions which lead to the increase of the cost in order
    to escape from low quality local optimal solutions.

    This algorithm needs an initial solution. If not provided, it is
    constructed by a simple greedy algorithm. At every iteration, the
    algorithm selects thoughtfully a neighbor solution.
    Consider $c(x)$ cost of current solution and $c(x')$ cost of a
    neighbor solution.
    If $c(x') - c(x) <= 0$ then the neighbor solution becomes the current
    solution for the next iteration. Otherwise, the algorithm accepts
    the neighbor solution with probability $p = exp - ([c(x') - c(x)] / temp)$.
    Otherwise the current solution is retained.

    `temp` is a parameter of the algorithm and represents temperature.

    Time complexity:
    For $N_i$ iterations of the inner loop and $N_o$ iterations of the
    outer loop, this algorithm has running time $O(N_i * N_o * |V|)$.

    For more information and how the algorithm is inspired see:
    http://en.wikipedia.org/wiki/Simulated_annealing
    """
    if source is None:
        source = next(iter(G))
    
    if init_cycle == "greedy":
        best_cycle = greedy_tsp(G, weight=weight, source=source)
    else:
        best_cycle = list(init_cycle)
    
    if move == "1-1":
        move_func = swap_two_nodes
    elif move == "1-0":
        move_func = move_one_node
    else:
        move_func = move
    
    rng = np.random.default_rng(seed)
    
    def cycle_cost(cycle):
        return sum(G[u][v].get(weight, 1) for u, v in nx.utils.pairwise(cycle))
    
    best_cost = cycle_cost(best_cycle)
    current_cycle = best_cycle.copy()
    current_cost = best_cost
    
    no_improvement = 0
    for _ in range(max_iterations):
        for _ in range(N_inner):
            candidate_cycle = move_func(current_cycle.copy(), rng)
            candidate_cost = cycle_cost(candidate_cycle)
            
            if candidate_cost < current_cost or rng.random() < math.exp((current_cost - candidate_cost) / temp):
                current_cycle = candidate_cycle
                current_cost = candidate_cost
                
                if current_cost < best_cost:
                    best_cycle = current_cycle.copy()
                    best_cost = current_cost
                    no_improvement = 0
                    break
        else:
            no_improvement += 1
        
        if no_improvement >= max_iterations:
            break
        
        temp *= (1 - alpha)
    
    return best_cycle


@py_random_state(9)
@nx._dispatchable(edge_attrs='weight')
def threshold_accepting_tsp(G, init_cycle, weight='weight', source=None,
    threshold=1, move='1-1', max_iterations=10, N_inner=100, alpha=0.1,
    seed=None):
    """Returns an approximate solution to the traveling salesman problem.

    This function uses threshold accepting methods to approximate the minimal cost
    cycle through the nodes. Starting from a suboptimal solution, threshold
    accepting methods perturb that solution, accepting any changes that make
    the solution no worse than increasing by a threshold amount. Improvements
    in cost are accepted, but so are changes leading to small increases in cost.
    This allows the solution to leave suboptimal local minima in solution space.
    The threshold is decreased slowly as iterations proceed helping to ensure
    an optimum. In summary, the function returns a cycle starting at `source`
    for which the total cost is minimized.

    Parameters
    ----------
    G : Graph
        `G` should be a complete weighted graph.
        The distance between all pairs of nodes should be included.

    init_cycle : list or "greedy"
        The initial solution (a cycle through all nodes returning to the start).
        This argument has no default to make you think about it.
        If "greedy", use `greedy_tsp(G, weight)`.
        Other common starting cycles are `list(G) + [next(iter(G))]` or the final
        result of `simulated_annealing_tsp` when doing `threshold_accepting_tsp`.

    weight : string, optional (default="weight")
        Edge data key corresponding to the edge weight.
        If any edge does not have this attribute the weight is set to 1.

    source : node, optional (default: first node in list(G))
        Starting node.  If None, defaults to ``next(iter(G))``

    threshold : int, optional (default=1)
        The algorithm's threshold parameter. It represents the initial
        threshold's value

    move : "1-1" or "1-0" or function, optional (default="1-1")
        Indicator of what move to use when finding new trial solutions.
        Strings indicate two special built-in moves:

        - "1-1": 1-1 exchange which transposes the position
          of two elements of the current solution.
          The function called is :func:`swap_two_nodes`.
          For example if we apply 1-1 exchange in the solution
          ``A = [3, 2, 1, 4, 3]``
          we can get the following by the transposition of 1 and 4 elements:
          ``A' = [3, 2, 4, 1, 3]``
        - "1-0": 1-0 exchange which moves an node in the solution
          to a new position.
          The function called is :func:`move_one_node`.
          For example if we apply 1-0 exchange in the solution
          ``A = [3, 2, 1, 4, 3]``
          we can transfer the fourth element to the second position:
          ``A' = [3, 4, 2, 1, 3]``

        You may provide your own functions to enact a move from
        one solution to a neighbor solution. The function must take
        the solution as input along with a `seed` input to control
        random number generation (see the `seed` input here).
        Your function should maintain the solution as a cycle with
        equal first and last node and all others appearing once.
        Your function should return the new solution.

    max_iterations : int, optional (default=10)
        Declared done when this number of consecutive iterations of
        the outer loop occurs without any change in the best cost solution.

    N_inner : int, optional (default=100)
        The number of iterations of the inner loop.

    alpha : float between (0, 1), optional (default=0.1)
        Percentage of threshold decrease when there is at
        least one acceptance of a neighbor solution.
        If no inner loop moves are accepted the threshold remains unchanged.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    cycle : list of nodes
        Returns the cycle (list of nodes) that a salesman
        can follow to minimize total weight of the trip.

    Raises
    ------
    NetworkXError
        If `G` is not complete the algorithm raises an exception.

    Examples
    --------
    >>> from networkx.algorithms import approximation as approx
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from(
    ...     {
    ...         ("A", "B", 3),
    ...         ("A", "C", 17),
    ...         ("A", "D", 14),
    ...         ("B", "A", 3),
    ...         ("B", "C", 12),
    ...         ("B", "D", 16),
    ...         ("C", "A", 13),
    ...         ("C", "B", 12),
    ...         ("C", "D", 4),
    ...         ("D", "A", 14),
    ...         ("D", "B", 15),
    ...         ("D", "C", 2),
    ...     }
    ... )
    >>> cycle = approx.threshold_accepting_tsp(G, "greedy", source="D")
    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    >>> cycle
    ['D', 'C', 'B', 'A', 'D']
    >>> cost
    31
    >>> incycle = ["D", "B", "A", "C", "D"]
    >>> cycle = approx.threshold_accepting_tsp(G, incycle, source="D")
    >>> cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    >>> cycle
    ['D', 'C', 'B', 'A', 'D']
    >>> cost
    31

    Notes
    -----
    Threshold Accepting is a metaheuristic local search algorithm.
    The main characteristic of this algorithm is that it accepts
    even solutions which lead to the increase of the cost in order
    to escape from low quality local optimal solutions.

    This algorithm needs an initial solution. This solution can be
    constructed by a simple greedy algorithm. At every iteration, it
    selects thoughtfully a neighbor solution.
    Consider $c(x)$ cost of current solution and $c(x')$ cost of
    neighbor solution.
    If $c(x') - c(x) <= threshold$ then the neighbor solution becomes the current
    solution for the next iteration, where the threshold is named threshold.

    In comparison to the Simulated Annealing algorithm, the Threshold
    Accepting algorithm does not accept very low quality solutions
    (due to the presence of the threshold value). In the case of
    Simulated Annealing, even a very low quality solution can
    be accepted with probability $p$.

    Time complexity:
    It has a running time $O(m * n * |V|)$ where $m$ and $n$ are the number
    of times the outer and inner loop run respectively.

    For more information and how algorithm is inspired see:
    https://doi.org/10.1016/0021-9991(90)90201-B

    See Also
    --------
    simulated_annealing_tsp

    """
    pass
