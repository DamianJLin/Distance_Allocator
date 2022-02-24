import graph_tool.all as gt
import networkx as nx

import fidls.fidls_d
from lib.distance_calculator import DistanceCalculator


def circuit_transformation(
        embedding: dict,
        circuit: list,
        arch_graph: gt.Graph,
        distance_calculator: DistanceCalculator
):
    from fidls.fidls_d import qct_old
    """
    Wrapper around fidls.fidls_d.qct_old().
    
    :param embedding: Map from logical qubits (int) to embedded qubits (gt.vertex).
    :param circuit: List of gates in the quantum circuit. Gates are composed of two logical qubits in a list.
    :param architecture_graph: gt.Graph of the quantum architecture.
    :return: New circuit capable of execution on the architecture, quality of the new circuit (number of gates as a
    factor of the input circuit).
    """
    arch_network = nx.Graph()

    # Add nodes and edges based on the indices of the vertices in architecture_graph.
    arch_network.add_nodes_from(
        arch_graph.iter_vertices()
    )
    arch_network.add_edges_from(
        arch_graph.iter_edges()
    )

    # Create unembedding, the inverse map of the embedding, which goes from physical qubits (now represented by ints),
    # to logical qubits (also ints). Since it is indexed by ints in the range [0, num of arch_graph.vertices() - 1], we
    # can use a list instead of a dictionary.
    for v in arch_graph.vertices():
        print(arch_graph.vertex_index[v], '  ', embedding[v])
    unembedding = [-1] * len(list(arch_graph.vertices()))
    for i, v in enumerate(arch_graph.vertices()):
        if embedding[v]:
            pass

    # Create the set of logical qubits in the circuit.
    logical_qubits = set()
    for gate in circuit:
        for qubit in gate:
            logical_qubits.add(qubit)

    # Shortest path length object. Needs to be indexed by a tuple of physical qubits (ints) to give the shortest path
    # length between those qubits. This is nearly already implemented, only needing turning the indexing into a
    # function call to DistanceCalculator.distance().
    shortest_path_finder = ShortestPathFinder(distance_calculator)

    # Call fidls.fidls_d.qct_old().
    circuit_transformed, quality = fidls.fidls_d.qct_old(
        tau=unembedding,
        C=circuit,
        Q=logical_qubits,
        G=arch_network,
        EG=arch_network.edges(),
        V=arch_network.nodes(),
        SPL=shortest_path_finder,
        QFilter_type='01y'  # This is apparently the preferred filter type.
    )

    return circuit_transformed, quality


class ShortestPathFinder:
    """
    Class wrapper around DistanceCalculator that allows distance call with tuple indexing.
    """
    def __init__(self, distance_calculator: DistanceCalculator):
        assert isinstance(distance_calculator, DistanceCalculator)

        self.distance_calculator = distance_calculator

    def __getitem__(self, item):
        if not (
            isinstance(item, tuple)
            and len(item) == 2
            and all(isinstance(i, int) for i in item)
        ):
            raise IndexError("ShortestPath is indexed by tuple of length 2.")
        u, v = item
        return self.distance_calculator.distance(u, v)
