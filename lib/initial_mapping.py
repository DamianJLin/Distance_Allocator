import graph_tool.all as gt
import numpy as np
from lib.utils import NoSubgraphIsomorphism


def get_layer(circuit, mask, logical_qubits):
    """
    Generates circuit[0] then all the gates (edges) in the top layer of circuit (all gates that can be run
    parallel to circuit[0]).

    :param logical_qubits: Set of logical qubits in the circuit.
    :param mask: Array of bools for whether gates are already allocated.
    :param circuit: List[List[Int]] the sublist of gates to generate from.
    :return: None
    """
    independent = logical_qubits.copy()

    for j, (x, y) in enumerate(circuit):

        # If a gate is already allocated, ignore and move on.
        if mask[j]:
            continue

        # If all logical qubits are depended on at this stage in the circuit, no more gates can be parallel.
        # Also true if all but one logical qubit is depended on (2 qubits needed for gate).
        if len(independent) <= 1:
            break

        # Otherwise, either it is parallel and can be returned, or is not parallel, and depends on a previous gate.
        gate = {x, y}
        if gate.issubset(independent):
            yield j, (x, y)

        independent -= gate

    return


def greedy_layer_initial_mapping(circuit, architecture_graph: gt.Graph):
    """
    Use the following algorithm to finding an initial subcircuit:

    while layer_forbids_embedding is False:
        search all gates in layer:
            add gate to subgraph
            if subgraph embeds:
                break and go to next layer
            if gate does not embed:
                fix up the graph
                continue to next gate in layer
        set layer_forbids_embedding True

    :param circuit:
    :param architecture_graph:
    :return:
    """

    # Set of logical qubits (represented by ints).
    logical_qubits = set()
    for u, v in circuit:
        logical_qubits.add(u)
        logical_qubits.add(v)

    # Dictionary for the physical qubit (vertex in sub) a logical qubit (int) is allocated to.
    logical_to_allocated = {}

    # Keep track of which gates are allocated
    gate_allocation = np.zeros(len(circuit), dtype=bool)

    # Keep list of gates in the initial circuit.
    initial_circuit = []  # List for fast append.

    # Initialise subgraph induced from adding circuits. Empty initially.
    sub = gt.Graph()
    sub.set_directed(False)
    sub.set_fast_edge_removal(fast=True)

    embeddings_init = None

    # If a layer is fully searched with no gates that can be allocated, then we have finished.
    layer_forbids_embedding = False
    while not layer_forbids_embedding:

        embedding_found_in_layer = False
        for i, (u_log, v_log) in get_layer(circuit, mask=gate_allocation, logical_qubits=logical_qubits):

            u_is_allocated = bool(u_log in logical_to_allocated)
            v_is_allocated = bool(v_log in logical_to_allocated)

            # u and v embedded
            if u_is_allocated and v_is_allocated:
                e = sub.add_edge(
                    logical_to_allocated[u_log],
                    logical_to_allocated[v_log]
                )
            # u embedded but v not
            elif u_is_allocated and not v_is_allocated:
                logical_to_allocated[v_log] = sub.add_vertex()
                e = sub.add_edge(
                    logical_to_allocated[u_log],
                    logical_to_allocated[v_log]
                )
            # u not embedded but v
            elif not u_is_allocated and v_is_allocated:
                logical_to_allocated[u_log] = sub.add_vertex()
                e = sub.add_edge(
                    logical_to_allocated[u_log],
                    logical_to_allocated[v_log]
                )
            # u nor v embedded
            else:
                logical_to_allocated[u_log] = sub.add_vertex()
                logical_to_allocated[v_log] = sub.add_vertex()
                e = sub.add_edge(
                    logical_to_allocated[u_log],
                    logical_to_allocated[v_log]
                )
            gt.remove_parallel_edges(sub)

            prev = embeddings_init
            try:
                embeddings_init = gt.subgraph_isomorphism(sub, architecture_graph, max_n=1, induced=False)

                if embeddings_init:
                    gate_allocation[i] = True
                    initial_circuit.append(np.asarray([u_log, v_log]))
                    embedding_found_in_layer = True
                    break
                else:
                    raise NoSubgraphIsomorphism

            # If no isomorphism was found, we need to clean up the added vertices and edges and the physical_qubit
            # dictionary.
            except NoSubgraphIsomorphism:
                if u_is_allocated and v_is_allocated:
                    sub.remove_edge(e)
                elif u_is_allocated and not v_is_allocated:
                    sub.remove_edge(e)
                    sub.remove_vertex(logical_to_allocated[v_log])
                    logical_to_allocated.pop(v_log)
                elif not u_is_allocated and v_is_allocated:
                    sub.remove_edge(e)
                    sub.remove_vertex(logical_to_allocated[u_log])
                    logical_to_allocated.pop(u_log)
                elif not u_is_allocated and not v_is_allocated:
                    sub.remove_edge(e)
                    sub.remove_vertex(logical_to_allocated[v_log])
                    sub.remove_vertex(logical_to_allocated[u_log])
                    logical_to_allocated.pop(v_log)
                    logical_to_allocated.pop(u_log)
                embeddings_init = prev

        if not embedding_found_in_layer:
            layer_forbids_embedding = True

    assert embeddings_init is not None
    embedding_init = embeddings_init[0]

    initial_circuit = np.asarray(initial_circuit)
    final_circuit = np.copy(circuit[gate_allocation == 0, :])

    return initial_circuit, final_circuit, embedding_init, sub, logical_to_allocated
