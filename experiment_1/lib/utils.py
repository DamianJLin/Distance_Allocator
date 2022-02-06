import pathlib
import graph_tool.all as gt


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


def save_embedding_image(
        subgraph: gt.Graph,
        graph: gt.Graph,
        mapping: gt.VertexPropertyMap,
        location: pathlib.Path | type(None),
        log_to_alloc: dict
):
    """
    Save image of subgraph embedded in graph with mapping at location.
    :param log_to_alloc: Map from logical vertex (int) to allocated vertex in sub.
    :param subgraph: Subgraph
    :param graph: Graph
    :param mapping: Map from Subgraph to graph.
    :param location: Path object or string location
    :return: None
    """
    vmask, emask = gt.mark_subgraph(graph, subgraph, mapping)

    # Setting the vertex text labels.
    label_on_sub = subgraph.new_vertex_property('string')
    for log, phys in log_to_alloc.items():
        label_on_sub[phys] = log
    label_on_graph = graph.new_vertex_property('string')
    for s in subgraph.vertices():
        label_on_graph[graph.vertex(mapping[s])] = label_on_sub[s]

    # Setting the position vertex property. Tailored to QTokyo only.
    def pos(n):
        y = n // 5
        x = n % 5
        return x, y

    display_pos = graph.new_vertex_property('vector<float>')
    for v in graph.vertices():
        display_pos[v] = pos(graph.vertex_index[v])

    if location is not None:
        output_path = str(location)  # Save as file.
    else:
        output_path = None  # Interactive window.

    gt.graph_draw(
        graph,
        pos=display_pos,
        vertex_text=label_on_graph,
        vertex_fill_color=vmask,
        edge_color=emask,
        output=output_path
    )


class NoSubgraphIsomorphism(Exception):
    """
    Custom exception raised when no isomorphic subgraph found.
    """
    pass
