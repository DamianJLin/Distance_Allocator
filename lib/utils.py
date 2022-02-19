import pathlib
import graph_tool.all as gt
import lib.architecture_layout
import numpy as np
import re


def qasm_as_array(qasm_path):
    with open(qasm_path, 'r') as qasm_file:
        qasm = qasm_file.read()
    pattern = r'\[?\[(\d*), (\d*)\]\]?'
    circuit = np.asarray(
        [
            np.asarray(list(map(int, tup)))
            for tup in re.findall(pattern, qasm)
        ]
    )

    return circuit


def save_embedding_image(
        subgraph: gt.Graph,
        graph: gt.Graph,
        architecture: str,
        mapping: gt.VertexPropertyMap,
        location: pathlib.Path | type(None),
        log_to_alloc: dict
):
    """
    Save image of subgraph embedded in graph with mapping at location.
    :param architecture: The string name of the architecture e.g. IBM_Tokyo.
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

    # Setting the position vertex property.
    display_pos = None

    if architecture == 'IBM_Tokyo':
        display_pos = lib.architecture_layout.tokyo_pos(graph)
    elif architecture == 'IBM_Rochester':
        display_pos = lib.architecture_layout.rochester_pos(graph)

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
