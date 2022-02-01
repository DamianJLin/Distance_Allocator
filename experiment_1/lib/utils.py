import graph_tool.all as gt


class NoSubgraphIsomorphismError(ValueError):
    """
    Custom error raised when no isomorphic subgraph found.
    """
    pass


def layer(subcircuit):
    """
    Generates the gates (edges) in the top layer of subcircuit (all gates that can be run parallel to subcircuit[0]).

    :param subcircuit: List[List[Int]] the sublist of gates to generate from.
    :return: None
    """
    u_init, v_init = subcircuit[0]
    yield subcircuit[0]
    k = 1
    while u_init not in subcircuit[k] and v_init not in subcircuit[k]:
        yield subcircuit[k]
        k += 1

    return


def save_embedding_image(subgraph, graph, mapping, location):
    """
    Save image of subgraph embedded in graph with mapping at location.
    :param subgraph: Subgraph
    :param graph: Graph
    :param mapping: Map from Subgraph to graph.
    :param location: Path object or string location
    :return: None
    """
    vmask, emask = gt.mark_subgraph(graph, subgraph, mapping)

    gt.graph_draw(
        graph,
        vertex_fill_color=vmask,
        edge_color=emask,
        output=str(location)
    )
