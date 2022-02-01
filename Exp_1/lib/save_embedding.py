import graph_tool.all as gt


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
