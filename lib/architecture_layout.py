import graph_tool.all as gt


def tokyo_pos(graph: gt.Graph):
    display_pos = graph.new_vertex_property('vector<float>')

    for v in graph.vertices():
        i = graph.vertex_index[v]
        x = i % 5
        y = i // 5
        display_pos[v] = (x, y)

    return display_pos


def rochester_pos(graph: gt.Graph):
    display_pos = graph.new_vertex_property('vector<float>')

    idx_to_pos = {
        0: (0, 0),
        1: (1, 0),
        2: (2, 0),
        3: (3, 0),
        4: (4, 0),

        5: (0, 1),
        6: (4, 1),

        7: (-2, 2),
        8: (-1, 2),
        9: (0, 2),
        10: (1, 2),
        11: (2, 2),
        12: (3, 2),
        13: (4, 2),
        14: (5, 2),
        15: (6, 2),

        16: (-2, 3),
        17: (2, 3),
        18: (6, 3),

        19: (-2, 4),
        20: (-1, 4),
        21: (0, 4),
        22: (1, 4),
        23: (2, 4),
        24: (3, 4),
        25: (4, 4),
        26: (5, 4),
        27: (6, 4),

        28: (0, 5),
        29: (4, 5),

        30: (-2, 6),
        31: (-1, 6),
        32: (0, 6),
        33: (1, 6),
        34: (2, 6),
        35: (3, 6),
        36: (4, 6),
        37: (5, 6),
        38: (6, 6),

        39: (-2, 7),
        40: (2, 7),
        41: (6, 7),

        42: (-2, 8),
        43: (-1, 8),
        44: (0, 8),
        45: (1, 8),
        46: (2, 8),
        47: (3, 8),
        48: (4, 8),
        49: (5, 8),
        50: (6, 8),

        51: (0, 9),
        52: (4, 9),
    }

    for v in graph.vertices():
        i = graph.vertex_index[v]
        display_pos[v] = idx_to_pos[i]

    return display_pos
