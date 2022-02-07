import graph_tool.all as gt
from lib.set_dist_visitor import SetDistVisitor


class DistanceCalculator:

    def __init__(self, graph: gt.Graph):
        self.graph = graph

        # The set of already computed maps of the distance from a given vertex.
        self.computed_dist_map = {}

    def distance(self, u: gt.Vertex, v: gt.Vertex):
        u_dist_already_computed = bool(u in self.computed_dist_map)
        v_dist_already_computed = bool(v in self.computed_dist_map)
        dist = None

        if u_dist_already_computed and v_dist_already_computed:
            dist_from_u_map = self.computed_dist_map[u]
            dist = dist_from_u_map[v]
        elif u_dist_already_computed and not v_dist_already_computed:
            dist_from_u_map = self.computed_dist_map[u]
            dist = dist_from_u_map[v]
        elif not u_dist_already_computed and v_dist_already_computed:
            dist_from_v_map = self.computed_dist_map[v]
            dist = dist_from_v_map[u]
        elif not u_dist_already_computed and not v_dist_already_computed:
            # Calculate property map with distance from u.
            new_dist_from_u_map = self.graph.new_vertex_property("int")
            gt.bfs_search(self.graph, u, SetDistVisitor(new_dist_from_u_map))
            self.computed_dist_map[u] = new_dist_from_u_map
            dist = new_dist_from_u_map[v]

        assert dist is not None
        return dist
