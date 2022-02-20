"""
The graph_tool package documentation may be helpful for understanding Visitor objects.
https://graph-tool.skewed.de/static/doc/search_module.html#
"""
import graph_tool.all as gt


class DistanceCalculator:
    """
    Calculator class that implements a distance function using memorisation to efficiently calculate distance between
    physical qubits.
    """

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


class SetDistVisitor(gt.BFSVisitor):
    """
    Visitor object that when used as the visitor object for a breath-first search sets the passed vertex property map to
    the distance in the shortest path from the source vertex. Assumes an integer distance function on paths equal to the
    number of edges (edges are unweighted).
    """

    def __init__(self, dist: gt.VertexPropertyMap):
        if not (dist.python_value_type() is int):
            raise TypeError(
                f'The property map dist must be of type int. '
                f'Encountered type: {dist.python_value_type}'
            )
        self.dist = dist
        self.start_flag = False

        # Overriding BFSVisitor.initialize_vertex() would be more pythonic, but it appears not to work (perhaps a bug).
        for u in dist.get_graph().vertices():
            dist[u] = -1

    def tree_edge(self, e):
        self.dist[e.target()] = self.dist[e.source()] + 1

    def discover_vertex(self, u):
        if not self.start_flag:
            self.start_flag = True
            self.dist[u] = 0


class VerboseSetDistVisitor(SetDistVisitor):
    """
    Subclass of DistVisitor object that also prints vertex examinations and discoveries to stdout.
    """

    def __init__(self, dist, name):
        super().__init__(dist)
        self.name = name

    def discover_vertex(self, u):
        super().discover_vertex(u)
        indent = self.dist[u]
        print('\t' * indent + f'{self.name[u]} has been discovered at d = {self.dist[u]}')

    def examine_vertex(self, u):
        indent = self.dist[u]
        print('\t' + '    ' * indent + f'{self.name[u]} has been examined ...')
