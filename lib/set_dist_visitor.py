"""
The graph_tool package documentation may be helpful for understanding Visitor objects.
https://graph-tool.skewed.de/static/doc/search_module.html#
"""
import graph_tool.all as gt


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
