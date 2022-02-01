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

    def __init__(self, dist):
        if not (dist.python_value_type() is int or dist.python_value_type is float):
            raise TypeError(
                f'The property map dist must be of type int or float. '
                f'Encountered type: {dist.python_value_type}'
            )
        self.dist = dist

    def tree_edge(self, e):
        self.dist[e.target()] = self.dist[e.source()] + 1


class VerboseSetDistVisitor(SetDistVisitor):
    """
    Subclass of DistVisitor object that also prints vertex examinations and discoveries to stdout.
    """

    def __init__(self, dist, name):
        super().__init__(dist)
        self.name = name

    def discover_vertex(self, u):
        indent = self.dist[u]
        print('\t' * indent + f'{self.name[u]} has been discovered at d = {self.dist[u]}')

    def examine_vertex(self, u):
        indent = self.dist[u]
        print('\t' + '    ' * indent + f'{self.name[u]} has been examined ...')

