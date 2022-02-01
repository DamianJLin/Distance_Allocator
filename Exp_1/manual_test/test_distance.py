# Standard packages.
import itertools
from pathlib import Path
from string import ascii_uppercase

# Non-standard packages.
import graph_tool.all as gt

# Custom packages/imports.
from lib.set_dist_visitor import SetDistVisitor


def all_strings_iter():
    for size in itertools.count(1):
        for s in itertools.product(ascii_uppercase, repeat=size):
            yield ''.join(s)


# Initialise graph.
ag_dir = Path(__file__).resolve().parent.parent / 'arch_graphs'
tokyo = gt.load_graph(str(ag_dir / 'AG_Tokyo.graphml'))

# Add vertex property.
name = tokyo.new_vertex_property("string")
dist = tokyo.new_vertex_property("int")

# Set name vertex property
all_strings = all_strings_iter()
for vertex in tokyo.vertices():
    name[vertex] = next(all_strings)

gt.bfs_search(tokyo, tokyo.vertex(0), SetDistVisitor(dist))

name_dist = tokyo.new_vertex_property("string")
for vertex in tokyo.vertices():
    name_dist[vertex] = name[vertex] + ', ' + str(dist[vertex])

gt.graph_draw(tokyo, vertex_text=name_dist)
