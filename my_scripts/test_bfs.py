"""
Graph karate from W. W. Zachary, An information flow model for conflict and fission in small groups, Journal of
Anthropological Research 33, 452-473 (1977). Retrieved from Mark Newman’s website.
"""
# Standard packages.
import itertools
from pathlib import Path
from string import ascii_uppercase

# Non-standard packages.
import graph_tool.all as gt

# Custom packages/imports.
from lib.set_dist_visitor import SetDistVisitor, VerboseSetDistVisitor


def all_strings_iter():
    for size in itertools.count(1):
        for s in itertools.product(ascii_uppercase, repeat=size):
            yield ''.join(s)


# Initialise tokyo.
ag_dir = Path(__file__).resolve().parent.parent / 'arch_graphs'
tokyo = gt.load_graph(str(ag_dir / 'AG_Tokyo.graphml'))

dc_1 = tokyo.add_vertex()
dc_2 = tokyo.add_vertex()
tokyo.add_edge(dc_1, dc_2)

# Add vertex property.
name = tokyo.new_vertex_property("string")
dist = tokyo.new_vertex_property("int")

# Set name vertex property
all_strings = all_strings_iter()
for vertex in tokyo.vertices():
    name[vertex] = next(all_strings)

gt.bfs_search(tokyo, tokyo.vertex(0), VerboseSetDistVisitor(dist, name))

name_dist = tokyo.new_vertex_property("string")
for vertex in tokyo.vertices():
    name_dist[vertex] = name[vertex] + ', ' + str(dist[vertex])

print('Displaying graph in new window..')
gt.graph_draw(tokyo, vertex_text=name_dist)


# Initialise karate.
karate = gt.collection.data['karate']

# Add vertex property.
name = karate.new_vertex_property("string")
dist = karate.new_vertex_property("int")

# Set name vertex property
all_strings = all_strings_iter()
for vertex in karate.vertices():
    name[vertex] = next(all_strings)

gt.bfs_search(karate, karate.vertex(0), VerboseSetDistVisitor(dist, name))

name_dist = karate.new_vertex_property("string")
for vertex in karate.vertices():
    name_dist[vertex] = name[vertex] + ', ' + str(dist[vertex])

print('Displaying graph in new window..')
gt.graph_draw(karate, vertex_text=name_dist)


