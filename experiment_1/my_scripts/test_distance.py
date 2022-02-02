"""
Graph adjnoun from M. E. J. Newman, Phys. Rev. E 74, 036104 (2006). Retrieved from Mark Newman’s website.
"""
# Non-standard packages.
import graph_tool.all as gt
import numpy as np
import time
import sys

# Custom packages/imports.
from lib.distance import DistanceCalculator

adjnoun = gt.collection.data['adjnoun']
label = adjnoun.vertex_properties['label']

dist_calc = DistanceCalculator(adjnoun)

v_list = list(adjnoun.vertices())

# My method
d_array = np.empty(
    (len(v_list), len(v_list)),
    int
)

start = time.time()
for i, u in enumerate(v_list):
    for j, v in enumerate(v_list):
        d_array[i][j] = dist_calc.distance(u, v)

delta = time.time() - start
np.set_printoptions(threshold=sys.maxsize)
print(f'Time with DistanceCaluclator: {delta}')


# Graph-tool method
g_array = np.empty(
    (len(v_list), len(v_list)),
    int
)
start = time.time()
for i, u in enumerate(v_list):
    for j, v in enumerate(v_list):
        g_array[i][j] = gt.shortest_distance(adjnoun, u, v)
delta = time.time() - start
print(f'Time with naive method: {delta}')
