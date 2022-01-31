from pathlib import Path
import re
import itertools
import graph_tool.all as gt
from lib.layer import layer
from lib.no_subgraph_isomorphism_error import NoSubgraphIsomorphismError
from lib.save_embedding import save_embedding_image

ag_dir = Path(__file__).resolve().parent / 'arch_graphs'
circ_dir = Path(__file__).resolve().parent / 'circuits'
out_dir = Path(__file__).resolve().parent / 'output'

ag_paths = list(ag_dir.glob('*.graphml'))
circ_paths = list(circ_dir.glob('*.qasm.txt'))

for ag_path, circ_path in itertools.product(ag_paths, circ_paths):
    print(str(ag_path.stem).rstrip('.graphml') + ' \u2A86 ' + str(circ_path.stem).rstrip('.qasm.txt'))

    ag = gt.load_graph(str(ag_path))

    with open(circ_path, 'r') as circ_file:
        circ_str = circ_file.read()
    pattern = r'\[?\[(\d*), (\d*)\]\]?'
    circuit = [
        list(map(int, tup))
        for tup in re.findall(pattern, circ_str)
    ]
    del ag.vertex_properties["_graphml_vertex_id"]
    del ag.edge_properties["_graphml_edge_id"]

    sub = gt.Graph()  # Subgraph induced by adding circuits.
    sub.set_directed(False)
    sub.set_fast_edge_removal(fast=True)

    iso_map = None

    # Keeping track of logical qubits by their number in the qasm files. This is because graph_tool node ids are always
    # ordered starting from 0.
    qubit_by_qasm_id = {}

    # Find maximal top sublist (largest circuit[:n] that can still be embedded as a subgraph).
    for i, (u_layer, v_layer) in enumerate(circuit):

        iso_found_in_layer = False

        for j, (u, v) in enumerate(layer(circuit[i:])):
            # Add necessary edges and nodes to graph.
            u_already_qubit = bool(u in qubit_by_qasm_id)
            v_already_qubit = bool(v in qubit_by_qasm_id)
            match (u_already_qubit, v_already_qubit):
                # u and v
                case (True, True):
                    e = sub.add_edge(qubit_by_qasm_id[u], qubit_by_qasm_id[v])
                # u but not v (and vice versa)
                case (True, False) | (False, True):
                    if v in qubit_by_qasm_id:  # Assume u is the one assigned
                        u, v = v, u
                    qubit_by_qasm_id[v] = sub.add_vertex()
                    e = sub.add_edge(qubit_by_qasm_id[u], qubit_by_qasm_id[v])
                # Neither u nor v
                case (False, False):
                    qubit_by_qasm_id[u] = sub.add_vertex()
                    qubit_by_qasm_id[v] = sub.add_vertex()
                    e = sub.add_edge(qubit_by_qasm_id[u], qubit_by_qasm_id[v])
                case _:
                    e = None
            try:
                iso_map = gt.subgraph_isomorphism(sub, ag, max_n=1, induced=False)
                if iso_map:
                    iso_found_in_layer = True
                    break
                else:
                    raise NoSubgraphIsomorphismError('No subgraph isomorphism between sub and ag.')
            # If there was no isomorphism, we need to clean up edges and vertices we tried to add, then break.
            except NoSubgraphIsomorphismError:
                match (u_already_qubit, v_already_qubit):
                    case (True, True):
                        sub.remove_edge(e)
                    case (True, False) | (False, True):
                        sub.remove_edge(e)
                        sub.remove_vertex(v)
                    case (False, False):
                        sub.remove_edge(e)
                        sub.remove_vertex(v)
                        sub.remove_vertex(u)
        if iso_found_in_layer:
            # Continue search with new mapping.
            continue
        else:
            # Finish search.
            break

    # Save image to file
    if iso_map:
        save_embedding_image(
            subgraph=sub,
            graph=ag,
            mapping=iso_map[0],
            location=out_dir / (str(ag_path.stem).rstrip('.graphml') + '_' +
                                str(circ_path.stem).rstrip('.qasm.txt') + '.pdf'),
        )

    iso_map_all = gt.subgraph_isomorphism(sub, ag, max_n=0, induced=False)

