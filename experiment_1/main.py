from pathlib import Path
import re
import itertools
from datetime import datetime
import time
import graph_tool.all as gt
from lib.utils import NoSubgraphIsomorphismError, layer, save_embedding_image

ag_dir = Path(__file__).resolve().parent / 'arch_graphs'
circ_dir = Path(__file__).resolve().parent / 'circuits'
out_dir = Path(__file__).resolve().parent / 'output'

ag_paths = sorted(list(
    ag_dir.glob('*.graphml')
))
circ_paths = sorted(list(
    circ_dir.glob('*.qasm.txt')
))

# Clear log path.
for path in out_dir.iterdir():
    path.unlink()

# Create log file and keep it open.
log_path = out_dir / 'experiment.log'
log_path.touch(exist_ok=True)
with open(log_path, 'w') as log_file:
    log_file.write(
        str(datetime.now()) + '\n\n\n'
    )

    # Loop through all combinations of architectures and circuits.
    for ag_path, circ_path in itertools.product(ag_paths, circ_paths):

        # Print and write to the log file.
        print(
            f'For architecture {str(ag_path.stem).rstrip(".graphml")}, '
            f'allocating circuit {str(circ_path.stem).rstrip(".qasm.txt")}...'
        )
        log_file.write(
            f'Architecture: {str(ag_path.stem).rstrip(".graphml")}, '
            f'Circuit: {str(circ_path.stem).rstrip(".qasm.txt")}\n'
        )

        # Initialise architecture graph.
        ag = gt.load_graph(str(ag_path))

        with open(circ_path, 'r') as circ_file:
            circ_str = circ_file.read()
        pattern = r'\[?\[(\d*), (\d*)\]\]?'
        circuit = [
            list(map(int, tup))
            for tup in re.findall(pattern, circ_str)
        ]
        # Remove redundant graphml properties.
        del ag.vertex_properties["_graphml_vertex_id"]
        del ag.edge_properties["_graphml_edge_id"]

        # Initialise subgraph induced from adding circuits. Empty initially.
        sub = gt.Graph()
        sub.set_directed(False)
        sub.set_fast_edge_removal(fast=True)

        iso_map = None

        # Keeping track of logical qubits by their number in the qasm files. This is because graph_tool node ids are
        # always ordered starting from 0, so we cannot directly use them to keep track of qubits.
        qubit_by_qasm_id = {}

        # Start timing.
        start_time = time.time()

        # Find maximal top sublist (largest circuit[:n] that can still be embedded as a subgraph). We do this by looping
        # over all gates in the circuit. We then look first at that gate, then all gates in the same layer (parallel
        # gates) and greedily add one to the top sublist.
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
                    iso_map_prev = iso_map  # Hold on to old map in case the new one fails, we can revert in except.
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
                    iso_map = iso_map_prev
            if iso_found_in_layer:
                # Continue search in next layer, with new mapping.
                continue
            else:
                # Finish search.
                break

        # Stop timing.
        end_time = time.time()
        delta_time = end_time - start_time

        # Logging, saving image to file.
        log_file.write(f'Time: {delta_time: .3g} s.\n')
        if iso_map:
            img_path = out_dir / (str(ag_path.stem).rstrip('.graphml') + '_'
                                  + str(circ_path.stem).rstrip('.qasm.txt') + '.png')
            save_embedding_image(
                subgraph=sub,
                graph=ag,
                mapping=iso_map[0],
                location=img_path,
            )
            log_file.write(
                f'|V| = {sub.num_vertices(ignore_filter=True)},\n'
                f'|E| = {sub.num_edges(ignore_filter=True)}.\n'
            )
            log_file.write(
                f'Graphic at {img_path.parent.name}/{img_path.name}.\n'
            )
        else:
            log_file.write(
                f'No map was found. Possible error.\n'
            )
        log_file.write('\n')

        # iso_map_all = gt.subgraph_isomorphism(sub, ag, max_n=0, induced=False)

