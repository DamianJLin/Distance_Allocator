import sys
from pathlib import Path
import re
import itertools
import datetime
import time
import numpy as np
import graph_tool.all as gt
from lib.utils import NoSubgraphIsomorphismError, layer, save_embedding_image
from lib.distance_calculator import DistanceCalculator

ag_dir = Path(__file__).resolve().parent / 'arch_graphs'
circ_dir = Path(__file__).resolve().parent / 'circuits'
out_dir = Path(__file__).resolve().parent / 'output'

ag_paths = sorted(list(
    ag_dir.glob('*.graphml')
))
circ_paths = sorted(list(
    circ_dir.glob('*.qasm.txt')
))

# Set verbosity.
verbose = False
if '-v' in sys.argv or '--verbose' in sys.argv:
    verbose = True

# Clear log path.
for path in out_dir.iterdir():
    path.unlink()

# Keep track of time spent.
time_initial_circuit_total = 0.0
time_find_embeddings_total = 0.0
time_calculate_distance_total = 0.0
time_find_min_distance_total = 0.0

# Create log file and keep it open.
log_path = out_dir / 'experiment.log'
log_path.touch(exist_ok=True)
with open(log_path, 'w') as log_file:
    log_file.write(
        str(datetime.datetime.now()) + '\n\n\n'
    )

    # Loop through all combinations of architectures and circuits.
    for ag_path, circ_path in itertools.product(ag_paths, circ_paths):

        # Print and write to the log file.
        print(
            f'Starting allocation for architecture {str(ag_path.stem).rstrip(".graphml")}, '
            f'circuit {str(circ_path.stem).rstrip(".qasm.txt")}...'
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

        if verbose:
            print('Finding initial subcircuit...')

        start_time = time.time()

        # Find maximal top sublist (largest circuit[:n] that can still be embedded as a subgraph). We do this by looping
        # over all gates in the circuit. We then look first at that gate, then all gates in the same layer (parallel
        # gates) and greedily add one to the top sublist.
        # TODO: Confirm not bugged, in that it cannot skip layers or anything silly.
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
        time_initial_circuit = end_time - start_time

        # Logging, printing, saving image to file.
        log_file.write(f'Compute initial circuit: {time_initial_circuit: .3g} s.\n')
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
            if verbose:
                print(
                    f'Initial subcircuit found with |V| = {sub.num_vertices(ignore_filter=True)}, '
                    f'|E| = {sub.num_edges(ignore_filter=True)} in {time_initial_circuit: .3g} s.')

        else:
            log_file.write(
                f'No initial subcircuit was found. Possible error.\n'
            )
            if verbose:
                print('Failed to find initial subcircuit.')
            continue  # Remaining algorithm is redundant.

        if verbose:
            print('Finding alternative embeddings of initial circuit...')

        start_time = time.time()

        # Find all embeddings.
        embeddings_all = gt.subgraph_isomorphism(sub, ag, max_n=1000, induced=False)

        end_time = time.time()
        time_find_embeddings = end_time - start_time

        log_file.write(f'Compute {len(embeddings_all)} embeddings: {time_find_embeddings: .3g} s.')
        if verbose:
            print(f'Found {len(embeddings_all)} embeddings of initial circuit in {time_find_embeddings: .3g} s.')

        # Initialize distance list with distances np.inf.
        distances = [np.inf for i, emb in enumerate(embeddings_all)]

        # Initialise DistanceCalculator object for efficiency with repeated distance calls.
        dc = DistanceCalculator(ag)

        if verbose:
            print(f'Calculating distance for each embedding and finding minimum...')

        start_time = time.time()

        # Calculate distance.
        for i, emb in enumerate(embeddings_all):

            dist_cuml = 0

            for (u_qasm, v_qasm) in circuit:
                if u_qasm in qubit_by_qasm_id and v_qasm in qubit_by_qasm_id:
                    dist_cuml += dc.distance(
                        qubit_by_qasm_id[u_qasm],
                        qubit_by_qasm_id[v_qasm]
                    )

            distances[i] = dist_cuml

        end_time = time.time()
        time_calculate_distance = end_time - start_time

        start_time = time.time()

        # Find embedding of minimum distance.
        idx_min, best_embedding_dist = min(enumerate(distances), key=lambda x: x[1])
        best_embedding = embeddings_all[idx_min]

        end_time = time.time()
        time_find_min_distance = end_time - start_time

        log_file.write(
            f'Compute embedding with min. dist. of {best_embedding_dist}: '
            f'{time_calculate_distance: .3g} s.'
        )
        if verbose:
            print(
                f'Calculated embedding with min. distance of {best_embedding_dist} in '
                f'{time_calculate_distance: .3g} s.'
            )
            print()

        # Add time spent.
        time_initial_circuit_total += time_initial_circuit
        time_find_embeddings_total += time_find_embeddings
        time_calculate_distance_total += time_calculate_distance
        time_find_min_distance_total += time_find_min_distance


total_time_all = time_initial_circuit_total + time_find_embeddings_total + time_calculate_distance_total +\
                 time_find_min_distance_total

if verbose:

    def format_time(seconds):
        return str(
            datetime.timedelta(seconds=round(seconds))
        )

    print('Time spent...')
    print(
        f'\tConstructing initial subcircuit/embedding: \t'
        f'{format_time(time_initial_circuit_total)}'
    )
    print(
        f'\tFinding other embeddings: \t\t\t'
        f'{format_time(time_find_embeddings_total)}'
    )
    print(
        f'\tComputing distance for all embeddings: \t\t'
        f'{format_time(time_calculate_distance_total)}'
    )
    print(
        f'\tFinding embedding with minimum distance: \t'
        f'{format_time(time_find_min_distance)}'
    )

    print()
    print(
        f'\tTotal: \t\t\t\t\t\t'
        f'{format_time(total_time_all)}'
    )
