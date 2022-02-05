import sys
from pathlib import Path
import re
import itertools
import datetime
import time
import numpy as np
import graph_tool.all as gt
from lib.utils import NoSubgraphIsomorphism, layer, save_embedding_image
from lib.distance_calculator import DistanceCalculator
from termcolor import colored

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
            f'Starting allocation for architecture {colored(str(ag_path.stem).rstrip(".graphml"), "blue")}, '
            f'circuit {colored(str(circ_path.stem).rstrip(".qasm.txt"), "blue")}...'
        )
        log_file.write(
            f'Architecture: {str(ag_path.stem).rstrip(".graphml")}, '
            f'Circuit: {str(circ_path.stem).rstrip(".qasm.txt")}\n'
        )

        # Initialise architecture graph.
        ag = gt.load_graph(str(ag_path))

        # Initialise gate list.
        with open(circ_path, 'r') as circ_file:
            circ_str = circ_file.read()
        pattern = r'\[?\[(\d*), (\d*)\]\]?'
        circuit = np.asarray(
            [
                list(map(int, tup))
                for tup in re.findall(pattern, circ_str)
            ]
        )

        # Remove redundant graphml properties.
        del ag.vertex_properties["_graphml_vertex_id"]
        del ag.edge_properties["_graphml_edge_id"]

        # Initialise subgraph induced from adding circuits. Empty initially.
        sub = gt.Graph()
        sub.set_directed(False)
        sub.set_fast_edge_removal(fast=True)
        embedding = None

        if verbose:
            print('Finding initial subcircuit...')

        start_time = time.time()

        # Set of logical qubits (represented by ints).
        logical_qubits = set()
        for u, v in circuit:
            logical_qubits.add(u)
            logical_qubits.add(v)

        # Dictionary for the physical qubit (vertex in sub) a logical qubit (int) is allocated to.
        physical_qubit = {}

        allocated = np.zeros(len(circuit), dtype=bool)

        # Below algorithm for finding initial subcircuit:
        # while layer_forbids_embedding is False:
        #   search all gates in layer:
        #       add gate to subgraph
        #       if subgraph embeds:
        #           break and go to next layer
        #       if gate does not embed:
        #           fix up the graph
        #           continue to next gate in layer
        #   set layer_forbids_embedding True

        # If a layer is fully searched with no gates that can be allocated, then we have finished.
        layer_forbids_embedding = False
        while not layer_forbids_embedding:

            # For every gate that is parallel to (in the same layer as) the first unallocated gate, check subgraph
            # isomorphism to see if it is a candidate for inclusion.
            embedding_found_in_layer = False
            for i, (u_logical, v_logical) in layer(circuit, mask=allocated, logical_qubits=logical_qubits):
                u_allocated = bool(u_logical in physical_qubit)
                v_allocated = bool(v_logical in physical_qubit)
                # u and v embedded
                if u_allocated and v_allocated:
                    e = sub.add_edge(
                        physical_qubit[u_logical],
                        physical_qubit[v_logical]
                    )
                # u embedded but v not
                elif u_allocated and not v_allocated:
                    physical_qubit[v_logical] = sub.add_vertex()
                    e = sub.add_edge(
                        physical_qubit[u_logical],
                        physical_qubit[v_logical]
                    )
                # u not embedded but v
                elif not u_allocated and v_allocated:
                    physical_qubit[u_logical] = sub.add_vertex()
                    e = sub.add_edge(
                        physical_qubit[u_logical],
                        physical_qubit[v_logical]
                    )
                # u nor v embedded
                elif not u_allocated and not v_allocated:
                    physical_qubit[u_logical] = sub.add_vertex()
                    physical_qubit[v_logical] = sub.add_vertex()
                    e = sub.add_edge(
                        physical_qubit[u_logical],
                        physical_qubit[v_logical]
                    )
                gt.remove_parallel_edges(sub)

                try:
                    embedding_prev = embedding
                    embedding = gt.subgraph_isomorphism(sub, ag, max_n=1, induced=False)

                    if embedding:
                        allocated[i] = True
                        embedding_found_in_layer = True
                        break
                    else:
                        raise NoSubgraphIsomorphism

                # If no isomorphism was found, we need to clean up the added vertices and edges.
                except NoSubgraphIsomorphism:
                    if u_allocated and v_allocated:
                        sub.remove_edge(e)
                    elif u_allocated and not v_allocated:
                        sub.remove_edge(e)
                        sub.remove_vertex(physical_qubit[v_logical])
                        physical_qubit.pop(v_logical)
                    elif not u_allocated and v_allocated:
                        sub.remove_edge(e)
                        sub.remove_vertex(physical_qubit[u_logical])
                        physical_qubit.pop(u_logical)
                    elif not u_allocated and not v_allocated:
                        sub.remove_edge(e)
                        sub.remove_vertex(physical_qubit[v_logical])
                        sub.remove_vertex(physical_qubit[u_logical])
                        physical_qubit.pop(v_logical)
                        physical_qubit.pop(u_logical)
                    embedding = embedding_prev

            if not embedding_found_in_layer:
                layer_forbids_embedding = True

        end_time = time.time()
        time_initial_circuit = end_time - start_time

        # Logging, printing, saving image to file.
        log_file.write(f'Compute initial circuit: {time_initial_circuit:.3g} s.\n')
        if embedding:
            img_path = out_dir / (
                str(ag_path.stem).rstrip('.graphml') + '_' + str(circ_path.stem).rstrip('.qasm.txt') + '.png'
            )
            save_embedding_image(
                subgraph=sub,
                graph=ag,
                mapping=embedding[0],
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
                    f'|E| = {sub.num_edges(ignore_filter=True)} in {time_initial_circuit:.3g} s.'
                )

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
        embeddings_all = gt.subgraph_isomorphism(sub, ag, max_n=10, induced=False)

        end_time = time.time()
        time_find_embeddings = end_time - start_time

        log_file.write(f'Compute {len(embeddings_all)} embeddings: {time_find_embeddings:.3g} s.')
        if verbose:
            print(f'Found {len(embeddings_all)} embeddings of initial circuit in {time_find_embeddings:.3g} s.')

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
                if u_qasm in physical_qubit and v_qasm in physical_qubit:
                    dist_cuml += dc.distance(
                        physical_qubit[u_qasm],
                        physical_qubit[v_qasm]
                    )

            distances[i] = dist_cuml

        end_time = time.time()
        time_calculate_distance = end_time - start_time

        start_time = time.time()

        # Find embedding of minimum distance.
        idx_min, best_embedding_dist = min(enumerate(distances), key=lambda pair: pair[1])
        best_embedding = embeddings_all[idx_min]

        end_time = time.time()
        time_find_min_distance = end_time - start_time

        log_file.write(
            f'Compute embedding with min. dist. of {best_embedding_dist}: '
            f'{time_calculate_distance:.3g} s.\n\n'
        )
        if verbose:
            print(
                f'Calculated embedding with min. distance of {best_embedding_dist} in '
                f'{time_calculate_distance:.3g} s.'
            )
            print()

        # Add time spent.
        time_initial_circuit_total += time_initial_circuit
        time_find_embeddings_total += time_find_embeddings
        time_calculate_distance_total += time_calculate_distance
        time_find_min_distance_total += time_find_min_distance

total_time_all = time_initial_circuit_total + time_find_embeddings_total + time_calculate_distance_total + \
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
