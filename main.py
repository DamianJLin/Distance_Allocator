import sys
from pathlib import Path
import re
import itertools
import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graph_tool.all as gt
from lib.utils import NoSubgraphIsomorphism, get_layer, save_embedding_image
from lib.distance_calculator import DistanceCalculator
from termcolor import colored

"""
Terminology:
============

    Logical Qubit:
    A qubit in a quantum circuit. They are represented by integers.
    
    Allocated Qubit:
    A vertex of the subgraph sub. However there are many possible embeddings of this subgraph into the architecture
    graph, chosen from only when this qubit becomes and embedded qubit. (This is neither a logical qubit and a physical
    qubit, but sits in between these definitions.)
    
    Embedded Qubit:
    An embedded qubit is a vertex of the graph ag (the architecture graph). In the literature this would be called a
    physical qubit.
    
"""

ag_dir = Path(__file__).resolve().parent / 'arch_graphs'
circ_dir = Path(__file__).resolve().parent / 'circuits'
out_dir = Path(__file__).resolve().parent / 'output'

ag_paths = sorted(list(
    ag_dir.glob('*.graphml')
))
circ_paths = sorted(list(
    circ_dir.glob('*.qasm.txt')
))
# This branch produced embeddings for a single circuit on a single graph.
assert len(circ_paths) == 1
assert len(ag_paths) == 1

# Script parameters.
# Set verbosity.
verbose = False
if '-v' in sys.argv or '--verbose' in sys.argv:
    verbose = True

# Number of embeddings.
n_embeddings = 1000
if '-e' in sys.argv:
    if len(sys.argv) < sys.argv.index('-e') + 2:
        raise ValueError('-e flag needs an argument.')
    n_embeddings = None
    try:
        n_embeddings = int(sys.argv[sys.argv.index('-e') + 1])
    except ValueError:
        raise ValueError('-e flag argument must be integer.')
assert isinstance(n_embeddings, int)

# Clear log path.
for path in out_dir.iterdir():
    path.unlink()

# Keep track of time spent.
time_initial_circuit_total = 0.0
time_find_embeddings_total = 0.0
time_calculate_min_dist_total = 0.0
time_find_min_distance_total = 0.0

# Create log file and keep it open.
log_path = out_dir / 'log.csv'
make_header = True
with open(log_path, 'a') as log_file:

    # Loop through all combinations of architectures and circuits.
    for comb_index, (ag_path, circ_path) in enumerate(itertools.product(ag_paths, circ_paths)):

        architecture_name = str(ag_path.name).removesuffix(".graphml")
        circuit_name = str(circ_path.name).removesuffix(".qasm.txt")

        # Write to log dict.
        log_data = {
            'architecture': architecture_name,
            'circuit': str(circ_path.name).removesuffix(".qasm.txt"),
        }
        print(
            f'Allocating architecture: {colored(architecture_name, "blue")}, '
            f'circuit {colored(circuit_name, "blue")}...',
            flush=True
        )

        # Initialise architecture graph.
        ag = gt.load_graph(str(ag_path))

        # Initialise gate list.
        with open(circ_path, 'r') as circ_file:
            circ_str = circ_file.read()
        pattern = r'\[?\[(\d*), (\d*)\]\]?'
        circuit = np.asarray(
            [
                np.asarray(list(map(int, tup)))
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
        embeddings_init = None

        if verbose:
            print('Finding initial subcircuit... ', end='', flush=True)

        start_time = time.time()

        # Set of logical qubits (represented by ints).
        logical_qubits = set()
        for u, v in circuit:
            logical_qubits.add(u)
            logical_qubits.add(v)

        # Dictionary for the physical qubit (vertex in sub) a logical qubit (int) is allocated to.
        logical_to_allocated = {}

        # Keep track of which gates are allocated
        gate_allocation = np.zeros(len(circuit), dtype=bool)
        circuit_allocated = []  # List for fast append.

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

            embedding_found_in_layer = False
            for i, (u_log, v_log) in get_layer(circuit, mask=gate_allocation, logical_qubits=logical_qubits):
                u_is_allocated = bool(u_log in logical_to_allocated)
                v_is_allocated = bool(v_log in logical_to_allocated)
                # u and v embedded
                if u_is_allocated and v_is_allocated:
                    e = sub.add_edge(
                        logical_to_allocated[u_log],
                        logical_to_allocated[v_log]
                    )
                # u embedded but v not
                elif u_is_allocated and not v_is_allocated:
                    logical_to_allocated[v_log] = sub.add_vertex()
                    e = sub.add_edge(
                        logical_to_allocated[u_log],
                        logical_to_allocated[v_log]
                    )
                # u not embedded but v
                elif not u_is_allocated and v_is_allocated:
                    logical_to_allocated[u_log] = sub.add_vertex()
                    e = sub.add_edge(
                        logical_to_allocated[u_log],
                        logical_to_allocated[v_log]
                    )
                # u nor v embedded
                elif not u_is_allocated and not v_is_allocated:
                    logical_to_allocated[u_log] = sub.add_vertex()
                    logical_to_allocated[v_log] = sub.add_vertex()
                    e = sub.add_edge(
                        logical_to_allocated[u_log],
                        logical_to_allocated[v_log]
                    )
                gt.remove_parallel_edges(sub)

                try:
                    prev = embeddings_init
                    embeddings_init = gt.subgraph_isomorphism(sub, ag, max_n=1, induced=False)

                    if embeddings_init:
                        gate_allocation[i] = True
                        circuit_allocated.append(np.asarray([u_log, v_log]))
                        embedding_found_in_layer = True
                        break
                    else:
                        raise NoSubgraphIsomorphism

                # If no isomorphism was found, we need to clean up the added vertices and edges and the physical_qubit
                # dictionary.
                except NoSubgraphIsomorphism:
                    if u_is_allocated and v_is_allocated:
                        sub.remove_edge(e)
                    elif u_is_allocated and not v_is_allocated:
                        sub.remove_edge(e)
                        sub.remove_vertex(logical_to_allocated[v_log])
                        logical_to_allocated.pop(v_log)
                    elif not u_is_allocated and v_is_allocated:
                        sub.remove_edge(e)
                        sub.remove_vertex(logical_to_allocated[u_log])
                        logical_to_allocated.pop(u_log)
                    elif not u_is_allocated and not v_is_allocated:
                        sub.remove_edge(e)
                        sub.remove_vertex(logical_to_allocated[v_log])
                        sub.remove_vertex(logical_to_allocated[u_log])
                        logical_to_allocated.pop(v_log)
                        logical_to_allocated.pop(u_log)
                    embeddings_init = prev

            if not embedding_found_in_layer:
                layer_forbids_embedding = True

        circuit_allocated = np.copy(np.asarray(circuit_allocated))
        circuit_unallocated = np.copy(circuit[gate_allocation == 0, :])

        end_time = time.time()
        time_initial_circuit = end_time - start_time

        # Logging, printing, saving image to file.
        if embeddings_init:
            init_emb_img_path = out_dir / (
                    architecture_name + '__' + circuit_name + '__' + 'initial_embedding' + '.png'
            )
            save_embedding_image(
                subgraph=sub,
                graph=ag,
                architecture=architecture_name,
                mapping=embeddings_init[0],
                location=init_emb_img_path,
                log_to_alloc=logical_to_allocated
            )
            log_data['num_gates'] = len(circuit)
            log_data['num_gates_allocated'] = len(circuit_allocated)
            log_data['num_gates_unallocated'] = len(circuit_unallocated)
            log_data['sub_vertices'] = sub.num_vertices(ignore_filter=True)
            log_data['sub_edges'] = sub.num_edges(ignore_filter=True)
            log_data['init_subcircuit_time'] = f'{time_initial_circuit:.3g} s'
            if verbose:
                print('done.', flush=True)
                print(
                    f'\t# Vertices: {sub.num_vertices(ignore_filter=True)},\n'
                    f'\t# Edges: {sub.num_edges(ignore_filter=True)}\n'
                    f'\tTime: {time_initial_circuit:.3g} s.',
                    flush=True
                )

        else:
            if verbose:
                print('failed.', flush=True)
            sys.exit(0)

        if verbose:
            print('Finding alternative embeddings of initial circuit... ', end='', flush=True)

        start_time = time.time()

        # Find all embeddings.
        embeddings_all = np.asarray(
            gt.subgraph_isomorphism(sub, ag, max_n=n_embeddings, induced=False)
        )

        end_time = time.time()
        time_find_embeddings = end_time - start_time

        log_data['num_embeddings'] = len(embeddings_all)
        log_data['embeddings_time'] = f'{time_find_embeddings:.3g} s'
        if verbose:
            print('done.', flush=True)
            print(
                f'\t# Embeddings: {len(embeddings_all)}\n'
                f'\tTime: {time_find_embeddings:.3g} s.',
                flush=True
            )

        # Initialize distance array with distances np.inf.
        distances = np.full_like(embeddings_all, np.inf)

        # Initialise DistanceCalculator object for efficiency with repeated distance calls.
        dc = DistanceCalculator(ag)

        if verbose:
            print(f'Calculating distance for each embedding and finding min... ', end='', flush=True)

        start_time = time.time()

        # Calculate distance.
        for i, emb in enumerate(embeddings_all):

            # Print percentage complete.
            if verbose:
                pc_str = str(int(i * 100 / len(embeddings_all))) + '%'
                print(pc_str + '\b' * len(pc_str), end='', flush=True)

            dist_cuml = 0
            num_gates_measured = 0

            for (u_log, v_log) in circuit_unallocated:
                if u_log in logical_to_allocated and v_log in logical_to_allocated:
                    # Find the allocation of u and v
                    u_allocated = logical_to_allocated[u_log]
                    v_allocated = logical_to_allocated[v_log]
                    # Find the embedding of u and v
                    u_embedded = ag.vertex(emb[u_allocated])
                    v_embedded = ag.vertex(emb[v_allocated])
                    # Find the distance between the u and v in ag when embedded via emb.
                    dist_cuml += dc.distance(
                        u_embedded,
                        v_embedded
                    )
                    num_gates_measured += 1
            distances[i] = dist_cuml

        # Create embedding output path and open it.
        emb_path = out_dir / '_inimap_list_rochester_custom_bigQ.txt'
        with open(emb_path, 'w') as emb_file:

            # Save all embeddings.
            emb_out = []
            for i, emb in enumerate(embeddings_all):
                emb_out.append(
                    [
                        i + 1,
                        [
                            [
                                v_log,
                                emb[logical_to_allocated[v_log]]
                            ] for v_log in logical_to_allocated
                        ]
                    ]
                )
            emb_file.write(str(emb_out))

        # Save distances to .npy file.
        dist_path = out_dir / 'distances.npy'
        np.save(str(dist_path), distances)

        # Find embedding of minimum distance.
        idx_min, best_embedding_dist = min(enumerate(distances), key=lambda pair: pair[1])
        best_embedding = embeddings_all[idx_min]
        if num_gates_measured == 0:
            best_embedding_efficiency = 1
        else:
            best_embedding_efficiency = best_embedding_dist / num_gates_measured

        end_time = time.time()
        time_calculate_min_dist = end_time - start_time

        log_data['best_dist'] = best_embedding_dist
        log_data['best_embedding_efficiency'] = f'{best_embedding_efficiency:.3g}'
        log_data['best_dist_time'] = f'{time_calculate_min_dist:.3g} s'
        if verbose:
            print('done.', flush=True)
            print(
                f'\tMin. dist.: {best_embedding_dist}\n'
                f'\tNo. gates.: {len(circuit)}\n'
                f'\tMin. dist. per gate: {best_embedding_dist / num_gates_measured:.3g}\n'
                f'\tTime: {time_calculate_min_dist:.3g} s.',
                flush=True
            )

        # Calculate dist histogram and save.
        fig, ax = plt.subplots()
        ax.hist(distances, 50)
        hist_path = out_dir / (
                architecture_name + '__' + circuit_name + '__' + 'dist_hist' + '.png'
        )
        fig.savefig(
            fname=hist_path,
            dpi='figure',
            format='png'
        )
        plt.close(fig)

        # Draw final embedding.
        final_emb_img_path = out_dir / (
                architecture_name + '__' + circuit_name + '__' + 'best_embedding' + '.png'
        )
        save_embedding_image(
            subgraph=sub,
            graph=ag,
            architecture=architecture_name,
            mapping=best_embedding,
            location=final_emb_img_path,
            log_to_alloc=logical_to_allocated
        )

        # Add time spent.
        time_initial_circuit_total += time_initial_circuit
        time_find_embeddings_total += time_find_embeddings
        time_calculate_min_dist_total += time_calculate_min_dist

        log_frame = pd.DataFrame(log_data, index=[comb_index])
        log_frame.to_csv(log_file, header=make_header)
        make_header = False
        if verbose:
            print(flush=True)

total_time_all = time_initial_circuit_total + time_find_embeddings_total + time_calculate_min_dist_total

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
        f'{format_time(time_calculate_min_dist_total)}'
    )

    print()
    print(
        f'\tTotal: \t\t\t\t\t\t'
        f'{format_time(total_time_all)}'
    )
