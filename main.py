import sys
from pathlib import Path
import itertools
import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graph_tool.all as gt

import lib.utils
from lib.initial_mapping import greedy_layer_initial_mapping
from lib.distance_calculator import DistanceCalculator
from lib.circuit_transformation import circuit_transformation
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

        # Initialise gate array.
        circuit = lib.utils.qasm_as_array(circ_path)

        # Remove redundant graphml properties.
        del ag.vertex_properties["_graphml_vertex_id"]
        del ag.edge_properties["_graphml_edge_id"]

        if verbose:
            print('Finding initial subcircuit... ', end='', flush=True)

        # Finding details of initial circuit and initial mapping. Use a greedy algorithm that adds gates in the current
        # layer.
        start_time = time.time()

        # Find the initial mapping
        circuit_allocated, circuit_unallocated, embedding_init, sub, logical_to_allocated =\
            greedy_layer_initial_mapping(circuit, ag)

        end_time = time.time()
        time_initial_circuit = end_time - start_time

        # Logging, printing, saving image to file.
        init_emb_img_path = out_dir / (
                architecture_name + '__' + circuit_name + '__' + 'initial_embedding' + '.png'
        )
        lib.utils.save_embedding_image(
            subgraph=sub,
            graph=ag,
            architecture=architecture_name,
            mapping=embedding_init,
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
        if verbose:
            print('Finding alternative embeddings of initial circuit... ', end='', flush=True)

        # Find all (up to n_embeddings) embeddings of graph induced by circuit_allocated in the architecture graph.
        start_time = time.time()

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

        if verbose:
            print(f'Calculating distance for each embedding and finding min... ', end='', flush=True)

        # Evaluate the distance function on each embedding.
        start_time = time.time()

        # Initialize distance array with distances np.inf.
        distances = np.full_like(embeddings_all, np.inf)
        # Initialise DistanceCalculator object for efficiency with repeated distance calls.
        distance_calculator = DistanceCalculator(ag)

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
                    dist_cuml += distance_calculator.distance(
                        u_embedded,
                        v_embedded
                    )
                    num_gates_measured += 1
            distances[i] = dist_cuml

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

        # Calculate distance histogram and save.
        fig, ax = plt.subplots()
        ax.hist(distances, n_embeddings // 25 + 1)
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
        lib.utils.save_embedding_image(
            subgraph=sub,
            graph=ag,
            architecture=architecture_name,
            mapping=best_embedding,
            location=final_emb_img_path,
            log_to_alloc=logical_to_allocated
        )

        # Calculate the quality of the circuit.
        for emb in embeddings_all:
            circuit_transformed, quality = circuit_transformation(emb, circuit, ag, distance_calculator)

        # Add time spent.
        time_initial_circuit_total += time_initial_circuit
        time_find_embeddings_total += time_find_embeddings
        time_calculate_min_dist_total += time_calculate_min_dist

        # Write a column to log file.
        log_frame = pd.DataFrame(log_data, index=[comb_index])
        log_frame.to_csv(log_file, header=make_header)
        make_header = False
        if verbose:
            print(flush=True)


# Create a time summary.
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
