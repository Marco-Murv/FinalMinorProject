#!/bin/python3

"""
Sub populations parallelisation of the Genetic Algorithm.

A number of separate GA's will be run on multiple processors, with each processor having their own small, independent
sub-population. The different processors can exchange pairs of clusters with each other to maintain population
diversity.
"""

import os
import ase
import time
import numpy as np

from genetic_algorithm import config_info, debug, fitness, generate_population
from genetic_algorithm import get_configuration, natural_selection_step
from genetic_algorithm import optimise_local, fitness, get_mutants
from genetic_algorithm import store_local_minima, store_results_database
from mating import mating
from ga_distributed import flatten_list
from mpi4py import MPI


def ga_sub_populations():
    # Raise real errors when numpy encounters a division by zero.
    np.seterr(divide='raise')

    # Initialising MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    # File to get default configuration / run information
    config_file = "run_config.yaml"

    # Parse possible terminal input and yaml file.
    c = get_configuration(config_file)
    if rank == 0:
        config_info(c)

    # =========================================================================
    # Initial population and variables
    # =========================================================================
    # TODO: good sub-population size? Atm normal population size divided by number of processes.
    sub_pop_size = np.ceil(c.pop_size / num_procs).astype(int)

    pop = generate_population(sub_pop_size, c.cluster_size, c.cluster_radius)
    energies = optimise_local(pop, c.calc, c.local_optimiser)

    # Keep track of global minima. Initialised with random cluster
    best_min = [pop[0]] # TODO: remove first entries at the end before storing prob?
    local_min = [pop[0]]
    energies_min = np.array(pop[0].get_potential_energy())

    # =========================================================================
    # Main loop
    # =========================================================================
    # Keep track of iterations
    gen = 0
    gen_no_success = 0

    # Used for swapping random clusters between sub-populations
    rng = np.random.default_rng()

    send_req_left = None
    send_req_right = None

    # TODO: using max_gen as in normal GA may lead to errors with message passing when sub-population is stopped
    while gen_no_success < c.max_no_success and gen < c.max_gen:

        # Exchange clusters with neighbouring processors
        # TODO: proper criteria to start exchanges between all processors? Now it's just set to every 10th gen.
        if (gen % 10) == 0:
            if send_req_left is not None:
                send_req_left.wait()
                send_req_right.wait()

            perc_pop_exchanged = 0.2
            num_exchanges = np.ceil(sub_pop_size * perc_pop_exchanged).astype(int) * 2  # TODO: how many to swap?
            cluster_indices = rng.choice(sub_pop_size, size=num_exchanges, replace=False)

            left_neighb = (rank - 1) % num_procs
            left_msg = [pop[i] for i in cluster_indices[:(num_exchanges // 2)]]
            right_neighb = (rank + 1) % num_procs
            right_msg = [pop[i] for i in cluster_indices[(num_exchanges // 2):]]

            send_req_left = comm.isend(left_msg, dest=left_neighb)
            send_req_right = comm.isend(right_msg, dest=right_neighb)

            recv_req_left = comm.irecv(source=left_neighb)
            recv_req_right = comm.irecv(source=right_neighb)
            left_clusters = recv_req_left.wait()
            right_clusters = recv_req_right.wait()

            debug(f"Generation {gen}: processor {rank} finished all exchanges!")

            pop = [cluster for idx, cluster in enumerate(pop) if idx not in cluster_indices]
            pop += left_clusters + right_clusters
            energies = [cluster.get_potential_energy() for cluster in pop]



            # perc_pop_exchanged = 0.2
            # num_exchanges = np.ceil(sub_pop_size * perc_pop_exchanged).astype(int)  # TODO: how many to swap?
            # cluster_indices = rng.choice(sub_pop_size, size=num_exchanges, replace=False)
            #
            # left_neighb = (rank - 1) % num_procs
            # right_neighb = (rank + 1) % num_procs
            # right_msg = [pop[i] for i in cluster_indices]
            #
            # send_req = comm.isend(right_msg, dest=right_neighb)
            # # debug(f"Generation {gen}: processor {rank} sent message to {right_neighb}!")
            # req_left = comm.irecv(source=left_neighb)
            # left_clusters = req_left.wait()
            # # debug(f"    Generation {gen}: processor {rank} received from {left_neighb}!")
            # # send_req.wait()
            #
            # debug(f"    Generation {gen}: processor {rank} finished all exchanges!")
            #
            # pop = [cluster for idx, cluster in enumerate(pop) if idx not in cluster_indices]
            # pop += left_clusters
            # energies = [cluster.get_potential_energy() for cluster in pop]

        # Get fitness values
        pop_fitness = fitness(energies, func=c.fitness_func)
        # Mating - get new population
        children = mating(pop, pop_fitness, c.children_perc, c.mating_method)
        # Mutating - get new mutants
        mutants = get_mutants(pop, c.cluster_radius, c.cluster_size)

        # Local minimisation and add to population
        newborns = children + mutants
        energies += optimise_local(newborns, c.calc, c.local_optimiser)
        pop += newborns

        # Keep track of new local minima
        local_min, energies_min = store_local_minima(newborns, energies, local_min, energies_min, c.dE_thr)

        # Natural selection
        pop, energies = natural_selection_step(pop, energies, sub_pop_size, c.dE_thr, c.fitness_func)

        # Store current best
        if energies[0] < best_min[-1].get_potential_energy():
            debug(f"Process {rank} in generation {gen}: new global minimum at {energies[0]}")
            best_min.append(pop[0])
            gen_no_success = 0  # This is success, so set to zero.
        else:
            gen_no_success += 1

        gen += 1

    # =========================================================================
    # Combine all results and store them
    # =========================================================================
    # Combine results
    best_min = comm.gather(best_min, root=0)
    local_min = comm.gather(local_min, root=0)

    if rank == 0:
        debug("All results have been combined!")
        best_min = flatten_list(best_min)
        local_min = flatten_list(local_min)

        # TODO: remove duplicates and find the GM

        # Connect to database
        # db_file = "genetic_algorithm_results.db"
        # db_file = os.path.join(os.path.dirname(__file__), db_file)
        # db = ase.db.connect(db_file)
        # store_results_database(best_min[-1], local_min, db, c)

    return 0


if __name__ == "__main__":
    ga_sub_populations()

# perc_pop_exchanged = 0.2
# num_exchanges = np.ceil(sub_pop_size * perc_pop_exchanged).astype(int) * 2  # TODO: how many to swap?
# cluster_indices = rng.choice(sub_pop_size, size=num_exchanges, replace=False)
#
# left_neighb = (rank - 1) % num_procs
# left_msg = [pop[i] for i in cluster_indices[:(num_exchanges // 2)]]
# right_neighb = (rank + 1) % num_procs
# right_msg = [pop[i] for i in cluster_indices[(num_exchanges // 2):]]
#
# comm.isend(left_msg, dest=left_neighb)
# comm.isend(right_msg, dest=right_neighb)
# req_left = comm.irecv(source=left_neighb)
# req_right = comm.irecv(source=right_neighb)
# left_clusters = req_left.wait()
# debug(f"Generation {gen}: processor {rank} received from {left_neighb}!")
# right_clusters = req_right.wait()
# debug(f"Generation {gen}: processor {rank} received from {right_neighb}!")
#
# debug(f"Generation {gen}: processor {rank} finished all exchanges!")
#
# pop = [cluster for idx, cluster in enumerate(pop) if idx not in cluster_indices]
# pop += left_clusters + right_clusters
# energies = [cluster.get_potential_energy() for cluster in pop]