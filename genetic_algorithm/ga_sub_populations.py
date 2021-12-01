#!/bin/python3

"""
Sub populations parallelisation of the Genetic Algorithm.

A number of separate GA's will be run on multiple processors, with each processor having their own small, independent
sub-population. The different processors can exchange pairs of clusters with each other to maintain population
diversity.
"""

import os
import sys
import inspect
import time

import ase
import numpy as np

from ase.io import Trajectory
from genetic_algorithm import config_info, debug, generate_population
from genetic_algorithm import get_configuration, natural_selection_step
from genetic_algorithm import optimise_local, fitness, get_mutants
from genetic_algorithm import store_results_database
from mating import mating
from ga_distributed import flatten_list
from mpi4py import MPI

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import process_data


def one_directional_exchange(pop, sub_pop_size, gen, comm, rank, send_req_right, perc_pop_exchanged=0.2):
    """
    Performs an exchange of clusters with a neighbouring processor (the neighbour has rank +1).

    @param pop: the sub-population of a processor
    @param sub_pop_size: the size of the sub-population
    @param gen: generation where the exchange occurs (mainly for printing progress of the communications)
    @param comm: the MPI communicator
    @param rank: rank of this processor
    @param send_req_right: send communication request handle of the previous cluster exchange with right neighbour
    @param perc_pop_exchanged: percentage of the sub-population that should be exchanged with each neighbour
    @return: population with the newly received clusters and corresponding energies
    """

    # Number of clusters to be exchanged with a single neighbour.
    num_exchanges = np.ceil(sub_pop_size * perc_pop_exchanged).astype(int)  # TODO: how many to swap?
    rng = np.random.default_rng()
    cluster_indices = rng.choice(sub_pop_size, size=num_exchanges, replace=False)

    num_procs = comm.Get_size()
    left_neighb = (rank - 1) % num_procs
    right_neighb = (rank + 1) % num_procs
    right_msg = [pop[i] for i in cluster_indices]

    # Wait to make sure that the neighbouring processor correctly received the previously sent clusters before
    # before exchanging a new group of clusters!
    if send_req_right is not None:
        send_req_right.wait()
    send_req_right = comm.isend(right_msg, dest=right_neighb, tag=1)
    req_left = comm.irecv(source=left_neighb, tag=1)
    left_clusters = req_left.wait()

    debug(f"    Generation {gen}: processor {rank} finished all exchanges!")

    # Filter out the exchanged clusters and instead add the newly received clusters from neighbouring populations.
    pop = [cluster for idx, cluster in enumerate(pop) if idx not in cluster_indices]
    pop += left_clusters
    energies = [cluster.get_potential_energy() for cluster in pop]

    return pop, energies, send_req_right


def bi_directional_exchange(pop, sub_pop_size, gen, comm, rank, send_req_left, send_req_right, perc_pop_exchanged=0.2):
    """
    Performs an exchange of clusters with both neighbouring processors (the neighbours have rank +/- 1).

    @param pop: the sub-population of a processor
    @param sub_pop_size: the size of the sub-population
    @param gen: generation where the exchange occurs (mainly for printing progress of the communications)
    @param comm: the MPI communicator
    @param rank: rank of this processor
    @param send_req_left: send communication request handle of the previous cluster exchange with left neighbour
    @param send_req_right: send communication request handle of the previous cluster exchange with right neighbour
    @param perc_pop_exchanged: percentage of the sub-population that should be exchanged with each neighbour
    @return: population with the newly received clusters and corresponding energies
    """

    print(f"\tGen {gen} processor {rank} starting comms!")

    # Number of clusters to be exchanged with a single neighbour.
    num_exchanges = np.ceil(sub_pop_size * perc_pop_exchanged).astype(int) * 2  # TODO: how many to swap?
    rng = np.random.default_rng()
    cluster_indices = rng.choice(sub_pop_size, size=num_exchanges, replace=False)

    num_procs = comm.Get_size()
    left_neighb = (rank - 1) % num_procs
    left_msg = [pop[i] for i in cluster_indices[:(num_exchanges // 2)]] # TODO: encountered a rare IOOB error once?
    right_neighb = (rank + 1) % num_procs
    right_msg = [pop[i] for i in cluster_indices[(num_exchanges // 2):]]

    # Wait to make sure that the neighbouring processors correctly received the previously sent clusters before
    # before exchanging a new group of clusters!
    if send_req_left is not None:
        send_req_left.wait()
        send_req_right.wait()

    # Tag 1 is used for cluster exchange messages.
    send_req_left = comm.isend(left_msg, dest=left_neighb, tag=1)
    send_req_right = comm.isend(right_msg, dest=right_neighb, tag=1)

    recv_req_left = comm.irecv(source=left_neighb, tag=1)
    recv_req_right = comm.irecv(source=right_neighb, tag=1)

    # TODO: use test here with the abort msg request as well, or maybe use barrier with the abort check?
    left_clusters = recv_req_left.wait()
    right_clusters = recv_req_right.wait()

    debug(f"Generation {gen}: processor {rank} finished all exchanges!")

    # Filter out the exchanged clusters and instead add the newly received clusters from neighbouring populations.
    pop = [cluster for idx, cluster in enumerate(pop) if idx not in cluster_indices]
    pop += left_clusters + right_clusters
    energies = [cluster.get_potential_energy() for cluster in pop]

    return pop, energies, send_req_left, send_req_right


def ga_sub_populations():
    """
    Performs a parallel execution of a Genetic Algorithm using the sub-populations method.

    Each processor has its own independent population of a smaller size and communicates with neighbouring processors
    to exchange clusters to prevent population stagnation.

    @return:
    """

    # =========================================================================
    # Algorithm setup
    # =========================================================================
    # Raise real errors when numpy encounters a division by zero.
    np.seterr(divide='raise')
    # np.random.seed(2543632)

    # Initialising MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    # File to get default configuration / run information
    config_file = "./config/ga_sub_populations_config.yaml"

    # Parse possible terminal input and yaml file.
    # TODO: Bcast config + random state?
    c = get_configuration(config_file)  # TODO: prob make own get_config, leads to some complications using the GA one
    if rank == 0:
        config_info(c)

    # Ensure all processors have exactly the same starting time for properly synchronising the maximum durations
    start_time = MPI.Wtime()
    start_time = comm.bcast(start_time, root=0)

    # =========================================================================
    # Initial population and variables
    # =========================================================================
    # TODO: good sub-population size? Atm normal population size divided by number of processes.
    sub_pop_size = np.ceil(c.pop_size / num_procs).astype(int)

    pop = generate_population(sub_pop_size, c.cluster_size, c.cluster_radius)
    energies = optimise_local(pop, c.calc, c.local_optimiser)

    # Keep track of global minima. Initialised with random cluster
    best_min = pop[0]
    local_min = [pop[0]]

    # Number of generations until cluster exchange happens
    exchange_gen = 10

    # Number of cluster exchanges without new lowest minima until algorithm abortion
    # Processor with rank 0 keeps track of this
    if rank == 0:
        max_num_exchanges = 5
        no_success_processes = np.zeros((num_procs, 1), dtype=int)

    # TODO: implement system for aborting when this number is reached
    # Send gen_no_success to rank 0, stores it in array with len(num_procs), if limit reached then abort algo
    # special abort tag, non-blocking recv for each process, stop looping if a msg been received

    # =========================================================================
    # Main loop
    # =========================================================================
    # Keep track of iterations
    gen = 0
    gen_no_success = 0
    gen_no_success_buffer = 0

    # Used for swapping random clusters between sub-populations
    send_req_left = None
    send_req_right = None

    # Non-blocking recv message waiting for potential abortion message (tag = 0) from processor 0
    abort = np.ones(1, dtype=int)
    if rank != 0:
        send_req_abort = None  # Rename this prob
        comm.Irecv(abort, source=0, tag=0)

    # TODO: using max_gen as in normal GA may lead to errors with message passing when sub-population is stopped
    while gen < c.max_gen:
        # Processor 0 check for stopping condition met or not and send abort msg if needed

        # If algo abort condition is met, set stopping condition to true
        if (MPI.Wtime() - start_time) > c.time_lim:
            print(f"\t\t\tGen {gen} processor {rank} reached time lim!")
            abort[0] = 0

        if abort[0] == 0:
            print(f"Processor {rank} aborted!")
            break

        # Exchange clusters with neighbouring processors
        if (gen % exchange_gen) == 0:
            # Synchronise all processors before communication to make sure they all abort simultaneously before comms
            req_barr = comm.Ibarrier()
            while not req_barr.get_status():
                if (MPI.Wtime() - start_time) > c.time_lim:
                    abort[0] = 0
                    break
                time.sleep(0.001)
            if abort[0] == 0:
                break

            # Send gen_no_success values to processor 0
            if rank != 0:
                if send_req_abort is not None:
                    send_req_abort.wait()
                # print(f"\t\t Gen {gen} processor {rank} gen_no_success: {gen_no_success}")
                gen_no_success_buffer = np.array([gen_no_success], dtype=int)
                send_req_abort = comm.Isend(gen_no_success_buffer, dest=0, tag=3)

            else:  # Processor 0
                # print(f"\t\t Gen {gen} processor {rank} gen_no_success: {gen_no_success}")
                no_success_processes[0] = np.array(gen_no_success, dtype=int)
                for i in range(1, num_procs):
                    comm.Recv(no_success_processes[i], source=i, tag=3)
                # print(f"\t\t Gen {gen} processor {rank} array: {no_success_processes}")

            # Exchange clusters with both neighbouring processors
            pop, energies, send_req_left, send_req_right = bi_directional_exchange(pop, sub_pop_size, gen, comm, rank,
                                                                                   send_req_left, send_req_right)
            # Exchange clusters with only a single neighbour
            # pop, energies, send_req_right = one_directional_exchange(pop, sub_pop_size, gen, comm, rank, send_req_right)

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
        local_min += newborns

        # Natural selection
        pop, energies = natural_selection_step(pop, energies, sub_pop_size, c.dE_thr)

        # Store current best
        if energies[0] < best_min.get_potential_energy():
            debug(f"Process {rank} in generation {gen}: new global minimum at {energies[0]}")
            best_min = pop[0]
            gen_no_success = 0  # This is success, so set to zero.
        else:
            gen_no_success += 1

        gen += 1

    # Delete first values from local_min as these were initialised with a random cluster
    local_min.pop(0)

    # =========================================================================
    # Combine all results and store them
    # =========================================================================
    local_min = comm.gather(local_min, root=0)

    # TODO: nicer to make separate function for this
    if rank == 0:
        debug("All results have been combined!")
        local_min = flatten_list(local_min)

        total_time = MPI.Wtime() - start_time
        print(f"\nExecution time for sub-population GA: {total_time}")

        # Filter minima
        local_min = process_data.select_local_minima(local_min)
        process_data.print_stats(local_min)

        # Write all local minima to trajectory file
        traj_file = Trajectory(f"{c.results_dir}/ga_sub_pop_{c.cluster_size}.traj", 'w')
        for cluster in local_min:
            traj_file.write(cluster)
        traj_file.close()

        # Connect to database
        db_file = os.path.join(os.path.dirname(__file__), c.results_dir+'/'+c.db_file)
        db = ase.db.connect(db_file)
        store_results_database(local_min[0], local_min, db, c)

    return 0


if __name__ == "__main__":
    ga_sub_populations()
