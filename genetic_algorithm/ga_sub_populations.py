#!/bin/python3

"""
Sub populations parallelisation of the Genetic Algorithm.

A number of separate GA's will be run on multiple processors, with each processor having their own small, independent
sub-population. The different processors can exchange pairs of clusters with each other to maintain population
diversity.

In order to run the sub-population GA algorithm, you need to use the ga_sub_populations.yaml file.
An example configuration:

    general:
      cluster_radius: 2.0
      cluster_size: 10
      delta_energy_thr: 0.1
      gens_until_exchange: 10
      pop_size: 20
    mating:
      children_perc: 0.8
      fitness_func: exponential
      mating_method: roulette
    results:
      db_file: genetic_algorithm_results.db
      results_dir: results
    run_id: 1
    stop_conditions:
      max_exchanges_no_success: 3
      max_gen: 100
      time_lim: 600

"""

import os
import sys
import ase
import time
import yaml
import inspect
import argparse
import numpy as np

from mpi4py import MPI
from mating import mating
from ase.io import Trajectory
from ase.optimize import LBFGS
from dataclasses import dataclass
from datetime import datetime as dt
from ga_distributed import flatten_list
from ase.calculators.lj import LennardJones
from genetic_algorithm import debug, generate_population
from genetic_algorithm import natural_selection_step
from genetic_algorithm import optimise_local, fitness, get_mutants

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import process_data


def config_info(config):
    """
    Log the most important configuration info to stdout.
    """
    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    n = 63
    print()
    print(" ---------------------------------------------------------------- ")
    print(f"| {f'Parallel Global Geometry Optimisation':{n}s}|")
    print(f"| {f'Genetic Algorithm':{n}s}|")
    print(" ================================================================ ")
    print(f"| {f'Timestamp          : {timestamp}':{n}s}|")
    print(f"| {f'cluster size       : {config.cluster_size}':{n}s}|")
    print(f"| {f'Population size    : {config.pop_size}':{n}s}|")
    print(f"| {f'Fitness function   : {config.fitness_func}':{n}s}|")
    print(f"| {f'Maximum generations: {config.max_gen}':{n}s}|")
    print(f"| {f'Max exchanges wo success : {config.max_exch_no_success}':{n}s}|")
    print(" ---------------------------------------------------------------- ")


@dataclass
class Config:
    cluster_size: int = None
    pop_size: int = None
    max_exch_no_success: int = None
    gens_until_exchange: int = None
    fitness_func: str = None
    mating_method: str = None
    results_dir: str = None
    children_perc: float = None
    cluster_radius: float = None
    max_gen: int = None
    dE_thr: float = None
    run_id: int = None
    db_file: str = None
    time_lim: float = None
    calc = LennardJones(sigma=1.0, epsilon=1.0)
    local_optimiser = LBFGS


def get_configuration(config_file):
    """
    Set the parameters for this run.

    @param config_file: Filename to the yaml configuration
    @type config_file: str
    @return: object with all the configuration parameters
    """

    # Get parameters from config file
    config_file = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_file) as f:
        yaml_conf = yaml.safe_load(os.path.expandvars(f.read()))

    # Create parser for terminal input
    parser = argparse.ArgumentParser(description='Genetic Algorithm PGGO')

    parser.add_argument('--cluster_size', type=int, metavar='',
                        help='Number of atoms per cluster')
    parser.add_argument('--pop_size', type=int, metavar='',
                        help='Number of clusters in the population')
    parser.add_argument('--fitness_func', metavar='',
                        help='Fitness function')
    parser.add_argument('--mating_method', metavar='',
                        help='Mating Method')
    parser.add_argument('--children_perc', type=float, metavar='',
                        help='Fraction of population that will have a child')
    parser.add_argument('--cluster_radius', type=float, metavar='',
                        help='Dimension of initial random clusters')
    parser.add_argument('--gens_until_exchange', type=int, metavar='',
                        help='Number of generations each sub-population runs before exchanging clusters')
    parser.add_argument('--max_exchanges_no_success', type=int, metavar='',
                        help='Number of consecutive cluster exchange rounds without any improvements')
    parser.add_argument('--max_gen', type=int, metavar='',
                        help='Maximum number of generations')
    parser.add_argument('--delta_energy_thr', type=float, metavar='',
                        help='Minimum difference in energy between clusters')
    parser.add_argument('--db_file', type=str, metavar='',
                        help="The database file to write results to")
    parser.add_argument('--results_dir', type=str, metavar='',
                        help="Directory to store results")
    parser.add_argument('--run_id', type=int, metavar='',
                        help="ID for the current run. Increments automatically")
    parser.add_argument('--time_lim', type=float, metavar='',
                        help="Time limit for the algorithm")

    p = parser.parse_args()

    c = Config()
    # Set variables to terminal input if possible, otherwise use config file
    c.cluster_size = p.cluster_size or yaml_conf['general']['cluster_size']
    c.dE_thr = p.delta_energy_thr or yaml_conf['general']['delta_energy_thr']
    c.cluster_radius = p.cluster_radius or yaml_conf['general']['cluster_radius']
    c.pop_size = p.pop_size or yaml_conf['general']['pop_size']
    c.gens_until_exchange = p.gens_until_exchange or yaml_conf['general']['gens_until_exchange']
    c.children_perc = p.children_perc or yaml_conf['mating']['children_perc']
    c.fitness_func = p.fitness_func or yaml_conf['mating']['fitness_func']
    c.mating_method = p.mating_method or yaml_conf['mating']['mating_method']
    c.max_gen = p.max_gen or yaml_conf['stop_conditions']['max_gen']
    c.max_exch_no_success = p.max_exchanges_no_success or yaml_conf['stop_conditions']['max_exchanges_no_success']
    c.time_lim = p.time_lim or yaml_conf['stop_conditions']['time_lim']
    c.results_dir = p.results_dir or yaml_conf['results']['results_dir']
    c.db_file = p.db_file or yaml_conf['results']['db_file']
    c.run_id = p.run_id or yaml_conf['run_id']

    # Increment run_id for next run
    yaml_conf['run_id'] += 1
    with open(config_file, 'w') as f:
        yaml.dump(yaml_conf, f, default_style=False)

    return c


def store_results_database(global_min, local_min, db, c):
    """
    Writes GA results to the database.

    @param global_min: the global minimum cluster
    @param local_min: list of all local minima found
    @param db: the database to write to
    @param c: the configuration information of the GA run
    @return: exit code 0
    """
    db.write(global_min, global_min=True, pop_size=c.pop_size, cluster_size=c.cluster_size, max_gens=c.max_gen,
             max_exch_no_success=c.max_exch_no_success, time_lim=c.time_lim, run_id=c.run_id)

    for cluster in local_min:
        db.write(cluster, global_min=False, pop_size=c.pop_size, cluster_size=c.cluster_size, max_gens=c.max_gen,
                 max_exch_no_success=c.max_exch_no_success, time_lim=c.time_lim, run_id=c.run_id)

    return 0


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
    num_exchanges = np.ceil(sub_pop_size * perc_pop_exchanged).astype(int)
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

    # Number of clusters to be exchanged with a single neighbour.
    num_exchanges = np.ceil(sub_pop_size * perc_pop_exchanged).astype(int) * 2
    rng = np.random.default_rng()
    cluster_indices = rng.choice(sub_pop_size, size=num_exchanges, replace=False)

    num_procs = comm.Get_size()
    left_neighb = (rank - 1) % num_procs
    left_msg = [pop[i] for i in cluster_indices[:(num_exchanges // 2)]]
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

    left_clusters = recv_req_left.wait()
    right_clusters = recv_req_right.wait()

    debug(f"    Generation {gen}: processor {rank} finished all exchanges!")

    # Filter out the exchanged clusters and instead add the newly received clusters from neighbouring populations.
    pop = [cluster for idx, cluster in enumerate(pop) if idx not in cluster_indices]
    pop += left_clusters + right_clusters
    energies = [cluster.get_potential_energy() for cluster in pop]

    return pop, energies, send_req_left, send_req_right


def sync_before_exchange(start_time, gen_no_success, c, rank, comm):
    """
    Synchronises all processes before starting the cluster exchange process.
    If the maximum time limit has been reached while waiting, then return abort code 0.
    If maximum generations without a new global minimum on any processor has been reached, return 1.
    Else, if everything is successful, return 2.

    @param start_time: start time of the algorithm
    @param gen_no_success: the number of generations without a new global minimum found
    @param c: the configuration object containing the run parameters
    @param rank: rank of the current process
    @param comm: MPI communication
    @return: abort code 0, 1, or 2
    """

    abort = 2
    req_barr = comm.Ibarrier()
    while not req_barr.get_status():
        # Check whether the time limit has been reached
        if (MPI.Wtime() - start_time) > c.time_lim:
            abort = 0
        time.sleep(0.001)

    # Check whether max number of generations have passed without improvements for any processor
    gen_no_success = np.array([gen_no_success], dtype='i')
    no_success_arr = np.empty([comm.Get_size(), 1], dtype='i')
    comm.Gather(gen_no_success, no_success_arr, root=0)
    if rank == 0:
        if (no_success_arr > c.max_exch_no_success * c.gens_until_exchange).all():
            abort = 1

    abort = comm.bcast(abort, root=0)
    return abort


def save_to_traj(local_min, c):
    """
    Saves the found local minima to a trajectory file in the folder designated for results.

    @param local_min: the list of local minima from the algorithm run
    @param c: configuration file containing the input parameters
    @return:
    """
    traj_file_path = os.path.join(os.path.dirname(__file__), f"{c.results_dir}/ga_sub_pop_{c.cluster_size}.traj")
    traj_file = Trajectory(traj_file_path, 'w')
    for cluster in local_min:
        traj_file.write(cluster)
    traj_file.close()

    return


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

    print(f"Started sub-populations on processor {rank}!")

    # Check whether enough processors (> 1)
    if num_procs < 2:
        if rank == 0:
            debug("Please use more than 1 processor for the parallel version!")
            debug("Aborting...")
        return

    # Parse possible terminal input and yaml file.
    c = None
    if rank == 0:
        config_file = "config/ga_sub_populations_config.yaml"
        c = get_configuration(config_file)
        config_info(c)
    c = comm.bcast(c, root=0)

    # Ensure all processors have exactly the same starting time for properly synchronising the maximum durations
    start_time = MPI.Wtime()
    start_time = comm.bcast(start_time, root=0)

    # =========================================================================
    # Initial population and variables
    # =========================================================================
    sub_pop_size = np.ceil(c.pop_size / num_procs).astype(int)

    # Check for sufficiently large sub-population size, abort if not large enough (>= 2)
    if sub_pop_size < 2:
        if rank == 0:
            debug("The sub-population size for each processor is less than 2 after splitting the entire population!")
            debug("Please provide a larger population size.")
            debug("Aborting the program...")
        MPI.Finalize()
        return

    pop = generate_population(sub_pop_size, c.cluster_size, c.cluster_radius)
    energies = optimise_local(pop, c.calc, c.local_optimiser)

    # Keep track of global minima. Initialised with random cluster
    best_min = pop[0]
    local_min = [pop[0]]

    # =========================================================================
    # Main loop
    # =========================================================================
    # Keep track of iterations
    gen = 0
    gen_no_success = 0

    # Abort codes: 0 for time limit reached, 1 for max generations without improvement, 2 for no stopping condition yet
    abort = 2

    # Used for swapping random clusters between sub-populations
    send_req_left = None
    send_req_right = None

    while gen < c.max_gen:
        # If max time has been reached, set stopping condition to 0
        if (MPI.Wtime() - start_time) > c.time_lim:
            abort = 0
        if abort == 0:
            debug(f"Processor {rank} aborted due to time limit!")
            break

        # Exchange clusters with neighbouring processors
        if (gen % c.gens_until_exchange) == 0:
            # Synchronise all processors before communication to make sure they all abort simultaneously before comms
            abort = sync_before_exchange(start_time, gen_no_success, c, rank, comm)
            if abort == 0:
                debug(f"Processor {rank} aborted due to time limit!")
                break
            if abort == 1:
                debug(f"Processor {rank} aborted in gen {gen} due to max generations without success reached!")
                break

            # Exchange clusters with both neighbouring processors
            if num_procs > 2:
                pop, energies, send_req_left, send_req_right \
                    = bi_directional_exchange(pop, sub_pop_size, gen, comm, rank, send_req_left, send_req_right)
            # Exchange clusters with only a single neighbour if 2 processors
            else:
                pop, energies, send_req_right = one_directional_exchange(pop, sub_pop_size, gen, comm, rank,
                                                                         send_req_right)

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

    if rank == 0:
        local_min = flatten_list(local_min)
        debug("All results have been combined!")
        total_time = MPI.Wtime() - start_time
        print(f"\nExecution time for sub-population GA: {total_time}")

        # Filter minima
        local_min = process_data.select_local_minima(local_min)
        process_data.print_stats(local_min)

        # Write all local minima to trajectory file
        save_to_traj(local_min, c)

        # Connect to database
        db_file = os.path.join(os.path.dirname(__file__), c.results_dir+'/'+c.db_file)
        db = ase.db.connect(db_file)
        store_results_database(local_min[0], local_min, db, c)

    return 0


if __name__ == "__main__":
    ga_sub_populations()
