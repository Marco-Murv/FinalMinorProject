#!/bin/python3
from dataclasses import dataclass

import numpy as np
import yaml
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import LBFGS
import ase.db
import os
from typing import List
import argparse
import sys
import employee_bee
import onlooker_bee
import scout_bee
from datetime import datetime as dt
import process
import time
from mpi4py import MPI
from ase.io.trajectory import Trajectory

cluster_str = 'H'


def debug(*args, **kwargs) -> None:
    """
    Alias for print() function.
    This can easily be redefined to disable all output.
    """
    print("[DEBUG]: ", flush=True, *args, **kwargs)


def config_info(config):
    """
    Log the most important info to stdout.
    """
    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    n = 63
    print(" ---------------------------------------------------------------- ")
    print(f"| {f'Parallel Global Geometry Optimisation':{n}s}|")
    print(f"| {f'Artificial Bee Colony Algorithm':{n}s}|")
    print(" ================================================================ ")
    print(f"| {f'Timestamp          : {timestamp}':{n}s}|")
    print(f"| {f'cluster size       : {config.cluster_size}':{n}s}|")
    print(f"| {f'Population size    : {config.pop_size}':{n}s}|")
    print(f"| {f'Cycle              : {config.cycle}':{n}s}|")

    print(" ---------------------------------------------------------------- ")


@dataclass
class Config:
    cluster_size: int = None
    pop_size: int = None
    cluster_radius: float = None
    run_id: int = None
    cycle: int = None
    calc = LennardJones(sigma=1.0, epsilon=1.0)  # TODO: Change parameters
    local_optimiser = LBFGS
    is_parallel = 0


def get_configuration(config_file):
    """Set the parameters for this run.

    :param config_file: Filename to the yaml configuration
    :type config_file: str
    :return: object with all the configuration parameters
    :rtype: Config
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
    parser.add_argument('--cluster_radius', default=2.0, type=float, metavar='',
                        help='Dimension of initial random clusters')
    parser.add_argument('--run_id', type=int, metavar='',
                        help="ID for the current run. Increments automatically")
    parser.add_argument('--cycle', type=int, metavar='',
                        help="size of cycle for the loop")
    parser.add_argument('--is_parallel', type=int, metavar='',
                        help="run in parallel")
    p = parser.parse_args()

    c = Config()
    # Set variables to terminal input if possible, otherwise use config file
    c.cluster_size = p.cluster_size or yaml_conf['cluster_size']
    c.pop_size = p.pop_size or yaml_conf['pop_size']
    c.cluster_radius = p.cluster_radius or yaml_conf['cluster_radius']
    c.run_id = p.run_id or yaml_conf['run_id']
    c.cycle = yaml_conf['cycle']
    c.is_parallel = yaml_conf['is_parallel']

    # Increment run_id for next run
    yaml_conf['run_id'] += 1
    with open(config_file, 'w') as f:
        yaml.dump(yaml_conf, f)

    return c


def generate_cluster_with_position(p, cluster_size) -> Atoms:
    return Atoms(cluster_str + str(cluster_size), p)


def generate_cluster(cluster_size, radius) -> Atoms:
    """Generate a random cluster with set number of atoms
    The atoms will be placed within a (radius x radius x radius) cube.
    Args:
        cluster_size (int)  : Number of atoms per cluster
        radius (float)      : dimension of the space where atoms can be placed.
    Returns:
        new_cluster (Atoms) : Randomly generated cluster
    """
    return Atoms(cluster_str + str(cluster_size),
                 np.random.uniform(-radius / 2, radius / 2, (cluster_size, 3)).tolist())


def generate_population(popul_size, cluster_size, radius) -> List[Atoms]:
    """Generate initial population.
    Args:
        popul_size (int)    : number of clusters in the population
        cluster_size (int)  : number of atoms in each cluster
        radius (float)      : dimension of the initial random clusters
    Returns:
        (List[Atoms])       : List of clusters
    """
    return [generate_cluster(cluster_size, radius) for i in range(popul_size)]


def optimise_local_each(cluster, calc, optimiser) -> Atoms:
    cluster.calc = calc
    try:
        optimiser(cluster, maxstep=0.2, logfile=None).run(steps=50)
        return cluster
    except:  # TODO: how to properly handle these error cases?
        print("FATAL ERROR: DIVISION BY ZERO ENCOUNTERED!")
        sys.exit("PROGRAM ABORTED: FATAL ERROR")


def optimise_local(population, calc, optimiser) -> List[Atoms]:
    """Local optimisation of the population. The clusters in the population
    are optimised and can be used after this function is called. Moreover,
    calculate and return the final optimised potential energy of the clusters.
    Args:
        population(List[Atoms]) : List of clusters to be locally optimised
        calc (Calculator)       : ASE Calculator for potential energy (e.g. LJ)
        optimiser (Optimiser)   : ASE Optimiser (e.g. LBFGS)
    Returns:
        (List[Atoms])           : Optimised population
    """

    """ Parallel version of this method. Currently gives a "Atom has no calculator" error in employee bee
    comm = MPI.COMM_WORLD

    splitted_population = split(population, comm.Get_size()) # divides the array into n parts to divide over processors
    population = comm.scatter(splitted_population, root=0)

    result = []
    for cluster in population:
        result.append(optimise_local_each(cluster, calc, optimiser).get_potential_energy())

    optimised_clusters = comm.gather(result, root=0)
    optimised_clusters = [item for sublist in optimised_clusters for item in sublist]

    return optimised_clusters
    """

    return [optimise_local_each(cluster, calc, optimiser).get_potential_energy() for cluster in population]


def store_results_database(pop, db, c, cycle):
    """
    Writes GA results to the database.

    @param global_min: the global minimum cluster
    @param local_min: list of all local minima found
    @param db: the database to write to
    @param c: the configuration information of the GA run
    @return: exit code 0
    """

    for cluster in pop:
        last_id = db.write(cluster, pop_size=c.pop_size,
                           cluster_size=c.cluster_size, run_id=c.run_id,
                           potential_energy=cluster.get_potential_energy(), cycle=cycle)
    return 0


def artificial_bee_colony_algorithm():
    tic = time.perf_counter()
    setup_start_time = MPI.Wtime()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    total_p = comm.Get_size()
    if rank == 0:
        # configure db
        db = ase.db.connect(os.path.join(os.path.dirname(__file__), "artificial_bee_colony_algorithm_results.db"))
        # Parse possible input, otherwise use default parameters
        p = get_configuration('run_config.yaml')
        config_info(p)
        # generate initial population
        population = generate_population(p.pop_size, p.cluster_size, p.cluster_radius)
        optimise_local(population, p.calc, p.local_optimiser)
        # Generate initial population and optimise locally
    else:
        db = None
        population = None
        p = None
    population = comm.bcast(population, root=0)
    p = comm.bcast(p, root=0)
    comm.Barrier()

    show_calc_min = 1
    eb_mutation_size = 3
    if rank == 0:
        debug(f"Set up took {MPI.Wtime() - setup_start_time}")
    cycle_start_time = MPI.Wtime()
    for i in range(p.cycle):
        i=i+1
        population = employee_bee.employee_bee_func(population, p.pop_size, p.cluster_size, p.calc, p.local_optimiser,
                                                    comm, rank, total_p, p.is_parallel, eb_mutation_size)

        if rank == 0:
            population = onlooker_bee.onlooker_bee_func(population, p.pop_size, p.cluster_size, p.calc,
                                                        p.local_optimiser)
            population = scout_bee.scout_bee_func(population, p.pop_size, p.cluster_size,
                                                 p.cluster_radius, p.calc, p.local_optimiser)
            if (i % show_calc_min) == 0:
                debug(f"Global optimisation at loop {i}:{np.min([cluster.get_potential_energy() for cluster in population])}")

        if p.is_parallel == 1:
            population = comm.bcast(population, root=0)
            comm.Barrier()

        toc = time.perf_counter()
        if toc - tic >= 100: # if algorithm didn't stop after x seconds, stop the algorithm
            debug(f"Function time exceeded. Stopping now")

            # filter out local minima that are too similar and print out the results
            local_minima = process.select_local_minima(population)
            process.print_stats(local_minima)

            db_start_time = MPI.Wtime()
            store_results_database(local_minima, db, p, p.cycle)
            debug(f"Saving to db took {MPI.Wtime() - db_start_time}")

            return

    if rank == 0:
        if p.is_parallel == 1:
            debug(f"It took {MPI.Wtime() - cycle_start_time} with {total_p} processors")
        else:
            debug(f"It took {MPI.Wtime()- cycle_start_time} without parallelization")

    if rank == 0:
        # filter out local minima that are too similar and print out the results
        local_minima = process.select_local_minima(population)
        process.print_stats(local_minima)

        trajFile = Trajectory(f"ga_{p.cluster_size}.traj", 'w')
        for cluster in local_minima:
            trajFile.write(cluster)
        trajFile.close()

        db_start_time = MPI.Wtime()
        store_results_database(local_minima, db, p, p.cycle)
        debug(f"Saving to db took {MPI.Wtime() - db_start_time}")


def split(a, n):
    """
    Splits an array 'a' into n parts.
    For example split the following array into 5 parts: [1,2,3,4,5,6,7,8,9,10] -> [[1,2],[3,4],[5,6],[7,8],[9,10]]
    """

    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == '__main__':
    artificial_bee_colony_algorithm()