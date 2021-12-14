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
from ase.visualize import view


def debug(*args, **kwargs) -> None:
    """
    Alias for print() function.
    This can easily be redefined to disable all output.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
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
    time_out = 0
    view_traj = 0


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
    c = Config()
    # Set variables to terminal input if possible, otherwise use config file
    c.cluster_size = yaml_conf['cluster_size']
    c.pop_size = yaml_conf['pop_size']
    c.cluster_radius = yaml_conf['cluster_radius']
    c.run_id = yaml_conf['run_id']
    c.cycle = yaml_conf['cycle']
    c.is_parallel = yaml_conf['is_parallel']

    # When time out is set to less than 0, it is set to infinity
    if yaml_conf['time_out'] <= 0:
        c.time_out = float('inf')
    else:
        c.time_out = yaml_conf['time_out']
    c.view_traj = yaml_conf['view_traj']
    # Increment run_id for next run
    yaml_conf['run_id'] += 1
    with open(config_file, 'w') as f:
        yaml.dump(yaml_conf, f)

    return c


def generate_cluster_with_position(p, cluster_size) -> Atoms:
    return Atoms('H' + str(cluster_size), p)


def generate_cluster(cluster_size, radius) -> Atoms:
    """Generate a random cluster with set number of atoms
    The atoms will be placed within a (radius x radius x radius) cube.
    Args:
        cluster_size (int)  : Number of atoms per cluster
        radius (float)      : dimension of the space where atoms can be placed.
    Returns:
        new_cluster (Atoms) : Randomly generated cluster
    """
    return Atoms('H' + str(cluster_size),
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


def optimise_local(population, calc, optimiser, size) -> List[Atoms]:
    """Local optimisation of the population. The clusters in the population
    are optimised and can be used after this function is called. Moreover,
    calculate and return the final optimised potential energy of the clusters.
    Args:
        population(List[Atoms]) : List of clusters to be locally optimised
        calc (Calculator)       : ASE Calculator for potential energy (e.g. LJ)
        optimiser (Optimiser)   : ASE Optimiser (e.g. LBFGS)
        size                    : the amount of processors that call this method
    Returns:
        (List[Atoms])           : Optimised population
    """

    if size == 1:
        return [optimise_local_each(cluster, calc, optimiser).get_potential_energy() for cluster in population]
    else:
        comm = MPI.COMM_WORLD

        splitted_population = split(population,
                                    comm.Get_size())  # divides the array into n parts to divide over processors
        population = comm.scatter(splitted_population, root=0)

        result = []
        if len(population) != 0:
            for cluster in population:
                cluster.calc = calc
                result.append(optimise_local_each(cluster, calc, optimiser).get_potential_energy())

        optimised_clusters = comm.gather(result, root=0)
        if comm.Get_rank() == 0:
            optimised_clusters = [i for i in optimised_clusters if i]
            optimised_clusters = [item for sublist in optimised_clusters for item in sublist]
            return optimised_clusters
        else:
            return []


def store_results_database(pop, db, c, cycle):
    """
    Writes GA results to the database.

    @param db: the database to write to
    @param c: the configuration information of the GA run
    @return: exit code 0
    """
    # TODO separation between local minimas and global minima
    for cluster in pop:
        last_id = db.write(cluster, pop_size=c.pop_size,
                           cluster_size=c.cluster_size, run_id=c.run_id,
                           potential_energy=cluster.get_potential_energy(), cycle=cycle)
    return 0


# TODO change the strutuer of config file , look at the genetic algorithm config
def artificial_bee_colony_algorithm():
    tic = time.perf_counter()
    setup_start_time = MPI.Wtime()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    total_p = comm.Get_size()
    if rank == 0:
        # Parse possible input, otherwise use default parameters
        # TODO directory and file names should also be in the config file
        p = get_configuration('config/run_config.yaml')
        config_info(p)
        # generate initial population
        population = generate_population(p.pop_size, p.cluster_size, p.cluster_radius)
        optimise_local(population, p.calc, p.local_optimiser, 1)
        # Generate initial population and optimise locally
    else:
        population = None
        p = None
    population = comm.bcast(population, root=0)
    p = comm.bcast(p, root=0)
    comm.Barrier()

    show_calc_min = 1
    eb_mutation_size = 3

    debug(f"Set up took {MPI.Wtime() - setup_start_time}")
    cycle_start_time = MPI.Wtime()
    break_loop = False
    # TODO stop loop when it does not show improvement for many loops
    # eg when the minima at lo
    for i in range(1, p.cycle + 1):
        population = employee_bee.employee_bee_func(population, p.pop_size, p.cluster_size, p.calc, p.local_optimiser,
                                                    comm, rank, total_p, p.is_parallel, eb_mutation_size)
        if rank == 0:
            population = onlooker_bee.onlooker_bee_func(population, p.pop_size, p.cluster_size, p.calc, p.local_optimiser)
    
        population = comm.bcast(population, root=0)
        population = scout_bee.scout_bee_func(population, p.pop_size, p.cluster_size, p.cluster_radius, p.calc,
                                             p.local_optimiser, comm, rank, 0.04, 0.65, i-1)
        population = comm.bcast(population, root=0)


        if rank == 0:
            if (i % show_calc_min) == 0:
                debug(
                    f"Global optimisation at loop {i}:{np.min([cluster.get_potential_energy() for cluster in population])}")

        if time.perf_counter() - tic >= p.time_out:  # if algorithm didn't stop after x seconds, stop the algorithm
            if rank == 0:
                debug(f"Function time exceeded. Stopping now")
                break_loop = True
            break_loop = comm.bcast(break_loop, root=0)

        if break_loop:
            break

    debug(f"It took {MPI.Wtime() - cycle_start_time} with {total_p} processors")

    if rank == 0:
        # filter out local minima that are too similar and print out the results
        local_minima = process.select_local_minima(population)
        process.print_stats(local_minima)

        root_directory = os.path.dirname(__file__)
        trajectory = Trajectory(root_directory + "/" + f"results/abc_{p.cluster_size}.traj", "w")
        for cluster in local_minima:
            trajectory.write(cluster)
        trajectory.close()
        if p.view_traj == 1:
            trajectory = Trajectory(root_directory + "/" + f"results/abc_{p.cluster_size}.traj")
            view(trajectory)

        db_start_time = MPI.Wtime()
        # TODO directory and file names should also be in the config file
        store_results_database(local_minima,
                               ase.db.connect(os.path.join(os.path.dirname(__file__),
                                                           "artificial_bee_colony_algorithm_results.db")), p, p.cycle)
        debug(f"Saving to db took {MPI.Wtime() - db_start_time}")
        debug(f"total time took {MPI.Wtime() - setup_start_time}")


def split(a, n):
    """
    Splits an array 'a' into n parts.
    For example split the following array into 5 parts: [1,2,3,4,5,6,7,8,9,10] -> [[1,2],[3,4],[5,6],[7,8],[9,10]]
    """

    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == '__main__':
    artificial_bee_colony_algorithm()
