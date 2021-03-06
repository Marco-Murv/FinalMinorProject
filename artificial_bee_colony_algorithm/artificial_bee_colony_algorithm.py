#!/bin/python3
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime as dt
from typing import List

import ase.db
import numpy as np
import yaml
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS
from ase.visualize import view
from mpi4py import MPI

import employee_bee
import onlooker_bee
import process
import scout_bee


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
    print(f"| {f'Cluster size       : {config.cluster_size}':{n}s}|")
    print(f"| {f'Population size    : {config.pop_size}':{n}s}|")
    print(f"| {f'Cycles             : {config.maximum_cycle}':{n}s}|")

    print(" ---------------------------------------------------------------- ")


@dataclass
class Config:
    cluster_size: int = None
    pop_size: int = None
    cluster_radius: float = None
    run_id: int = None
    minimum_cycle: int = None
    maximum_cycle: int = None
    calc = LennardJones(sigma=1.0, epsilon=1.0)  # TODO: Change parameters
    local_optimiser = LBFGS
    is_parallel = 0
    time_out = 0
    view_traj = 0
    eb_search_method = 0
    eb_search_size = 4
    monte_carlo_search_f = -1
    eb_enable = 1
    ob_enable = 1
    sb_enable = 1
    auto_stop = -1
    auto_stop_sf = 0
    energy_diff = 0.04
    energy_abnormal = 0.65
    check_energies_every_x_loops = 20
    update_energies = 1
    sb_count =4


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
    c.cluster_size = yaml_conf['cluster_config']['cluster_size']
    c.pop_size = yaml_conf['cluster_config']['pop_size']
    c.cluster_radius = yaml_conf['cluster_config']['cluster_radius']
    c.run_id = yaml_conf['run_config']['run_id']
    c.minimum_cycle = yaml_conf['run_config']['minimum_cycle']
    c.maximum_cycle = yaml_conf['run_config']['maximum_cycle']
    c.is_parallel = yaml_conf['run_config']['is_parallel']
    c.auto_stop = yaml_conf['run_config']['auto_stop']
    c.auto_stop_sf = yaml_conf['run_config']['auto_stop_sf']

    # When time out is set to less than 0, it is set to infinity
    if yaml_conf['run_config']['time_out'] <= 0:
        c.time_out = float('inf')
    else:
        c.time_out = yaml_conf['run_config']['time_out']
    c.view_traj = yaml_conf['run_config']['view_traj']
    # Increment run_id for next run
    yaml_conf['run_config']['run_id'] += 1

    c.eb_search_method = yaml_conf['employed_bee_config']['search_method']
    c.monte_carlo_search_f = yaml_conf['employed_bee_config']['monte_carlo_search_f']
    c.eb_search_size = yaml_conf['employed_bee_config']['search_size']
    c.eb_enable = yaml_conf['employed_bee_config']['enable']

    c.ob_enable = yaml_conf['onlooker_bee_config']['enable']

    c.sb_enable = yaml_conf['scout_bee_config']['enable']
    c.energy_diff = yaml_conf['scout_bee_config']['energy_difference']
    c.energy_abnormal = yaml_conf['scout_bee_config']['energy_abnormal']
    c.check_energies_every_x_loops = yaml_conf['scout_bee_config']['check_energies_every_x_loops']
    c.update_energies = yaml_conf['scout_bee_config']['update_energies']
    c.update_energies = yaml_conf['scout_bee_config']['count']
    with open(config_file, 'w') as f:
        yaml.dump(yaml_conf, f)

    return c


def generate_cluster_with_position(p, cluster_size, counter) -> Atoms:
    a = Atoms('C' + str(cluster_size),
              p)
    tag = a.get_tags()
    tag[0] = counter
    a.set_tags(tag)
    return a


def generate_cluster(cluster_size, radius, counter) -> Atoms:
    """Generate a random cluster with set number of atoms
    The atoms will be placed within a (radius x radius x radius) cube.
    Args:
        cluster_size (int)  : Number of atoms per cluster
        radius (float)      : dimension of the space where atoms can be placed.
    Returns:
        new_cluster (Atoms) : Randomly generated cluster
    """
    a = Atoms('C' + str(cluster_size),
                 np.random.uniform(-radius / 2, radius / 2, (cluster_size, 3)).tolist())
    p = a.get_tags()
    p[0] = counter
    a.set_tags(p)
    return a


def generate_population(popul_size, cluster_size, radius, counter) -> List[Atoms]:
    """Generate initial population.
    Args:
        popul_size (int)    : number of clusters in the population
        cluster_size (int)  : number of atoms in each cluster
        radius (float)      : dimension of the initial random clusters
    Returns:
        (List[Atoms])       : List of clusters
    """
    return [generate_cluster(cluster_size, radius, counter) for i in range(popul_size)]


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
        # Generate initial population and optimise locally
        population = generate_population(p.pop_size, p.cluster_size, p.cluster_radius, p.sb_count)
        optimise_local(population, p.calc, p.local_optimiser, 1)

        # Set scout bee variable
        scout_bee.check_every_loop = p.check_energies_every_x_loops
    else:
        population = None
        p = None
    population = comm.bcast(population, root=0)
    p = comm.bcast(p, root=0)
    removed_clusters = []
    comm.Barrier()

    show_calc_min = 1
    debug(f"Set up took {MPI.Wtime() - setup_start_time}")
    cycle_start_time = MPI.Wtime()
    break_loop = False
    min_potential_energies = np.array([])

    for i in range(1, p.maximum_cycle + 1):
        if p.eb_enable == 1:
            population = employee_bee.employee_bee_func(population, p.pop_size, p.cluster_size, p.calc,
                                                        p.local_optimiser,
                                                        comm, rank, total_p, p.is_parallel, p.eb_search_size,
                                                        p.eb_search_method, p.monte_carlo_search_f, p.sb_count)
        if (p.ob_enable == 1) & (rank == 0):
            population = onlooker_bee.onlooker_bee_func(population, p.pop_size, p.cluster_size, p.calc,
                                                        p.local_optimiser, p.sb_count)
        population = comm.bcast(population, root=0)

        if p.sb_enable == 1:
            population, removed_clusters = scout_bee.scout_bee_func(population, p.pop_size, p.cluster_size, p.cluster_radius, p.calc,
                                                  p.local_optimiser, comm, rank, p.energy_diff, p.energy_abnormal, i - 1, p.is_parallel, p.update_energies, p.sb_count, removed_clusters)
            population = comm.bcast(population, root=0)

        if rank == 0:
            if (i % show_calc_min) == 0:
                min_e = np.min([cluster.get_potential_energy() for cluster in population])
                min_potential_energies = np.append(min_potential_energies, min_e)
                debug(
                    f"Global optimisation at loop {i}:{min_e}")

        if (time.perf_counter() - tic >= p.time_out) & (i > p.minimum_cycle):  # if algorithm didn't stop after x seconds, stop the algorithm
            if rank == 0:
                debug(f"Function time exceeded. Stopping now")
                break_loop = True

        if (rank == 0) & (p.auto_stop > 0) & (i > p.minimum_cycle):
            p_index = int(i / p.auto_stop)
            if int(min_potential_energies[p_index - 1] * (10 ** p.auto_stop_sf)) == int(min_potential_energies[i - 1] * (10 ** p.auto_stop_sf)):
                debug(f"compared to loop {p_index} where min energy is {min_potential_energies[p_index - 1]} "
                      f" no big improvements have been found, thus stopping calculation " )
                break_loop = True
        break_loop = comm.bcast(break_loop, root=0)
        if break_loop:
            break

    debug(f"It took {MPI.Wtime() - cycle_start_time} with {total_p} processors")

    if rank == 0:
        local_minima = process.select_local_minima(population) # filter out local minima that are too similar
        for cluster in removed_clusters: local_minima.append(cluster) # add clusters that were discarded by the scout bee
        process.print_stats(local_minima) # print out the results

        traj_file_path = os.path.join(os.path.dirname(__file__), "results/abc_5.traj")
        trajectory = Trajectory(traj_file_path, "w")
        for cluster in local_minima:
            trajectory.write(cluster)
        trajectory.close()
        if p.view_traj == 1:
            trajectory = Trajectory(traj_file_path)
            view(trajectory)

        db_start_time = MPI.Wtime()
        # TODO directory and file names should also be in the config file
        db_file = os.path.join(os.path.dirname(__file__), "artificial_bee_colony_algorithm_results.db")
        db = ase.db.connect(db_file)
        store_results_database(local_minima, db, p, p.maximum_cycle)

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
