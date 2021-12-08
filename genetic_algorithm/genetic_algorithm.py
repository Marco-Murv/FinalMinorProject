#!/bin/python3

"""
Genetic algorithm for geometry optimisation of atomic clusters.

This program requires a file called `ga_config.yaml` in the same directory.
Example ga_config.yaml:
```yaml
    general:
        cluster_radius: 2.0
        cluster_size: 4
        delta_energy_thr: 0.1
        pop_size: 5
    mating:
        children_perc: 0.8
        fitness_func: exponential
        mating_method: roulette
    results:
        db_file: genetic_algorithm_results.db
        results_dir: results
    reuse_state: false
    run_id: 22
    show_plot: true
    stop_conditions:
        max_gen: 50
        max_no_success: 50
        time_lim: 100
```
"""

import os
import sys
import yaml
import time
import ase.db
import pickle
import inspect
import argparse
import mutators
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime as dt

from ase import Atoms
from ase.optimize import LBFGS
from ase.io.trajectory import Trajectory
from ase.calculators.lj import LennardJones

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from mating import mating
import process_data

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
    print()
    print(" ---------------------------------------------------------------- ")
    print(f"| {f'Parallel Global Geometry Optimisation':{n}s}|")
    print(f"| {f'Genetic Algorithm':{n}s}|")
    print(" ================================================================ ")
    print(f"| {f'Timestamp          : {timestamp}':{n}s}|")
    print(f"| {f'Time limit         : {config.time_lim} sec':{n}s}|")
    print(f"| {f'cluster size       : {config.cluster_size}':{n}s}|")
    print(f"| {f'Population size    : {config.pop_size}':{n}s}|")
    print(f"| {f'Fitness function   : {config.fitness_func}':{n}s}|")
    print(f"| {f'Max gen wo success : {config.max_no_success}':{n}s}|")
    print(f"| {f'Maximum generations: {config.max_gen}':{n}s}|")
    print(" ---------------------------------------------------------------- ")


@dataclass
class Config:
    cluster_size: int = None
    pop_size: int = None
    max_no_success: int = None
    fitness_func: str = None
    mating_method: str = None
    results_dir: str = None
    children_perc: float = None
    cluster_radius: float = None
    max_gen: int = None
    dE_thr: float = None
    run_id: int = None
    reuse_state: bool = None
    show_plot: bool = None
    db_file: str = None
    time_lim: float = None
    calc = LennardJones(sigma=1.0, epsilon=1.0)
    local_optimiser = LBFGS


def get_configuration(config_file):
    """Set the parameters for this run.

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
                        help='Fraction of opulation that will have a child')
    parser.add_argument('--cluster_radius', type=float, metavar='',
                        help='Dimension of initial random clusters')
    parser.add_argument('--max_no_success', type=int, metavar='',
                        help='Consecutive generations without new minimum')
    parser.add_argument('--max_gen', type=int, metavar='',
                        help='Maximum number of generations')
    parser.add_argument('--delta_energy_thr', type=float, metavar='',
                        help='Minimum difference in energy between clusters')
    parser.add_argument('--reuse_state', type=bool, metavar='',
                        help="Reuse the same random state from previous run")
    parser.add_argument('--show_plot', type=bool, metavar='',
                        help="Show Evolutionary Progress Plot")
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
    c.children_perc = p.children_perc or yaml_conf['mating']['children_perc']
    c.fitness_func = p.fitness_func or yaml_conf['mating']['fitness_func']
    c.mating_method = p.mating_method or yaml_conf['mating']['mating_method']
    c.max_gen = p.max_gen or yaml_conf['stop_conditions']['max_gen']
    c.max_no_success = p.max_no_success or yaml_conf['stop_conditions']['max_no_success']
    c.time_lim = p.time_lim or yaml_conf['stop_conditions']['time_lim']
    c.results_dir = p.results_dir or yaml_conf['results']['results_dir']
    c.db_file = p.db_file or yaml_conf['results']['db_file']
    c.reuse_state = p.reuse_state or yaml_conf['reuse_state']
    c.show_plot = p.show_plot or yaml_conf['show_plot']
    c.run_id = p.run_id or yaml_conf['run_id']

    # Increment run_id for next run
    yaml_conf['run_id'] += 1
    with open(config_file, 'w') as f:
        yaml.dump(yaml_conf, f, default_style=False)

    return c


def generate_cluster(cluster_size, radius) -> Atoms:
    """
    Generate a random cluster with set number of atoms
    The atoms will be placed within a (radius x radius x radius) cube.

    @param cluster_size: number of atoms per cluster
    @param radius: dimension of the space where atoms can be placed.
    @return: new random cluster
    """
    coords = np.random.uniform(-radius / 2, radius / 2, (cluster_size, 3)).tolist()

    # TODO: Can we use "mathematical dots" instead of H-atoms
    new_cluster = Atoms('H' + str(cluster_size), coords)

    return new_cluster


def generate_population(popul_size, cluster_size, radius) -> List[Atoms]:
    """
    Generate initial population.

    @param popul_size: number of clusters in the population
    @param cluster_size: number of atoms in each cluster
    @param radius: dimension of the initial random clusters
    @return: List of clusters
    """
    return [generate_cluster(cluster_size, radius) for i in range(popul_size)]


def optimise_local(population, calc, optimiser) -> List[Atoms]:
    """
    Local optimisation of the population. The clusters in the population
    are optimised and can be used after this function is called. Moreover,
    calculate and return the final optimised potential energy of the clusters.

    @param population: List of clusters to be locally optimised
    @param calc: ASE Calculator for potential energy (e.g. LJ)
    @param optimiser: ASE Optimiser (e.g. LBFGS)
    @returns: -> Optimised population
    """
    invalid_indices = []
    for idx, cluster in enumerate(population):
        cluster.calc = calc

        try:
            optimiser(cluster, maxstep=0.2, logfile=None).run(steps=50)
        
        # deletes cluster from pop if division by zero error encountered.
        except FloatingPointError:
            invalid_indices.append(idx)
            debug("DIVIDE BY ZERO REMOVED FROM POPULATION!")

        except Exception as e:
            invalid_indices.append(idx)
            debug(f"[ERROR]: CAUGHT EXCEPTION: {e}")
            debug(f"\tREMOVED FROM POPULATION - FIX PROBLEM")

    return [cluster.get_potential_energy() for idx, cluster in enumerate(population) if idx not in invalid_indices]


def fitness(energies, func="exponential") -> np.ndarray:
    """
    Calculate the fitness of the clusters in the population

    @param energies: List of cluster energies
    @param func: Fitness function ("exponential" / "linear" / "hyperbolic")
    @return: fitness values of population
    """
    # Normalise the energies
    normalised_energies = (np.array(energies) - np.min(energies)) / (np.max(energies) - np.min(energies))

    if func == "exponential":
        alpha = 3
        return np.exp(- alpha * normalised_energies)

    elif func == "linear":
        return 1 - 0.7 * normalised_energies

    elif func == "hyperbolic":
        return 0.5 * (1 - np.tanh(2 * energies - 1))

    else:
        print(f"'{func}' is not a valid fitness function. Using default")
        return fitness(energies)


def plot_EPP(lowest_energies, highest_energies, average_energies, c):
    """
    This function will show the EPP (Evolutionary Progress Plot) of this GA run.

    @param lowest_energies: list containing the minimum energy in each generation
    @param highest_energies: list containing the highest energy in each generation
    @param average_energies: list containing the average energy in each generation
    @param c: Configuration
    @return:
    """
    plot_file = os.path.join(os.path.dirname(
        __file__), f'{c.results_dir}/EPP_run_{c.run_id}.png')
    gens = np.arange(0, len(lowest_energies))

    plt.figure(1)
    plt.plot(gens, lowest_energies, 'b', marker='o')
    plt.plot(gens, highest_energies, 'r', marker='o')
    plt.plot(gens, average_energies, 'g', marker='o')
    plt.legend(['min energy', 'max energy', 'avg energy'], loc="upper right")
    plt.title("The lowest, highest, and average energy in each generation")
    plt.xlabel("generations")
    plt.ylabel("energy")
    plt.savefig(plot_file)
    plt.show()

    return 0


def get_mutants(pop, cluster_radius, cluster_size, p_static=0.05, p_dynamic=0.05,
                p_rotation=0.05, p_replacement=0.05, p_mirror=0.05):
    """
    Generates all the mutants for the given population.
    The default probability of each mutation happening to any individual cluster in the population is set to 5%.

    @param pop: population of clusters
    @param cluster_radius: cluster radius used when generating the initial clusters
    @param cluster_size: number of atoms in a cluster
    @param p_static: probability of static displacement mutation happening to a cluster
    @param p_dynamic: probability of dynamic displacement mutation happening to a cluster
    @param p_rotation: probability of a rotation mutation happening to a cluster
    @param p_replacement: probability of a replacement mutation happening to a cluster
    @param p_mirror: probability of a mirror and shift mutation happening to a cluster
    @return: a list of mutated clusters
    """
    mutants = mutators.displacement_static(pop, p_static, cluster_radius)
    mutants += mutators.displacement_dynamic(pop, p_dynamic, cluster_radius)
    mutants += mutators.rotation(pop, p_rotation)
    mutants += mutators.replacement(pop, cluster_size,
                                    cluster_radius, p_replacement)
    mutants += mutators.mirror_shift(pop, cluster_size, p_mirror)

    return mutants


def natural_selection_step(pop, energies, pop_size, dE_thr):
    """
    Applies a natural selection step to the given population.

    @param pop: population of clusters
    @param energies: energies corresponding to each cluster in population
    @param pop_size: maximum population size
    @param dE_thr: minimum energy threshold for clusters with nearly equal energies
    @return: sorted smaller population after natural selection with the corresponding energy and fitness values
    """
    # Sort based on energies, check if not too close (DeltaEnergy) and select popul_size best
    pop_sort_i = np.argsort(energies)

    count = 0
    new_pop = [pop[pop_sort_i[count]]]
    new_energies = [new_pop[0].get_potential_energy()]

    while len(new_pop) < pop_size and count < len(pop_sort_i) - 1:
        count += 1
        sorted_i = pop_sort_i[count]
        candidate = pop[sorted_i]
        cand_energy = energies[sorted_i]

        # Add candidate if energy not too close to one that is already there
        if abs(cand_energy - new_energies[-1]) > dE_thr:
            new_pop.append(candidate)
            new_energies.append(cand_energy)

    # Store newly formed population
    pop = new_pop.copy()
    energies = new_energies.copy()

    return pop, energies


def store_results_database(global_min, local_min, db, c):
    """
    Writes GA results to the database.

    @param global_min: the global minimum cluster
    @param local_min: list of all local minima found
    @param db: the database to write to
    @param c: the configuration information of the GA run
    @return: exit code 0
    """
    db.write(global_min, global_min=True, pop_size=c.pop_size,
             cluster_size=c.cluster_size, max_gens=c.max_gen,
             max_no_success=c.max_no_success, run_id=c.run_id)

    for cluster in local_min:
        db.write(cluster, global_min=False, pop_size=c.pop_size, cluster_size=c.cluster_size, max_gens=c.max_gen,
                 max_no_success=c.max_no_success, run_id=c.run_id)

    return 0


def store_or_reuse_state(reuse=False):
    state_file = "random_state.txt"
    state_file = os.path.join(os.path.dirname(__file__), state_file)

    if not reuse:
        with open(state_file, 'wb+') as f:
            random_state = np.random.get_state()
            pickle.dump(random_state, f)
    else:
        debug("REUSING RANDOM STATE FROM PREVIOUS RUN\n")
        with open(state_file, 'rb') as f:
            reuse_state = pickle.load(f)
            np.random.set_state(reuse_state)


def genetic_algorithm() -> None:
    """
    The main genetic algorithm
    """
    # np.random.seed(241)
    np.seterr(divide='raise')

    # =========================================================================
    # Parameters
    # =========================================================================

    # File to get default configuration / run information
    config_file = "config/ga_config.yaml"

    # Parse terminal input
    c = get_configuration(config_file)

    # Output the run info to stdout
    config_info(c)

    # Either store current np.random state or retrieve state from previous run
    store_or_reuse_state(reuse=c.reuse_state)

    # Lists for EPP plots
    lowest_energies = []
    highest_energies = []
    average_energies = []

    # Start timing
    ga_start_time = time.time()

    # =========================================================================
    # Initial population
    # =========================================================================

    # Generate initial population and optimise locally
    pop = generate_population(c.pop_size, c.cluster_size, c.cluster_radius)
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

    while gen_no_success < c.max_no_success and gen < c.max_gen:
        if time.time() - ga_start_time > c.time_lim:
            debug("REACHED TIME LIMIT")
            break

        # Get fitness values
        pop_fitness = fitness(energies, func=c.fitness_func)
        # Mating - get new population
        children = mating(pop, pop_fitness, c.children_perc, c.mating_method)
        # Mutating - get new mutants
        mutants = get_mutants(pop, c.cluster_radius, c.cluster_size)

        # Local minimisation and add to population
        newborns = children + mutants

        #  debug(f"Local optimisation of {len(newborns)} newborns")
        energies += optimise_local(newborns, c.calc, c.local_optimiser)
        pop += newborns

        # Add new local minima to the list
        local_min += newborns

        # Natural selection
        pop, energies = natural_selection_step(
            pop, energies, c.pop_size, c.dE_thr)

        # Store info about lowest, average, and highest energy of this gen for EPP
        lowest_energies.append(energies[0])
        highest_energies.append(energies[-1])
        average_energies.append(np.mean(energies))

        # Store current best
        if energies[0] < best_min.get_potential_energy():
            debug(f"New global minimum in generation {gen:2d}: ", energies[0])
            best_min = pop[0]
            gen_no_success = 0  # This is success, so set to zero.
        else:
            gen_no_success += 1

        gen += 1

    # Stop timer ga
    ga_time = time.time() - ga_start_time
    print(f"\nGenetic Algorithm took {ga_time:.2f} seconds to execute\n")

    # Store / report
    local_min = process_data.select_local_minima(local_min)
    process_data.print_stats(local_min)

    traj_file_path = os.path.join(os.path.dirname(
        __file__), f"{c.results_dir}/ga_{c.cluster_size}.traj")
    traj_file = Trajectory(traj_file_path, 'w')
    for cluster in local_min:
        traj_file.write(cluster)
    traj_file.close()

    # Connect to database and store results
    db_file = os.path.join(os.path.dirname(__file__),
                           c.results_dir + '/' + c.db_file)
    db = ase.db.connect(db_file)
    store_results_database(local_min[0], local_min[1:], db, c)

    # Show EPP plot if desired
    if c.show_plot:
        plot_EPP(lowest_energies, highest_energies, average_energies, c)

    return 0


if __name__ == '__main__':
    genetic_algorithm()
