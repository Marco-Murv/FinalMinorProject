#!/bin/python3

"""
Genetic algorithm for geometry optimisation of atomic clusters.
We can add more information later.

This program requires a file called `run_config.yaml` in the same directory.
Example run_config.yaml:
```yaml
    children_perc: 0.8
    cluster_radius: 2.0
    cluster_size: 5
    delta_energy_thr: 0.001
    fitness_func: exponential
    mating_method: roulette
    max_gen: 50
    max_no_success: 5
    pop_size: 12
    run_id: 1
```
"""

import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import ase.db
import mutators
import argparse
import ase.db

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import LBFGS
from ase.visualize import view
from ase.io import write
from typing import List
from mating import mating
from datetime import datetime as dt
from dataclasses import dataclass


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
    print(f"| {f'Genetic Algorithm':{n}s}|")
    print(" ================================================================ ")
    print(f"| {f'Timestamp          : {timestamp}':{n}s}|")
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
    fitness_func: str = None
    mating_method: str = None
    children_perc: float = None
    cluster_radius: float = None
    max_no_success: int = None
    max_gen: int = None
    dE_thr: float = None
    run_id: int = None
    calc = LennardJones(sigma=1.0, epsilon=1.0)  # TODO: Change parameters
    local_optimiser = LBFGS


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
    parser.add_argument('--fitness_func', metavar='',
                        help='Fitness function')
    parser.add_argument('--mating_method', metavar='',
                        help='Mating Method')
    parser.add_argument('--children_perc', type=float, metavar='',
                        help='Fraction of opulation that will have a child')
    parser.add_argument('--cluster_radius', default=2.0, type=float, metavar='',
                        help='Dimension of initial random clusters')
    parser.add_argument('--max_no_success', default=10, type=int, metavar='',
                        help='Consecutive generations without new minimum')
    parser.add_argument('--max_gen', type=int, metavar='',
                        help='Maximum number of generations')
    parser.add_argument('--delta_energy_thr', type=float, metavar='',
                        help='Minimum difference in energy between clusters')
    parser.add_argument('--run_id', type=int, metavar='',
                        help="ID for the current run. Increments automatically")

    p = parser.parse_args()

    c = Config()
    # Set variables to terminal input if possible, otherwise use config file
    c.cluster_size = p.cluster_size or yaml_conf['cluster_size']
    c.pop_size = p.pop_size or yaml_conf['pop_size']
    c.fitness_func = p.fitness_func or yaml_conf['fitness_func']
    c.mating_method = p.mating_method or yaml_conf['mating_method']
    c.children_perc = p.children_perc or yaml_conf['children_perc']
    c.cluster_radius = p.cluster_radius or yaml_conf['cluster_radius']
    c.max_no_success = p.max_no_success or yaml_conf['max_no_success']
    c.max_gen = p.max_gen or yaml_conf['max_gen']
    c.dE_thr = p.delta_energy_thr or yaml_conf['delta_energy_thr']
    c.run_id = p.run_id or yaml_conf['run_id']

    # Increment run_id for next run
    yaml_conf['run_id'] += 1
    with open(config_file, 'w') as f:
        yaml.dump(yaml_conf, f)

    return c

    calc = LennardJones(sigma=1.0, epsilon=1.0)
    local_optimiser = LBFGS


def generate_cluster(cluster_size, radius) -> Atoms:
    """
    Generate a random cluster with set number of atoms
    The atoms will be placed within a (radius x radius x radius) cube.

    @param cluster_size: number of atoms per cluster
    @param radius: dimension of the space where atoms can be placed.
    @return: new random cluster
    """
    coords = np.random.uniform(-radius / 2, radius / 2,
                               (cluster_size, 3)).tolist()

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
    for cluster in population:
        cluster.calc = calc
        try:
            optimiser(cluster, maxstep=0.2, logfile=None).run(steps=50)
        except:  # delete cluster from population if invalid energy value
            population.remove(cluster)
            debug("DIVIDE BY ZERO REMOVED FROM POPULATION!")

    return [cluster.get_potential_energy() for cluster in population]


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


def get_configuration(config_file):
    """
    Set the parameters for this run.

    @param config_file: Filename to the yaml configuration
    @type config_file: str
    @return: object with all the configuration parameters
    @rtype: Config
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
    parser.add_argument('--cluster_radius', default=2.0, type=float, metavar='',
                        help='Dimension of initial random clusters')
    parser.add_argument('--max_no_success', default=10, type=int, metavar='',
                        help='Consecutive generations without new minimum')
    parser.add_argument('--max_gen', type=int, metavar='',
                        help='Maximum number of generations')
    parser.add_argument('--delta_energy_thr', type=float, metavar='',
                        help='Minimum difference in energy between clusters')
    parser.add_argument('--run_id', type=int, metavar='',
                        help="ID for the current run. Increments automatically")

    p = parser.parse_args()

    c = Config()
    # Set variables to terminal input if possible, otherwise use config file
    c.cluster_size = p.cluster_size or yaml_conf['cluster_size']
    c.pop_size = p.pop_size or yaml_conf['pop_size']
    c.fitness_func = p.fitness_func or yaml_conf['fitness_func']
    c.mating_method = p.mating_method or yaml_conf['mating_method']
    c.children_perc = p.children_perc or yaml_conf['children_perc']
    c.cluster_radius = p.cluster_radius or yaml_conf['cluster_radius']
    c.max_no_success = p.max_no_success or yaml_conf['max_no_success']
    c.max_gen = p.max_gen or yaml_conf['max_gen']
    c.dE_thr = p.delta_energy_thr or yaml_conf['delta_energy_thr']
    c.run_id = p.run_id or yaml_conf['run_id']

    # Increment run_id for next run
    yaml_conf['run_id'] += 1
    with open(config_file, 'w') as f:
        yaml.dump(yaml_conf, f)

    return c


def plot_EPP(lowest_energies, highest_energies, average_energies, run_id):
    """
    This function will show the EPP (Evolutionary Progress Plot) of this GA run.

    @param lowest_energies: list containing the minimum energy in each generation
    @param highest_energies: list containing the highest energy in each generation
    @param average_energies: list containing the average energy in each generation
    @param run_id: the id of this GA run
    @return:
    """
    plot_file = os.path.join(os.path.dirname(__file__), f'EPP_run_{run_id}.png')
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
    mutants += mutators.replacement(pop, cluster_size, cluster_radius, p_replacement)
    mutants += mutators.mirror_shift(pop, cluster_size, p_mirror)

    return mutants


def natural_selection_step(pop, energies, pop_size, dE_thr, fitness_func):
    """
    Applies a natural selection step to the given population.

    @param pop: population of clusters
    @param energies: energies corresponding to each cluster in population
    @param pop_size: maximum population size
    @param dE_thr: minimum energy threshold for clusters with nearly equal energies
    @param fitness_func: function used for calculating fitness values
    @return: smaller population after natural selection with the corresponding energy and fitness values
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

        if abs(cand_energy - new_energies[-1]) > dE_thr:
            new_pop.append(candidate)
            new_energies.append(cand_energy)

    # Store newly formed population
    pop = new_pop.copy()
    energies = new_energies.copy()

    return pop, energies


def store_local_minima(newborns, energies, local_min, energies_min, dE_thr):
    """
    Stores the new local minima from the newborns list into local_min list if they differ sufficiently from
    previously found local minima.

    @param newborns: list of new cluster with new local minima
    @param energies: energies of the newborns clusters
    @param local_min: previously found local minima clusters
    @param energies_min: energies of previously found local minima
    @param dE_thr: minimum energy threshold for different local minima
    @return: updated list containing new local minima and the corresponding energies
    """

    for i in range(len(newborns)):
        too_close = np.isclose(energies_min, energies[-(i + 1)], atol=dE_thr)
        if not np.any(too_close):
            local_min.append(newborns[i])
            energies_min = np.append(energies_min, energies[i])

    return local_min, energies_min


def store_results_database(global_min, local_min, db, c):
    """
    Writes GA results to the database.

    @param global_min: the global minimum cluster
    @param local_min: list of all local minima found
    @param db: the database to write to
    @param c: the configuration information of the GA run
    @return: exit code 0
    """

    for cluster in local_min:
        is_global_min = False
        if cluster == global_min:
            is_global_min = True

        last_id = db.write(cluster, global_min=is_global_min, pop_size=c.pop_size,
                           cluster_size=c.cluster_size, max_gens=c.max_gen,
                           max_no_success=c.max_no_success, run_id=c.run_id)

    return 0


def genetic_algorithm() -> None:
    """
    The main genetic algorithm
    """
    np.random.seed(241)
    np.seterr(divide='raise')

    # Provide file name
    db_file = "genetic_algorithm_results.db"

    # File to get default configuration / run information
    config_file = "run_config.yaml"

    # =========================================================================
    # Parameters and database
    # =========================================================================

    # Connect to database
    db_file = os.path.join(os.path.dirname(__file__), db_file)
    db = ase.db.connect(db_file)

    # Parse terminal input
    c = get_configuration(config_file)

    # Output the run info to stdout
    config_info(c)

    # Lists for EPP plots
    show_EPP = True
    lowest_energies = []
    highest_energies = []
    average_energies = []

    # =========================================================================
    # Initial population
    # =========================================================================

    # Generate initial population and optimise locally
    pop = generate_population(c.pop_size, c.cluster_size, c.cluster_radius)
    energies = optimise_local(pop, c.calc, c.local_optimiser)

    # Keep track of global minima. Initialised with random cluster
    best_min = [pop[0]]
    local_min = [pop[0]]
    energies_min = np.array(pop[0].get_potential_energy())

    # =========================================================================
    # Main loop
    # =========================================================================
    # Keep track of iterations
    gen = 0
    gen_no_success = 0

    while gen_no_success < c.max_no_success and gen < c.max_gen:
        debug(f"Generation {gen:2d} - Population size = {len(pop)}")

        # Get fitness values
        pop_fitness = fitness(energies, func=c.fitness_func)
        # Mating - get new population
        children = mating(pop, pop_fitness, c.children_perc, c.mating_method)
        # Mutating - get new mutants
        mutants = get_mutants(pop, c.cluster_radius, c.cluster_size)

        # Local minimisation and add to population
        newborns = children + mutants

        debug(f"Local optimisation of {len(newborns)} newborns")
        energies += optimise_local(newborns, c.calc, c.local_optimiser)
        pop += newborns

        # Keep track of new local minima
        local_min, energies_min = store_local_minima(newborns, energies, local_min, energies_min, c.dE_thr)

        # Natural selection
        pop, energies = natural_selection_step(pop, energies, c.pop_size, c.dE_thr, c.fitness_func)

        # Store info about lowest, average, and highest energy of this gen for EPP
        lowest_energies.append(energies[0])
        highest_energies.append(energies[-1])
        average_energies.append(np.mean(energies))

        # Store current best
        if energies[0] < best_min[-1].get_potential_energy():
            debug("New global minimum: ", energies[0])
            best_min.append(pop[0])
            gen_no_success = 0  # This is success, so set to zero.
        else:
            gen_no_success += 1

        gen += 1

    # Store / report
    debug(f"Found {len(local_min)} local minima in total.")
    debug("The evolution of the global minimum:")
    debug([cluster.get_potential_energy() for cluster in best_min])

    store_results_database(best_min[-1], local_min, db, c)

    # Show EPP plot if desired
    if show_EPP:
        plot_EPP(lowest_energies, highest_energies, average_energies, c.run_id)

    return 0


if __name__ == '__main__':
    genetic_algorithm()
