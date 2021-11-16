#!/bin/python3

"""
Genetic algorithm for geometry optimisation of atomic clusters.
We can add more information later.
"""

"""NOTES
    * TODO: Something might be wrong with the fitness values
    * FIXME: Why are the potential energies negative? What is 0?
    * TODO: See other TODOs in the file.
    * TODO: Collect parameter values in another file.
    * TODO: Use less .get_potential_energy() (calculates every time)
"""




import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import LBFGS
from ase.visualize import view
from ase.io import write
import ase.db
from typing import List
from mating import mating
import mutators
import argparse
import yaml
import os
import sys
def debug(*args, **kwargs) -> None:
    """Alias for print() function.
    This can easily be redefined to disable all output.
    """
    print("[DEBUG]: ", *args, **kwargs)


def generate_cluster(cluster_size, radius) -> Atoms:
    """Generate a random cluster with set number of atoms
    The atoms will be placed within a (radius x radius x radius) cube.

    Args:
        cluster_size (int)  : Number of atoms per cluster
        radius (float)      : dimension of the space where atoms can be placed.

    Returns:
        new_cluster (Atoms) : Randomly generated cluster
    """

    coords = np.random.uniform(-radius/2, radius/2, (cluster_size, 3)).tolist()
    # TODO: Can we use "mathematical dots" instead of H-atoms
    new_cluster = Atoms('H'+str(cluster_size), coords)

    return new_cluster


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
    for cluster in population:
        cluster.calc = calc
        try:
            optimiser(cluster, maxstep=0.2, logfile=None).run(steps=50)
        except:  # TODO: how to properly handle these error cases?
            print("FATAL ERROR: DIVISION BY ZERO ENCOUNTERED!")
            sys.exit("PROGRAM ABORTED: FATAL ERROR")

        # TODO: Maybe change steps? This is just a guess

    return [cluster.get_potential_energy() for cluster in population]


def fitness(population, func="exponential") -> np.ndarray:
    """Calculate the fitness of the clusters in the population

    Args:
        population(List[Atoms]) : List of clusters to calculate fitness
        func (str)              : Fitness function
                                    - "exponential" / "linear" / "hyperbolic"
        optimiser (Optimiser)   : ASE Optimiser (e.g. LBFGS)

    Returns:
        (List[Atoms])           : Optimised population
    """
    # Normalise the energies
    energies = np.array([cluster.get_potential_energy()
                        for cluster in population])

    normalised_energies = (energies - np.min(energies)) / \
        (np.max(energies) - np.min(energies))

    if func == "exponential":
        alpha = 3  # TODO: How general is this value? Change?
        return np.exp(- alpha * normalised_energies)

    elif func == "linear":
        return 1 - 0.7 * normalised_energies

    elif func == "hyperbolic":
        return 0.5 * (1 - np.tanh(2 * energies - 1))

    else:
        print(f"'{func}' is not a valid fitness function. Using default")
        return fitness(population)


def parse_args():
    """Parsing the most important parameters
    This will make it easy to run with different values (e.g. on a cluster)
    """
    parser = argparse.ArgumentParser(description='Genetic Algorithm PGGO')
    parser.add_argument('--cluster_size', default=10, type=int,
                        help='Number of atoms per cluster', metavar='')
    parser.add_argument('--pop_size', default=5, type=int,
                        help='Number of clusters in the population', metavar='')
    parser.add_argument('--fitness_func', default="exponential",
                        help='Fitness function', metavar='')
    parser.add_argument('--mating_method', default="roulette",
                        help='Mating Method', metavar='')
    parser.add_argument('--children_perc', default=0.8, type=float,
                        help='Fraction of the population that will have a child', metavar='')
    parser.add_argument('--cluster_radius', default=2.0, type=float,
                        help='Dimension of initial random clusters', metavar='')
    parser.add_argument('--max_no_success', default=10, type=int,
                        help='Consecutive generations allowed without new minimum', metavar='')
    parser.add_argument('--max_gen', default=50, type=int,
                        help='Maximum number of generations', metavar='')
    parser.add_argument('--delta_energy_thr', default=0.01, type=float,
                        help='Minimum difference in energy between clusters (DeltaE threshold)', metavar='')

    args = parser.parse_args()
    return args


def main() -> None:
    # np.random.seed(241)
    np.seterr(divide='raise')

    # Connect to database
    db = ase.db.connect('./genetic_algorithm_results.db')

    # Get configuration of parameters
    config_f_name = "./run_config.yaml"
    with open(config_f_name) as f:
        conf = yaml.safe_load(os.path.expandvars(f.read()))

    cluster_size = conf['cluster_size']
    pop_size = conf['pop_size']
    fitness_func = conf['fitness_func']
    mating_method = conf['mating_method']
    children_perc = conf['children_perc']
    cluster_radius = conf['cluster_radius']
    max_no_success = conf['max_no_success']
    max_gen = conf['max_gen']
    delta_energy_thr = conf['delta_energy_thr']
    run_id = conf['run_id']

    # Make local optimisation Optimiser and calculator
    calc = LennardJones(sigma=1.0, epsilon=1.0)  # TODO: Change parameters
    local_optimiser = LBFGS

    # Generate initial population and optimise locally
    population = generate_population(
        pop_size, cluster_size, cluster_radius)
    energies = optimise_local(population, calc, local_optimiser)

    # Determine fitness
    population_fitness = fitness(population, fitness_func)

    # Keep track of global minima. Initialised with random cluster
    best_minima = [population[0]]
    local_minima = [population[0]]
    energies_minima = np.array(population[0].get_potential_energy())

    # Keep track of iterations
    gen = 0
    gen_no_success = 0

    while gen_no_success < max_no_success and gen < max_gen:
        debug(f"Generation {gen:2d} - Population size = {len(population)}")

        # Mating - get new population
        children = mating(population, population_fitness, children_perc, mating_method)

        # Mutating (Choose 1 out of 4 mutators)
        mutants = mutators.displacement_static(population, 0.05, cluster_radius)
        mutants += mutators.displacement_dynamic(population, 0.05, cluster_radius)
        mutants += mutators.rotation(population, 0.05)
        mutants += mutators.replacement(population, cluster_size, cluster_radius, 0.05)
        mutants += mutators.mirror_shift(population, cluster_size, 0.05)

        # Local minimisation and add to population
        newborns = children + mutants



        energies += optimise_local(newborns, calc, local_optimiser)

        for i in range(len(newborns)):
            if not np.any(np.isclose(energies_minima, energies[-(i+1)], atol=delta_energy_thr)):
                local_minima.append(newborns[i])
                energies_minima = np.append(energies_minima, energies[i])

        population += newborns

        # Natural selection
        population_fitness = fitness(population, fitness_func)

        # Sort based on fitness, check if not too close (DeltaEnergy)
        # and select popul_size best
        pop_sort_i = np.argsort(-population_fitness)

        pop_i = 0
        new_population = [population[pop_sort_i[pop_i]]]
        new_energies = [new_population[0].get_potential_energy()]

        while len(new_population) < pop_size and pop_i < len(pop_sort_i)-1:
            pop_i += 1
            candidate = population[pop_sort_i[pop_i]]

            if abs(candidate.get_potential_energy() - new_energies[-1]) > delta_energy_thr:
                new_population.append(candidate)
                new_energies.append(candidate.get_potential_energy())

        population = new_population.copy()
        energies = new_energies.copy()


        # Store current best
        if energies[0] < best_minima[-1].get_potential_energy():
            best_minima.append(population[0]) 
            debug("New global minimum: ", energies[0])

            gen_no_success = 0

        else:
            gen_no_success += 1

        gen += 1

    # Store / report
    debug("All the minima we found:")
    debug([cluster.get_potential_energy() for cluster in best_minima])

    for cluster in local_minima:
        global_min = False
        if cluster == best_minima[-1]:
            global_min = True

        last_id = db.write(cluster, global_min=global_min,
                                pop_size=pop_size,
                                cluster_size=cluster_size, max_gens=max_gen,
                                max_no_success=max_no_success, run_id=run_id)

    # How to retrieve atoms:
    # atom_db = db.get(natoms=cluster_size, pop_size=10, ...).toatoms()
    #  view(best_minima[-1])

    conf['run_id'] += 1
    with open(config_f_name, 'w') as f:
        yaml.dump(conf, f)

    return


if __name__ == '__main__':
    main()
