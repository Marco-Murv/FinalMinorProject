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

"""NOTES
    * TODO: See other TODOs in the file.
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

    :param cluster_size: number of atoms per cluster
    :param radius: dimension of the space where atoms can be placed.
    :returns: -> new random cluster

    """

    coords = np.random.uniform(-radius/2, radius/2, (cluster_size, 3)).tolist()

    # TODO: Can we use "mathematical dots" instead of H-atoms
    new_cluster = Atoms('H'+str(cluster_size), coords)

    return new_cluster


def generate_population(popul_size, cluster_size, radius) -> List[Atoms]:
    """Generate initial population.

    :param popul_size: number of clusters in the population
    :param cluster_size: number of atoms in each cluster
    :param radius: dimension of the initial random clusters
    :returns: -> List of clusters

    """
    return [generate_cluster(cluster_size, radius) for i in range(popul_size)]


def optimise_local(population, calc, optimiser) -> List[Atoms]:
    """Local optimisation of the population. The clusters in the population
    are optimised and can be used after this function is called. Moreover,
    calculate and return the final optimised potential energy of the clusters.

    :param population: List of clusters to be locally optimised
    :param calc: ASE Calculator for potential energy (e.g. LJ)
    :param optimiser: ASE Optimiser (e.g. LBFGS)
    :returns: -> Optimised population

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

    :param population: List of clusters to calculate fitness
    :param func: Fitness function ("exponential" / "linear" / "hyperbolic")
    :param optimiser: ASE Optimiser (e.g. LBFGS)
    :returns: -> Optimised population

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
                        help='Fraction of opulation that will have a child',
                        metavar='')
    parser.add_argument('--cluster_radius', default=2.0, type=float,
                        help='Dimension of initial random clusters', metavar='')
    parser.add_argument('--max_no_success', default=10, type=int,
                        help='Consecutive generations without new minimum',
                        metavar='')
    parser.add_argument('--max_gen', default=50, type=int,
                        help='Maximum number of generations', metavar='')
    parser.add_argument('--delta_energy_thr', default=0.01, type=float,
                        help='Minimum difference in energy between clusters (DeltaE threshold)', metavar='')

    args = parser.parse_args()
    return args


def main() -> None:
    """The main genetic algorithm 

    """
    # np.random.seed(241)
    np.seterr(divide='raise')

    # Provide file name
    db_file = "genetic_algorithm_results.db"
    config_file = "run_config.yaml"

    # Connect to database and get parameters
    db_file = os.path.join(os.path.dirname(__file__), db_file)
    db = ase.db.connect('./genetic_algorithm_results.db')

    config_file = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_file) as f:
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
    pop = generate_population(
        pop_size, cluster_size, cluster_radius)
    energies = optimise_local(pop, calc, local_optimiser)

    # Determine fitness
    pop_fitness = fitness(pop, fitness_func)

    # Keep track of global minima. Initialised with random cluster
    best_min = [pop[0]]
    local_min = [pop[0]]
    energies_min = np.array(pop[0].get_potential_energy())

    # Keep track of iterations
    gen = 0
    gen_no_success = 0

    while gen_no_success < max_no_success and gen < max_gen:
        debug(f"Generation {gen:2d} - Population size = {len(pop)}")

        # Mating - get new population
        children = mating(pop, pop_fitness, children_perc, mating_method)

        # Mutating (Choose 1 out of 4 mutators)
        mutants = mutators.displacement_static(pop, 0.05, cluster_radius)
        mutants += mutators.displacement_dynamic(pop, 0.05, cluster_radius)
        mutants += mutators.rotation(pop, 0.05)
        mutants += mutators.replacement(pop,cluster_size, cluster_radius, 0.05)
        mutants += mutators.mirror_shift(pop, cluster_size, 0.05)

        # Local minimisation and add to population
        newborns = children + mutants

        energies += optimise_local(newborns, calc, local_optimiser)

        for i in range(len(newborns)):
            too_close = np.isclose(
                energies_min, energies[-(i+1)], atol=delta_energy_thr)
            if not np.any(too_close):
                local_min.append(newborns[i])
                energies_min = np.append(energies_min, energies[i])

        pop += newborns

        # Natural selection
        pop_fitness = fitness(pop, fitness_func)

        # Sort based on fitness, check if not too close (DeltaEnergy)
        # and select popul_size best
        pop_sort_i = np.argsort(-pop_fitness)

        count = 0
        new_pop = [pop[pop_sort_i[count]]]
        new_energies = [new_pop[0].get_potential_energy()]

        while len(new_pop) < pop_size and count < len(pop_sort_i)-1:
            count += 1
            sorted_i = pop_sort_i[count]
            candidate = pop[sorted_i]
            cand_energy = energies[sorted_i]

            if abs(cand_energy - new_energies[-1]) > delta_energy_thr:
                new_pop.append(candidate)
                new_energies.append(cand_energy)

        # Store newly formed population
        pop = new_pop.copy()
        energies = new_energies.copy()

        # Store current best
        if energies[0] < best_min[-1].get_potential_energy():
            best_min.append(pop[0])
            debug("New global minimum: ", energies[0])

            gen_no_success = 0 # This is success, so set to zero.

        else:
            gen_no_success += 1

        gen += 1

    # Store / report
    debug(f"Found {len(local_min)} local minima. in total")
    debug("The evolution of the global minimum:")
    debug([cluster.get_potential_energy() for cluster in best_min])

    for cluster in local_min:
        global_min = False
        if cluster == best_min[-1]:
            global_min = True

        last_id = db.write(cluster, global_min=global_min,
                           pop_size=pop_size,
                           cluster_size=cluster_size, max_gens=max_gen,
                           max_no_success=max_no_success, run_id=run_id)

    # How to retrieve atoms:
    # atom_db = db.get(natoms=cluster_size, pop_size=10, ...).toatoms()
    #  view(best_minima[-1])

    conf['run_id'] += 1
    with open(config_file, 'w') as f:
        yaml.dump(conf, f)

    return


if __name__ == '__main__':
    main()
