#!/bin/python3

"""
Genetic algorithm for geometry optimisation of atomic clusters.
We can add more information later.
"""

"""NOTES
    * TODO: Refactor "atoms" -> "cluster"
    * TODO: Write to a file to keep everything we run
    * TODO: Something might be wrong with the fitness values
    * FIXME: Why are the potential energies negative? What is 0?
    * TODO: See other TODOs in the file.
    * TODO: Collect parameter values in another file.
"""


import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.ga.startgenerator import StartGenerator
from ase.optimize import LBFGS
from typing import List
from mating import mating
import mutators
import time


def debug(*args, **kwargs):
    print(*args, **kwargs)


def generate_cluster(cluster_size, radius) -> Atoms:
    """Generate a random cluster with set number of atoms
    The atoms will be placed within a (radius x radius x radius) cube.
    """

    coords = np.random.uniform(-radius/2, radius/2, (cluster_size, 3)).tolist()
    # TODO: Don't use H atoms
    new_cluster = Atoms('H'+str(cluster_size), coords)

    return new_cluster


def generate_population(popul_size, cluster_size, radius) -> List[Atoms]:
    """Generate initial population
    """
    return [generate_cluster(cluster_size, radius) for i in range(popul_size)]


def optimise_local(population, calc, optimiser) -> None:
    """Local optimisation of the population
    """

    for atoms in population:
        atoms.set_calculator(calc)
        optimiser(atoms, logfile=None).run()

    return


def fitness(population, func="exponential") -> np.ndarray:
    """Calculate the fitness of the clusters in the population
    """
    # Normalise the energies
    energy = np.array([atoms.get_potential_energy() for atoms in population])

    normalised_energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))

    if func == "exponential":
        alpha = 3  # TODO: How general is this value?
        return np.exp(- alpha * normalised_energy)

    elif func == "linear":
        return 1 - 0.7 * normalised_energy

    elif func == "hyperbolic":
        return 0.5 * (1 - np.tanh(2 * energy - 1))

    else:
        print(f"'{func}' is not a valid fitness function. Using default")
        return fitness(population)


def main() -> None:
    # TODO: REMOVE THIS
    #  np.random.seed(52) # FIXME: Problem
    np.random.seed(63)
    #  np.random.seed(82) # FIXME: Problem (Different)

    # Parse possible input, otherwise use default parameters
    # Set parameters (change None)
    delta_energy_threshold = 0.1  # TODO: Change this
    fitness_func = "exponential"
    mating_method = "roulette"
    local_optimiser = LBFGS
    children_perc = 0.8  # TODO: Change later
    cluster_radius = 2  # [Angstroms] TODO: Change this
    cluster_size = 3
    popul_size = 5

    max_no_success = 5  # TODO: Change
    max_gen = 10  # TODO: Change

    # Make local optimisation calculator
    calc = LennardJones(sigma=1.0, epsilon=1.0)  # TODO: Change parameters

    # Generate initial population
    population = generate_population(popul_size, cluster_size, cluster_radius)

    print(population)
    optimise_local(population, calc, local_optimiser)

    # Determine fitness
    population_fitness = fitness(population, fitness_func)

    best_minima = [population[0]]  # Initialised with random cluster

    gen = 0
    gen_no_success = 0

    while gen_no_success < max_no_success and gen < max_gen:
        debug(f"Generation {gen}")

        # Mating - get new population
        children = mating(population, population_fitness, children_perc, mating_method)

        # Mutating (Choose 1 out of 4 mutators)
        # mutants = mutators.FUNCTION_1(population+children, mutation_rate_1)
        # mutants = mutators.FUNCTION_1(population+children, mutation_rate_2)
        mutants = []

        # Local minimisation
        optimise_local(children + mutants, calc, local_optimiser)
        population +=  children + mutants

        # Natural selection
        population_fitness = fitness(population, fitness_func)

        # Sort based on fitness, check if not too close (DeltaEnergy)
        # and select popul_size best
        pop_sort_i = np.argsort(population_fitness)
        population_fitness = np.take_along_axis(population_fitness, pop_sort_i, axis=0)
        population = [population[i] for i in pop_sort_i]


        #  population = [cluster for _, cluster in sorted(zip(population_fitness, population))]
        #  population_fitness = sorted(population_fitness)

        # Store current best
        if population[-1].get_potential_energy() < best_minima[-1].get_potential_energy():
            best_minima.append(population[-1])
            debug("New local minimum: ", population[-1].get_potential_energy())
            gen_no_success = 0
        else:
            gen_no_success += 1

        gen += 1

        # Store current best

    # Store / report
    debug("All the minima we found:")
    debug([cluster.get_potential_energy() for cluster in best_minima])

    return


if __name__ == '__main__':
    main()
