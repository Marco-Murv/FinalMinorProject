#!/bin/python3

"""
Genetic algorithm for geometry optimisation of atomic clusters.
We can add more information later.
"""

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.ga.startgenerator import StartGenerator
from ase.optimize import LBFGS

import mating
import mutators


def generate_cluster(cluster_size, radius) -> Atoms:
    """Generate a clus"""

    coords = np.random.uniform(-radius, radius, (cluster_size, 3)).tolist()

    # TODO: Don't use H atoms
    new_cluster = Atoms('H'+str(cluster_size), coords)

    return new_cluster


def generate_population(popul_size, cluster_size, radius):
    """Generate initial population
    """
    return [generate_cluster(cluster_size, radius) for i in range(popul_size)]


def optimise_local(population, calc, optimiser) -> None:
    """Local optimisation of the population
    """
    [atoms.set_calculator(calc) for atoms in population]
    [optimiser(atoms, logfile=None).run() for atoms in population]

    return 


def main() -> None:
    # Parse possible input, otherwise use default parameters
    # Set parameters (change None)
    delta_energy_threshold = 0.1  # TODO: Change this
    local_optimiser = LBFGS
    children_perc = 0.8  # TODO: Change later
    fitness_func = None
    cluster_radius = 2  # Angstroms TODO: Change this
    cluster_size = 3
    popul_size = 2

    # Make local optimisation calculator
    calc = LennardJones(sigma=1.0, epsilon=1.0)  # TODO: Change parameters

    # Generate initial population
    population = generate_population(popul_size, cluster_size, cluster_radius)

    optimise_local(population, calc, local_optimiser)
    # Determine fitness
    population_fitness = None

    # while not done (max iterations / optimum not changing)
    for i in range(10):

        # Mating - get new population
        # children = mating(population, population_fitness, children_perc)
        children = []

        # Mutating (Choose 1 out of 4 mutators)
        # mutants = mutators.FUNCTION_1(population+children, mutation_rate_1)
        # mutants = mutators.FUNCTION_1(population+children, mutation_rate_2)
        mutants = []

        # Local minimisation
        optimise_local(children + mutants, calc, local_optimiser)
        population += children + mutants

        # Natural selection
        population_fitness = None
        # Sort based on fitness, check if not too close (DeltaEnergy) and select popul_size best
        population = population

    # Store / report

    return


if __name__ == '__main__':
    main()

    # atoms = generate_cluster(3, 3)
    # atoms.set_calculator(LennardJones())
    # print(atoms.get_potential_energy())
    # dyn = LBFGS(atoms)
    # dyn.run()
    # print(atoms.get_potential_energy())
