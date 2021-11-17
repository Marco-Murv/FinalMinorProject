#!/bin/python3

"""
Distributed parallelisation of the Genetic Algorithm
"""

import genetic_algorithm as ga

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def organiser(c):

    # Provide file name
    db_file = "genetic_algorithm_results.db"


    # Create initial population
    pop = ga.generate_population(c.pop_size)
    
    # Broadcast initial population
    # Also distribute how large their sub-population should be 


    # Receive coordinates and energies

    # Perform natural selection


    # Initiate lists for local minima and their energies

    pass

def worker(c):
    # Receive complete population coordinates
    pop = None
    energies = None

    # Mating - get new population
        children = ga.mating(pop, c.pop_fitness, c.children_perc, c.mating_method)

        # Mutating (Choose 1 out of 4 mutators)
        mutants = ga.mutators.displacement_static(pop, 0.05, c.cluster_radius)
        mutants += ga.mutators.displacement_dynamic(pop, 0.05, c.cluster_radius)
        mutants += ga.mutators.rotation(pop, 0.05)
        mutants += ga.mutators.replacement(pop,
                                        c.cluster_size, c.cluster_radius, 0.05)
        mutants += ga.mutators.mirror_shift(pop, c.cluster_size, 0.05)

        # Local minimisation and add to population
        newborns = children + mutants

        energies += ga.optimise_local(newborns, c.calc, c.local_optimiser)

        for i in range(len(newborns)):
            too_close = np.isclose(
                energies_min, energies[-(i + 1)], atol=c.dE_thr)
            if not np.any(too_close):
                local_min.append(newborns[i])
                energies_min = np.append(energies_min, energies[i])

        pop += newborns

        # Natural selection

        pop_fitness = ga.fitness(energies, c.fitness_func)
    pass

def ga_distributed():
    np.seterr(divide='raise')

    # File to get default configuration / run information
    config_file = "run_config.yaml"

    c = ga.get_configuration(config_file)

    if rank == 0:
        organiser(c)

    else:
        worker(c)

    return 0



if __name__ == "__main__":
    ga_distributed()