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
    comm.bcast() 
    comm.bcast()
    
    # Also distribute how large their sub-population should be 


    # Receive coordinates and energies

    # Perform natural selection


    # Initiate lists for local minima and their energies

    pass

def worker(c):
    # Receive complete population coordinates
    pop = comm.recv()
    energies = comm.recv()

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
    new_energies = ga.optimise_local(newborns, c.calc, c.local_optimiser)

    # Send back newborns and new_energies

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