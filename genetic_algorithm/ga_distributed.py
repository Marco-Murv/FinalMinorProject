#!/bin/python3

"""
Distributed parallelisation of the Genetic Algorithm
"""

import genetic_algorithm as ga

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def organiser():

    # Provide file name
    db_file = "genetic_algorithm_results.db"

    # File to get default configuration / run information
    config_file = "run_config.yaml"

    c = ga.get_configuration(config_file)
    # Create initial population
    pop = ga.generate_population(c.pop_size)
    # Broadcast initial population

    # Initiate lists for local minima and their energies

    pass

def worker():
    pass

def ga_distributed():
    np.seterr(divide='raise')

    if rank == 0:
        organiser()

    else:
        worker()

    return 0



if __name__ == "__main__":
    ga_distributed()