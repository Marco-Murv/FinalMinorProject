#!/bin/python3

"""
Sub populations parallelisation of the Genetic Algorithm.

A number of separate GA's will be run on multiple processors, with each processor having their own small, independent
sub-population. The different processors can exchange pairs of clusters with each other to maintain population
diversity.
"""

import os
import ase
import numpy as np

from genetic_algorithm import config_info, debug, fitness, generate_population
from genetic_algorithm import get_configuration, natural_selection_step
from genetic_algorithm import optimise_local, fitness, get_mutants
from genetic_algorithm import store_local_minima, store_results_database
from mating import mating
from ga_distributed import flatten_list
from mpi4py import MPI


def cluster_exchange(pop, comm, rank, num_procs):
    return


def ga_sub_populations():
    # Raise real errors when numpy encounters a division by zero.
    np.seterr(divide='raise')

    # Initialising MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    # File to get default configuration / run information
    config_file = "run_config.yaml"

    # Parse possible terminal input and yaml file.
    c = get_configuration(config_file)

    # =========================================================================
    # Initial population and variables
    # =========================================================================
    # TODO: good sub-population size? Atm normal population size divided by number of processes.
    sub_pop_size = np.ceil(c.pop_size / num_procs)

    pop = generate_population(c.pop_size, c.cluster_size, c.cluster_radius)
    energies = optimise_local(pop, c.calc, c.local_optimiser)

    # Keep track of global minima. Initialised with random cluster
    best_min = [pop[0]] # TODO: remove first entries at the end before storing prob?
    local_min = [pop[0]]
    energies_min = np.array(pop[0].get_potential_energy())

    # =========================================================================
    # Main loop
    # =========================================================================
    # Keep track of iterations
    gen = 0
    gen_no_success = 0

    while gen_no_success < c.max_no_success and gen < c.max_gen:

        # Get fitness values
        pop_fitness = fitness(energies, func=c.fitness_func)
        # Mating - get new population
        children = mating(pop, pop_fitness, c.children_perc, c.mating_method)
        # Mutating - get new mutants
        mutants = get_mutants(pop, c.cluster_radius, c.cluster_size)

        # Local minimisation and add to population
        newborns = children + mutants
        energies += optimise_local(newborns, c.calc, c.local_optimiser)
        pop += newborns

        # Keep track of new local minima
        local_min, energies_min = store_local_minima(newborns, energies, local_min, energies_min, c.dE_thr)

        # Natural selection
        pop, energies = natural_selection_step(pop, energies, c.pop_size, c.dE_thr, c.fitness_func)

        # TODO: condition for cluster exchanges?

        # Store current best
        if energies[0] < best_min[-1].get_potential_energy():
            debug("New global minimum: ", energies[0])
            best_min.append(pop[0])
            gen_no_success = 0  # This is success, so set to zero.
        else:
            gen_no_success += 1

        gen += 1

    # =========================================================================
    # Combine all results and store them
    # =========================================================================
    if rank == 0:
        # Combine results
        best_min = comm.gather(best_min, root=0)
        local_min = comm.gather(local_min, root=0)
        best_min = flatten_list(best_min)
        local_min = flatten_list(local_min)

        # TODO: remove duplicates and find the GM

        # Connect to database
        db_file = "genetic_algorithm_results.db"
        db_file = os.path.join(os.path.dirname(__file__), db_file)
        db = ase.db.connect(db_file)
        store_results_database(best_min[-1], local_min, db, c)

    return 0


if __name__ == "__main__":
    ga_sub_populations()