#!/bin/python3

"""
Distributed parallelisation of the Genetic Algorithm
"""

"""
TODO:
    * Solve cluster size problem -> mutators
    * Fix population length problem
    
"""

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import ase

import process_data
import genetic_algorithm as ga
from mating import mating

from genetic_algorithm import config_info, debug, fitness, generate_population
from genetic_algorithm import get_configuration, natural_selection_step
from genetic_algorithm import optimise_local, fitness, get_mutants
from genetic_algorithm import store_results_database

import numpy as np
from mpi4py import MPI


def flatten_list(lst):
    """
    Convert 2d list to 1d list

    @param lst: 2d list to be flattened
    @return: float 1d list
    """
    return [item for sublst in lst for item in sublst]


def ga_distributed():
    """
    Main genetic algorithm (distributed)
    """
    np.seterr(divide='raise')

    # Initialising MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    # File to get default configuration / run information
    config_file = "run_config.yaml"

    # Parse possible terminal input and yaml file.
    c = get_configuration(config_file)

    # Start timer
    if rank == 0:
        ga_start_time = MPI.Wtime()

    # =========================================================================
    # Initial population
    # =========================================================================
    # Initialise variables
    pop = None
    energies = None

    if rank == 0:
        config_info(c)

        # Create initial population
        pop = generate_population(c.pop_size, c.cluster_size, c.cluster_radius)

        # FIX: Make this parallel already
        energies = optimise_local(pop, c.calc, c.local_optimiser)

        # Keep track of global minima. Initialised with random cluster
        best_min = [pop[0]]
        local_min = [pop[0]]

    # =========================================================================
    # Main loop
    # =========================================================================
    # Keep track of iterations
    gen = 0
    gen_no_success = 0
    done = False

    while not done:
        if rank == 0:
            debug(f"Generation {gen:2d} - Population size = {len(pop)}")

        # Broadcast initial population
        pop = comm.bcast(pop, root=0)
        energies = comm.bcast(energies, root=0) # TODO: use Bcast instead?

        # Mating - get new population
        pop_fitness = fitness(energies, c.fitness_func)
        children = mating(pop, pop_fitness, c.children_perc /
                          num_procs, c.mating_method)

        # Define sub-populaiton on every rank (only for mutating)
        chunk = len(pop) // num_procs  # TODO:
        sub_pop = pop[rank * chunk:(rank + 1) * chunk] or pop[-2:]


        # Mutating - get new mutants
        mutants = get_mutants(sub_pop, c.cluster_radius, c.cluster_size)

        # Local minimisation and add to population
        newborns = children + mutants
        new_energies = optimise_local(newborns, c.calc, c.local_optimiser)

        newborns = comm.gather(newborns, root=0)
        new_energies = comm.gather(new_energies, root=0)

        if rank == 0:
            newborns = flatten_list(newborns)
            new_energies = flatten_list(new_energies)

            pop += newborns
            energies.extend(new_energies)

            # Add new local minima to the list
            local_min += newborns

            # Natural Selection
            pop, energies = natural_selection_step(pop, energies, c.pop_size,
                                                   c.dE_thr)

            # Store current best
            if energies[0] < best_min[-1].get_potential_energy():
                debug("New global minimum: ", energies[0])
                best_min.append(pop[0])
                gen_no_success = 0  # This is success, so set to zero.
            else:
                gen_no_success += 1

            gen += 1

            # Check if done
            done = gen_no_success > c.max_no_success or gen > c.max_gen

        gen = comm.bcast(gen, root=0)
        gen_no_success = comm.bcast(gen_no_success, root=0)
        done = comm.bcast(done, root=0)

    if rank == 0:
        # Stop timer
        ga_time = MPI.Wtime() - ga_start_time
        print(f"\nga_distributed took {ga_time} seconds to execute")

        # Process and report local minima
        local_min = process_data.select_local_minima(local_min)
        process_data.print_stats(local_min)

        # Store in Database
        db_file = os.path.join(os.path.dirname(__file__), c.db_file)
        db = ase.db.connect(db_file)
        store_results_database(best_min[-1], local_min, db, c)

    return 0


if __name__ == "__main__":

    ga_distributed()
