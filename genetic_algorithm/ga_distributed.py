#!/bin/python3

"""
Distributed parallelisation of the Genetic Algorithm
"""

import os

import ase
import genetic_algorithm as ga
from genetic_algorithm import config_info, debug, fitness, generate_population
from genetic_algorithm import get_configuration, natural_selection_step
from genetic_algorithm import optimise_local, fitness, get_mutants
from genetic_algorithm import store_local_minima, store_results_database
from mating import mating

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = comm.Get_size()


def flatten_list(lst):
    return [item for sublst in lst for item in sublst]


def ga_distributed():
    np.seterr(divide='raise')

    # File to get default configuration / run information
    config_file = "run_config.yaml"

    c = get_configuration(config_file)

    # Initialise variables (for parallel run)
    pop = None
    pop_fitness = None

    if rank == 0:
        config_info(c)

        # Provide file name
        db_file = "genetic_algorithm_results.db"

        # Connect to database
        db_file = os.path.join(os.path.dirname(__file__), db_file)
        db = ase.db.connect(db_file)

        # Create initial population
        pop = generate_population(c.pop_size, c.cluster_size, c.cluster_radius)

        # TODO: Make this parallel already
        energies = optimise_local(pop, c.calc, c.local_optimiser)
        pop_fitness = fitness(energies, c.fitness_func)

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
        if rank == 0:
            debug(f"Generation {gen:2d} - Population size = {len(pop)}")
        
        # Broadcast initial population
        pop = comm.bcast(pop, root=0)
        pop_fitness = comm.bcast(pop_fitness, root=0)  # TODO: use Bcast instead (numpy)

        # Mating - get new population
        children = mating(pop, pop_fitness, c.children_perc / num_procs, c.mating_method)

        # Define sub-populaiton on every rank (only for mutating)
        chunk = len(pop) // num_procs # TODO:
        sub_pop = pop[rank*chunk:(rank+1) * chunk]

        # Mutating - get new mutants 
        mutants = get_mutants(sub_pop, c.cluster_radius, c.cluster_size)

        # FIX: TOO MANY MUTANTS NOW
        # Local minimisation and add to population
        newborns = children + mutants

        debug(f"\tRank {rank:2d} - Local optimisation of {len(newborns)} newb")
        new_energies = optimise_local(newborns, c.calc, c.local_optimiser)
        
        newborns = comm.gather(newborns, root=0)
        new_energies = comm.gather(new_energies, root=0)

        if rank == 0:
            newborns = flatten_list(newborns)
            new_energies = flatten_list(new_energies)
      
            pop +=  newborns
            energies.extend(new_energies)

            # Keep track of new local minima
            local_min, energies_min = store_local_minima(newborns, energies, local_min, energies_min, c.dE_thr)
            
            # Natural Selection
            debug(f"\tRank {rank:2d} - Natural Selection")
            pop, energies, pop_fitness = natural_selection_step(pop, energies, pop_fitness, c.pop_size, c.dE_thr, c.fitness_func)

            # Store current best
            if energies[0] < best_min[-1].get_potential_energy():
                debug("New global minimum: ", energies[0])
                best_min.append(pop[0])
                gen_no_success = 0  # This is success, so set to zero.
            else:
                gen_no_success += 1

            gen += 1
        
        gen = comm.bcast(gen, root=0)
        gen_no_success = comm.bcast(gen_no_success, root=0)
    
    if rank == 0:
        # Store / report
        debug(f"Found {len(local_min)} local minima in total.")
        debug("The evolution of the global minimum:")
        debug([cluster.get_potential_energy() for cluster in best_min])

        store_results_database(best_min[-1], local_min, db, c)

    return 0


if __name__ == "__main__":
    ga_distributed()
