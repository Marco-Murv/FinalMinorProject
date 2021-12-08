#!/bin/python3

"""
Distributed parallelisation of the Genetic Algorithm

This program requires a file called `ga_config.yaml` in the same directory.
Example ga_config.yaml:
```yaml
    general:
        cluster_radius: 2.0
        cluster_size: 4
        delta_energy_thr: 0.1
        pop_size: 5
    mating:
        children_perc: 0.8
        fitness_func: exponential
        mating_method: roulette
    results:
        db_file: genetic_algorithm_results.db
        results_dir: results
    reuse_state: false
    run_id: 22
    show_plot: true
    stop_conditions:
        max_gen: 50
        max_no_success: 50
        time_lim: 100
```
"""

import os
import sys
import ase
import inspect
import numpy as np
from mpi4py import MPI
from ase.io.trajectory import Trajectory


from mating import mating
import genetic_algorithm as ga
from genetic_algorithm import store_results_database
from genetic_algorithm import optimise_local, fitness, get_mutants
from genetic_algorithm import get_configuration, natural_selection_step
from genetic_algorithm import config_info, debug, fitness, generate_population
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import process_data


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

    # Parse possible terminal input and yaml file.
    c = None
    if rank == 0:
        config_file = "config/ga_distributed_config.yaml"
        c = get_configuration(config_file)

    # Distribute configuration data to all of the processors
    c = comm.bcast(c, root=0)

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
        # Broadcast initial population
        pop = comm.bcast(pop, root=0)
        energies = comm.bcast(energies, root=0)

        # Mating - get new population
        pop_fitness = fitness(energies, c.fitness_func)
        children = mating(pop, pop_fitness, c.children_perc /
                          num_procs, c.mating_method)
        
        # Define sub-population on every rank (only for mutating)
        chunk = len(pop) // num_procs
        sub_pop = pop[rank * chunk:(rank + 1) * chunk]

        # Mutating - get new mutants
        mutants = []
        if len(sub_pop) >= 2:
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
                debug(f"New global minimum in generation {gen:2d}: ", energies[0])
                best_min.append(pop[0])
                gen_no_success = 0  # This is success, so set to zero.
            else:
                gen_no_success += 1

            gen += 1

            # Check if done
            done = gen_no_success > c.max_no_success or gen > c.max_gen
            if MPI.Wtime() - ga_start_time > c.time_lim:
                debug("REACHED TIME LIMIT")
                done = True

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

        traj_file_path = os.path.join(os.path.dirname(__file__), f"{c.results_dir}/ga_distr_{c.cluster_size}.traj")
        traj_file = Trajectory(traj_file_path, 'w')
        for cluster in local_min:
            traj_file.write(cluster)
        traj_file.close()

        # Store in Database
        db_file = os.path.join(os.path.dirname(__file__), c.results_dir+'/'+c.db_file)
        db = ase.db.connect(db_file)
        store_results_database(best_min[-1], local_min, db, c)

    return 0


if __name__ == "__main__":
    ga_distributed()
