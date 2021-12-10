#!/bin/python3
import time

import numpy as np

import artificial_bee_colony_algorithm
import random
from mpi4py import MPI
import math

# TODO refactoring
def employee_bee_func(pop, s_n, cluster_size, calc, local_optimiser, comm, rank, total_p, is_parallel,
                      eb_mutation_size):
    """
        employed bee discovers neighbor structure and creates new structure

        pop: the given population
        s_n: number of population
        cluster_size: size of the cluster
        calc: method of the calculator
        local_optimiser: function which optimises locally
        eb_mutation_size: eg trigonometric-> 3
        """

    if is_parallel == 1:
        buffer_size = math.floor(s_n / total_p)
        start_i = rank * buffer_size
        if rank == total_p - 1:
            end_i = s_n
        else:
            end_i = (rank + 1) * buffer_size
        pop_final = []
        for i in range(start_i, end_i):
            sum_e = 0.0
            # array to save the atom to mutate
            e = np.array([])
            random_index = 0
            # if sum is 0, it leads to n/0 error in the later step
            while sum_e == 0:
                # select random fpr the mutation
                random_index = random.sample([x for x in range(s_n) if x != i], eb_mutation_size)
                for j in range(eb_mutation_size):
                    e = np.append(e, np.abs(pop[random_index[j]].get_potential_energy()))
                sum_e = np.sum(e)


            # TODO different search method could be possible
            # p =  |e[m]| / (|e[1]|+|e[1]|+ |e[1]|) where m = 1, 2, 3
            p = np.array([])
            for j in range(eb_mutation_size):
                p = np.append(p, e[j] / sum_e)

            # get position of atoms through mutation
            new_atoms_position = 0.0
            for j in range(eb_mutation_size):
                new_atoms_position += pop[random_index[j]].get_positions()
            new_atoms_position /= eb_mutation_size

            for j in range(eb_mutation_size - 1):
                new_atoms_position += (p[j + 1] - p[j]) * (pop[random_index[j]].get_positions()
                                                           - pop[j + 1].get_positions())
            new_atoms_position += (p[0] - p[eb_mutation_size - 1]) * (
                        pop[random_index[eb_mutation_size - 1]].get_positions() -
                        pop[random_index[0]].get_positions())

            # create new Atoms with new positions and locally optimise
            new_x = artificial_bee_colony_algorithm.optimise_local_each(
                artificial_bee_colony_algorithm.generate_cluster_with_position(new_atoms_position, cluster_size),
                calc, local_optimiser)

            # if new Atoms has better potential energy than current Atoms replace, otherwise keep
            if new_x.get_potential_energy() <= pop[i].get_potential_energy():
                pop_final.append(new_x)
            else:
                pop_final.append(pop[i])

        pop_final = comm.gather(pop_final, root=0)
        if rank == 0:
            return sum(pop_final, [])
        else:
            return None

    else:
        if rank == 0:
            # not parallelized version
            pop = employee_bee_mutation(pop, s_n, cluster_size, calc, local_optimiser, eb_mutation_size)
        return pop


def employee_bee_mutation(pop, s_n, cluster_size, calc, local_optimiser, mutation_n):
    """
            employed bee discovers neighbor structure and creates new structure

            pop: the given population
            s_n: number of population
            cluster_size: size of the cluster
            calc: method of the calculator
            local_optimiser: function which optimises locally
            mutation_n: eg trigonometric-> 3
            """
    pop_copy = pop.copy()
    for cluster in pop_copy:
        cluster.calc = calc
    for i in range(len(pop)):
        sum_e = 0.0
        e = np.array([])
        random_index = random.sample(range(0, s_n), mutation_n)
        while sum_e == 0:
            while i in random_index:
                random_index = random.sample(range(0, s_n), mutation_n)
            for j in range(mutation_n):
                e = np.append(e, np.abs(pop_copy[random_index[j]].get_potential_energy()))
            sum_e = np.sum(e)

        p = np.array([])
        for j in range(mutation_n):
            p = np.append(p, e[j] / sum_e)
        new_x_position = 0.0

        for j in range(mutation_n):
            new_x_position += pop_copy[random_index[j]].get_positions()
        new_x_position /= mutation_n

        for j in range(mutation_n):
            if j == mutation_n - 1:
                new_x_position += (p[0] - p[j]) * (pop_copy[random_index[j]].get_positions() -
                                                   pop_copy[random_index[0]].get_positions())
            else:
                new_x_position += (p[j + 1] - p[j]) * (pop_copy[random_index[j]].get_positions()
                                                       - pop_copy[j + 1].get_positions())

        new_x = artificial_bee_colony_algorithm.optimise_local_each(
            artificial_bee_colony_algorithm.generate_cluster_with_position(new_x_position, cluster_size),
            calc, local_optimiser)
        if new_x.get_potential_energy() <= pop[i].get_potential_energy():
            pop[i] = new_x
    return pop
