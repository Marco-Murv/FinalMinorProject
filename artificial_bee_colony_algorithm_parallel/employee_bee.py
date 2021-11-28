#!/bin/python3
import numpy as np

import artificial_bee_colony_algorithm
import random
from mpi4py import MPI
import math


def employee_bee_func(pop, s_n, cluster_size, calc, local_optimiser, comm, rank, total_p, is_parallel):
    """
        employed bee discovers neighbor structure and creates new structure

        pop: the given population
        s_n: number of population
        cluster_size: size of the cluster
        calc: method of the calculator
        local_optimiser: function which optimises locally
        """

    mutation_n = 3

    if is_parallel == 1:
        if rank == 0:
            buffer_size = math.floor(s_n/(total_p-1))
            #print(buffer_size)
            for i in range(total_p-1):

                crr = i + 1
                if crr != total_p-1 :
                    comm.send([(crr-1)*buffer_size,crr*buffer_size ], dest=crr)
                else:
                    comm.send([(crr - 1) *buffer_size, s_n], dest=crr)

            final_pop = []
            for i in range(total_p - 1):
                rec = comm.recv(source=MPI.ANY_SOURCE)
                #print(rec[0])
                final_pop = final_pop+rec

            #print(len(final_pop))
            return final_pop
        else:
            pop_final_re = []
            index_start_end = comm.recv(source=0)
            #print(index_start_end [0])
            #print(index_start_end[1])
            for i in range(index_start_end[0], index_start_end[1]):
                sum_e = 0.0
                e = np.array([])
                random_index = random.sample(range(0, s_n), mutation_n)
                while sum_e == 0:
                    while i in random_index:
                        random_index = random.sample(range(0, s_n), mutation_n)
                    for j in range(mutation_n):
                        e = np.append(e, np.abs(pop[random_index[j]].get_potential_energy()))
                    sum_e = np.sum(e)

                p = np.array([])
                for j in range(mutation_n):
                    p = np.append(p, e[j] / sum_e)
                new_x_position = 0.0
                for j in range(mutation_n):
                    new_x_position += pop[random_index[j]].get_positions()
                new_x_position /= mutation_n

                for j in range(mutation_n):
                    if j == mutation_n - 1:
                        new_x_position += (p[0] - p[j]) * (pop[random_index[j]].get_positions() -
                                                           pop[random_index[0]].get_positions())
                    else:
                        new_x_position += (p[j + 1] - p[j]) * (pop[random_index[j]].get_positions()
                                                               - pop[j + 1].get_positions())

                new_x = artificial_bee_colony_algorithm.optimise_local_each(
                    artificial_bee_colony_algorithm.generate_cluster_with_position(new_x_position, cluster_size),
                    calc, local_optimiser)
                if new_x.get_potential_energy() <= pop[i].get_potential_energy():
                    pop_final_re.append(new_x)
                    #print(new_x)
                else:
                    pop_final_re.append( pop[i])

           # print(pop_final_re[0])
            comm.isend(pop_final_re, dest=0)
            return pop





    else:
        print("dasdasd")
        if rank ==0:
            pop = employee_bee_mutation(pop, s_n, cluster_size, calc, local_optimiser, 3)
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
            while i in random_index :
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
            if j == mutation_n-1:
                new_x_position += (p[0] - p[j]) * (pop_copy[random_index[j]].get_positions() -
                                                   pop_copy[random_index[0]].get_positions())
            else:
                new_x_position += (p[j+1] - p[j]) * (pop_copy[random_index[j]].get_positions()
                                                     - pop_copy[j+1].get_positions())

        new_x = artificial_bee_colony_algorithm.optimise_local_each(
            artificial_bee_colony_algorithm.generate_cluster_with_position(new_x_position, cluster_size),
            calc, local_optimiser)
        if new_x.get_potential_energy() <= pop[i].get_potential_energy():
            pop[i] = new_x
    return pop
