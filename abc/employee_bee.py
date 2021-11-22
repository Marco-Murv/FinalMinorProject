#!/bin/python3
import numpy as np

import artificial_bee_colony_algorithm
import random


def employee_bee_func(pop, s_n, cluster_size, calc, local_optimiser):
    """
        employed bee discovers neighbor structure and creates new structure

        pop: the given population
        s_n: number of population
        cluster_size: size of the cluster
        calc: method of the calculator
        local_optimiser: function which optimises locally
        """
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
                e = np.append(e, np.abs(pop_copy[j].get_potential_energy()))
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
