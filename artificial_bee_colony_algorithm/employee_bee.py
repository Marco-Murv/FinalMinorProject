import numpy as np
import artificial_bee_colony_algorithm
import random
from random import randrange
import math


# TODO refactoring
def employee_bee_func(pop, s_n, cluster_size, calc, local_optimiser, comm, rank, total_p, is_parallel,
                      eb_mutation_size, search_method, monte_carlo_f, counter):
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
        return employee_bee_mutation_parallel(pop, s_n, cluster_size, calc, local_optimiser, eb_mutation_size, comm,
                                              rank, total_p, search_method, monte_carlo_f, counter)
    else:
        if rank == 0:
            # not parallelized version
            return employee_bee_mutation_non_parallel(pop, s_n, cluster_size, calc, local_optimiser, eb_mutation_size,
                                                      comm, rank, total_p, counter, monte_carlo_f)


def select_random_cluster_mutation(population, n, crr_i):
    e = np.array([])
    while True:
        # select random fpr the mutation
        random_index = random.sample([x for x in range(len(population)) if x != crr_i], n)
        for j in range(n):
            e = np.append(e, np.abs(population[random_index[j]].get_potential_energy()))
        e_sum = np.sum(e)
        p = np.array([])
        if np.sum(e) != 0:
            for j in range(n):
                p = np.append(p, e[j] / e_sum)
            return p, random_index, np.sum(e)


def calculate_new_position_mutation(i, pop, n):
    p, random_index, e_sum = select_random_cluster_mutation(pop, n, i)
    new_atoms_position = 0.0
    for j in range(n):
        new_atoms_position += pop[random_index[j]].get_positions()
    new_atoms_position /= n

    for j in range(n):
        k = j + 1
        if j == n - 1:
            k = 0
        new_atoms_position += (p[k] - p[j]) * (pop[random_index[j]].get_positions() - pop[random_index[k]].get_positions())
    return new_atoms_position


def calculate_new_position_monte_carlo(i, pop, s_n, n, f):
    if f < 0:
        f = randrange(1000) / 1000.0
    if n % 2 != 0:
        n += 1
    random_index = random.sample(range(s_n), n)
    new_position = pop[i].get_positions()
    for j in range(int(n / 2)):
        new_position += f * pop[random_index[j]].get_positions()
    for j in range(int(n / 2), n):
        new_position -= f * pop[random_index[j]].get_positions()

    return new_position


def employee_bee_mutation_parallel(pop, s_n, cluster_size, calc, local_optimiser, eb_mutation_size, comm, rank,
                                   total_p, search_method, monte_carlo_f, counter):
    end_i = (rank + 1) * math.floor(s_n / total_p)
    if rank == total_p - 1:
        end_i = s_n
    pop_final = []
    for i in range(rank * math.floor(s_n / total_p), end_i):
        # array to save the atom to mutate
        # TODO different search method could be possible
        # p =  |e[m]| / (|e[1]|+|e[1]|+ |e[1]|) where m = 1, 2, 3
        # get position of atoms through mutation
        # create new Atoms with new positions and locally optimise
        if search_method == 0:
            new_position = calculate_new_position_mutation(i, pop, eb_mutation_size)
        else:
            new_position = calculate_new_position_monte_carlo(i, pop, s_n, eb_mutation_size, monte_carlo_f)
        new_x = artificial_bee_colony_algorithm.optimise_local_each(
            artificial_bee_colony_algorithm.generate_cluster_with_position(
                new_position, cluster_size, counter),
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


def employee_bee_mutation_non_parallel(pop, s_n, cluster_size, calc, local_optimiser, mutation_n, comm, rank, total_p, counter, monte_carlo_f):
    """
            employed bee discovers neighbor structure and creates new structure

            pop: the given population
            s_n: number of population
            cluster_size: size of the cluster
            calc: method of the calculator
            local_optimiser: function which optimises locally
            mutation_n: eg trigonometric-> 3
            """
    if rank == 0:
        pop_copy = pop.copy()
        for cluster in pop_copy:
            cluster.calc = calc
        for i in range(len(pop)):
            new_x = artificial_bee_colony_algorithm.optimise_local_each(
                artificial_bee_colony_algorithm.generate_cluster_with_position(
                    calculate_new_position_monte_carlo(i, pop, s_n, mutation_n, monte_carlo_f), cluster_size, counter),
                calc, local_optimiser)
            if new_x.get_potential_energy() <= pop_copy[i].get_potential_energy():
                pop_copy[i] = new_x
        return pop_copy
    else:
        return None
