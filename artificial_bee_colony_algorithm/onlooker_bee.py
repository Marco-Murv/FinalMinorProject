import random
import sys
from random import randrange

import artificial_bee_colony_algorithm
import numpy as np


def onlooker_bee_func(pop, pop_size, cluster_size, calc, local_optimiser):
    '''

    :param pop: poputlaion
    :param pop_size: size of the population
    :param cluster_size: size of the cluster
    :param calc: calculation method
    :param local_optimiser: local optimization method
    :return:
    '''
    if True:
        pop = search_neighbor_monte_carlo(pop, pop_size, cluster_size, calc, local_optimiser)
    return pop


def search_neighbor_monte_carlo(pop, pop_size, cluster_size, calc, local_optimiser):
    # select random index
    selected_index = get_index(pop)
    random_index2 = random.sample(range(pop_size), 4)
    f = randrange(1000) / 1000.0
    new_x = artificial_bee_colony_algorithm.optimise_local_each(
        artificial_bee_colony_algorithm.generate_cluster_with_position(pop[selected_index].get_positions() + f *
                                                                       (pop[random_index2[0]].get_positions() +
                                                                        pop[
                                                                            random_index2[1]].get_positions()
                                                                        - pop[random_index2[2]].get_positions()
                                                                        - pop[
                                                                            random_index2[3]].get_positions()),
                                                                       cluster_size), calc, local_optimiser)
    if new_x.get_potential_energy() <= pop[selected_index].get_potential_energy():
        pop[selected_index] = new_x
    elif False:
        # if it current index did not improved find other index to improve
        return search_neighbor_monte_carlo(pop, pop_size, cluster_size, calc, local_optimiser)

    return pop


def get_index(pop):
    random_n = randrange(10)
    if random_n > 2:
        return get_index_best(pop)
    else:
        return get_index_random(pop)


def get_index_random(pop) -> int:
    """
         Retrieve the random index of the cluster

         pop: the given population
         """
    return randrange(len(pop))


def get_index_best(pop) -> int:
    """
         Retrieve the index of the cluster with the lowest potential energy

         pop: the given population
         """
    index_best = 0
    for i in range(len(pop)):
        if pop[i].get_potential_energy() < pop[index_best].get_potential_energy():
            index_best = i
    return index_best
