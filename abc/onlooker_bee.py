import random
import sys
from random import randrange

import artificial_bee_colony_algorithm
import numpy as np


def onlooker_bee_func(pop, pop_size, cluster_size, calc, local_optimiser):
    if True:
        pop = search_neighbor_monte_carlo(pop, pop_size, cluster_size, calc, local_optimiser)
    return pop


def search_neighbor_monte_carlo(pop, pop_size, cluster_size, calc, local_optimiser):
    selected_index = get_index(pop)
    random_index2 = random.sample(range(pop_size), 4)
    f = randrange(10) / 100.0
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
    return pop


def get_index(pop):
    random_n = randrange(10)
    if random_n > 4:
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


#def onlooker_bee_func(pop, pop_size, cluster_size, calc, local_optimiser):
#    minimal_pe = sys.maxsize  # lowest potential energy
#
#    for cluster in pop:
#        pe = cluster.get_potential_energy()
#        if pe < minimal_pe: minimal_pe = pe
#
#    new_pop = []
#    for cluster in pop:
#        if (cluster.get_potential_energy() / minimal_pe) >= 0.4:
#            new_pop.append(cluster)
#
#    return new_pop
