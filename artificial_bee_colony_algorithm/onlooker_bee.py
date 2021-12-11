
from random import randrange
import artificial_bee_colony_algorithm
import employee_bee


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
    # TODO more than one cluster could be choosed to be updated ?
    selected_index = get_index(pop)
    f = randrange(1000) / 1000.0
    new_x = artificial_bee_colony_algorithm.optimise_local_each(
        artificial_bee_colony_algorithm.generate_cluster_with_position(employee_bee.calculate_new_position_monte_carlo(selected_index, pop, pop_size, 4, f),
                                                                       cluster_size), calc, local_optimiser)
    if new_x.get_potential_energy() <= pop[selected_index].get_potential_energy():
        pop[selected_index] = new_x
    elif False:
        # TODO to config
        # if it current index did not improved find other index to improve
        return search_neighbor_monte_carlo(pop, pop_size, cluster_size, calc, local_optimiser)

    return pop


def get_index(pop):
    # TODO probability should be a config variable
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
