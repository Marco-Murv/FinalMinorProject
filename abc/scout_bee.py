#!/bin/python3
import numpy as np

import artificial_bee_colony_algorithm
import random


def scout_bee_func(pop, Sn, cluster_size, calc, local_optimiser):
    pop = sorted(pop, key=lambda student: student.get_potential_energy(), reverse=True)
    for i in range(5):
        pop[i] = artificial_bee_colony_algorithm.generate_cluster(
            cluster_size, artificial_bee_colony_algorithm.parse_args().cluster_size)
    return pop
