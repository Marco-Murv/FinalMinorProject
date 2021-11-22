import random
import sys

import artificial_bee_colony_algorithm
import numpy as np

def onlooker_bee_func(pop, pop_size, cluster_size, calc, local_optimiser):
    minimal_pe = sys.maxsize  # lowest potential energy

    for cluster in pop:
        pe = cluster.get_potential_energy()
        if pe < minimal_pe: minimal_pe = pe

    new_pop = []
    for cluster in pop:
        if (cluster.get_potential_energy() / minimal_pe) >= 0.4:
            new_pop.append(cluster)

    return new_pop
