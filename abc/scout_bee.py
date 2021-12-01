#!/bin/python3

import sys
import artificial_bee_colony_algorithm


def scout_bee_func(pop, Sn, cluster_size, cluster_radius, calc, local_optimiser):
    minimal_pe = sys.maxsize  # lowest potential energy

    for cluster in pop:
        pe = cluster.get_potential_energy()
        if pe < minimal_pe: minimal_pe = pe

    new_pop = []
    for cluster in pop:
        if (cluster.get_potential_energy() / minimal_pe) >= 0.8:
            new_pop.append(cluster)

    if len(pop) != len(new_pop):  # replace the old removed clusters with new clusters
        # if n clusters were removed, then n new clusters are added
        new_clusters = artificial_bee_colony_algorithm.generate_population(Sn, cluster_size, cluster_radius)[:len(pop)-len(new_pop)]
        artificial_bee_colony_algorithm.optimise_local(new_clusters, calc, local_optimiser)
        new_pop += new_clusters

    return new_pop
