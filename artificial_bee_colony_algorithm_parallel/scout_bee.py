#!/bin/python3

import sys
import artificial_bee_colony_algorithm
import numpy as np


def scout_bee_func(pop, s_n, cluster_size, cluster_radius, calc, local_optimiser):
    minimal_pe = sys.maxsize  # lowest potential energy

    for cluster in pop:
        pe = cluster.get_potential_energy()
        if pe < minimal_pe: minimal_pe = pe

    new_pop = []
    for cluster in pop:
        if (cluster.get_potential_energy() / minimal_pe) >= 0.65:
            if cluster.get_potential_energy() < 0:
                new_pop.append(cluster)
    #print(len(pop) - len(new_pop))
    energy_diff = 0.04
    energies = [cluster.get_potential_energy() for cluster in new_pop]
    combined = list(zip(new_pop, energies))
    minima = np.array(combined, dtype=object)

    ind = np.argsort(minima[:, -1])
    sorted_clusters = minima[ind]

    local_minima = [sorted_clusters[0]]
    for i in range(sorted_clusters.shape[0]):
        energy = sorted_clusters[i, 1]
        if np.abs(local_minima[-1][1] - energy) > energy_diff:
            local_minima.append(sorted_clusters[i])

    new_pop = [cluster for [cluster, energy] in local_minima]
    if len(pop) != len(new_pop):  # replace the old removed clusters with new clusters
        #print(len(pop) - len(new_pop))
        # if n clusters were removed, then n new clusters are added
        new_clusters = artificial_bee_colony_algorithm.generate_population(s_n, cluster_size, cluster_radius)[
                       :len(pop) - len(new_pop)]
        artificial_bee_colony_algorithm.optimise_local(new_clusters, calc, local_optimiser)
        new_pop += new_clusters

    return new_pop
