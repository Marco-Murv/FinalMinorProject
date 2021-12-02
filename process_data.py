#!/bin/python3

import numpy as np


def select_local_minima(minima, energy_diff=0.1):
    """
    Removes similar local minima from the list using the given energy threshold.

    @param minima: list of atoms objects
    @param energy_diff: energy threshold for considering two atoms objects too similar
    @return: list of filtered local minima which differ sufficiently
    """
    energies = [cluster.get_potential_energy() for cluster in minima]
    combined = list(zip(minima, energies))
    minima = np.array(combined, dtype=object)

    ind = np.argsort(minima[:, -1])
    sorted_clusters = minima[ind]

    local_minima = [sorted_clusters[0]]
    for i in range(sorted_clusters.shape[0]):
        energy = sorted_clusters[i, 1]
        if energy > 0:
            break
        if np.abs(local_minima[-1][1] - energy) > energy_diff:
            local_minima.append(sorted_clusters[i])

    return [cluster for [cluster, energy] in local_minima]


def print_stats(local_minima):
    """
    Prints some information about the minima found during a global optimisation run.

    @param local_minima: list of the local minima found
    @return:
    """

    n = 63
    print(" ---------------------------------------------------------------- ")
    print(f"| {f'Global Geometry Optimisation - Results':{n}s}|")
    print(" ================================================================ ")
    print(f"| {f'Lowest energy local minima :':{n}s}|")
    size = min(5, len(local_minima))
    for i in range(size):
        print(f"| {f'   {local_minima[i].get_potential_energy():.2f}':{n}s}|")
    print(f"| {f'':{n}s}|")
    print(f"| {f'Number of local minima found   : {len(local_minima)}':{n}s}|")
    print(" ---------------------------------------------------------------- ")

    return 0
