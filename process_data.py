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
    minima = np.array(combined)

    ind = np.argsort(minima[:, -1])
    sorted_clusters = minima[ind]

    local_minima = [sorted_clusters[0]]
    for i in range(sorted_clusters.shape[0]):
        energy = sorted_clusters[i, 1]
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
    print(f"| {f'Global Geometry Optimisation results':{n}s}|")
    print(" ================================================================ ")
    print(f"| {f'Global minimum potential energy: {local_minima[0].get_potential_energy():.2f}':{n}s}|")
    if len(local_minima) >= 5:
        print(f"| {f'Second lowest potential energy: {local_minima[1].get_potential_energy():.2f}':{n}s}|")
        print(f"| {f'Third lowest potential energy: {local_minima[2].get_potential_energy():.2f}':{n}s}|")
        print(f"| {f'Fourth lowest potential energy: {local_minima[3].get_potential_energy():.2f}':{n}s}|")
        print(f"| {f'Fifth lowest potential energy: {local_minima[4].get_potential_energy():.2f}':{n}s}|")
    print(f"| {f'Number of local minima found: {len(local_minima)}':{n}s}|")
    print(" ---------------------------------------------------------------- ")

    return 0
