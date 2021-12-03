#!/bin/python3

import sys
import artificial_bee_colony_algorithm
import numpy as np
from mpi4py import MPI


def scout_bee_func(pop, s_n, cluster_size, cluster_radius, calc, local_optimiser, comm, rank, is_parallel):
    minimal_pe = sys.maxsize  # lowest potential energy

    for cluster in pop:
        pe = cluster.get_potential_energy()
        if pe < minimal_pe: minimal_pe = pe

    # Serial version of below for loop. Just sitting here because MAYBE I need it later
    # new_pop = []
    # for cluster in pop:
    #     if (cluster.get_potential_energy() / minimal_pe) >= 0.65:
    #         if cluster.get_potential_energy() < 0:
    #             new_pop.append(cluster)

    splitted_pop = split(pop, comm.Get_size()) # divides the array into n parts to divide over processors
    pop = comm.scatter(splitted_pop, root=0)
    result = []
    for cluster in pop:
        if (cluster.get_potential_energy() / minimal_pe) >= 0.65:
            if cluster.get_potential_energy() < 0:
                result.append(cluster)
    new_pop = comm.gather(result, root=0) # gather the result from all processes to master
    if rank == 0:
        new_pop = [i for i in new_pop if i]
        new_pop = [item for sublist in new_pop for item in sublist] # flatten array to 1D array
    new_pop = comm.bcast(new_pop)

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

    # This is the parallel version of the above for loop. Master-worker pattern.
    """
    local_minima = [sorted_clusters[0]]
    if rank == 0:
        for i in range(sorted_clusters.shape[0]):
            comm.send(sorted_clusters[i, 1], dest=i, tag=i)
            local_minima = comm.recv(source=i)
    else:
        data = comm.recv(source=0, tag=rank)
        energy = data
        if np.abs(local_minima[-1][1] - energy) > energy_diff:
            local_minima.append(sorted_clusters[rank])
        comm.send(local_minima, dest=0)
    """

    new_pop = [cluster for [cluster, energy] in local_minima]
    if len(pop) != len(new_pop):  # replace the old removed clusters with new clusters
        # if n clusters were removed, then n new clusters are added
        new_clusters = artificial_bee_colony_algorithm.generate_population(s_n, cluster_size, cluster_radius)[:len(pop) - len(new_pop)]
        for cluster in new_clusters: cluster.calc = calc
        artificial_bee_colony_algorithm.optimise_local(new_clusters, calc, local_optimiser, comm.Get_size())
        new_pop += new_clusters

    return new_pop


def split(a, n):
    """
    Splits an array 'a' into n parts.
    For example split the following array into 5 parts: [1,2,3,4,5,6,7,8,9,10] -> [[1,2],[3,4],[5,6],[7,8],[9,10]]
    """

    k, m = divmod(len(a), n)
    return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))