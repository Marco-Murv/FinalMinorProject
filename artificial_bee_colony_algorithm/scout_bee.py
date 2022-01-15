#!/bin/python3
import sys
import artificial_bee_colony_algorithm
import numpy as np

check_every_loop = 20  # this is the standard value. The actual value depends on the value in config.
local_minima_per_loop = np.zeros(check_every_loop, dtype=object)

def scout_bee_func(pop, s_n, cluster_size, cluster_radius, calc, local_optimiser, comm, rank, energy_diff, energy_abnormal, loop_index, is_parallel, update_energies, counter, removed_clusters):
    if is_parallel == 1:
        return scout_bee_func_parallel(pop, s_n, cluster_size, cluster_radius, calc, local_optimiser, comm, rank, energy_diff, energy_abnormal, loop_index, update_energies, counter, removed_clusters)
    else:
        return scout_bee_func_serial(pop, s_n, cluster_size, cluster_radius, calc, local_optimiser, comm, energy_diff, energy_abnormal, loop_index, update_energies, counter, removed_clusters)


def scout_bee_func_parallel(pop, s_n, cluster_size, cluster_radius, calc, local_optimiser, comm, rank, energy_diff, energy_abnormal, loop_index, update_energies, counter, removed_clusters):
    minimal_pe = sys.maxsize  # lowest potential energy
    popcp = pop.copy()
    for cluster in pop:
        pe = cluster.get_potential_energy()
        if pe < minimal_pe: minimal_pe = pe

    result1 = []
    for i in range(s_n):
        if (pop[i].get_tags()[0] >= 0) | (minimal_pe ==pop[i].get_potential_energy()):
            p = pop[i].get_tags()
            p[0] = p[0] - 1
            pop[i].set_tags(p)
            result1.append(pop[i])
    pop = result1
    # TODO removed value here might be the local minima?

    # splitted_pop = split(pop, comm.Get_size())  # divides the array into n parts to divide over processors
    # pop = comm.scatter(splitted_pop, root=0)
    # result = []
    # rc = []
    # for cluster in pop:
    #     if (cluster.get_potential_energy() / minimal_pe) >= energy_abnormal:
    #         if cluster.get_potential_energy() < 0:
    #             result.append(cluster)
    #     else: rc.append(cluster)
    #
    # new_pop = comm.gather(result, root=0)  # gather the result from all processes to master
    # removed_clusters = comm.gather(rc, root=0)
    # if rank == 0:
    #     new_pop = [i for i in new_pop if i]
    #     new_pop = [item for sublist in new_pop for item in sublist]  # flatten array to 1D array
    #
    # removed_clusters = comm.bcast(removed_clusters)
    # new_pop = comm.bcast(new_pop)

    new_pop = []
    for cluster in pop:
        if (cluster.get_potential_energy() / minimal_pe) >= energy_abnormal:
            if cluster.get_potential_energy() < 0:
                new_pop.append(cluster)
        else:
            if rank == 0:
                removed_clusters.append(cluster)

    energy_diff = energy_diff
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
    #TODO
    pop = popcp
    if len(pop) != len(new_pop):  # replace the old removed clusters with new clusters
        # if n clusters were removed, then n new clusters are added
        new_clusters = artificial_bee_colony_algorithm.generate_population(s_n, cluster_size, cluster_radius, counter)[
                       :len(pop) - len(new_pop)]
        for cluster in new_clusters: cluster.calc = calc
        artificial_bee_colony_algorithm.optimise_local(new_clusters, calc, local_optimiser, comm.Get_size())
        new_pop += new_clusters

    if update_energies == 1 & rank == 0:
        if loop_index >= check_every_loop:  # if a local minima hasn't been updated for 'check_every_loop' loops, then replace with new cluster
            for idx, a in enumerate(new_pop):
                if new_pop[idx].get_potential_energy() in local_minima_per_loop[loop_index % check_every_loop]:
                    removed_clusters.append(new_pop[idx])
                    new_cluster = artificial_bee_colony_algorithm.generate_population(s_n, cluster_size, cluster_radius, counter)[0]
                    new_cluster.calc = calc
                    artificial_bee_colony_algorithm.optimise_local([new_cluster], calc, local_optimiser, comm.Get_size())
                    new_pop[idx] = new_cluster

        local_minima = np.array([])
        for cluster in new_pop:
            local_minima = np.append(local_minima, cluster.get_potential_energy())
        local_minima_per_loop[loop_index % check_every_loop] = local_minima

    return new_pop, removed_clusters


def scout_bee_func_serial(pop, s_n, cluster_size, cluster_radius, calc, local_optimiser, comm, energy_diff, energy_abnormal, loop_index, update_energies, counter, removed_clusters):
    minimal_pe = sys.maxsize  # lowest potential energy

    for cluster in pop:
        pe = cluster.get_potential_energy()
        if pe < minimal_pe: minimal_pe = pe

    new_pop = []
    for cluster in pop:
        if (cluster.get_potential_energy() / minimal_pe) >= energy_abnormal:
            if cluster.get_potential_energy() < 0:
                new_pop.append(cluster)
        else: removed_clusters.append(cluster)

    energy_diff = energy_diff
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
        # if n clusters were removed, then n new clusters are added
        new_clusters = artificial_bee_colony_algorithm.generate_population(s_n, cluster_size, cluster_radius, counter)[
                       :len(pop) - len(new_pop)]

        for cluster in new_clusters: cluster.calc = calc
        artificial_bee_colony_algorithm.optimise_local(new_clusters, calc, local_optimiser, comm.Get_size())
        new_pop += new_clusters

    if update_energies == 1:
        if loop_index >= check_every_loop:  # if a local minima hasn't been updated for 'check_every_loop' loops, then replace with new cluster
            for idx, a in enumerate(new_pop):
                if new_pop[idx].get_potential_energy() in local_minima_per_loop[loop_index % check_every_loop]:
                    removed_clusters.append(new_pop[idx])
                    new_cluster = artificial_bee_colony_algorithm.generate_population(s_n, cluster_size, cluster_radius, counter)[0]
                    new_cluster.calc = calc
                    artificial_bee_colony_algorithm.optimise_local([new_cluster], calc, local_optimiser, comm.Get_size())
                    new_pop[idx] = new_cluster

        local_minima = np.array([])
        for cluster in new_pop:
            local_minima = np.append(local_minima, cluster.get_potential_energy())
        local_minima_per_loop[loop_index % check_every_loop] = local_minima

    return new_pop, removed_clusters


def split(a, n):
    """
    Splits an array 'a' into n parts.
    For example split the following array into 5 parts: [1,2,3,4,5,6,7,8,9,10] -> [[1,2],[3,4],[5,6],[7,8],[9,10]]
    """

    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
