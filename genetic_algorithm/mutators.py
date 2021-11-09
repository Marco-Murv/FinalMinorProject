"""
Possible mutation operations for the Genetic Algorithm.
"""

import numpy as np
from ase import Atoms

def single_displacement_static(cluster, radius, displacements_per_cluster):
    """
    Applies a static displacement to a certain number of atoms within the cluster.

    @param cluster: cluster of atoms to mutate
    @param radius: the scale of atom positions used for generating new random positions
    @param displacements_per_cluster: probability of atoms mutated within a cluster
    @return: mutated cluster with some displaced atoms
    """

    for i in range(len(cluster.positions)):
        if np.random.uniform(-radius, radius) < displacements_per_cluster:
            cluster.positions[i] = np.random.uniform(-radius, radius)

    return cluster


def displacement_static(population, mutation_rate, radius, displacements_per_cluster=0.1):
    """
    Mutates population by replacing a certain number of atoms within a cluster with new
    randomly generated coordinates.

    @param population: list of atom to potentially apply mutations on
    @param mutation_rate: probability of mutation occurring in a cluster
    @param radius: the scale of atom positions used for generating new random positions
    @param displacements_per_cluster: probability of atoms mutated within a cluster
    @return: list of mutated clusters with some of their atoms displaced
    """

    return [single_displacement_static(cluster, radius, displacements_per_cluster) for cluster in population if np.random.uniform() < mutation_rate]


def rotation():
    return


def type_swap():
    return


def mirror_shift():
    return
