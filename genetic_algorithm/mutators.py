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


def displacement_static(population, mutation_rate, radius, displacements_per_cluster=0.2):
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


def single_displacement_dynamic(cluster, radius, max_moves, scale_stdev=0.1):
    """
    Applies a dynamic mutation to a cluster where the atoms are randomly moved
    in a random direction a random number of times.

    @param cluster: cluster of atoms to mutate
    @param radius: the scale of atom positions used for moving atoms
    @param max_moves: (maximum) amount of times to randomly move each atom
    @return: mutated cluster with the atoms displaced
    """

    num_moves = np.random.randint(max_moves)
    stdev = scale_stdev * radius
    for i in range(num_moves):
        # rattle uses normal distribution with specified standard deviation.
        # 10% of radius is used as standard deviation, can be modified/tuned.
        cluster.rattle(stdev=stdev)

    return cluster


def displacement_dynamic(population, mutation_rate, radius, max_moves=10): #TODO: how many atoms to change here? all?
    """
    Mutates population by moving the atoms a random distance in a random direction a random number of times.

    @param population: list of atom to potentially apply mutations on
    @param mutation_rate: probability of mutation occurring in a cluster
    @param radius: the scale of atom positions used for moving atoms
    @param max_moves: (maximum) amount of times to randomly move each atom
    @return: list of mutated clusters with their atoms randomly moved
    """

    return [single_displacement_dynamic(cluster, radius, max_moves) for cluster in population if np.random.uniform() < mutation_rate]


def single_rotation(cluster):
    """
    Mutates a single cluster by taking half of the cluster and rotating it by a random amount
    along the z-axis.

    @param cluster: the cluster to have half of it rotated
    @return: mutated cluster with a random rotation applied
    """

    #TODO: use rotation matrices or scipy

    return cluster


def rotation(population, mutation_rate):
    """
    Mutates population by splitting the cluster in 2 halves and randomly rotating one half around the z-axis

    @param population: list of atom to potentially apply mutations on
    @param mutation_rate: probability of mutation occurring in a cluster
    @return: list of mutated clusters where some of the structures have a part of them rotated
    """

    return [single_rotation(cluster) for cluster in population if np.random.uniform() < mutation_rate]


def single_replacement(cluster_type, cluster_size, radius):
    """
    Generates a new cluster

    @param cluster_type: atom type of the cluster
    @param cluster_size: size of a single cluster
    @param radius: the scale of atom positions used for moving atoms
    @return: newly created cluster
    """

    coords = np.random.uniform(-radius, radius, (cluster_size, 3))

    return Atoms(cluster_type, coords)


def replacement(population, cluster_size, radius, mutation_rate):
    """
    Mutates population by replacing some clusters with newly generated clusters

    @param population: list of atom to potentially apply mutations on
    @param cluster_size: size of a single cluster
    @param radius: the scale of atom positions used for moving atoms
    @param mutation_rate: probability of mutation occurring in a cluster
    @return: list of mutated clusters which are newly generated
    """
    cluster_type = population[0].get_chemical_symbols
    return [single_replacement(cluster_type, cluster_size, radius) for cluster in population if np.random.uniform() < mutation_rate]


def type_swap():
    return


def mirror_shift():
    return
