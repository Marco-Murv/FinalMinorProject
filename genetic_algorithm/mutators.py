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
        # 10% of radius is used as standard deviation for normal distribution, can be modified/tuned.
        cluster.rattle(stdev=stdev) # TODO: check if original population member is unchanged

    return cluster


def displacement_dynamic(population, mutation_rate, radius, max_moves=10):
    """
    Mutates population by moving the atoms a random distance in a random direction a random number of times.

    @param population: list of atom to potentially apply mutations on
    @param mutation_rate: probability of mutation occurring in a cluster
    @param radius: the scale of atom positions used for moving atoms
    @param max_moves: (maximum) amount of times to randomly move each atom
    @return: list of mutated clusters with their atoms randomly moved
    """

    return [single_displacement_dynamic(cluster, radius, max_moves) for cluster in population if np.random.uniform() < mutation_rate]


def single_rotation(cluster, cluster_type):
    """
    Mutates a single cluster by taking half of the cluster and rotating it by a random amount
    along the z-axis.

    @param cluster: the cluster to have half of it rotated
    @param cluster_type: atom type of the cluster
    @return: mutated cluster with a random rotation applied to its top half
    """

    cluster.center()
    coords = np.array(cluster.get_positions())
    median_z = np.median(coords[:, 2])
    top_half = coords[coords[:, 2] > median_z, :]
    bottom_half = coords[coords[:, 2] < median_z, :]

    top_cluster = Atoms(cluster_type+str(top_half.shape[0]), top_half)
    bottom_cluster = Atoms(cluster_type+str(bottom_half.shape[0]), bottom_half)

    top_cluster.rotate(np.random.randint(360), (0, 0, 1))
    top_cluster.extend(bottom_cluster)

    return top_cluster


def rotation(population, cluster_type, mutation_rate): #TODO: Test this, stuff like correct cluster size, original cluster not modified, etc.
    """
    Mutates population by splitting the cluster in 2 halves and randomly rotating one half around the z-axis.

    @param population: list of atom to potentially apply mutations on
    @param cluster_type: atom type of the cluster
    @param mutation_rate: probability of mutation occurring in a cluster
    @return: list of mutated clusters where some of the structures have a part of them rotated
    """

    return [single_rotation(cluster, cluster_type) for cluster in population if np.random.uniform() < mutation_rate]


def single_replacement(cluster_type, cluster_size, radius):
    """
    Generates a completely new cluster.

    @param cluster_type: atom type of the cluster
    @param cluster_size: size of a single cluster
    @param radius: the scale of atom positions used for moving atoms
    @return: newly created cluster
    """

    coords = np.random.uniform(-radius, radius, (cluster_size, 3))

    return Atoms(cluster_type, coords)


def replacement(population, cluster_size, radius, mutation_rate):
    """
    Mutates population by replacing some clusters with newly generated clusters.

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


def single_mirror_shift(cluster, cluster_size):
    """
    Mutates a cluster by mirroring half of it which creates symmetric structures.

    @param cluster: cluster to apply mirror-shift mutation on
    @param cluster_size: size of a single cluster
    @return: list of mutated clusters which are mirrored
    """

    # TODO: get correct num of atoms (take into account even/odd!), check final cluster for correct size, add small shift
    coords = np.array(cluster.get_positions())
    normal = np.random.uniform(-1, 1, 3)
    normalised_norm = normal / np.linalg.norm(normal)

    # Obtain mirrored coordinates of all atoms
    mirrored_coords = coords - 2 * np.outer(coords.dot(normalised_norm), normalised_norm)
    mirrored_cluster = Atoms('H' + str(coords.shape[0]), np.concatenate(coords, mirrored_coords))

    return mirrored_cluster


def mirror_shift(population, cluster_size, mutation_rate):
    """
    Mutates population by mirroring half of some clusters which creates symmetric structures.

    @param population: list of atom to potentially apply mutations on
    @param cluster_size: size of a single cluster
    @param mutation_rate: probability of mutation occurring in a cluster
    @return: list of mutated clusters which are mirrored
    """

    return [single_mirror_shift(cluster, cluster_size) for cluster in population if np.random.uniform() < mutation_rate]
