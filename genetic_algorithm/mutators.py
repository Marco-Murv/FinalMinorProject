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

    coords = np.array(cluster.get_positions())
    median_z = np.median(coords[:, 2])
    top_half = coords[coords[:, 2] > median_z, :]
    bottom_half = coords[coords[:, 2] < median_z, :]

    top_cluster = Atoms(cluster_type+str(top_half.shape[0]), top_half)
    bottom_cluster = Atoms(cluster_type+str(bottom_half.shape[0]), bottom_half)

    top_cluster.rotate(np.random.randint(360), (0, 0, 1), center='cop') # TODO: check 'cop' vs 'com' versions
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


def single_type_swap(cluster, max_swaps=5):
    """
    Mutates a cluster by swapping some of its atom types.

    @param cluster: the cluster to mutate
    @param max_swaps: maximum amount of pairs to swap atom types of
    @return: mutated cluster with some of its atomic types swapped
    """

    num_atoms = len(cluster)
    num_swaps = np.random.randint(max_swaps)
    rng = np.random.default_rng()

    for i in range(num_swaps):
        atom_indices = rng.choice(num_atoms, size=2, replace=False)
        symbol1 = cluster[atom_indices[0]].symbol
        cluster[atom_indices[0]].symbol = cluster[atom_indices[1]].symbol
        cluster[atom_indices[1]].symbol = symbol1

    return cluster


def type_swap(population, mutation_rate): # TODO: test function, also if original clusters remain unchanged
    """
    Mutates population by swapping atomic types of some atom pairs of a certain number of clusters.

    @param population: list of atom to potentially apply mutations on
    @param mutation_rate: probability of mutation occurring in a cluster
    @return: list of mutated clusters which some types of pairs of atoms swapped
    """

    return [single_type_swap(cluster) for cluster in population if np.random.uniform() < mutation_rate]


def single_mirror_shift(cluster, cluster_size, shift=0.1): # TODO: maybe change how much shift/what amount would be fitting
    """
    Mutates a cluster by mirroring half of it which creates symmetric structures.

    @param cluster: cluster to apply mirror-shift mutation on
    @param cluster_size: size of a single cluster
    @param shift: small shift for the mirrored part such that atoms are not too close
    @return: list of mutated clusters which are mirrored
    """

    cluster.center()
    coords = np.array(cluster.get_positions())
    normal = np.random.uniform(-1, 1, 3)
    normalised_norm = normal / np.linalg.norm(normal)

    # Obtain mirrored coordinates of all atoms lying on the same side of the mirror plane as the normal
    dot_products = coords.dot(normalised_norm)
    # TODO: nice way to deal with odd-sized clusters: add random coordinate? take ceil(size/2) and delete one random coordinate at end?
    # TODO: check if correct number of atoms after operation below and how to deal with case where it's not correct
    coords = coords[dot_products > 0, :]
    mirrored_coords = coords - (shift + 2 * np.outer(dot_products[dot_products > 0], normalised_norm))

    mirrored_cluster = Atoms('H' + str(2 * coords.shape[0]), np.concatenate(coords, mirrored_coords))

    return mirrored_cluster # TODO: test and check if original clusters unchanged


def mirror_shift(population, cluster_size, mutation_rate):
    """
    Mutates population by mirroring half of some clusters which creates symmetric structures.

    @param population: list of atom to potentially apply mutations on
    @param cluster_size: size of a single cluster
    @param mutation_rate: probability of mutation occurring in a cluster
    @return: list of mutated clusters which are mirrored
    """

    return [single_mirror_shift(cluster, cluster_size) for cluster in population if np.random.uniform() < mutation_rate]
