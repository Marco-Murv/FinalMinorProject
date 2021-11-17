"""
Mating procedure

The parent selection can happen in one of two ways: Roulette wheel or tournament
When there are enough parents, the children are created one by one.
"""

"""
NOTES here
    * TODO:
"""




import numpy as np
from ase import Atoms
from typing import List
import math as m
def make_child(parent1, parent2, atol=1e-8) -> List[Atoms]:
    """Making child from two parents

    :param parent1: First parent cluster
    :param parent2: Second parent cluster
    :param atol:  (Default value = 1e-8)
    :returns: child-> Offspring cluster

    """

    cluster_size = len(parent1.positions)

    coords_p1 = np.array(parent1.positions)
    coords_p2 = np.array(parent2.positions)

    # Find the division line to split in two.
    z_ctr_p1 = np.median(coords_p1[:, 2])
    z_ctr_p2 = np.median(coords_p2[:, 2])

    # Take half of one parent and half of the other. for odd N atoms, p1 favored
    coords = np.concatenate((coords_p1[coords_p1[:, 2] >= z_ctr_p1],
                             coords_p2[coords_p2[:, 2] < z_ctr_p2]))

    for i in range(len(coords)):
        for j in range(len(coords[:i])):
            while np.allclose(coords[i], coords[j], atol=atol):
                print("Too close!!")
                coords[i] = [coord + atol * np.random.rand()
                             for coord in coords[i]]

    if coords.size < cluster_size:
        print("PROBLEM IN make_child: not enough atoms in the child.")
        return None

    elif len(coords) > cluster_size:
        print("PROBLEM IN make_child: too many atoms in the child.")
        return None

    child = Atoms('H'+str(len(coords)), coords)
    return child


def mating(pop, pop_fitness, child_perc, method="roulette",
           tournament_size=2) -> List[Atoms]:
    """Generate children for the given population

    :param population: Pulation
    :param population_fitness: the fitness values
    :param children_perc: : fraction of the population in [0, 1]
    :param method: in {"roulette", "tournament"} (Default value = "roulette")
    :param tournament_size: parents per tournament (Default value = 2)
    :returns: children ([Atoms])

    """

    num_children = m.ceil(child_perc * len(pop))
    children = []
    parents = []

    while len(children) < num_children:
        while len(parents) < num_children * 2:
            if method == "roulette":
                # Pick one of the clusters
                cluster_i = np.random.randint(0, len(pop)-1)

                # Randomly decide if it can be a parent or not.
                if pop_fitness[cluster_i] > np.random.random():
                    parents.append(pop[cluster_i])

            elif method == "tournament":
                # Pick a set of cluster indices. FIXME: Prevent twice the same.
                subset_i = [np.random.randint(0, len(pop)-1)
                            for i in range(tournament_size)]
                subset_fitness = [pop_fitness[i] for i in subset_i]

                # Decide on a winner
                winner_i = subset_i[subset_fitness.index(max(subset_fitness))]
                winner = pop[winner_i]

                parents.append(winner)

        new_child = make_child(parents.pop(), parents.pop())

        if new_child != None:
            children.append(new_child)

    return children
