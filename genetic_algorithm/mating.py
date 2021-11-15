"""
Mating procedure

The parent selection can happen in one of two ways: Roulette wheel or tournament
When there are enough parents, the children are created one by one.
"""

"""
NOTES here
    * TODO: Make sure the atoms are not on top of each other.
    * 
"""




import numpy as np
from ase import Atoms
from typing import List
import math as m
def make_child(parent1, parent2) -> List[Atoms]:
    """Making child from two parents

    Args:
        parent1 (List[Atoms]) : First parent cluster
        parent2 (List[Atoms]) : Second parent cluster

    Returns:
        child (Atoms) : Offspring cluster
    """

    cluster_size = len(parent1.positions)

    coords_p1 = np.array(parent1.positions)
    coords_p2 = np.array(parent2.positions)

    # Find the division line to split in two.
    z_center_p1 = np.median(coords_p1[:, 2])
    z_center_p2 = np.median(coords_p2[:, 2])

    # Take half of one parent and half of the other. for odd N atoms, p1 favored
    coords = np.concatenate((coords_p1[coords_p1[:, 2] >= z_center_p1],
                             coords_p2[coords_p2[:, 2] < z_center_p2]))

    if coords.size < cluster_size:
        print("PROBLEM IN make_child: not enough atoms in the child.")
        return None

    elif len(coords) > cluster_size:
        print("PROBLEM IN make_child: too many atoms in the child.")
        return None

    child = Atoms('H'+str(len(coords)), coords)
    return child


def mating(population, population_fitness, children_perc, method="roulette", tournament_size=2) -> List[Atoms]:
    """Generate children for the given population

    Args:
        population (List[Atoms])        : Pulation
        population_fitness (List[float]): the fitness values
        children_perc (float):          : fraction of the population in [0, 1]
        method (string)                 : in {"roulette", "tournament"}
        tournament_size (int)           : parents per tournament

    Returns:
        children ([Atoms])
    """

    num_children = m.ceil(children_perc * len(population))
    children = []
    parents = []

    while len(children) < num_children:
        while len(parents) < num_children * 2:
            if method == "roulette":
                # Pick one of the clusters
                cluster_i = np.random.randint(0, len(population)-1)

                # Randomly decide if it can be a parent or not.
                if population_fitness[cluster_i] > np.random.random():
                    parents.append(population[cluster_i])

            elif method == "tournament":
                # Pick a set of cluster indices. FIXME: Prevent twice the same.
                subset_i = [np.random.randint(0, len(population)-1) for i in range(tournament_size)]
                subset_fitness = [population_fitness[i] for i in subset_i]

                # Decide on a winner
                winner_i = subset_i[subset_fitness.index(max(subset_fitness))]
                winner = population[winner_i]

                parents.append(winner)

        new_child = make_child(parents.pop(), parents.pop())
        if new_child != None:
            children.append(new_child)

    return children  # TODO: Convert to np.array !
