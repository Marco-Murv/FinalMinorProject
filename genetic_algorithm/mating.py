"""
Mating procedure

The parent selection can happen in one of two ways: Roulette wheel or tournament
When there are enough parents, the children are created one by one.
"""


import numpy as np
from ase import Atoms
from typing import List
import random
import math as m


def make_child(parent1, parent2) -> List[Atoms]:
    """Making child from two parents
    """
    print("Making Child")

    coords_parent1 = parent1.positions[np.argsort(parent1.positions[:,2])]
    coords_parent2 = parent2.positions[np.argsort(parent1.positions[:,2])]

    np.sort

    cluster_size = len(parent1.positions)

    coords = coords_parent1[:cluster_size//2] + coords_parent2[cluster_size//2:]
   
    if len(coords) < len(parent1.positions):
        print("PROBLEM IN make_child: not enough atoms in the child.")

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

    if method == "roulette":

        while len(parents) < num_children * 2:
            # Pick one of the clusters
            cluster_i = random.randint(0, len(population)-1)

            # Randomly decide if it can be a parent or not.
            if population_fitness[cluster_i] > random.random():
                parents.append(population[cluster_i])

    elif method == "tournament":

        while len(parents) < num_children * 2:
            # Pick a set of cluster indices. FIXME: Prevent twice the same.
            subset_i = [random.randint(0, len(population)-1) for i in range(tournament_size)]
            subset_fitness = [population_fitness[i] for i in subset_i]

            # Decide on a winner
            winner_i = subset_i[subset_fitness.index(max(subset_fitness))]
            winner = population[winner_i]

            parents.append(winner)

    children = [make_child(parents.pop(), parents.pop())
                for i in range(num_children)]

    return children
