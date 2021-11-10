import numpy as np
from ase import Atoms
from typing import List
import random


def make_child(parent1, parent2) -> List[Atoms]:

    cluster_size = len(parent1.positions)
    coords = parent1.positions[:cluster_size//2] + \
        parent2.positions[cluster_size//2:]

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
    num_children = children_perc * len(population)
    children = []
    parents = []

    if method == "roulette":

        while len(parents) < num_children * 2:
            cluster_i = random.randint(0, num_children-1)

            if population_fitness[cluster_i] > random.random():
                parents.append(population[cluster_i])

    elif method == "tournament":

        while len(parents) < num_children * 2:
            subset_i = [random.randint(0, num_children-1)
                        for i in range(tournament_size)]
            subset_fitness = [population_fitness[i] for i in subset_i]

            winner_i = subset_i[subset_fitness.index(max(subset_fitness))]
            winner = population[winner_i]

            parents.append(winner)

    children = [make_child(parents.pop(), parents.pop())
                for i in range(num_children)]

    return children
