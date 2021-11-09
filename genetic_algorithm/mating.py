import numpy as np
from ase import Atoms
from typing import List


def mating(population, population_fitness, children_perc) -> List[Atoms]:
    """Generate children for the given population

    Args:
        population ([Atoms])
        population_fitness ([float])
        children_perc (float)

    Returns:
        children ([Atoms])

    """

    # Select parents based on selection criterion

    # Generate children population with given size

    children = population  # TODO: CHANGE

    return children
