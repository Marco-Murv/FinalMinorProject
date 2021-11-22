#!/bin/python3
import numpy as np

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import LBFGS
from ase.visualize import view
from ase.io import write
import ase.db
import sys
from typing import List
# from mating import mating
# import mutators
import argparse
import sys
import employee_bee
import onlooker_bee
import random
import math

cluster_str = 'H'


def parse_args():
    """Parsing the most important parameters
    This will make it easy to run with different values (e.g. on a cluster)
    """
    parser = argparse.ArgumentParser(description='Genetic Algorithm PGGO')
    parser.add_argument('--cluster_size', default=12, type=int, help='Number of atoms per cluster', metavar='')
    parser.add_argument('--pop_size', default=10, type=int, help='Number of clusters in the population', metavar='')
    parser.add_argument('--fitness_func', default="exponential", help='Fitness function', metavar='')
    parser.add_argument('--mating_method', default="roulette", help='Mating Method', metavar='')
    parser.add_argument('--children_perc', default=0.8, type=float,
                        help='Fraction of the population that will have a child', metavar='')
    parser.add_argument('--cluster_radius', default=3.0, type=float, help='Dimension of initial random clusters',
                        metavar='')
    parser.add_argument('--max_no_success', default=10, type=int,
                        help='Consecutive generations allowed without new minimum', metavar='')
    parser.add_argument('--max_gen', default=50, type=int, help='Maximum number of generations', metavar='')
    parser.add_argument('--delta_energy_thr', default=0.01, type=float,
                        help='Minimum difference in energy between clusters (DeltaE threshold)', metavar='')

    args = parser.parse_args()
    return args


def generate_cluster_with_position(p, cluster_size) -> Atoms:
    return Atoms(cluster_str + str(cluster_size), p)


def generate_cluster(cluster_size, radius) -> Atoms:
    """Generate a random cluster with set number of atoms
    The atoms will be placed within a (radius x radius x radius) cube.
    Args:
        cluster_size (int)  : Number of atoms per cluster
        radius (float)      : dimension of the space where atoms can be placed.
    Returns:
        new_cluster (Atoms) : Randomly generated cluster
    """
    # TODO: Can we use "mathematical dots" instead of H-atoms
    return Atoms(cluster_str + str(cluster_size),
                 np.random.uniform(-radius / 2, radius / 2, (cluster_size, 3)).tolist())


def generate_population(popul_size, cluster_size, radius) -> List[Atoms]:
    """Generate initial population.
    Args:
        popul_size (int)    : number of clusters in the population
        cluster_size (int)  : number of atoms in each cluster
        radius (float)      : dimension of the initial random clusters
    Returns:
        (List[Atoms])       : List of clusters
    """
    return [generate_cluster(cluster_size, radius) for i in range(popul_size)]


def optimise_local_each(cluster, calc, optimiser) -> Atoms:
    cluster.calc = calc
    try:
        optimiser(cluster, maxstep=0.2, logfile=None).run(steps=50)
        return cluster
    except:  # TODO: how to properly handle these error cases?
        print("FATAL ERROR: DIVISION BY ZERO ENCOUNTERED!")
        sys.exit("PROGRAM ABORTED: FATAL ERROR")


def optimise_local(population, calc, optimiser) -> List[Atoms]:
    """Local optimisation of the population. The clusters in the population
    are optimised and can be used after this function is called. Moreover,
    calculate and return the final optimised potential energy of the clusters.
    Args:
        population(List[Atoms]) : List of clusters to be locally optimised
        calc (Calculator)       : ASE Calculator for potential energy (e.g. LJ)
        optimiser (Optimiser)   : ASE Optimiser (e.g. LBFGS)
    Returns:
        (List[Atoms])           : Optimised population
    """
    return [optimise_local_each(cluster, calc, optimiser).get_potential_energy() for cluster in population]


def scout_bee(pop, Sn, cluster_size, calc, local_optimiser):
    return pop



if __name__ == "__main__":
    # np.random.seed(241)
    np.seterr(divide='raise')

    # Parse possible input, otherwise use default parameters
    p = parse_args()

    # Make local optimisation Optimiser and calculator
    calc = LennardJones(sigma=1.0, epsilon=1.0)  # TODO: Change parameters
    local_optimiser = LBFGS

    # Generate initial population and optimise locally
    population = generate_population(p.pop_size, p.cluster_size, p.cluster_radius)
    optimise_local(population, calc, local_optimiser)
    for cluster in population:
        cluster.calc = calc
    for i in range(100):
        population = employee_bee.employee_bee_func(population, p.pop_size, p.cluster_size, calc, local_optimiser)
        population = onlooker_bee.onlooker_bee_func(population, p.pop_size, p.cluster_size, calc, local_optimiser)
        for cluster in population:
            cluster.calc = calc
        print(np.min([cluster.get_potential_energy() for cluster in population]))
        # $print(population[1].get_potential_energy())
    for cluster in population:
        cluster.calc = calc
    print(np.min([cluster.get_potential_energy() for cluster in population]))


