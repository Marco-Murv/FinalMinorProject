#!/bin/python3
import numpy as np

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import LBFGS
from ase.visualize import view
from ase.io import write
import ase.db

from typing import List
#from mating import mating
#import mutators
import argparse
import sys

import random
import math




def parse_args():
    """Parsing the most important parameters
    This will make it easy to run with different values (e.g. on a cluster)
    """
    parser = argparse.ArgumentParser(description='Genetic Algorithm PGGO')
    parser.add_argument('--cluster_size', default=10, type=int, help='Number of atoms per cluster', metavar='')
    parser.add_argument('--pop_size', default=20, type=int, help='Number of clusters in the population', metavar='')
    parser.add_argument('--fitness_func', default="exponential", help='Fitness function', metavar='')
    parser.add_argument('--mating_method', default="roulette", help='Mating Method', metavar='')
    parser.add_argument('--children_perc', default=0.8, type=float, help='Fraction of the population that will have a child', metavar='')
    parser.add_argument('--cluster_radius', default=2.0, type=float, help='Dimension of initial random clusters', metavar='')
    parser.add_argument('--max_no_success', default=10, type=int, help='Consecutive generations allowed without new minimum', metavar='')
    parser.add_argument('--max_gen', default=50, type=int, help='Maximum number of generations', metavar='')
    parser.add_argument('--delta_energy_thr', default=0.01, type=float, help='Minimum difference in energy between clusters (DeltaE threshold)', metavar='')

    args = parser.parse_args()
    return args



def create_cluster_defined(p,  cluster_size) -> Atoms:
    return Atoms('H' + str(cluster_size), p)


def generate_cluster(cluster_size, radius) -> Atoms:
    """Generate a random cluster with set number of atoms
    The atoms will be placed within a (radius x radius x radius) cube.
    Args:
        cluster_size (int)  : Number of atoms per cluster
        radius (float)      : dimension of the space where atoms can be placed.
    Returns:
        new_cluster (Atoms) : Randomly generated cluster
    """

    coords = np.random.uniform(-radius/2, radius/2, (cluster_size, 3)).tolist()
    # TODO: Can we use "mathematical dots" instead of H-atoms
    new_cluster = Atoms('H'+str(cluster_size), coords)
   # print(new_cluster.get_positions())
   # print(coords)
    return new_cluster


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
    for cluster in population:
        cluster.calc = calc
        try:
            optimiser(cluster, maxstep=0.2, logfile=None).run(steps=50)
        except:  # TODO: how to properly handle these error cases?
            print("FATAL ERROR: DIVISION BY ZERO ENCOUNTERED!")
            sys.exit("PROGRAM ABORTED: FATAL ERROR")

        # TODO: Maybe change steps? This is just a guess

    return [cluster.get_potential_energy() for cluster in population]




def sphere(x):
    ans = 0
    for i in range(len(x)):
        ans += x[i] ** 2
    return ans



def EB(pop, Sn, calc):
    pop_copy=pop.copy()
    for cluster in pop_copy:
        cluster.calc = calc
    for i in range(len(pop)):

        random_index = random.sample(range(0, Sn), 3)
        while (random_index[0]==i|random_index[1]==i|random_index[2]==i):
            random_index = random.sample(range(0, Sn), 3)
        E_1 =np.abs( pop_copy[0].get_potential_energy())
        E_2 =np.abs( pop_copy[1].get_potential_energy())
        E_3 =np.abs( pop_copy[2].get_potential_energy())
        sum_E = (E_1 +E_2 + E_3)
        if (sum_E!=0):
            p_1 = E_1 /sum_E
            p_2 = E_2 / sum_E
            p_3 = E_3 / sum_E
            new_x = create_cluster_defined((1.0/3.0)*(pop_copy[0].get_positions()
                                                      + pop_copy[1].get_positions() + pop_copy[2].get_positions())
                                           + (p_2-p_1)*(pop_copy[0].get_positions()-pop_copy[1].get_positions())
                                           + (p_3-p_2)*(pop_copy[1].get_positions()-pop_copy[2].get_positions())
                                           + (p_1-p_3)*(pop_copy[2].get_positions()-pop_copy[0].get_positions()), 10)
            new_x.calc = calc
            if (new_x.get_potential_energy()<=pop[i].get_potential_energy()):
                pop[i]= new_x

    return pop

def OL(pop, Sn, calc):
    random_index1 = random.sample(range(0, Sn), 1)
    random_index2 = random.sample(range(0, Sn), 4)
    new_x= create_cluster_defined(pop[random_index1[0]].get_positions() +
                                                  (pop[random_index2[0]].get_positions()+ pop[random_index2[1]].get_positions()
                                                     -pop[random_index2[2]].get_positions()
                                                   - pop[random_index2[3]].get_positions()), 10)
    new_x.calc = calc
    if (new_x.get_potential_energy() <= pop[random_index1[0]].get_potential_energy()):
        pop[random_index1[0]] = new_x

    return pop



# Employee Bee
def EBee(X, f, trials):
    for i in range(len(X)):
        V = []
        R = X.copy()
        R.remove(X[i])
        r = random.choice(R)

        for j in range(len(X[0])):
            V.append((X[i][j] + random.uniform(-1, 1) * (X[i][j] - r[j])))

        if f(X[i]) < f(V):
            trials[i] += 1
        else:
            X[i] = V
            trials[i] = 0
    return X, trials


def P(X, f):
    P = []
    sP = sum([1 / (1 + f(i)) for i in X])
    for i in range(len(X)):
        P.append((1 / (1 + f(X[i]))) / sP)

    return P


# Onlooker Bee
def OBee(X, f, trials):
    Pi = P(X, f)

    for i in range(len(X)):
        if random.random() < Pi[i]:
            V = []
            R = X.copy()
            R.remove(X[i])
            r = random.choice(R)

            for j in range(len(X[0])):  # x[0] or number of dimensions
                V.append((X[i][j] + random.uniform(-1, 1) * (X[i][j] - r[j])))

            if f(X[i]) < f(V):
                trials[i] += 1
            else:
                X[i] = V
                trials[i] = 0
    return X, trials




# Scout Bee
def SBee(X, trials, bounds, limit=3):
    for i in range(len(X)):
        if trials[i] > limit:
            trials[i] = 0
            X[i] = [bounds[i][0] + (random.uniform(0, 1) * (bounds[i][1] - bounds[i][0])) for i in range(len(X[0]))]
    return X

def main() -> None:
    # np.random.seed(241)
    np.seterr(divide='raise')

    # Parse possible input, otherwise use default parameters
    p = parse_args()

    # Make local optimisation Optimiser and calculator
    calc = LennardJones(sigma=1.0, epsilon=1.0)  # TODO: Change parameters
    local_optimiser = LBFGS

    # Generate initial population and optimise locally
    population = generate_population(p.pop_size, p.cluster_size, p.cluster_radius)
    # energy = optimise_local(population, calc, local_optimiser)
    #print(add_cluster(population[0], population[0], 10).get_positions())
    for cluster in population:
        cluster.calc = calc
    print(np.min([cluster.get_potential_energy() for cluster in population]))
    for i in range((50)):
        population = EB(population, 20, calc)
        population = OL(population, 20, calc)

    for cluster in population:
        cluster.calc = calc
    print(np.min([cluster.get_potential_energy() for cluster in population]))


    #energies= optimise_local(population, calc, local_optimiser)

'''

    while runs > 0:
        X, Trials = EBee(X, f, Trials)
    
        X, Trials = OBee(X, f, Trials)

        X = SBee(X, Trials, bounds, limit)

        runs -= 1

'''

main()
