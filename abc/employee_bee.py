#!/bin/python3
import numpy as np

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import LBFGS
from ase.visualize import view
from ase.io import write
import ase.db

from typing import List
# from mating import mating
# import mutators
import argparse
import artificial_bee_colony_algorithm
import sys

import random
import math




def employee_bee_func(pop, Sn, cluster_size, calc, local_optimiser):

    pop_copy = pop.copy()
    for cluster in pop_copy:
        cluster.calc = calc
    for i in range(len(pop)):
        sum_E = 0
        E_1 = 0
        E_2 = 0
        E_3 = 0
        while sum_E == 0:
            random_index = random.sample(range(0, Sn), 3)
            while random_index[0] == i | random_index[1] == i | random_index[2] == i:
                random_index = random.sample(range(0, Sn), 3)
            E_1 = np.abs(pop_copy[0].get_potential_energy())
            E_2 = np.abs(pop_copy[1].get_potential_energy())
            E_3 = np.abs(pop_copy[2].get_potential_energy())
            sum_E = (E_1 + E_2 + E_3)

        p_1 = E_1 / sum_E
        p_2 = E_2 / sum_E
        p_3 = E_3 / sum_E
        new_x = artificial_bee_colony_algorithm.generate_cluster_with_position((1.0 / 3.0) * (pop_copy[0].get_positions()
                                                                  + pop_copy[1].get_positions() + pop_copy[
                                                                      2].get_positions())
                                                   + (p_2 - p_1) * (pop_copy[0].get_positions() - pop_copy[
                1].get_positions())
                                                   + (p_3 - p_2) * (pop_copy[1].get_positions() - pop_copy[
                2].get_positions())
                                                   + (p_1 - p_3) * (pop_copy[2].get_positions() - pop_copy[
                0].get_positions()),
                                                   cluster_size)
        new_x = artificial_bee_colony_algorithm.optimise_local_each(new_x, calc, local_optimiser)
        if new_x.get_potential_energy() <= pop[i].get_potential_energy():
            pop[i] = new_x

    return pop
