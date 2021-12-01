#!/bin/python3

"""
Sub populations parallelisation of the Genetic Algorithm
"""

import genetic_algorithm as ga

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def ga_sub_populations():
    return 0

if __name__ == "__main__":
    ga_sub_populations()