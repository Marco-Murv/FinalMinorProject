#!/bin/python3

"""
Distributed parallelisation of the Genetic Algorithm
"""

import genetic_algorithm as ga

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def ga_distributed():
    return 0

if __name__ == "__main__":
    ga_distributed()