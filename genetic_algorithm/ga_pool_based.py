#!/bin/python3
"""
Pool-based parallelisation of the genetic algorithm
"""

import genetic_algorithm as ga

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def pool_based():
    return 0

if __name__ == "__main__":
    pool_based()