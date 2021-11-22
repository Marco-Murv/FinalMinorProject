#!/bin/python3

import unittest
import mating
import math as m

import genetic_algorithm as ga



class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        self.popul_size = 5;
        self.cluster_size = 10;
        self.radius = 2;
        self.population = ga.generate_population(self.popul_size, self.cluster_size, self.radius)


    def test_population(self):
        self.assertEqual(len(self.population), self.popul_size)
        self.assertEqual(len(self.population[0].positions), self.cluster_size)

    def test_mating(self):
        pass



if __name__ == '__main__':
    unittest.main()