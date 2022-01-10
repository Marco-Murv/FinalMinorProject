#!/bin/python3


import numpy as np
import matplotlib.pyplot as plt

rho = np.linspace(0, 1, 100)
flin = 1 - 0.7 * rho
fexp = np.exp(-3 * rho)
fhyper = 1/2 * (1 - np.tanh(2*rho - 1))


plt.plot(rho, flin, '-', label=r"$f_i^{linear}$")
plt.plot(rho, fexp, ':', label=r"$f_i^{exponential}$")
plt.plot(rho, fhyper, '-.',label=r"$f_i^{hyperbolic}$")


plt.xlabel(r"Normalised energy $\rho_i$")
plt.ylabel(r"Fitness value $f_i$")

# plt.ylim(ymin=0, ymax=2)
plt.tight_layout()
plt.legend()
plt.show()
