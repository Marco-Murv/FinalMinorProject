#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def LJ(r, sigma, epsilon):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)


if __name__ == "__main__":

    rmin = 0.99
    rmax = 3
    N = 1000

    sigma = 1
    epsilon = 1

    rVals = np.linspace(rmin, rmax, N)
    eVals = LJ(rVals, sigma, epsilon)

    plt.plot(rVals, eVals, 'k--',
             label=rf'$\sigma = {sigma}$, $\varepsilon = {epsilon}$')
    plt.axhline(0, color='k', linestyle='-')
    plt.axvline(0, color='k', linestyle='-')
    plt.xlabel(r"Relative distance $r_{ij}$")
    plt.ylabel(r"Potential Energy $E$")
    plt.legend()

    plt.annotate(s=r'', xy=(0, 0.01 * rmax), xytext=(sigma, 0.01 * rmax),
                 arrowprops=dict(arrowstyle='<->'))
    plt.annotate(s=r"$\sigma$", xy=(0.5, 0.02 * rmax))

    rMinE = rVals[np.argmin(eVals)]
    plt.annotate(s=r'', xy=(rMinE, 0), xytext=(rMinE, -epsilon),
                 arrowprops=dict(arrowstyle='<->'))
    plt.annotate(s=r"$\epsilon$", xy=(rMinE + rmax * 0.01, -epsilon/2))

    plt.tight_layout()
    plt.savefig("lj-potential")
    plt.show()
