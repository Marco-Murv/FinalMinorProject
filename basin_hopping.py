
from mpi4py import MPI
import numpy as np
import math

from ase import Atoms
from ase.optimize import LBFGS
from ase.calculators.lj import LennardJones

# Initialize MPI related constants
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Constants
# Max iterations
MAX_ITERATIONS = 500
# Cluster size
N = 5
# Cluster accept rate
ACCEPT_RATE = 0.5
# Step size adjustment interval
INTERVAL = 10
# Stepsize adjustment factor
FACTOR = 0.9
# Radius of initial configuration in Angstrom
R = 5.5
# Temperature for metropolis acceptance criterion
BOLTZMANN = 8.61733034e-5 # eV/K
T = 100 * BOLTZMANN

# Global variables
stepSize = 0.5
nAccept = 1
nTotal = 1

def generateConfiguration():
    # Generate normally distributed points on the surface of the sphere
    XYZ = np.random.normal(size=(N, 3))
    XYZ /= np.linalg.norm(XYZ, axis=1)[:,np.newaxis]
    # Generate uniformly distributed points
    U = R*np.cbrt(np.random.uniform(size=(N)))
    # Calculate uniformly distributed points inside the sphere
    return XYZ*U[:,np.newaxis]

def displaceConfiguration(X):
    # Displace points uniformly
    return X + np.random.uniform(-stepSize, stepSize, (N, 3))

def localMinimisation(atoms):
    optimiser = LBFGS(atoms, logfile=None)
    optimiser.run(steps=50)

# Metropolis acceptance criterion
def accept(dE):
    # If ΔE <= 0, always accept
    if dE <= 0: return True
    # If ΔE > 0, sometimes accept
    return math.exp(-dE/T) >= np.random.uniform()

def adjustStepSize():
    global stepSize
    if nAccept / nTotal > ACCEPT_RATE:
        # Too many steps are accepted, increase step size.
        stepSize /= FACTOR
    else:
        # Too few steps are accepted, decrease step size.
        stepSize *= FACTOR

def basinHopping(X, verbose=False):
    global nTotal, nAccept

    atoms = Atoms(positions=X, calculator=LennardJones())
    localMinimisation(atoms)
    minX = X
    prevE = atoms.get_potential_energy()
    minE = prevE
    steps_left = 100

    # Max iterations
    for _ in range(MAX_ITERATIONS):
        newX = displaceConfiguration(X)
        atoms.set_positions(newX)
        # Local minimisation
        localMinimisation(atoms)
        # Potential energy
        E = atoms.get_potential_energy()
        if E < minE:
            minE = E
            minX = newX
            steps_left = 100
        else:
            steps_left -= 1
        dE = E - prevE
        # Acceptance
        acc = accept(dE)
        # Increase step totals
        nTotal += 1
        nAccept += 1 if acc else 0
        if nTotal % INTERVAL == 0: adjustStepSize()
        # Print intermediate result
        if verbose:
            print(f"potential energy = {E}, accepted = {acc}, accept rate = {nAccept / nTotal}")
        # If configuration is not accepted, continue with previous configuration
        if not acc: continue
        # Else, continue with new configuration
        X = newX
        prevE = E
    
    return minE, minX

def main():
    # Generate initial configurations
    data = [generateConfiguration() for _ in range(SIZE)] if RANK == 0 else None

    # Distribute initial configurations
    X = COMM.scatter(data)

    # Run basin hopping algorithm
    print(f"Process {RANK} started")
    E, X = basinHopping(X, True)
    print(f"Process {RANK} finished")

    # Gather results
    potential_energies = COMM.gather(E)
    configurations = COMM.gather(X)

    # Print results
    if RANK == 0:
        print(potential_energies)
        print(configurations)

if __name__ == "__main__":
    main()