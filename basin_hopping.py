
from mpi4py import MPI
import numpy as np
import math

# Initialize MPI related constants
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Constants
# Cluster size
N = 10
# Cluster accept rate
ACCEPT_RATE = 0.5
# Step size adjustment interval
INTERVAL = 10
# Stepsize adjustment factor
FACTOR = 0.9
# Radius of initial configuration in Angstrom
R = 5.5
# Temperature for metropolis acceptance criterion
T = 0.8

# Global variables
stepSize = 0.5
nAccept = 0.0
nTotal = 0.0

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

def basinHopping(X):
    global nTotal, nAccept

    while True:
        newX = displaceConfiguration(X)
        # Local minimisation
        dE = ...
        # Acceptance
        acc = accept(dE)
        # Increase step totals
        nTotal += 1
        nAccept += 1 if acc else 0
        if nTotal % INTERVAL == 0: adjustStepSize()
        # If configuration is not accepted, continue with previous configuration
        if not acc: continue
        # Else, continue with new configuration
        X = newX
        break
    
    return X

def main():
    # Generate initial configurations
    data = [generateConfiguration() for _ in range(SIZE)] if RANK == 0 else None

    # Distribute initial configurations
    X = COMM.scatter(data)

    # Run basin hopping algorithm
    # X = basinHopping(X)

    # Gather results
    result = COMM.gather(X)

    # Print results
    if RANK == 0:
        print(result)

main()