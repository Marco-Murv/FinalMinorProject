
from mpi4py import MPI
import numpy as np
import math
from typing import List, Tuple

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import LBFGS
from ase.optimize.optimize import Optimizer
from ase.units import kB
from ase.visualize import view

# Initialize MPI related constants
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

class CubeConstraint:
    """Constrain atoms inside of a cube."""
    def __init__(self, radius) -> None:
        self.radius = radius
    
    def adjust_positions(self, atoms, positions):
        positions[:] = np.clip(positions, -self.radius, self.radius)
    
    def adjust_forces(self, atoms, forces):
        pass

class BasinHopping:
    """
    Implementation of the basin hopping algorithm described by Wales, D. J., & Doye, J. P. K. (1997).

    Parameters
    ----------
    atoms: Atoms object
        The atoms object to perform basin hopping on.
    optimizer: Optimizer object, optional
        The optimizer used for local optimization.
    optimizer_logfile: string, None, optional
        The logfile to write the output of the optimizer to.
        Use '-' for stdout
        If None (default), the output of the optimizer is discarded.
    temperature: float, optional
        The temperature parameter for the Metropolis acceptance criterion.
    step_size: float, optional
        The initial value of the step size.
    accept_rate: float, optional
        The desired step acceptance rate.
    factor: float, optional
        The factor to multiply and divide the step size by.
    interval: float, optional
        The interval for how often to update the step size.
    """
    def __init__(self,
                 atoms: Atoms,
                 optimizer: Optimizer=LBFGS,
                 optimizer_logfile: str=None,
                 temperature: float=100*kB,
                 step_size: float=0.5,
                 accept_rate: float=0.5,
                 factor: float=0.9,
                 interval: int=50) -> None:
        self.atoms = atoms
        self.optimizer = optimizer
        self.optimizer_logfile = optimizer_logfile
        self.temperature = temperature
        self.step_size = step_size
        self.accept_rate = accept_rate
        self.step_size_factor = factor
        self.step_size_interval = interval
        self.nTotal = self.nAccept = 0
        # Minima
        self.min_potential_energy = self.atoms.get_potential_energy()
        self.min_atoms = self.atoms.copy()
        self.minima = [(self.min_potential_energy, self.min_atoms)]
    
    def run(self, max_steps: int=5000, stop_steps: int=None, verbose: bool=True) -> Tuple[float, Atoms, List[Tuple[float, List[List[float]]]]]:
        """
        Run the basin hopping algorithm.

        Parameters
        ----------
        max_steps: int, optional
            The maximum number of steps the algorithm will take.
        stop_steps: int, optional
            The number of steps, without there being a new minimum, the algorithm will take before stopping.
            If None (default), the algorithm will run for the maximum number of steps.
        verbose: bool, optional
            Print information about each step.
        """
        stop_step_count = 0
        old_potential_energy = self.min_potential_energy
        if verbose: print("{:s} {:>5s} {:>16s} {:>8s}".format(" "*13, "Step", "Energy", "Accept"))
        for i in range(max_steps):
            old_positions = self.atoms.get_positions()
            # Displace atoms
            self.displace_atoms()
            # Get the potential energy
            new_potential_energy = self.get_potential_energy()
            # Check if new global minimum was found
            if new_potential_energy < self.min_potential_energy:
                self.min_atoms = self.atoms.copy()
                self.min_potential_energy = new_potential_energy
                stop_step_count = 0
            else:
                stop_step_count += 1
            # Acceptance criterion
            accept = self.accept(old_potential_energy, new_potential_energy)
            # Step size adjustment
            if accept: self.nAccept += 1
            self.nTotal += 1
            if self.nTotal % self.step_size_interval: self.adjust_step_size()
            # Log step
            if verbose: print(f"BasinHopping: {i:5d} {new_potential_energy:15.6f}* {str(accept):>8s}")
            # Set old values and save configuration
            if accept:
                old_potential_energy = new_potential_energy
                self.minima.append((new_potential_energy, self.atoms.get_positions()))
            else:
                self.atoms.set_positions(old_positions)
            # Stop condition
            if stop_steps is not None and stop_step_count >= stop_steps: break
        
        print(f"Stopped at iteration {i}.")
        return self.min_potential_energy, self.min_atoms, self.minima
        
    
    def displace_atoms(self) -> None:
        dX = np.random.uniform(-self.step_size, self.step_size, (len(self.atoms), 3))
        self.atoms.translate(dX)
    
    def get_potential_energy(self) -> float:
        # Ignore divide by zero errors
        with np.errstate(divide='ignore', invalid='ignore'):
            self.optimizer(self.atoms, logfile=self.optimizer_logfile).run(steps=50)
            return self.atoms.get_potential_energy()
    
    def accept(self, old_potential_energy: float, new_potential_energy: float) -> bool:
        """Metropolis acceptance criterion.
        If ΔE <= 0, always accept.
        If ΔE > 0, sometimes accept.
        """
        dE = new_potential_energy - old_potential_energy
        return True if dE <= 0 else math.exp(-dE/self.temperature) >= np.random.uniform()
    
    def adjust_step_size(self) -> None:
        if self.nAccept / self.nTotal > self.accept_rate:
            self.step_size /= self.step_size_factor
        else:
            self.step_size *= self.step_size_factor

def generateConfiguration(N: int, radius: int) -> np.ndarray:
    # Generate normally distributed points on the surface of the sphere
    XYZ = np.random.normal(size=(N, 3))
    XYZ /= np.linalg.norm(XYZ, axis=1)[:,np.newaxis]
    # Generate uniformly distributed points
    U = radius*np.cbrt(np.random.uniform(size=(N)))
    # Calculate uniformly distributed points inside the sphere
    return XYZ*U[:,np.newaxis]

def main():
    atoms = Atoms(positions=generateConfiguration(N=3, radius=1), constraint=CubeConstraint(radius=1.5), calculator=LennardJones())
    potential_energy, atoms, minima = BasinHopping(atoms=atoms).run(1000, None)

    potential_energy = COMM.gather(potential_energy)
    atoms = COMM.gather(atoms)
    minima = COMM.gather(minima)

    if RANK == 0:
        min_potential_energy = np.min(potential_energy)
        min_atoms = atoms[np.argmin(potential_energy)]
        # Flatten minima
        minima = [j for i in minima for j in i]
        # Sort minima
        minima = minima.sort(key=lambda x:x[0])
        # Display global minimum
        view(min_atoms)
        print(f"Global minimum = {min_potential_energy}")
        print(min_atoms.get_positions())

if __name__ == "__main__":
    main()