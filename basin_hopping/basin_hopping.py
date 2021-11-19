
import argparse
import math
import numpy as np
import sys
from typing import Any, Dict, List, Optional, Type
import yaml

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones
from ase.optimize import LBFGS
from ase.optimize.optimize import Optimizer
from ase.units import kB
from ase.visualize import view

class DummyMPI:
    def __init__(self) -> None:
        pass

    def Get_rank(self):
        return 0
    
    def Get_size(self):
        return 1
    
    def gather(self, x):
        return [x]

if 'mpi4py' in sys.modules:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
else:
    COMM = DummyMPI()

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
        The temperature parameter for the Metropolis acceptance criterion. (default = 100*kB)
    step_size: float, optional
        The initial value of the step size. (default = 0.5)
    accept_rate: float, optional
        The desired step acceptance rate. (default = 0.5)
    step_size_factor: float, optional
        The factor to multiply and divide the step size by. (default = 0.9)
    step_size_interval: int, optional
        The interval for how often to update the step size. (default = 50)
    """
    def __init__(self,
                 atoms: Atoms,
                 optimizer: Type[Optimizer]=LBFGS,
                 optimizer_logfile: Optional[str]=None,
                 temperature: float=100*kB,
                 step_size: float=0.5,
                 accept_rate: float=0.5,
                 step_size_factor: float=0.9,
                 step_size_interval: int=50) -> None:
        self.atoms = atoms
        self.optimizer = optimizer
        self.optimizer_logfile = optimizer_logfile
        self.temperature = temperature
        self.step_size = step_size
        self.accept_rate = accept_rate
        self.step_size_factor = step_size_factor
        self.step_size_interval = step_size_interval
        self.nTotal = self.nAccept = 0
        # Minima
        self.min_potential_energy = self.atoms.get_potential_energy()
        self.old_potential_energy = self.min_potential_energy
        self.min_atoms = self.atoms.copy()
        self.minima = [[0, self.min_potential_energy, self.min_atoms.copy()]]
    
    def run(self, max_steps: int=5000, stop_steps: Optional[None]=None, verbose: bool=False) -> None:
        """
        Run the basin hopping algorithm.

        Parameters
        ----------
        max_steps: int, optional
            The maximum number of steps the algorithm will take. (default = 5000)
        stop_steps: int, None, optional
            The number of steps, without there being a new minimum, the algorithm will take before stopping.
            If None (default), the algorithm will run for the maximum number of steps.
        verbose: bool, optional
            Print information about each step.
        """
        stop_step_count = 0
        if verbose: print("{:s} {:>5s} {:>16s} {:>8s}".format(" "*13, "Step", "Energy", "Accept"))
        for i in range(max_steps):
            old_positions = self.atoms.get_positions()
            # Displace atoms
            self.displace_atoms()
            # Get the potential energy
            new_potential_energy = self.get_potential_energy()
            # Check if new global minimum was found
            if new_potential_energy < self.min_potential_energy:
                self.minima.append([self.nTotal+1, new_potential_energy, self.atoms.copy()])
                self.min_atoms = self.atoms.copy()
                self.min_potential_energy = new_potential_energy
                stop_step_count = 0
            else:
                stop_step_count += 1
            # Acceptance criterion
            accept = self.accept(self.old_potential_energy, new_potential_energy)
            # Step size adjustment
            if accept: self.nAccept += 1
            self.nTotal += 1
            if self.nTotal % self.step_size_interval: self.adjust_step_size()
            # Log step
            if verbose: print(f"BasinHopping: {i:5d} {new_potential_energy:15.6f}* {str(accept):>8s}")
            # Set values for next step
            if accept:
                self.old_potential_energy = new_potential_energy
            else:
                self.atoms.set_positions(old_positions)
            # Stop condition
            if stop_steps is not None and stop_step_count >= stop_steps: break
        print(f"Stopped at iteration {i}.")
    
    def displace_atoms(self) -> None:
        dX = np.random.uniform(-self.step_size, self.step_size, (len(self.atoms), 3))
        self.atoms.translate(dX)
    
    def get_potential_energy(self) -> float:
        # Ignore divide by zero errors
        with np.errstate(divide='ignore', invalid='ignore'):
            self.optimizer(self.atoms, logfile=self.optimizer_logfile).run(steps=50)
            return self.atoms.get_potential_energy() or 1.e23
    
    def accept(self, old_potential_energy: float, new_potential_energy: float) -> bool:
        """Metropolis acceptance criterion. \n
        If ΔE <= 0, always accept. \n
        If ΔE > 0, sometimes accept.
        """
        dE = new_potential_energy - old_potential_energy
        return True if dE <= 0 else math.exp(-dE/self.temperature) >= np.random.uniform()
    
    def adjust_step_size(self) -> None:
        if self.nAccept / self.nTotal > self.accept_rate:
            self.step_size /= self.step_size_factor
        else:
            self.step_size *= self.step_size_factor
    
    def get_min_potential_energy(self) -> float:
        """Get minimum potential energy."""
        return self.min_potential_energy
    
    def get_min_atoms(self) -> Atoms:
        """Get minimum potential energy atom configuration"""
        return self.min_atoms
    
    def get_minima(self) -> List[List[Any]]:
        """
        List of all global minimum energy configurations found with the following columns:
        step: int
            The step the new global minimum was found.
        potential energy: float
            The potential energy of the new global minimum.
        atom configuration: Atoms
            The atom configuration of the new global minimum.
        """
        return self.minima
    
    @staticmethod
    def generate_initial_configuration(cluster_size: int, radius: float=1.0, max_radius: Optional[float]=None, calculator: Type[Calculator]=LennardJones) -> Atoms:
        """
        Generate a random configuration of atoms uniformly distributed in a sphere.

        Parameters
        ----------
        cluster_size: int
            The size of the cluster.
        radius: float, optional
            The radius of the sphere the atoms are uniformly distributed in.
            (default = 1.0)
        max_radius: float, None, optional
            The maximum radius of the cube to constrain the atoms in.
            If None, no constraint is placed on the atoms. (default = None)
        calculator: type(Calculator), optional
            The calculator used to calculate the potential energy surface. (default = LennardJones)
        """
        # Generate normally distributed points on the surface of the sphere
        XYZ = np.random.normal(size=(cluster_size, 3))
        XYZ /= np.linalg.norm(XYZ, axis=1)[:,np.newaxis]
        # Generate uniformly distributed points
        U = radius*np.cbrt(np.random.uniform(size=(cluster_size)))
        # Calculate uniformly distributed points inside the sphere
        positions = XYZ*U[:,np.newaxis]
        # Initialize atoms object
        constraint = None if max_radius is None else CubeConstraint(max_radius)
        return Atoms(positions=positions, constraint=constraint, calculator=calculator())

def main(args: argparse.Namespace):
    atoms = BasinHopping.generate_initial_configuration(args.cluster_size, args.radius, args.max_radius)
    basin_hopping = BasinHopping(atoms=atoms, temperature=args.temperature, step_size=args.step_size, accept_rate=args.accept_rate,
                                 step_size_factor=args.step_size_factor, step_size_interval=args.step_size_interval)
    basin_hopping.run(args.max_steps, args.stop_steps, args.verbose)
    potential_energy = basin_hopping.get_min_potential_energy()
    atoms = basin_hopping.get_min_atoms()
    minima = basin_hopping.get_minima()

    rank = COMM.Get_rank()
    potential_energy = COMM.gather(potential_energy)
    atoms = COMM.gather(atoms)
    minima = COMM.gather(minima)

    if rank == 0:
        min_potential_energy = np.min(potential_energy)
        min_atoms = atoms[np.argmin(potential_energy)]
        # Flatten minima
        minima = [j for i in minima for j in i]
        # Sort minima
        minima = minima.sort(key=lambda x:x[1])
        # Display global minimum
        view(min_atoms)
        print(f"Global minimum = {min_potential_energy}")
        print(min_atoms.get_positions())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the basin hopping algorithm.")
    config_type = parser.add_mutually_exclusive_group(required=True)
    config_type.add_argument("-f", "--config-file", type=str, help="The location of the config file")
    config_type.add_argument("-n", "--cluster-size", type=int, help="The size of the cluster")
    #
    parser.add_argument("-r", "--radius", type=float, default=1, help="Radius of the sphere the initial atoms configuration is uniformly distributed in")
    parser.add_argument("-mr", "--max-radius", type=float, default=None, help="The maximum radius of the cube to constrain the atoms in. If not set, no constraint is placed on the atoms")
    #
    parser.add_argument("-ms", "--max-steps", type=int, default=5000, help="The maximum number of steps the algorithm will take")
    parser.add_argument("-ss", "--stop-steps", type=int, default=None, help="The number of steps, without there being a new minimum, the algorithm will take before stopping. If not set, the algorithm will run for the maximum number of steps")
    #
    parser.add_argument("-T", "--temperature", type=float, default=100*kB, help="The temperature parameter for the Metropolis acceptance criterion")
    parser.add_argument("-s", "--step-size", type=float, default=0.5, help="The initial value of the step size")
    parser.add_argument("-a", "--accept-rate", type=float, default=0.5, help="The desired step acceptance rate")
    parser.add_argument("-sf", "--step-size-factor", type=float, default=0.9, help="The factor to multiply and divide the step size by")
    parser.add_argument("-si", "--step-size-interval", type=int, default=50, help="The interval for how often to update the step size")
    #
    parser.add_argument("-v", "--verbose", action="store_true", help="Print information about each step")

    args = parser.parse_args()
    if args.config_file is not None:
        config = yaml.load(open(args.config_file), Loader=yaml.FullLoader)
        vars(args).update(config)
    
    print(args)

    main(args)