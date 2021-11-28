
import argparse
import json
import math
import numpy as np
from typing import Optional, Type
import yaml

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones
from ase.constraints import Hookean
from ase.db import connect
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS
from ase.optimize.optimize import Optimizer
from ase.units import kB
from ase.visualize import view

from filter_results import filter_trajectory

class DummyMPI:
    def __init__(self) -> None:
        pass

    def Get_rank(self):
        return 0
    
    def Get_size(self):
        return 1
    
    def gather(self, x):
        return [x]
    
    def bcast(self, x):
        return x

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
except:
    COMM = DummyMPI()

class CubeConstraint:
    """Constrain atoms inside of a cube."""
    def __init__(self, radius) -> None:
        self.radius = radius
    
    def adjust_positions(self, atoms, positions):
        positions[:] = np.clip(positions, -self.radius, self.radius)
    
    def adjust_forces(self, atoms, forces):
        pass

    def todict(self):
        # Workaround for ase hardcoding constraint names
        return {'name': 'FixAtoms',
                'kwargs': {'indices': [0]}}

class BasinHopping:
    """
    Implementation of the basin hopping algorithm described by Wales, D. J., & Doye, J. P. K. (1997).

    Parameters
    ----------
    atoms: Atoms object
        The atoms object to perform basin hopping on.
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
    optimizer: Optimizer object, optional
        The optimizer used for local optimization.
    optimizer_logfile: string, None, optional
        The logfile to write the output of the optimizer to.
        Use '-' for stdout
        If None (default), the output of the optimizer is discarded.
    """
    def __init__(self,
                 atoms: Atoms,
                 temperature: float=100*kB,
                 step_size: float=0.5,
                 accept_rate: float=0.5,
                 step_size_factor: float=0.9,
                 step_size_interval: int=50,
                 optimizer: Type[Optimizer]=LBFGS,
                 trajectory: Optional[str]=None,
                 optimizer_logfile: Optional[str]=None) -> None:
        self.atoms = atoms
        self.optimizer = optimizer
        self.temperature = temperature
        self.step_size = step_size
        self.accept_rate = accept_rate
        self.step_size_factor = step_size_factor
        self.step_size_interval = step_size_interval
        self.trajectory = None if trajectory is None else Trajectory(trajectory, 'w')
        self.optimizer_logfile = optimizer_logfile
        # Initialise
        self.nTotal = self.nAccept = 0
        self.min_potential_energy = self.atoms.get_potential_energy()
        self.old_potential_energy = self.min_potential_energy
        self.min_atoms = self.atoms.copy()
        if self.trajectory is not None:
            self.trajectory.write(self.atoms)
    
    def run(self, max_steps: int=500, stop_steps: Optional[int]=None, verbose: bool=False) -> None:
        """
        Run the basin hopping algorithm.

        Parameters
        ----------
        max_steps: int, optional
            The maximum number of steps the algorithm will take. (default = 500)
        stop_steps: int, None, optional
            The number of steps, without there being a new minimum, the algorithm will take before stopping.
            If None (default), the algorithm will run for the maximum number of steps.
        verbose: bool, optional
            Print information about each step.
        """
        rank = COMM.Get_rank()
        stop_step_count = 0
        if verbose and rank == 0: print("{:s} {:>5s} {:>16s} {:>8s}".format(" "*13, "Step", "Energy", "Accept"))
        for i in range(max_steps):
            old_positions = self.atoms.get_positions()
            # Displace atoms
            self.displace_atoms()
            # Get the potential energy
            new_potential_energy = self.get_potential_energy()
            # Gather results
            atoms = COMM.gather(self.atoms)
            min_atoms = None if atoms is None else atoms[np.argmin([atom.get_potential_energy() for atom in atoms])]
            self.atoms = COMM.bcast(min_atoms)
            # Update potential energy
            new_potential_energy = self.atoms.get_potential_energy()
            # Check if new global minimum was found
            if new_potential_energy < self.min_potential_energy:
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
            if verbose and rank == 0: print(f"BasinHopping: {i:5d} {new_potential_energy:15.6f}* {str(accept):>8s}")
            # Write to trajectory
            if self.trajectory is not None and rank == 0:
                for x in atoms:
                    self.trajectory.write(x)
            # Set values for next step
            if accept:
                self.old_potential_energy = new_potential_energy
            else:
                self.atoms.set_positions(old_positions)
            # Stop condition
            if stop_steps is not None and stop_step_count >= stop_steps: break
        
        if verbose and rank == 0: print(f"Stopped at iteration {i}.")
    
    def displace_atoms(self) -> None:
        dX = np.random.uniform(-self.step_size, self.step_size, (len(self.atoms), 3))
        self.atoms.set_positions(self.atoms.get_positions() + dX)
    
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
        
        Returns
        -------
        atoms: Atoms
            the generated initial configuration
        """
        # Generate normally distributed points on the surface of the sphere
        XYZ = np.random.normal(size=(cluster_size, 3))
        XYZ /= np.linalg.norm(XYZ, axis=1)[:,np.newaxis]
        # Generate uniformly distributed points
        U = radius*np.cbrt(np.random.uniform(size=(cluster_size)))
        # Calculate uniformly distributed points inside the sphere
        positions = XYZ*U[:,np.newaxis]
        # Initialize atoms object
        # constraint = [Hookean(i, (0,0,0), 15, max_radius) for i in range(cluster_size)]
        constraint = None if max_radius is None else CubeConstraint(max_radius)
        return Atoms(positions=positions, constraint=constraint, calculator=calculator())

def main(**kwargs):
    size = COMM.Get_size()
    rank = COMM.Get_rank()

    if kwargs.get('cluster_size') is None:
        print("Cluster size not given. Please set the cluster size.")
        return

    if size > 1: print(f"Starting basin hopping in rank {rank}")
    atoms = None if rank > 0 else BasinHopping.generate_initial_configuration(kwargs.get('cluster_size'), kwargs.get('radius', 1), kwargs.get('max_radius'))
    atoms = COMM.bcast(atoms)
    basin_hopping = BasinHopping(atoms, kwargs.get('temperature', 100*kB), kwargs.get('step_size', 0.5), kwargs.get('accept_rate', 0.5),
                                 kwargs.get('step_size_factor', 0.9), kwargs.get('step_size_interval', 50), trajectory=kwargs.get('trajectory'))
    basin_hopping.run(kwargs.get('max_steps', 500), kwargs.get('stop_steps'), kwargs.get('verbose', False))
    if size > 1: print(f"Basin hopping completed in rank {rank}")

    if rank == 0:
        min_potential_energy = basin_hopping.get_min_potential_energy()
        min_atoms = basin_hopping.get_min_atoms()
        # Calculate energy to include it in the database
        min_atoms.set_constraint()
        min_atoms.set_calculator(LennardJones())
        min_atoms.get_potential_energy()
        # Filter local minima
        if kwargs.get('filter_type') is not None and kwargs.get('trajectory') is not None:
            filter_trajectory(kwargs.get('trajectory'), kwargs.get('filtered-trajectory'), kwargs.get('filter_type'), kwargs.get('significant_figures', 2), kwargs.get('difference', 0.1))
        # Update or write
        if kwargs.get('database') is not None:
            db = connect(kwargs.get('database'), type="db")
            try:
                row = db.get(natoms=kwargs.get('cluster_size'))
                if min_potential_energy < row.energy:
                    print("Lower global minimum found")
                    db.update(row.id, min_atoms)
            except:
                db.write(min_atoms)
        # Display global minimum
        if kwargs.get('filtered-trajectory') is not None:
            trajectory = Trajectory(kwargs.get('filtered-trajectory'))
            view(trajectory)
        elif kwargs.get('trajectory') is not None:
            trajectory = Trajectory(kwargs.get('trajectory'))
            view(trajectory)
        else:
            view(min_atoms)
        
        print(f"Global minimum = {min_potential_energy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the basin hopping algorithm.")
    # Require either a config file or the cluster size
    parser.add_argument("-f", "--config", type=str, default=None, help="The location of the config file")
    parser.add_argument("-n", "--cluster-size", type=int, default=None, help="The size of the cluster")
    # 
    parser.add_argument("-r", "--radius", type=float, default=1, help="Radius of the sphere the initial atoms configuration is uniformly distributed in")
    parser.add_argument("-c", "--max-radius", type=float, default=None, help="The maximum radius of the cube to constrain the atoms in. If not set, no constraint is placed on the atoms")
    #
    parser.add_argument("-m", "--max-steps", type=int, default=500, help="The maximum number of steps the algorithm will take")
    parser.add_argument("-s", "--stop-steps", type=int, default=None, help="The number of steps, without there being a new minimum, the algorithm will take before stopping. If not set, the algorithm will run for the maximum number of steps")
    #
    parser.add_argument("--temperature", type=float, default=100*kB, help="The temperature parameter for the Metropolis acceptance criterion")
    parser.add_argument("--step-size", type=float, default=0.5, help="The initial value of the step size")
    parser.add_argument("--accept-rate", type=float, default=0.5, help="The desired step acceptance rate")
    parser.add_argument("--step-size-factor", type=float, default=0.9, help="The factor to multiply and divide the step size by")
    parser.add_argument("--step-size-interval", type=int, default=50, help="The interval for how often to update the step size")
    #
    filter_types = parser.add_mutually_exclusive_group()
    filter_types.add_argument("-fn", "--no-filter", action="store_const", dest="filter_type", const=None, default=None, help="Use no filter")
    filter_types.add_argument("-fs", "--filter-significant-figures", action="store_const", dest="filter_type", const="s", help="Use significant figures filter")
    filter_types.add_argument("-fd", "--filter-difference", action="store_const", dest="filter_type", const="d", help="Use difference filter")
    parser.add_argument("-sf", "--significant-figures", type=int, default=2, help="Significant figures to round the potential energy to to check for uniqueness")
    parser.add_argument("-d", "--difference", type=float, default=0.1, help="Minimum potential energy difference between unique local minima to check for uniqueness")
    #
    parser.add_argument("-db", "--database", default=None, help="Database file for storing global minima")
    parser.add_argument("-tr", "--trajectory", default=None, help="Trajectory file for storing local minima")
    parser.add_argument("-trf", "--filtered-trajectory", default=None, help="Trajectory file for storing filtered local minima. If None, filtered local minima are stored in the original trajectory file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print information about each step")

    # Parse args
    args = parser.parse_args()
    # Load config file if some file was provided
    if args.config is not None:
        config = None
        with open(args.config, 'r') as file:
            if args.config.endswith('.yaml'):
                config = yaml.load(file, Loader=yaml.FullLoader)
            if args.config.endswith('.json'):
                config = json.load(file)
        vars(args).update(config)
    # Run main
    main(**vars(args))