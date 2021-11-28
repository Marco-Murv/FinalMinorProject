
import argparse
import numpy as np
import numpy.typing as npt
from typing import List

from ase.atoms import Atoms
from ase.io.trajectory import Trajectory

def filter_significant_figures(atoms: List[Atoms], significant_figures: int) -> npt.NDArray:
    """
    Remove similar local minima from the list based on the relevant number significant figures of the potential energy.

    Parameters
    ----------
    atoms: List[Atoms]
        List of all local minima found
    significant_figures: int
        Number of significant figures to round the potential energy to to find the unique local minima
    
    Returns
    -------
    atoms: ndarray
        List of all unique local minima
    """
    # Initialise atoms array first because ase.atoms.Atoms will be unpacked by numpy into an array of ase.atom.Atom
    _atoms = np.empty(len(atoms), dtype=object)
    # Sort atoms by ascending potential energy
    _atoms[:] = sorted(atoms, key=lambda a: a.get_potential_energy())
    # Get all potential energies
    potential_energies = np.array([a.get_potential_energy() for a in _atoms])
    # Remove values above zero
    indexes = potential_energies < 0
    _atoms = _atoms[indexes]
    potential_energies = potential_energies[indexes]

    # Rounding
    # 1. Calculate order of magnitude in 10^i
    order_of_magnitudes = np.floor(np.log10(np.abs(potential_energies))).astype(np.int32)
    # 2. Calculate position of msd in relation to the decimal point
    positions_msd = -(order_of_magnitudes + 1)
    # 3. Calculate position for rounding to significant figures in relation to the decimal point
    rounding = positions_msd + significant_figures
    # 4. Round energies
    rounded = np.array([round(energy, d) for (energy, d) in zip(potential_energies, rounding)])

    # Only keep unique
    _, indexes = np.unique(rounded, return_index=True)
    _atoms = _atoms[indexes]
    potential_energies = potential_energies[indexes]
    
    return _atoms

def filter_difference(atoms: List[Atoms], difference: float) -> npt.NDArray:
    """
    Remove similar local minima from the list based on the minimum difference between the potential energy.

    Parameters
    ----------
    atoms: List[Atoms]
        List of all local minima found
    difference: float
        Minimum potential energy difference between unique local minima to check for uniqueness
    
    Returns
    -------
    atoms: ndarray
        List of all unique local minima
    """
    # Sort atoms by ascending potential energy
    atoms = sorted(atoms, key=lambda a: a.get_potential_energy())

    # Always select first element
    E = atoms[0].get_potential_energy()
    new_atoms = list()
    new_atoms.append(atoms[0])
    # Append new elements if potential energy difference is greater than diff
    for a in atoms[1:]:
        potential_energy = a.get_potential_energy()
        if potential_energy >= 0: break
        if potential_energy - E >= difference:
            E = potential_energy
            new_atoms.append(a)

    # Initialise atoms array first because ase.atoms.Atoms will be unpacked by numpy into an array of ase.atom.Atom
    _atoms = np.empty(len(new_atoms), dtype=object)
    _atoms[:] = new_atoms

    return _atoms

def filter_trajectory(input: str, output: str=None, filter_type: str="s", significant_figures: int=2, difference: float=0.1):
    """
    Remove similar local minima from a trajectory based on the relevant number significant figures of the potential energy.

    Parameters
    ----------
    input: str
        File path to the input trajectory
    output: str, None, optional
        File path to the trajectory to store filtered local minima
        If None, the input file will be replaced by the new trajectory with filtered local minima
    filter_type: {'s', 'd'}, optional
        Which filter type to use:
        - s : significant figures filter
        - d : difference filter \n
        (default = 's')
    significant_figures: int, optional
        Number of significant figures to round the potential energy to to find the unique local minima
        (default = 2)
    difference: float
        Minimum potential energy difference between unique local minima to check for uniqueness
        (default = 0.1)
    """
    if output is None:
        output = input
    # Load input trajectory
    trajectory = Trajectory(input)
    # Filter atoms
    if filter_type == "s":
        atoms = filter_significant_figures(trajectory[:], significant_figures)
    elif filter_type == "d":
        atoms = filter_difference(trajectory[:], difference)
    else:
        raise ValueError("Invalid value for filter_type")
    # Close trajectory
    trajectory.close()

    # Load output trajectory
    trajectory = Trajectory(output, 'w')
    # Write atoms
    for a in atoms:
        trajectory.write(a)
    # Close trajectory
    trajectory.close()

def main(**kwargs):
    filter_trajectory(kwargs.get('input'), kwargs.get('output', None), kwargs.get('filter_type', 's'), kwargs.get('significant_figures', 2), kwargs.get('difference', 0.1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file")
    filter_types = parser.add_mutually_exclusive_group()
    filter_types.add_argument("-fs", "--filter-significant-figures", action="store_const", dest="filter_type", const="s", default="s", help="Use significant figures filter")
    filter_types.add_argument("-fd", "--filter-difference", action="store_const", dest="filter_type", const="d", help="Use difference filter")
    parser.add_argument("-sf", "--significant-figures", type=int, default=2, help="Significant figures to round the potential energy to to check for uniqueness")
    parser.add_argument("-d", "--difference", type=float, default=0.1, help="Minimum potential energy difference between unique local minima to check for uniqueness")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file (default = input file)")

    args = parser.parse_args()

    main(**vars(args))