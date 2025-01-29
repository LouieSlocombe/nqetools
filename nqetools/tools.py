import os
import sys
import time

import numpy as np
from ase import Atoms
from ase.build import minimize_rotation_and_translation
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.test.fio.vasp.test_vasp_out import atoms


def add_ipi_paths(base=os.path.expanduser("~") + "/i-pi/"):
    """
    Adds i-PI paths to the system path and environment variables.

    Parameters:
    base (str, optional): The base directory for i-PI. Default is the i-PI directory in the user's home directory.

    Returns:
    None
    """
    sys.path.append(base)
    os.environ['PATH'] += f":{base}/bin/"
    return None


def rm_ipi_tmp(tmp=r"/tmp/ipi_localhost"):
    """
    Removes the i-PI temporary file if it exists.

    Parameters:
    tmp (str, optional): The path to the temporary file. Default is "/tmp/ipi_localhost".

    Returns:
    None
    """
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
    return None


def has_pbc(atoms):
    """
    Checks if an ASE atoms object has periodic boundary conditions (PBC).

    Parameters:
    atoms (ase.Atoms): The ASE atoms object to check.

    Returns:
    bool: True if the atoms object has PBC, False otherwise.
    """
    return any(atoms.pbc)


def remove_pbc(atoms):
    """
    Removes the periodic boundary conditions from an ASE atoms object.
    This is important for calculations involving isolated molecules where
    periodicity (like in crystals) is not desired. It ensures that the system
    is treated as an isolated entity without interactions from periodic images.

    Parameters:
    atoms (ase.Atoms): The ASE atoms object from which to remove periodic boundary conditions.

    Returns:
    None
    """
    atoms.set_cell([0, 0, 0])  # Setting the cell size to zero
    atoms.set_pbc([0, 0, 0])  # Turning off periodic boundary conditions
    return None


def add_hydrogen_halfway(atoms, index1, index2):
    """
    Add a hydrogen atom halfway between two atoms in an Atoms object.

    Parameters:
    atoms (Atoms): The ASE Atoms object.
    index1 (int): The index of the first atom.
    index2 (int): The index of the second atom.

    Returns:
    Atoms: The updated Atoms object with the hydrogen atom added.
    """
    atoms = atoms.copy()
    # Get the positions of the two atoms
    pos1 = atoms.positions[index1]
    pos2 = atoms.positions[index2]

    # Calculate the midpoint
    midpoint = (pos1 + pos2) / 2.0

    # Add a hydrogen atom at the midpoint
    atoms += Atoms('H', positions=[midpoint])

    return atoms


def move_atom_halfway(atoms, atom_index, target_index1, target_index2):
    """
    Move an atom to be halfway between two target atoms in an Atoms object.

    Parameters:
    atoms (Atoms): The ASE Atoms object.
    atom_index (int): The index of the atom to move.
    target_index1 (int): The index of the first target atom.
    target_index2 (int): The index of the second target atom.

    Returns:
    Atoms: The updated Atoms object with the atom moved.
    """
    atoms = atoms.copy()
    # Get the positions of the target atoms
    pos1 = atoms.positions[target_index1]
    pos2 = atoms.positions[target_index2]

    # Calculate the midpoint
    midpoint = (pos1 + pos2) / 2.0

    # Move the atom to the midpoint
    atoms.positions[atom_index] = midpoint

    return atoms


def optimise_atom_halfway(atoms, atom_index, target_index1, target_index2, calc, fmax=0.05):
    """
    Move an atom to be halfway between two target atoms, fix the positions of the three atoms,
    perform a geometry optimization, and return the final result without any constraints.

    Parameters:
    atoms (Atoms): The ASE Atoms object.
    atom_index (int): The index of the atom to move.
    target_index1 (int): The index of the first target atom.
    target_index2 (int): The index of the second target atom.
    calc (Calculator): The calculator to be used for the optimization.
    fmax (float): The maximum force criterion for the optimization. Default is 0.05 eV/Å.

    Returns:
    Atoms: The optimized Atoms object without any constraints.
    """
    # Move the atom to be halfway between the two target atoms
    atoms = move_atom_halfway(atoms, atom_index, target_index1, target_index2)

    # Fix the positions of the three atoms
    constraint = FixAtoms(indices=[atom_index, target_index1, target_index2])
    atoms.set_constraint(constraint)

    # Set the calculator
    atoms.set_calculator(calc)

    # Perform the geometry optimization
    BFGS(atoms).run(fmax=fmax)

    # Get the final configuration
    atoms = atoms[-1]

    # Remove the constraints
    atoms.set_constraint()

    return atoms


def add_hydrogen_at_distance(atoms, index1, index2, distance):
    """
    Add a hydrogen atom at a specified distance from one atom along the line between two atoms in an Atoms object.

    Parameters:
    atoms (Atoms): The ASE Atoms object.
    index1 (int): The index of the first atom.
    index2 (int): The index of the second atom.
    distance (float): The distance from the first atom to place the hydrogen atom.

    Returns:
    Atoms: The updated Atoms object with the hydrogen atom added.
    """
    atoms = atoms.copy()
    # Get the positions of the two atoms
    pos1 = atoms.positions[index1]
    pos2 = atoms.positions[index2]

    # Calculate the direction vector from atom1 to atom2
    direction = pos2 - pos1
    direction /= np.linalg.norm(direction)  # Normalize the direction vector

    # Calculate the position of the hydrogen atom
    hydrogen_position = pos1 + direction * distance

    # Add a hydrogen atom at the calculated position
    atoms += Atoms('H', positions=[hydrogen_position])

    return atoms


def swap_bonding_configuration(atoms, donor_index, hydrogen_index, acceptor_index):
    """
    Swap the bonding configuration from O-H...O to O...H-O in an Atoms object.

    Parameters:
    atoms (Atoms): The ASE Atoms object.
    donor_index (int): The index of the donor oxygen atom.
    hydrogen_index (int): The index of the hydrogen atom.
    acceptor_index (int): The index of the acceptor oxygen atom.

    Returns:
    Atoms: The updated Atoms object with the swapped bonding configuration.
    """
    atoms = atoms.copy()
    # Get the positions of the donor, hydrogen, and acceptor atoms
    donor_pos = atoms.positions[donor_index]
    hydrogen_pos = atoms.positions[hydrogen_index]
    acceptor_pos = atoms.positions[acceptor_index]

    # Calculate the new position for the hydrogen atom
    direction = acceptor_pos - donor_pos
    direction /= np.linalg.norm(direction)  # Normalize the direction vector
    new_hydrogen_pos = acceptor_pos - direction * np.linalg.norm(hydrogen_pos - donor_pos)

    # Update the position of the hydrogen atom
    atoms.positions[hydrogen_index] = new_hydrogen_pos

    return atoms


def time_force_call(atoms, calc, n_reps=3):
    """
    Measure the time taken to calculate forces on an atomic structure multiple times.

    Parameters:
    atoms (ase.Atoms): ASE Atoms object containing the atomic structure.
    calc (ase.Calculator): Calculator to be used for the force calculations.
    n_reps (int, optional): Number of repetitions for the force calculation. Default is 3.

    Returns:
    None
    """
    times = np.zeros(n_reps, dtype=float)
    t0 = time.time()
    for i in range(n_reps):
        tmp = atoms.copy()
        tmp.calc = calc
        tmp.get_forces()
        times[i] = time.time() - t0
        print('Time taken={:.3} s'.format(times[i]), flush=True)
    print('Average time taken={:.3} s std={:.3}'.format(np.mean(times), np.std(times)), flush=True)


def get_fmax(atoms):
    """
    Calculate the maximum force on any atom in the given Atoms object.

    Parameters:
    atoms (ase.Atoms): ASE Atoms object containing the atomic structure.

    Returns:
    float: The maximum force on any atom.
    """
    return np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max())


def align_mols(atoms1, atoms2):
    """
    Align two molecular structures by minimizing their rotational and translational differences.

    Parameters:
    atoms1 (ase.Atoms): The first ASE Atoms object.
    atoms2 (ase.Atoms): The second ASE Atoms object.

    Returns:
    tuple: A tuple containing the aligned ASE Atoms objects (atoms1, atoms2).
    """
    atoms1 = atoms1.copy()
    atoms2 = atoms2.copy()
    atoms1.center()
    atoms2.center()
    # Minimize the rotation and translation
    minimize_rotation_and_translation(atoms1, atoms2)
    return atoms1, atoms2


def align_principal_axis(atoms: Atoms, axis: str = 'z') -> Atoms:
    """
    Rotate the given Atoms object so that its principal axis with the largest
    moment of inertia is aligned along the specified axis ('x', 'y', or 'z').

    Parameters
    ----------
    atoms : ase.Atoms
        The Atoms object to be aligned.
    axis : str, optional
        The Cartesian axis to align to, one of 'x', 'y', or 'z'. Default is 'z'.

    Returns
    -------
    aligned_atoms : ase.Atoms
        The rotated (aligned) Atoms object.
    """

    # Map axis strings to direction vectors
    directions = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0]),
    }

    # Validate the requested axis
    if axis not in directions:
        raise ValueError("axis must be one of 'x', 'y', or 'z'")

    # Ensure the Atoms object is centered
    atoms = atoms.copy()
    atoms.center()

    # Compute principal axes
    # evalues are sorted ascending, so evecs[2] is the axis with the largest eigenvalue
    evalues, evecs = atoms.get_moments_of_inertia(vectors=True)

    # This is the principal axis with the largest moment of inertia
    principal_axis = evecs[2]

    # Rotate the Atoms so that 'principal_axis' aligns with the chosen axis
    target_vector = directions[axis]
    atoms.rotate(principal_axis, target_vector, center='COM')

    return atoms


def round_sf(value, sig_figs=3):
    """
    Round a float to a specified number of significant figures using NumPy.

    Parameters:
    value (float): The number to be rounded.
    sig_figs (int): The number of significant figures to round to.

    Returns:
    float: The rounded number.
    """
    if value == 0:
        return 0
    else:
        return np.round(value, sig_figs - int(np.floor(np.log10(abs(value)))) - 1)


def get_file_extension(file_path):
    """
    Returns the file extension for the given file path.

    Parameters:
    file_path (str): The path to the file.

    Returns:
    str: The file extension.
    """
    _, file_extension = os.path.splitext(file_path)
    return file_extension
