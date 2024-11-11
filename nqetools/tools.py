import os
import sys

import numpy as np
from ase import Atoms
from scipy.constants import physical_constants


# import mace off


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


# Conversion factor from Bohr to Angstrom
bohr_to_angstrom = physical_constants["Bohr radius"][0] * 1e10
