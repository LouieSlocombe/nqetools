import os

from ase.data.pubchem import pubchem_atoms_search
from ase.visualize import view
import numpy as np
from ase import Atoms
from ase.build import molecule
# import mace off
from mace.calculators import mace_off
from ase.optimize import BFGS

from ase.io import read, write


import nqetools as nqe
from nqetools import get_ts_image


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

atoms = read("malonaldehyde.traj")
view(atoms)
exit()


fmax = 0.01
calc = mace_off(model="large", )
#
# atoms = pubchem_atoms_search(smiles="C(C=O)C=O")
# reactant = add_hydrogen_at_distance(atoms, 0, 1, 1.0)
# product = add_hydrogen_at_distance(atoms, 1, 0, 1.0)
# # view(reactant)
# # view(product)

reactant = nqe.optimise_geom(reactant, calc, fmax=fmax)
view(reactant)

product = nqe.optimise_geom(product, calc, fmax=fmax)
view(product)

neb = nqe.prepare_neb(reactant, product, calc, n_images=5)
view(neb.images)

ts_path = nqe.optimise_neb(neb, fmax=fmax)
view(ts_path)

ts_image = nqe.get_ts_image(ts_path,calc)
view(ts_image)