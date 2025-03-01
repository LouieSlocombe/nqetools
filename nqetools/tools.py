import glob
import os
import sys
import time

import ipi
import numpy as np
from ase import Atoms
from ase.build import minimize_rotation_and_translation
from ase.constraints import FixAtoms
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.optimize import BFGS


def add_ipi_paths(base: str = None) -> None:
    """
    Adds i-PI paths to the system path and environment variables.

    Parameters:
    base (str, optional): The base directory for i-PI. Default is the i-PI directory in the user's home directory.

    Returns:
    None
    """
    if base is None:
        base = os.path.expanduser("~") + "/i-pi/"
    sys.path.append(base)
    os.environ['PATH'] += f":{base}/bin/"
    return None


def get_ipi_driver() -> str:
    """
    Get the path to the i-PI driver executable.

    This function locates the i-PI driver executable by finding the path
    to the i-PI package and appending the relative path to the executable.

    Returns:
    str: The full path to the i-PI driver executable.
    """
    tmp = ipi.__file__.split('__init__.py')[0]
    return os.path.join(tmp, 'bin', 'i-pi-driver')


def rm_ipi_tmp(tmp_dir: str = "/tmp") -> None:
    """
    Removes any file in the given directory that starts with 'ipi_'.

    Parameters:
    tmp_dir (str, optional): The directory to search for files. Default is '/tmp'.

    Returns:
    None
    """
    # Search for files starting with 'ipi_'
    files = glob.glob(os.path.join(tmp_dir, 'ipi_*'))

    # Remove each file found
    for file in files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
    return None


def has_pbc(atoms: Atoms) -> bool:
    """
    Checks if an ASE atoms object has periodic boundary conditions (PBC).

    Parameters:
    atoms (ase.Atoms): The ASE atoms object to check.

    Returns:
    bool: True if the atoms object has PBC, False otherwise.
    """
    return any(atoms.pbc)


def remove_pbc(atoms: Atoms) -> None:
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


def cluster_atoms(atoms: Atoms, multi=1.0) -> list[Atoms]:
    """
    Clusters atoms based on their natural cutoffs.

    This function uses the NeighborList to determine clusters of atoms
    based on their natural cutoffs. It iterates through each atom,
    checking if it has been visited, and if not, it performs a depth-first
    search to find all connected atoms, forming a cluster.

    Parameters:
    atoms (ase.Atoms): The ASE Atoms object to be clustered.

    Returns:
    list[ase.Atoms]: A list of ASE Atoms objects, each representing a cluster.
    """
    cutoffs = natural_cutoffs(atoms)

    # Adjust cutoffs by a factor
    cutoffs = [cutoff * multi for cutoff in cutoffs]

    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    visited = [False] * len(atoms)
    clusters_indices = []

    for i in range(len(atoms)):
        if visited[i]:
            continue
        cluster, stack = [], [i]
        while stack:
            j = stack.pop()
            if not visited[j]:
                visited[j] = True
                cluster.append(j)
                indices, _ = nl.get_neighbors(j)
                stack.extend(neighbor for neighbor in indices if not visited[neighbor])
        clusters_indices.append(cluster)

    return [atoms[indices] for indices in clusters_indices]


def cluster_non_hydrogen_atoms(atoms: Atoms) -> tuple[list[int], list[int]]:
    """
    Clusters non-hydrogen atoms based on their natural cutoffs and returns
    indices of atoms in each cluster.

    This function uses the cluster_atoms function to cluster atoms, and then
    returns the indices of non-hydrogen atoms in each cluster.

    Parameters:
    atoms (ase.Atoms): The ASE Atoms object to be clustered.

    Returns:
    tuple[list[int], list[int]]: Two lists containing the indices of atoms
                               in each of the two non-hydrogen clusters.
    """
    # Cluster all atoms
    clusters = cluster_atoms(atoms)
    assert len(clusters) == 2

    # Get indices of atoms in each cluster
    idx1 = [i for i in clusters[0] if atoms.get_chemical_symbols()[i] != "H"]
    idx2 = [i for i in clusters[1] if atoms.get_chemical_symbols()[i] != "H"]

    return idx1, idx2


def reindex_atoms_by_cluster(atoms: Atoms) -> Atoms:
    """
    Reindex atoms by their clusters.

    This function takes an ASE Atoms object, clusters the atoms using the
    `cluster_atoms` function, and then rejoins the clusters into a single
    Atoms object.

    Parameters:
    atoms (ase.Atoms): The ASE Atoms object to be reindexed by clusters.

    Returns:
    ase.Atoms: A new ASE Atoms object with atoms reindexed by clusters.
    """
    clusters = cluster_atoms(atoms)
    # Rejoin the clusters into a single Atoms object
    joined_atoms = clusters[0]
    for cluster in clusters[1:]:
        joined_atoms += cluster
    return joined_atoms


def move_com_to_origin(atoms: Atoms) -> Atoms:
    """
    Moves a set of atoms so that its center of mass is at the origin.

    Parameters:
    atoms (ase.Atoms): The ASE Atoms object.

    Returns:
    ase.Atoms: The modified Atoms object.
    """
    com = atoms.get_center_of_mass()
    atoms.positions -= com
    return atoms


def move_clusters_to_distance(cluster1: Atoms,
                              cluster2: Atoms,
                              index1: int,
                              index2: int,
                              target_distance: float) -> Atoms:
    """
    Moves two clusters of atoms along the vector connecting two target atoms
    such that the target atoms are separated by a given distance.

    Parameters:
        cluster1 (Atoms): The first set of atoms.
        cluster2 (Atoms): The second set of atoms.
        index1 (int): Index of the reference atom in cluster1.
        index2 (int): Index of the reference atom in cluster2.
        target_distance (float): The desired final distance between the target atoms.

    Returns:
        Atoms: A new Atoms object containing both clusters, repositioned.
    """
    # Get the current positions of the target atoms
    pos1 = cluster1.positions[index1]
    pos2 = cluster2.positions[index2]

    # Compute the vector from atom1 to atom2
    vec = np.subtract(pos2, pos1)
    current_distance = np.linalg.norm(vec)

    if current_distance == 0:
        raise ValueError("The selected atoms are at the same position; cannot determine a valid direction.")

    # Normalize the vector
    unit_vec = vec / current_distance

    # Compute the shift needed to achieve the target distance
    shift = (np.subtract(target_distance, current_distance)) / 2

    # Move clusters in opposite directions along the vector
    cluster1.positions -= shift * unit_vec  # Move cluster1 backward
    cluster2.positions += shift * unit_vec  # Move cluster2 forward

    # Combine the moved clusters into a single Atoms object
    combined_atoms = cluster1 + cluster2

    return combined_atoms


def move_to_distances(atoms: Atoms,
                      index1: int,
                      index2: int,
                      distances: list[float]) -> list[Atoms]:
    # Split the atoms into two clusters
    clusters = cluster_atoms(atoms)
    if len(clusters) != 2:
        raise ValueError("The input Atoms object must contain exactly two clusters.")

    cluster1, cluster2 = clusters

    index2 = index2 - len(cluster2)

    # Move the clusters to the specified distances
    moved_atoms_list = []
    for distance in distances:
        moved_atoms = move_clusters_to_distance(cluster1, cluster2, index1, index2, distance)
        moved_atoms_list.append(moved_atoms)

    return moved_atoms_list


def get_fes_times(timestep: float, total_steps: int, fes_arrays: list[np.ndarray]) -> list[float]:
    """
    Calculate time stamps for FES arrays based on simulation parameters.

    Parameters:
    timestep (float): Timestep in femtoseconds.
    total_steps (int): Total number of simulation steps.
    fes_arrays (list[np.ndarray]): List of FES arrays.

    Returns:
    list[float]: List of time points in picoseconds for each FES array.
    """
    n_arrays = len(fes_arrays)
    if n_arrays == 0:
        return []

    # Total time in ps (convert fs to ps by dividing by 1000)
    total_time = (timestep * total_steps) / 1000.0

    # Calculate time points evenly spaced across the simulation
    time_points = [i * total_time / (n_arrays - 1) if n_arrays > 1 else total_time
                   for i in range(n_arrays)]

    return [round_sf(t, sig_figs=2) for t in time_points]


def make_dimer(atoms, translate=None):
    if translate is None:
        translate = [0.0, 3.4, 0.0]

    # Center the first molecule
    atoms.center()

    # Make a copy
    atoms2 = atoms.copy()

    # Rotate the second molecule
    atoms2.rotate(180, 'z', rotate_cell=False)

    # Translate the second molecule
    atoms2.translate(translate)

    # Combine the two molecules
    combined = atoms + atoms2
    return combined
