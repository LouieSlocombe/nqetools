"""General-purpose structure and environment helpers.

Two loosely related groups. The first deals with the i-PI installation
itself: locating the driver executable, putting it on the path, and
clearing stale unix sockets left behind by interrupted runs.

The second manipulates ASE structures - adding and moving atoms,
measuring distances, clustering by connectivity, and checking for periodic
boundary conditions - as preparation for the calculations set up
elsewhere in the package.
"""

import glob
import inspect
import os
import sys
import tempfile
import textwrap

import ipi
import numpy as np
from ase.atoms import Atoms
from ase.build import minimize_rotation_and_translation
from ase.constraints import FixAtoms
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.optimize import BFGS


def add_ipi_paths(base: str | None = None) -> None:
    """Adds i-PI paths to the system path and environment variables.

    Parameters
    ----------
    base : str, optional
        The base directory for i-PI. Default is the i-PI directory in the user's home directory.
    """
    if base is None:
        base = os.path.expanduser("~") + "/i-pi/"
    sys.path.append(base)
    os.environ["PATH"] += f":{base}/bin/"


def get_ipi_driver() -> str:
    """Get the path to the i-PI driver executable.

    This function locates the i-PI driver executable by finding the path
    to the i-PI package and appending the relative path to the executable.

    Returns
    -------
    str
        The full path to the i-PI driver executable.
    """
    tmp = ipi.__file__.split("__init__.py")[0]
    return os.path.join(tmp, "bin", "i-pi-driver")


def rm_ipi_tmp(tmp_dir: str | None = None, address: str | None = None) -> None:
    """Removes stale i-PI unix sockets belonging to the current user.

    Only sockets owned by the calling user are considered, so this will not
    disturb i-PI runs started by anyone else sharing the machine.

    Parameters
    ----------
    tmp_dir : str, optional
        The directory to search for sockets. Defaults to the system temporary
        directory (honouring TMPDIR).
    address : str, optional
        Socket address to target, matching only 'ipi_<address>'. If None, every
        'ipi_*' socket owned by the current user is removed - including those of
        the caller's own concurrent runs, so prefer passing an address when one
        is known.
    """
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()

    pattern = f"ipi_{address}" if address is not None else "ipi_*"
    files = glob.glob(os.path.join(tmp_dir, pattern))

    uid = os.getuid()
    for file in files:
        try:
            if os.stat(file).st_uid != uid:
                continue  # Belongs to another user; leave their run alone
            os.remove(file)
        except FileNotFoundError:
            pass


def has_pbc(atoms: Atoms) -> bool:
    """Checks if an ASE atoms object has periodic boundary conditions (PBC).

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE atoms object to check.

    Returns
    -------
    bool
        True if the atoms object has PBC, False otherwise.
    """
    return any(atoms.pbc)


def add_hydrogen_halfway(atoms, index1, index2):
    """Add a hydrogen atom halfway between two atoms in an Atoms object.

    Parameters
    ----------
    atoms : Atoms
        The ASE Atoms object.
    index1 : int
        The index of the first atom.
    index2 : int
        The index of the second atom.

    Returns
    -------
    Atoms
        The updated Atoms object with the hydrogen atom added.
    """
    atoms = atoms.copy()
    pos1 = atoms.positions[index1]
    pos2 = atoms.positions[index2]

    midpoint = (pos1 + pos2) / 2.0

    atoms += Atoms("H", positions=[midpoint])

    return atoms


def move_atom_halfway(atoms, atom_index, target_index1, target_index2):
    """Move an atom to be halfway between two target atoms in an Atoms object.

    Parameters
    ----------
    atoms : Atoms
        The ASE Atoms object.
    atom_index : int
        The index of the atom to move.
    target_index1 : int
        The index of the first target atom.
    target_index2 : int
        The index of the second target atom.

    Returns
    -------
    Atoms
        The updated Atoms object with the atom moved.
    """
    atoms = atoms.copy()
    pos1 = atoms.positions[target_index1]
    pos2 = atoms.positions[target_index2]

    midpoint = (pos1 + pos2) / 2.0

    atoms.positions[atom_index] = midpoint

    return atoms


def optimise_atom_halfway(
    atoms, atom_index, target_index1, target_index2, calc, fmax=0.05
):
    """Relax a structure around an atom placed halfway between two others.

    The moved atom and its two targets are held fixed while the rest of
    the structure relaxes, giving a reasonable transition state guess.
    The constraints are removed before returning.

    Parameters
    ----------
    atoms : Atoms
        The ASE Atoms object.
    atom_index : int
        The index of the atom to move.
    target_index1 : int
        The index of the first target atom.
    target_index2 : int
        The index of the second target atom.
    calc : Calculator
        The calculator to be used for the optimisation.
    fmax : float
        The maximum force criterion for the optimisation. Default is 0.05 eV/Å.

    Returns
    -------
    Atoms
        The optimised Atoms object without any constraints.
    """
    atoms = move_atom_halfway(atoms, atom_index, target_index1, target_index2)

    constraint = FixAtoms(indices=[atom_index, target_index1, target_index2])
    atoms.set_constraint(constraint)

    atoms.set_calculator(calc)

    BFGS(atoms).run(fmax=fmax)

    atoms = atoms[-1]

    atoms.set_constraint()

    return atoms


def add_hydrogen_at_distance(atoms, index1, index2, distance):
    """Add a hydrogen atom at a specified distance from one atom along the line between two atoms in an Atoms object.

    Parameters
    ----------
    atoms : Atoms
        The ASE Atoms object.
    index1 : int
        The index of the first atom.
    index2 : int
        The index of the second atom.
    distance : float
        The distance from the first atom to place the hydrogen atom.

    Returns
    -------
    Atoms
        The updated Atoms object with the hydrogen atom added.
    """
    atoms = atoms.copy()
    pos1 = atoms.positions[index1]
    pos2 = atoms.positions[index2]

    direction = pos2 - pos1
    direction /= np.linalg.norm(direction)

    hydrogen_position = pos1 + direction * distance

    atoms += Atoms("H", positions=[hydrogen_position])

    return atoms


def align_mols(atoms1, atoms2):
    """Align two molecular structures by minimising their rotational and translational differences.

    Parameters
    ----------
    atoms1 : ase.Atoms
        The first ASE Atoms object.
    atoms2 : ase.Atoms
        The second ASE Atoms object.

    Returns
    -------
    tuple
        A tuple containing the aligned ASE Atoms objects (atoms1, atoms2).
    """
    atoms1 = atoms1.copy()
    atoms2 = atoms2.copy()
    atoms1.center()
    atoms2.center()
    minimize_rotation_and_translation(atoms1, atoms2)
    return atoms1, atoms2


def align_principal_axis(atoms: Atoms, axis: str = "z") -> Atoms:
    """Align a structure's largest principal axis with a Cartesian axis.

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

    directions = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }

    if axis not in directions:
        raise ValueError("axis must be one of 'x', 'y', or 'z'")

    atoms = atoms.copy()
    atoms.center()

    # evalues are sorted ascending, so evecs[2] is the axis with the largest eigenvalue
    _evalues, evecs = atoms.get_moments_of_inertia(vectors=True)

    principal_axis = evecs[2]

    target_vector = directions[axis]
    atoms.rotate(principal_axis, target_vector, center="COM")

    return atoms


def round_sf(value, sig_figs=3):
    """Round a float to a specified number of significant figures using NumPy.

    Parameters
    ----------
    value : float
        The number to be rounded.
    sig_figs : int
        The number of significant figures to round to.

    Returns
    -------
    float
        The rounded number.
    """
    if value == 0:
        return 0
    else:
        return np.round(value, sig_figs - int(np.floor(np.log10(abs(value)))) - 1)


def get_file_extension(file_path):
    """Returns the file extension for the given file path.

    Parameters
    ----------
    file_path : str
        The path to the file.

    Returns
    -------
    str
        The file extension.
    """
    _, file_extension = os.path.splitext(file_path)
    return file_extension


def _connected_groups(
    atoms: Atoms, cutoff_scale: float = 1.0, skin: float = 0.3
) -> list[list[int]]:
    """Connected components of the bonding graph, in discovery order.

    Bonds are inferred from ASE's natural cutoffs rather than from any
    explicit topology, so this works on bare coordinates. The traversal is a
    depth-first search from each unvisited atom in turn.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to analyse.
    cutoff_scale : float, optional
        Multiplier on the natural cutoffs. Values above 1.0 make bonding
        more permissive. Default is 1.0.
    skin : float, optional
        Extra separation, in Angstrom, still counted as bonded. Default is
        0.3, matching ASE's own default for :class:`NeighborList`.

    Returns
    -------
    list of list of int
        One list of atom indices per connected group. Empty if `atoms`
        contains no atoms.
    """
    if len(atoms) == 0:
        return []

    cutoffs = [cutoff_scale * c for c in natural_cutoffs(atoms)]
    neighbours = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=skin)
    neighbours.update(atoms)

    visited = [False] * len(atoms)
    groups = []

    for start in range(len(atoms)):
        if visited[start]:
            continue
        group, stack = [], [start]
        while stack:
            current = stack.pop()
            if visited[current]:
                continue
            visited[current] = True
            group.append(current)
            indices, _offsets = neighbours.get_neighbors(current)
            stack.extend(i for i in indices if not visited[i])
        groups.append(group)

    return groups


def cluster_atoms(atoms: Atoms, multi=1.0) -> list[Atoms]:
    """Clusters atoms based on their natural cutoffs.

    Groups atoms by connectivity, inferring bonds from ASE's natural
    cutoffs, and returns each group as its own Atoms object. Atoms within a
    group keep the order the search found them in.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to be clustered.
    multi : float, optional
        Multiplier on the natural cutoffs. Values above 1.0 make bonding
        more permissive, merging clusters that are close but not bonded.
        Default is 1.0.

    Returns
    -------
    list[ase.Atoms]
        A list of ASE Atoms objects, each representing a cluster.
    """
    return [atoms[group] for group in _connected_groups(atoms, multi)]


def cluster_non_hydrogen_atoms(atoms: Atoms) -> tuple[list[int], list[int]]:
    """Cluster atoms by connectivity, reporting only the heavy atoms.

    Wraps :func:`cluster_atoms` and filters hydrogens out of each
    cluster, leaving the heavy-atom skeleton.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to be clustered.

    Returns
    -------
    tuple[list[int], list[int]]
        Two lists containing the indices of atoms in each of the two
        non-hydrogen clusters.
    """
    clusters = cluster_atoms(atoms)
    assert len(clusters) == 2

    idx1 = [i for i in clusters[0] if atoms.get_chemical_symbols()[i] != "H"]
    idx2 = [i for i in clusters[1] if atoms.get_chemical_symbols()[i] != "H"]

    return idx1, idx2


def reindex_atoms_by_cluster(atoms: Atoms) -> Atoms:
    """Reindex atoms by their clusters.

    This function takes an ASE Atoms object, clusters the atoms using the
    `cluster_atoms` function, and then rejoins the clusters into a single
    Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to be reindexed by clusters.

    Returns
    -------
    ase.Atoms
        A new ASE Atoms object with atoms reindexed by clusters.
    """
    clusters = cluster_atoms(atoms)
    joined_atoms = clusters[0]
    for cluster in clusters[1:]:
        joined_atoms += cluster
    return joined_atoms


def move_com_to_origin(atoms: Atoms) -> Atoms:
    """Moves a set of atoms so that its centre of mass is at the origin.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object.

    Returns
    -------
    ase.Atoms
        The modified Atoms object.
    """
    com = atoms.get_center_of_mass()
    atoms.positions -= com
    return atoms


def move_clusters_to_distance(
    cluster1: Atoms, cluster2: Atoms, index1: int, index2: int, target_distance: float
) -> Atoms:
    """Separate two clusters to a set distance between chosen atoms.

    Both clusters move rigidly along the vector joining the two target
    atoms, so their internal geometry is unchanged.

    Parameters
    ----------
    cluster1 : Atoms
        The first set of atoms.
    cluster2 : Atoms
        The second set of atoms.
    index1 : int
        Index of the reference atom in cluster1.
    index2 : int
        Index of the reference atom in cluster2.
    target_distance : float
        The desired final distance between the target atoms.

    Returns
    -------
    Atoms
        A new Atoms object containing both clusters, repositioned. The inputs
        are not modified.
    """
    cluster1 = cluster1.copy()
    cluster2 = cluster2.copy()

    pos1 = cluster1.positions[index1]
    pos2 = cluster2.positions[index2]

    vec = np.subtract(pos2, pos1)
    current_distance = np.linalg.norm(vec)

    if current_distance == 0:
        raise ValueError(
            "The selected atoms are at the same position; cannot determine a valid direction."
        )

    unit_vec = vec / float(current_distance)

    shift = (np.subtract(target_distance, current_distance)) / 2

    cluster1.positions -= shift * unit_vec
    cluster2.positions += shift * unit_vec

    combined_atoms = cluster1 + cluster2

    return combined_atoms


def move_to_distances(
    atoms: Atoms, index1: int, index2: int, distances: list[float]
) -> list[Atoms]:
    """Move two clusters of atoms to specified distances along the vector connecting two target atoms.

    This function splits the input `Atoms` object into two clusters, calculates the vector
    connecting the specified atoms in the clusters, and moves the clusters to achieve the
    specified distances between the target atoms.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object containing the atomic structure.
    index1 : int
        The index of the reference atom in the first cluster.
    index2 : int
        The index of the reference atom in the second cluster.
    distances : list[float]
        A list of target distances to move the clusters.

    Returns
    -------
    list[ase.Atoms]
        A list of ASE Atoms objects, each representing the clusters moved to the specified distances.
    """
    clusters = cluster_atoms(atoms)
    if len(clusters) != 2:
        raise ValueError("The input Atoms object must contain exactly two clusters.")

    cluster1, cluster2 = clusters

    # index2 is a global index into `atoms`; convert it to an index within
    # cluster2 by subtracting the number of atoms that precede it
    index2 = index2 - len(cluster1)

    moved_atoms_list = []
    for distance in distances:
        moved_atoms = move_clusters_to_distance(
            cluster1, cluster2, index1, index2, distance
        )
        moved_atoms_list.append(moved_atoms)

    return moved_atoms_list


def get_fes_times(
    timestep: float, total_steps: int, fes_arrays: list[np.ndarray]
) -> list[float]:
    """Calculate time stamps for FES arrays based on simulation parameters.

    Parameters
    ----------
    timestep : float
        Timestep in femtoseconds.
    total_steps : int
        Total number of simulation steps.
    fes_arrays : list[np.ndarray]
        List of FES arrays.

    Returns
    -------
    list[float]
        List of time points in picoseconds for each FES array.
    """
    n_arrays = len(fes_arrays)
    if n_arrays == 0:
        return []

    # Total time in ps (convert fs to ps by dividing by 1000)
    total_time = (timestep * total_steps) / 1000.0

    time_points = [
        i * total_time / (n_arrays - 1) if n_arrays > 1 else total_time
        for i in range(n_arrays)
    ]

    return [round_sf(t, sig_figs=2) for t in time_points]


def make_dimer(atoms, translate=None, angle=180, axis="z"):
    """Create a dimer by combining two copies of a molecule.

    This function takes an ASE Atoms object, centers the first molecule,
    creates a copy, rotates the copy by a specified angle around a given axis,
    translates the copy, and combines the two molecules into a single Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object representing the molecule.
    translate : list[float], optional
        Translation vector for the second molecule. Default is [0.0, 3.4, 0.0].
    angle : float, optional
        Rotation angle in degrees for the second molecule. Default is 180 degrees.
    axis : str, optional
        Axis of rotation ('x', 'y', or 'z'). Default is 'z'.

    Returns
    -------
    ase.Atoms
        The combined Atoms object representing the dimer.
    """
    if translate is None:
        translate = [0.0, 3.4, 0.0]

    atoms.center()

    atoms2 = atoms.copy()

    atoms2.rotate(angle, axis, rotate_cell=False)

    atoms2.translate(translate)

    combined = atoms + atoms2
    return combined


def convert_code_to_string(code):
    """Converts a Python function or code object to its string representation.

    This function uses the `inspect.getsource` method to retrieve the source
    code of the given function or code object and removes any common leading
    whitespace using `textwrap.dedent`.

    Parameters
    ----------
    code : object
        The Python function or code object to convert.

    Returns
    -------
    str
        The string representation of the source code.
    """
    return textwrap.dedent(inspect.getsource(code))


def get_distance(atoms: Atoms, idx1: int, idx2: int) -> float:
    """Calculate the Euclidean distance between two atoms in an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object containing the atomic structure.
    idx1 : int
        The index of the first atom.
    idx2 : int
        The index of the second atom.

    Returns
    -------
    float
        The Euclidean distance between the two specified atoms.
    """
    pos1 = atoms.positions[idx1]
    pos2 = atoms.positions[idx2]
    return np.linalg.norm(pos1 - pos2)


def closest_corresponding_index(super_atoms, sub_atoms, sub_idx):
    """Find the index of the atom in `super_atoms` that is closest to a specific atom in `sub_atoms`.

    This function calculates the Euclidean distance between the position of a given atom
    in `sub_atoms` (specified by `sub_idx`) and all atoms in `super_atoms`. It then returns
    the index of the atom in `super_atoms` with the smallest distance.

    Parameters
    ----------
    super_atoms : ASE Atoms object
        The larger set of atoms to search within.
    sub_atoms : ASE Atoms object
        The smaller set of atoms containing the target atom.
    sub_idx : int
        The index of the target atom in `sub_atoms`.

    Returns
    -------
    int
        The index of the closest atom in `super_atoms` to the specified atom in `sub_atoms`.
    """
    diff = super_atoms.positions - sub_atoms.positions[sub_idx]
    norm = np.linalg.norm(diff, axis=1)
    return np.argmin(norm)


def _bonded_groups(atoms: Atoms, cutoff_scale: float = 1.0) -> list[list[int]]:
    """Find connected components of the bonding graph.

    Bonds are inferred from ASE's natural cutoffs rather than from any
    explicit topology, so this works on bare coordinates.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to analyse.
    cutoff_scale : float, optional
        Multiplier on the natural cutoffs. Values above 1.0 make bonding
        more permissive. Default is 1.0.

    Returns
    -------
    list of list of int
        One list of atom indices per connected group. Empty if `atoms`
        contains no atoms.
    """
    # skin=0.0 holds this to the natural cutoffs exactly. cluster_atoms takes
    # ASE's default 0.3 A of slack instead and so bonds a little more readily,
    # which is why the two callers do not share a cutoff policy.
    return [sorted(group) for group in _connected_groups(atoms, cutoff_scale, skin=0.0)]


def combine_without_overlaps(
    A: Atoms,
    B: Atoms,
    *,
    bond_cutoff_scale: float = 1.0,
    overlap_cutoff_scale: float = 1.0,
) -> Atoms:
    """Merge two structures, dropping whole molecules of B that clash with A.

    The usual case is inserting a solute into a pre-equilibrated solvent
    box. Removal is by molecule rather than by atom, so no partial
    solvent fragments are left behind.

    Two atoms are considered to clash when their separation falls below
    ``overlap_cutoff_scale * (r_i + r_j)``, with ``r`` taken from ASE's
    natural cutoffs.

    Parameters
    ----------
    A : ase.Atoms
        Structure kept intact. Not modified.
    B : ase.Atoms
        Structure to remove clashing molecules from. Not modified.
    bond_cutoff_scale : float, optional
        Scale factor on the cutoffs used to group B into molecules.
        Default is 1.0.
    overlap_cutoff_scale : float, optional
        Scale factor on the cutoffs used to detect clashes between A and
        B. Raising it slightly, to around 1.1, removes clashes more
        aggressively. Default is 1.0.

    Returns
    -------
    ase.Atoms
        The merged structure, with A first and the surviving atoms of B
        appended.

    Notes
    -----
    A and B are assumed to share a coordinate frame and cell.
    """
    A = A.copy()
    B = B.copy()

    # Grouped so that a single clashing atom takes its whole molecule with it,
    # rather than leaving a fragment behind
    groups_B = _bonded_groups(B, cutoff_scale=bond_cutoff_scale)
    index_to_group_B = {}
    for gidx, group in enumerate(groups_B):
        for i in group:
            index_to_group_B[i] = gidx

    if len(A) and len(B):
        # Concatenated so one NeighborList covers both sets under a single cell;
        # this assumes A and B already share a coordinate frame
        AB = A + B

        ov_cutoffs = [overlap_cutoff_scale * c for c in natural_cutoffs(AB)]
        nl_ab = NeighborList(
            ov_cutoffs, self_interaction=False, bothways=True, skin=0.0
        )
        nl_ab.update(AB)

        nA = len(A)
        b_atoms_to_remove: set[int] = set()
        for iA in range(nA):
            js, _ = nl_ab.get_neighbors(iA)
            for j in js:
                if j >= nA:  # Indices at or past nA belong to B
                    jB = j - nA
                    gidx = index_to_group_B.get(jB)
                    if gidx is not None:
                        b_atoms_to_remove.update(groups_B[gidx])
                    else:
                        b_atoms_to_remove.add(jB)  # Isolated atom, no molecule to take

        if b_atoms_to_remove:
            keep = [i for i in range(len(B)) if i not in b_atoms_to_remove]
            B = B[keep]

    merged = A + B

    return merged


def largest_bonded_cluster_indices(atoms: Atoms) -> list[int]:
    """Finds the indices of the largest bonded cluster of atoms in an ASE Atoms object.

    This function uses ASE's NeighborList to determine bonded groups of atoms based on
    natural covalent radii. It identifies all connected components (clusters) of atoms
    and returns the indices of the atoms in the largest cluster. If there are ties in
    cluster size, the cluster with the smallest index is chosen.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object containing the atomic structure.

    Returns
    -------
    List[int]
        A sorted list of indices representing the largest bonded cluster of atoms.
        Returns an empty list if the input Atoms object is empty.
    """
    n = len(atoms)
    if n == 0:
        return []

    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
    nl.update(atoms)

    # Build adjacency as a list of sets for speed
    adj = [set() for _ in range(n)]
    for i in range(n):
        idxs, _ = nl.get_neighbors(i)
        for j in idxs:
            # bothways=True should already ensure symmetry, but make it explicit
            adj[i].add(int(j))
            adj[int(j)].add(i)

    # Find connected components (clusters) with an iterative DFS
    visited = [False] * n
    largest_cluster = []

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        cluster = []
        visited[start] = True
        while stack:
            u = stack.pop()
            cluster.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        # Keep the largest; break ties by smallest index in the cluster
        if len(cluster) > len(largest_cluster) or (
            len(cluster) == len(largest_cluster)
            and cluster
            and min(cluster)
            < (min(largest_cluster) if largest_cluster else float("inf"))
        ):
            largest_cluster = cluster

    return sorted(largest_cluster)
