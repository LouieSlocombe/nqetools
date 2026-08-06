import copy
import numpy as np
import os
import time
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read
from ase.mep import NEB
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from itertools import permutations
from mace.calculators import mace_off
from scipy.interpolate import CubicSpline
from sella import IRC, Sella
from ase.data import covalent_radii
import geodesic_interpolate as gi
from .tools import get_fmax


def get_neb_path(images):
    """Calculate the path length between each image in a Nudged Elastic Band (NEB) calculation.

    The function calculates the path length by summing the Euclidean distances between consecutive images.

    Parameters
    ----------
    images : list
        A list of images. Each image is an instance of the Atoms class.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the cumulative path length at each image.
    """

    positions = [atoms.positions for atoms in images]
    path = [0] + [np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]
    return np.cumsum(path)


def stitch_path(path1, path2, f_reverse_path=False):
    """This function stitches together two paths, reversing the order of the first path and removing the first image of the second path.
    It also provides an option to reverse the order of the whole stitched path.

    Parameters
    ----------
    path1 : list
        The first path to be stitched. This path will be reversed.
    path2 : list
        The second path to be stitched. The first image of this path will be removed before stitching.
    f_reverse_path : bool, optional
        If True, the order of the whole stitched path is reversed. Default is False.

    Returns
    -------
    list
        The stitched path.
    """
    irc = list(path1)[::-1] + list(path2)[1:]
    if f_reverse_path:
        irc = irc[::-1]
    return irc


def resample_path(path, n_resample):
    """This function resamples a given path with a specified number of points.

    Parameters
    ----------
    path : list
        The original path to be resampled. Each element of the list is an instance of the Atoms class.
    n_resample : int
        The number of points in the resampled path.

    Returns
    -------
    list
        The resampled path. Each element of the list is an instance of the Atoms class.
    """

    path_distance = get_neb_path(path)
    path_interp = np.linspace(0, path_distance[-1], n_resample)
    positions = np.array([image.positions for image in path])
    positions_interp = CubicSpline(path_distance, positions)(path_interp)

    irc_resampled = [path[0]]  # keep the first image exact, not spline-interpolated
    for ii in range(1, n_resample - 1):
        atoms = path[0].copy()
        atoms.positions = positions_interp[ii, :, :]
        irc_resampled.append(atoms)
    irc_resampled.append(path[-1])  # keep the last image exact, not spline-interpolated

    return irc_resampled


def optimise_geom(atoms, calc,
                  fmax=0.01,
                  steps=1000,
                  opti_traj='opti.traj'):
    """Optimise the geometry of the given atomic structure using the BFGS algorithm.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object containing the atomic structure to be optimised.
    calc : ase.Calculator
        Calculator to be used for the optimisation.
    fmax : float, optional
        Maximum force tolerance for the optimisation. Default is 0.01 eV/Å.
    steps : int, optional
        Maximum number of optimisation steps. Default is 1000.
    opti_traj : str, optional
        Filename for saving the optimisation trajectory. Default is 'opti.traj'.

    Returns
    -------
    ase.Atoms
        Optimized ASE Atoms object.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    t0 = time.time()
    BFGS(atoms, trajectory=opti_traj).run(fmax=fmax, steps=steps)
    t1 = time.time()
    print('Time taken: {:.3} s'.format(t1 - t0), flush=True)
    atoms = read(opti_traj, index=-1)
    os.remove(opti_traj)
    return atoms


def optimise_reactant_product(reactant, product, calc,
                              fmax=0.01,
                              steps=1000,
                              reactant_opti='reactant_opti.traj',
                              product_opti='product_opti.traj'):
    """Optimise the geometries of the reactant and product structures.

    Parameters
    ----------
    reactant : ase.Atoms
        ASE Atoms object containing the reactant structure.
    product : ase.Atoms
        ASE Atoms object containing the product structure.
    calc : ase.Calculator
        Calculator to be used for the optimisation.
    fmax : float, optional
        Maximum force tolerance for the optimisation. Default is 0.01 eV/Å.
    steps : int, optional
        Maximum number of optimisation steps. Default is 1000.
    reactant_opti : str, optional
        Filename for saving the reactant optimisation trajectory. Default is 'reactant_opti.traj'.
    product_opti : str, optional
        Filename for saving the product optimisation trajectory. Default is 'product_opti.traj'.

    Returns
    -------
    tuple
        A tuple containing the optimised reactant and product ASE Atoms objects.
    """
    print('Optimising reactant...', flush=True)
    reactant = optimise_geom(reactant, calc,
                             fmax=fmax,
                             steps=steps,
                             opti_traj=reactant_opti)

    print('Optimizing product...', flush=True)
    product = optimise_geom(product, calc,
                            fmax=fmax,
                            steps=steps,
                            opti_traj=product_opti)
    return reactant, product


def prepare_neb(reactant, product, calc,
                n_images=5,
                climb=True,
                rm_ro_trans=True,
                geo_int=True,
                k=2.0):
    neb_images = [reactant]
    for ii in range(n_images - 2):
        neb_images.append(reactant.copy())
    neb_images.append(product)

    if geo_int:
        neb_images = gi.geodesic_interpolate(neb_images, n_images=n_images)

    for image in neb_images:
        image.calc = copy.copy(calc)
        image.get_potential_energy()

    neb = NEB(neb_images,
              climb=climb,
              remove_rotation_and_translation=rm_ro_trans,
              k=k)
    if not geo_int:
        neb.interpolate()
        neb.interpolate("idpp")
    return neb


def optimise_neb(neb,
                 fmax=0.01,
                 steps=1000,
                 ts_traj='ts.traj',
                 n_images=5):
    """Optimise the geometry of a Nudged Elastic Band (NEB) calculation using the BFGS algorithm.

    Parameters
    ----------
    neb : ase.neb.NEB
        NEB object containing the images to be optimised.
    fmax : float, optional
        Maximum force tolerance for the optimisation. Default is 0.01 eV/Å.
    steps : int, optional
        Maximum number of optimisation steps. Default is 1000.
    ts_traj : str, optional
        Filename for saving the transition state trajectory. Default is 'ts.traj'.
    n_images : int, optional
        Number of images in the NEB calculation. Default is 5.

    Returns
    -------
    list
        A list of ASE Atoms objects representing the optimised images.
    """
    t0 = time.time()
    BFGS(neb, trajectory=ts_traj).run(fmax=fmax, steps=steps)
    t1 = time.time()
    print('Time taken: {:.3} s'.format(t1 - t0), flush=True)
    # Read the trajectory of the last images
    return read(ts_traj, index=f"-{n_images}:")


def get_ts_image(neb_images, calc):
    """Find the transition state (TS) image with the highest energy from a list of NEB images.

    Parameters
    ----------
    neb_images : list
        A list of ASE Atoms objects representing the NEB images.
    calc : ase.Calculator
        Calculator to be used for the energy calculations.

    Returns
    -------
    ase.Atoms
        The NEB image with the highest energy.
    """
    for image in neb_images:
        image.calc = copy.copy(calc)

    index = np.argmax([image.get_potential_energy() for image in neb_images])
    return neb_images[index]


def optimise_ts(ts_image, calc,
                fmax=0.01,
                steps=1000,
                eta=1e-4,
                gamma=0.1,
                sella_traj='sella.traj'):
    """Optimise the transition state (TS) image using the Sella algorithm.

    Parameters
    ----------
    ts_image : ase.Atoms
        ASE Atoms object representing the transition state image.
    calc : ase.Calculator
        Calculator to be used for the optimisation.
    fmax : float, optional
        Maximum force tolerance for the optimisation. Default is 0.01 eV/Å.
    steps : int, optional
        Maximum number of optimisation steps. Default is 1000.
    eta : float, optional
        Step size parameter for the Sella algorithm. Default is 1e-4.
    gamma : float, optional
        Damping parameter for the Sella algorithm. Default is 0.1.
    sella_traj : str, optional
        Filename for saving the Sella optimisation trajectory. Default is 'sella.traj'.

    Returns
    -------
    ase.Atoms
        Optimized ASE Atoms object representing the refined transition state image.
    """
    print('Running Sella TS search', flush=True)
    ts_image.calc = calc

    print('Initial energy: {:.3} eV'.format(ts_image.get_potential_energy()), flush=True)
    print('Initial max force: {:.3} eV/A'.format(get_fmax(ts_image)), flush=True)

    sella_ts = Sella(ts_image,
                     trajectory=sella_traj,
                     eta=eta,
                     gamma=gamma)
    sella_ts.run(fmax=fmax, steps=steps)

    ts_image_refined = read(sella_traj, index=-1)
    return ts_image_refined


def optimise_irc(ts_image, calc,
                 fmax=0.01,
                 steps=1000,
                 dx=0.1,
                 eta=1e-4,
                 gamma=0.1,
                 keep_going=True,
                 irc_f_traj='irc_f.traj',
                 irc_r_traj='irc_r.traj'):
    """Optimise the Intrinsic Reaction Coordinate (IRC) path using the Sella algorithm.

    Parameters
    ----------
    ts_image : ase.Atoms
        ASE Atoms object representing the transition state image.
    calc : ase.Calculator
        Calculator to be used for the optimisation.
    fmax : float, optional
        Maximum force tolerance for the optimisation. Default is 0.01 eV/Å.
    steps : int, optional
        Maximum number of optimisation steps. Default is 1000.
    dx : float, optional
        Step size parameter for the IRC algorithm. Default is 0.1.
    eta : float, optional
        Step size parameter for the Sella algorithm. Default is 1e-4.
    gamma : float, optional
        Damping parameter for the Sella algorithm. Default is 0.1.
    keep_going : bool, optional
        If True, continue the optimisation until convergence. Default is True.
    irc_f_traj : str, optional
        Filename for saving the forward IRC trajectory. Default is 'irc_f.traj'.
    irc_r_traj : str, optional
        Filename for saving the reverse IRC trajectory. Default is 'irc_r.traj'.

    Returns
    -------
    None
    """
    irc_f = ts_image.copy()
    irc_f.calc = calc
    print("Running IRC forward", flush=True)
    sella_irc_f = IRC(irc_f,
                      trajectory=irc_f_traj,
                      dx=dx,
                      eta=eta,
                      gamma=gamma,
                      keep_going=keep_going)
    sella_irc_f.run(fmax=fmax,
                    steps=steps,
                    direction='forward')

    irc_r = ts_image.copy()
    irc_r.calc = calc

    print("Running IRC reverse", flush=True)
    sella_irc_r = IRC(irc_r,
                      trajectory=irc_r_traj,
                      dx=dx,
                      eta=eta,
                      gamma=gamma,
                      keep_going=keep_going)
    sella_irc_r.run(fmax=fmax,
                    steps=steps,
                    direction='reverse')
    return None


def get_vibrations(atoms, calc):
    """Calculate the vibrational frequencies of the given atomic structure.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object containing the atomic structure.
    calc : ase.Calculator
        Calculator to be used for the vibrational frequency calculations.

    Returns
    -------
    numpy.ndarray
        An array of vibrational frequencies.
    """
    atoms.calc = calc
    vib = Vibrations(atoms)
    # Make sure the folder is clean
    vib.clean()
    vib.run()
    vib.summary()
    freqs = vib.get_frequencies()
    vib.clean()
    return freqs


def quick_guess_ts(reactant, product, n_images=25):
    """Generate a quick guess for the transition state (TS) between a reactant and product.

    This function uses geodesic interpolation to create a series of intermediate images
    between the reactant and product structures. The center image of the interpolated
    path is selected as the guessed transition state.

    Parameters
    ----------
    reactant : ase.Atoms
        ASE Atoms object representing the reactant structure.
    product : ase.Atoms
        ASE Atoms object representing the product structure.
    n_images : int, optional
        Number of images to generate during interpolation. Default is 25.

    Returns
    -------
    ase.Atoms
        ASE Atoms object representing the guessed transition state.
    """
    atoms_ts = gi.geodesic_interpolate([reactant, product], n_images=n_images)
    atoms_ts = atoms_ts[n_images // 2]

    return atoms_ts


def bonded_cluster_indices_no_anchor_hub(atoms: Atoms,
                                         anchor: int,
                                         mult: float = 1.0,
                                         multi_h: float = 1.3) -> list[int]:
    n = len(atoms)
    if not (0 <= anchor < n):
        raise IndexError(f"Anchor index {anchor} out of range for {n} atoms.")

    cutoffs = natural_cutoffs(atoms, mult=mult)
    for i, atom in enumerate(atoms):
        if atom.symbol == 'H':
            cutoffs[i] = covalent_radii[atom.number] * multi_h

    nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)

    # First step: collect immediate neighbors of the anchor
    first_neighbors, _ = nl.get_neighbors(anchor)
    first_neighbors = [i for i in first_neighbors if i != anchor - 2]

    # Seed traversal with first-shell neighbors; mark anchor as visited so we never traverse through it
    visited = set([anchor]) | set(first_neighbors)
    stack = list(first_neighbors)

    # Explore without ever stepping onto the anchor again
    while stack:
        i = stack.pop()
        nbrs, _ = nl.get_neighbors(i)
        for j in nbrs:
            if j == anchor:
                continue  # do not traverse through anchor beyond the first step
            if j not in visited:
                visited.add(j)
                stack.append(j)

    return sorted(visited)


def get_dimer_bonded_cluster_indices(atoms: Atoms,
                                     anchors: list[int],
                                     mults=None,
                                     multi_h: float = 1.3) -> list[int]:
    if mults is None:
        mults = [1.0, 1.0]

    if len(anchors) != 2:
        raise ValueError("Anchors list must contain exactly two indices.")

    if len(mults) != 2:
        raise ValueError("Mults list must contain exactly two values.")

    base_a = bonded_cluster_indices_no_anchor_hub(atoms, anchors[0], mult=mults[0], multi_h=multi_h)
    base_b = bonded_cluster_indices_no_anchor_hub(atoms, anchors[1], mult=mults[1], multi_h=multi_h)

    return list(sorted(set(base_a + base_b)))


def _pca_frame(positions):
    pts = np.asarray(positions)
    origin = pts.mean(axis=0)
    X = pts - origin
    # PCA via SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Principal axes are rows of Vt; use columns as unit vectors
    x = Vt[0]  # largest variance axis
    y = Vt[1]
    z = Vt[2]  # normal to the base plane (smallest variance)
    # Re-orthonormalize into RH frame (numerical safety)
    x = x / np.linalg.norm(x)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)
    R = np.vstack([x, y, z]).T  # columns are axes
    return origin, R


def _orient_normal_toward(R, origin, target_point):
    z = R[:, 2]
    d = np.asarray(target_point) - np.asarray(origin)
    if np.dot(z, d) < 0.0:
        # flip both y and z to keep RH: x stays, y->-y, z->-z
        R = np.column_stack((R[:, 0], -R[:, 1], -R[:, 2]))
    return R


def _rigid_transform(points, anchor_pos, R_target, new_anchor_pos):
    P = np.asarray(points) - anchor_pos
    P_rot = P @ R_target.T
    return P_rot + new_anchor_pos


def flip_and_face_bases(
        atoms: Atoms,
        baseA_idxs: list,
        baseB_idxs: list,
        anchors: list,
        rot_matrix: list = None,
) -> Atoms:
    anchorA_idx = anchors[0]
    anchorB_idx = anchors[1]
    atoms = atoms.copy()

    pos = atoms.get_positions()
    baseA = np.array(baseA_idxs, dtype=int)
    baseB = np.array(baseB_idxs, dtype=int)

    # Anchors and base centroids
    anchorA = pos[anchorA_idx].copy()
    anchorB = pos[anchorB_idx].copy()

    # Local frames via PCA
    originA, RA = _pca_frame(pos[baseA])
    originB, RB = _pca_frame(pos[baseB])

    # Make each normal point toward the other base (for a consistent 'face')
    RB = _orient_normal_toward(RB, originB, originA)
    RA = _orient_normal_toward(RA, originA, originB)

    # Reflection matrix that flips the normal and y while keeping x,
    # so after swap the bases "face" each other with aligned x-axes.
    if rot_matrix is None:
        rot_matrix = [-1.0, 1.0, -1.0]
    M = np.diag(rot_matrix)  # x, -y, -z (right-handed overall mapping)

    # World-space rotations to map frames
    # For A -> B: R_target_A satisfies: R_target_A * RA ≈ RB * M
    # => R_target_A = RB * M * RA^T
    R_target_A = RB @ M @ RA.T

    # For B -> A: symmetric
    R_target_B = RA @ M @ RB.T

    # Apply rigid transforms about anchors, translated onto the opposite anchor
    # A goes to anchorB; B goes to anchorA
    newA = _rigid_transform(pos[baseA], anchorA, R_target_A, anchorB)
    newB = _rigid_transform(pos[baseB], anchorB, R_target_B, anchorA)

    # Write back
    new_pos = pos.copy()
    new_pos[baseA] = newA
    new_pos[baseB] = newB
    atoms.set_positions(new_pos)
    return atoms


def optimize_with_fixed_anchors(atoms: Atoms,
                                baseA_idxs: list,
                                baseB_idxs: list,
                                anchor_indices: list,
                                fmax: float = 0.05) -> Atoms:
    # Create a copy to avoid modifying the original
    atoms_opt = atoms.copy()
    calc = mace_off(model='small', device="cpu", default_dtype="float64")
    constraint = FixAtoms(indices=anchor_indices)
    atoms_opt.set_constraint(constraint)

    # Select only the atoms in baseA and baseB for optimization
    atoms_opt = atoms_opt[baseA_idxs + baseB_idxs]

    atoms_opt.calc = calc
    optimizer = BFGS(atoms_opt)
    optimizer.run(fmax=fmax)
    atoms_out = atoms.copy()
    atoms_out[baseA_idxs + baseB_idxs].set_positions(atoms_opt.get_positions())

    return atoms_out


def get_best_flip_and_face_bases(
        atoms: Atoms,
        baseA_idxs: list,
        baseB_idxs: list,
        anchors: list,
        optimise_after: bool = True,
) -> Atoms:
    rot_matrix_permutations = list(set(list(permutations([-1.0, 1.0, 1.0])) + list(permutations([-1.0, -1.0, 1.0]))))
    print(f"All permutations of rot_matrix: {rot_matrix_permutations}", flush=True)

    # loop over permutations to see which gives the least COM movement
    best_rot_matrix = None
    best_dist_after = float('inf')
    for rot_matrix in rot_matrix_permutations:
        rot_matrix = list(rot_matrix)
        print(f"Trying rot_matrix: {rot_matrix}", flush=True)
        swapped = flip_and_face_bases(
            atoms,
            baseA_idxs=baseA_idxs,
            baseB_idxs=baseB_idxs,
            anchors=anchors,
            rot_matrix=rot_matrix,
        )

        com_a_before = atoms[baseA_idxs].get_center_of_mass()
        com_b_before = atoms[baseB_idxs].get_center_of_mass()
        com_a_after = swapped[baseA_idxs].get_center_of_mass()
        com_b_after = swapped[baseB_idxs].get_center_of_mass()

        dist_before = np.linalg.norm(com_a_before - com_b_before)
        dist_after = np.linalg.norm(com_a_after - com_b_after)

        print(f"dist_before COM: {dist_before}", flush=True)
        print(f"dist_after COM:  {dist_after}", flush=True)
        print(flush=True)

        if dist_after < best_dist_after:
            best_dist_after = dist_after
            best_rot_matrix = rot_matrix

    print("Best rot_matrix:", best_rot_matrix)

    swapped = flip_and_face_bases(
        atoms,
        baseA_idxs=baseA_idxs,
        baseB_idxs=baseB_idxs,
        anchors=anchors,
        rot_matrix=best_rot_matrix,
    )
    if optimise_after:
        swapped = optimize_with_fixed_anchors(
            swapped,
            baseA_idxs=baseA_idxs,
            baseB_idxs=baseB_idxs,
            anchor_indices=anchors,
        )

    return swapped
