import os
import copy
import time

import numpy as np
from ase.io import read
from ase.mep import NEB
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from scipy.interpolate import CubicSpline
from sella import IRC
from sella import Sella


def get_fmax(atoms):
    return np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max())


def get_neb_path(images):
    """
    Calculate the path length between each image in a Nudged Elastic Band (NEB) calculation.

    The function calculates the path length by summing the Euclidean distances between consecutive images.

    Parameters:
    images (list): A list of images. Each image is an instance of the Atoms class.

    Returns:
    numpy.ndarray: A numpy array containing the cumulative path length at each image.
    """

    # Get the positions of all atoms in each image
    positions = [atoms.positions for atoms in images]

    # Calculate the Euclidean distance between consecutive images
    path = [0] + [np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]

    # Return the cumulative path length at each image
    return np.cumsum(path)


def stitch_path(path1, path2, f_reverse_path=False):
    """
    This function stitches together two paths, reversing the order of the first path and removing the first image of the second path.
    It also provides an option to reverse the order of the whole stitched path.

    Parameters:
    path1 (list): The first path to be stitched. This path will be reversed.
    path2 (list): The second path to be stitched. The first image of this path will be removed before stitching.
    f_reverse_path (bool, optional): If True, the order of the whole stitched path is reversed. Default is False.

    Returns:
    list: The stitched path.
    """
    # Reverse the order of path1 and remove the first image of path2
    irc = list(path1)[::-1] + list(path2)[1:]
    if f_reverse_path:
        # Reverse the order of the whole path
        irc = irc[::-1]
    return irc


def resample_path(path, N_resample):
    """
    This function resamples a given path with a specified number of points.

    Parameters:
    path (list): The original path to be resampled. Each element of the list is an instance of the Atoms class.
    N_resample (int): The number of points in the resampled path.

    Returns:
    list: The resampled path. Each element of the list is an instance of the Atoms class.
    """

    # Get the path length
    path_distance = get_neb_path(path)

    # Interpolate the path length
    path_interp = np.linspace(0, path_distance[-1], N_resample)

    # Get the atom positions
    positions = np.array([image.positions for image in path])

    # Interpolate and resample the positions
    positions_interp = CubicSpline(path_distance, positions)(path_interp)

    # Create the resampled images
    irc_resampled = [path[0]]  # Ensure the first image is the same
    for ii in range(1, N_resample - 1):
        atoms = path[0].copy()
        # Set the positions to the interpolated positions
        atoms.positions = positions_interp[ii, :, :]
        irc_resampled.append(atoms)
    irc_resampled.append(path[-1])  # Ensure the last image is the same

    return irc_resampled


def optimise_geom(atoms, calc,
                  fmax=0.01,
                  steps=1000,
                  opti_traj='opti.traj'):
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
    print('Optimizing reactant...', flush=True)
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


def prepare_neb(reactant, product, calc, n_images=5, climb=True, rm_ro_trans=True, k=2.0):
    # Construct the NEB images
    neb_images = [reactant]
    for ii in range(n_images - 2):
        neb_images.append(reactant.copy())
    neb_images.append(product)

    # Attach the calculator to the images
    for image in neb_images:
        image.calc = copy.copy(calc)
        image.get_potential_energy()

    # create the NEB object
    neb = NEB(neb_images,
              climb=climb,
              remove_rotation_and_translation=rm_ro_trans,
              k=k)
    # interpolate the images
    neb.interpolate()
    neb.interpolate("idpp")
    return neb


def optimise_neb(neb, fmax=0.01, steps=1000, ts_traj='ts.traj', n_images=5):
    t0 = time.time()
    BFGS(neb, trajectory=ts_traj).run(fmax=fmax, steps=steps)
    t1 = time.time()
    print('Time taken: {:.3} s'.format(t1 - t0), flush=True)
    # Read the trajectory of the last images
    return read(ts_traj, index=f"-{n_images}:")


def get_ts_image(neb_images, calc):
    # Attach the calculator to the images
    for image in neb_images:
        image.calc = copy.copy(calc)

    # Find the image with the highest energy
    index = np.argmax([image.get_potential_energy() for image in neb_images])
    return neb_images[index]


def optimise_ts(ts_image, calc,
                fmax=0.01,
                steps=1000,
                eta=1e-4,
                gamma=0.1,
                sella_traj='sella.traj'):
    print('Running Sella TS search', flush=True)
    ts_image.calc = calc

    # Get the initial forces
    print('Initial energy: {:.3} eV'.format(ts_image.get_potential_energy()), flush=True)
    print('Initial max force: {:.3} eV/A'.format(get_fmax(ts_image)), flush=True)

    # Run Sella TS search
    sella_ts = Sella(ts_image,
                     trajectory=sella_traj,
                     eta=eta,
                     gamma=gamma)
    sella_ts.run(fmax=fmax, steps=steps)

    # Read the trajectory
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
    # Read the trajectory
    ts_image.calc = calc

    print("Running IRC forward", flush=True)
    sella_irc_f = IRC(ts_image,
                      trajectory=irc_f_traj,
                      dx=dx,
                      eta=eta,
                      gamma=gamma,
                      keep_going=keep_going)
    sella_irc_f.run(fmax=fmax,
                    steps=steps,
                    direction='forward')

    print("Running IRC reverse", flush=True)
    sella_irc_r = IRC(ts_image,
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
    # Set the calculator
    atoms.calc = calc
    # Get the vibrational frequencies
    vib = Vibrations(atoms)
    # Make sure the folder is clean
    vib.clean()
    vib.run()
    vib.summary()
    freqs = vib.get_frequencies()
    # Clean the folder
    vib.clean()
    return freqs
