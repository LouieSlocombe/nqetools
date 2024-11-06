import  copy

import matplotlib.pyplot as plt
import numpy as np
from ase.io import  read
from ase.neb import   NEB
from ase.optimize import BFGS
from ase.visualize import view

import copy
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from ase.calculators.nwchem import NWChem
from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile
from ase.io import read, write
from ase.neb import NEB
from ase.optimize import BFGS
from ase.visualize import view
from scipy.interpolate import CubicSpline
from sella import IRC
from sella import Sella
from ase.vibrations import Vibrations

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


def optimise_reactant_product(reactant, product, calc,
                              fmax=0.01,
                              steps=1000,
                              reactant_opti='reactant_opti.traj',
                              product_opti='product_opti.traj'):
    reactant.calc = calc
    print('Optimizing reactant...', flush=True)
    t0 = time.time()
    BFGS(reactant, trajectory=reactant_opti).run(fmax=fmax, steps=steps)
    t1 = time.time()
    print('Time taken: {:.3} s'.format(t1 - t0), flush=True)
    reactant = read(reactant_opti, index=-1)

    product.calc = calc
    print('Optimizing product...', flush=True)
    t0 = time.time()
    BFGS(product, trajectory=product_opti).run(fmax=fmax, steps=steps)
    t1 = time.time()
    print('Time taken: {:.3} s'.format(t1 - t0), flush=True)
    product = read(product_opti, index=-1)
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
    neb = NEB(neb_images, climb=climb, remove_rotation_and_translation=rm_ro_trans, k=k)
    # interpolate the images
    neb.interpolate()
    neb.interpolate("idpp")
    return None

def optimise_neb(neb, fmax=0.01, steps=1000, ts_traj='ts.traj', n_images=5):
    BFGS(neb, trajectory=ts_traj).run(fmax=fmax, steps=steps)
    # Read the trajectory of the last images
    return read(ts_traj, index=f"-{n_images}:")

def get_ts_image(neb_images):
    # Find the image with the highest energy
    index = np.argmax([image.get_potential_energy() for image in neb_images])
    return neb_images[index]

def optimise_ts(ts_image, calc, fmax=0.01, steps=1000, sella_traj='sella.traj'):
    print('Running Sella TS search', flush=True)
    ts_image.calc = calc

    # Get the initial forces
    print('Initial energy: {:.3} eV'.format(ts_image.get_potential_energy()), flush=True)
    print('Initial max force: {:.3} eV/A'.format(get_fmax(ts_image)), flush=True)

    # Run Sella TS search
    sella_ts = Sella(ts_image, trajectory=sella_traj)
    sella_ts.run(fmax=fmax, steps=steps)

    # Read the trajectory
    ts_image_refined = read(sella_traj, index=-1)
    return ts_image_refined

def optimise_irc():
    # Read the trajectory
    ts_image_refined = read(sella_traj, index=-1)
    ts_image_refined.calc = calc

    # Run Sella IRC
    print("Running IRC forward", flush=True)
    sella_irc_f = IRC(ts_image_refined, trajectory=irc_f_traj, dx=dx)
    sella_irc_f.run(fmax=f_max_path, steps=steps, direction='forward')
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