import copy
import os
import time

import numpy as np
from ase.io import read
from ase.mep import NEB
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from scipy.interpolate import CubicSpline
from sella import IRC
from sella import Sella

from .tools import get_fmax


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


def resample_path(path, n_resample):
    """
    This function resamples a given path with a specified number of points.

    Parameters:
    path (list): The original path to be resampled. Each element of the list is an instance of the Atoms class.
    n_resample (int): The number of points in the resampled path.

    Returns:
    list: The resampled path. Each element of the list is an instance of the Atoms class.
    """

    # Get the path length
    path_distance = get_neb_path(path)

    # Interpolate the path length
    path_interp = np.linspace(0, path_distance[-1], n_resample)

    # Get the atom positions
    positions = np.array([image.positions for image in path])

    # Interpolate and resample the positions
    positions_interp = CubicSpline(path_distance, positions)(path_interp)

    # Create the resampled images
    irc_resampled = [path[0]]  # Ensure the first image is the same
    for ii in range(1, n_resample - 1):
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
    """
    Optimise the geometry of the given atomic structure using the BFGS algorithm.

    Parameters:
    atoms (ase.Atoms): ASE Atoms object containing the atomic structure to be optimised.
    calc (ase.Calculator): Calculator to be used for the optimisation.
    fmax (float, optional): Maximum force tolerance for the optimisation. Default is 0.01 eV/Å.
    steps (int, optional): Maximum number of optimisation steps. Default is 1000.
    opti_traj (str, optional): Filename for saving the optimisation trajectory. Default is 'opti.traj'.

    Returns:
    ase.Atoms: Optimized ASE Atoms object.
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
    """
    Optimise the geometries of the reactant and product structures.

    Parameters:
    reactant (ase.Atoms): ASE Atoms object containing the reactant structure.
    product (ase.Atoms): ASE Atoms object containing the product structure.
    calc (ase.Calculator): Calculator to be used for the optimisation.
    fmax (float, optional): Maximum force tolerance for the optimisation. Default is 0.01 eV/Å.
    steps (int, optional): Maximum number of optimisation steps. Default is 1000.
    reactant_opti (str, optional): Filename for saving the reactant optimisation trajectory. Default is 'reactant_opti.traj'.
    product_opti (str, optional): Filename for saving the product optimisation trajectory. Default is 'product_opti.traj'.

    Returns:
    tuple: A tuple containing the optimised reactant and product ASE Atoms objects.
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
                k=2.0):
    """
    Prepare the Nudged Elastic Band (NEB) calculation by constructing and interpolating the NEB images.

    Parameters:
    reactant (ase.Atoms): ASE Atoms object representing the reactant structure.
    product (ase.Atoms): ASE Atoms object representing the product structure.
    calc (ase.Calculator): Calculator to be used for the NEB calculation.
    n_images (int, optional): Number of images in the NEB calculation. Default is 5.
    climb (bool, optional): If True, use the climbing image method. Default is True.
    rm_ro_trans (bool, optional): If True, remove rotation and translation during the NEB calculation. Default is True.
    k (float, optional): Spring constant for the NEB calculation. Default is 2.0 eV/Å².

    Returns:
    ase.neb.NEB: NEB object containing the prepared NEB images.
    """
    # Construct the NEB images
    neb_images = [reactant]
    for ii in range(n_images - 2):
        neb_images.append(reactant.copy())
    neb_images.append(product)

    # Attach the calculator to the images
    for image in neb_images:
        image.calc = copy.copy(calc)
        image.get_potential_energy()

    # Create the NEB object
    neb = NEB(neb_images,
              climb=climb,
              remove_rotation_and_translation=rm_ro_trans,
              k=k)
    # Interpolate the images
    neb.interpolate()
    neb.interpolate("idpp")
    return neb


def optimise_neb(neb,
                 fmax=0.01,
                 steps=1000,
                 ts_traj='ts.traj',
                 n_images=5):
    """
    Optimise the geometry of a Nudged Elastic Band (NEB) calculation using the BFGS algorithm.

    Parameters:
    neb (ase.neb.NEB): NEB object containing the images to be optimised.
    fmax (float, optional): Maximum force tolerance for the optimisation. Default is 0.01 eV/Å.
    steps (int, optional): Maximum number of optimisation steps. Default is 1000.
    ts_traj (str, optional): Filename for saving the transition state trajectory. Default is 'ts.traj'.
    n_images (int, optional): Number of images in the NEB calculation. Default is 5.

    Returns:
    list: A list of ASE Atoms objects representing the optimised images.
    """
    t0 = time.time()
    BFGS(neb, trajectory=ts_traj).run(fmax=fmax, steps=steps)
    t1 = time.time()
    print('Time taken: {:.3} s'.format(t1 - t0), flush=True)
    # Read the trajectory of the last images
    return read(ts_traj, index=f"-{n_images}:")


def get_ts_image(neb_images, calc):
    """
    Find the transition state (TS) image with the highest energy from a list of NEB images.

    Parameters:
    neb_images (list): A list of ASE Atoms objects representing the NEB images.
    calc (ase.Calculator): Calculator to be used for the energy calculations.

    Returns:
    ase.Atoms: The NEB image with the highest energy.
    """
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
    """
    Optimise the transition state (TS) image using the Sella algorithm.

    Parameters:
    ts_image (ase.Atoms): ASE Atoms object representing the transition state image.
    calc (ase.Calculator): Calculator to be used for the optimisation.
    fmax (float, optional): Maximum force tolerance for the optimisation. Default is 0.01 eV/Å.
    steps (int, optional): Maximum number of optimisation steps. Default is 1000.
    eta (float, optional): Step size parameter for the Sella algorithm. Default is 1e-4.
    gamma (float, optional): Damping parameter for the Sella algorithm. Default is 0.1.
    sella_traj (str, optional): Filename for saving the Sella optimisation trajectory. Default is 'sella.traj'.

    Returns:
    ase.Atoms: Optimized ASE Atoms object representing the refined transition state image.
    """
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
    """
    Optimise the Intrinsic Reaction Coordinate (IRC) path using the Sella algorithm.

    Parameters:
    ts_image (ase.Atoms): ASE Atoms object representing the transition state image.
    calc (ase.Calculator): Calculator to be used for the optimisation.
    fmax (float, optional): Maximum force tolerance for the optimisation. Default is 0.01 eV/Å.
    steps (int, optional): Maximum number of optimisation steps. Default is 1000.
    dx (float, optional): Step size parameter for the IRC algorithm. Default is 0.1.
    eta (float, optional): Step size parameter for the Sella algorithm. Default is 1e-4.
    gamma (float, optional): Damping parameter for the Sella algorithm. Default is 0.1.
    keep_going (bool, optional): If True, continue the optimisation until convergence. Default is True.
    irc_f_traj (str, optional): Filename for saving the forward IRC trajectory. Default is 'irc_f.traj'.
    irc_r_traj (str, optional): Filename for saving the reverse IRC trajectory. Default is 'irc_r.traj'.

    Returns:
    None
    """
    # Read the trajectory
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
    """
    Calculate the vibrational frequencies of the given atomic structure.

    Parameters:
    atoms (ase.Atoms): ASE Atoms object containing the atomic structure.
    calc (ase.Calculator): Calculator to be used for the vibrational frequency calculations.

    Returns:
    numpy.ndarray: An array of vibrational frequencies.
    """
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
