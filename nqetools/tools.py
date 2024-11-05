import os
import sys

from scipy.constants import physical_constants


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


# Conversion factor from Bohr to Angstrom
bohr_to_angstrom = physical_constants["Bohr radius"][0] * 1e10
