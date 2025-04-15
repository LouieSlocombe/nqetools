import glob
import importlib.util
import os
import re
import shutil
import xml.etree.ElementTree as ET

import ase.io
import numpy as np
from ipi.utils.io import read_file

from .conversions import convert_atom_list_bohr_to_angstrom, eV_to_kJpermol


def read_ipi_xyz(filename, convert_units=True):
    """
    Reads a file in xyz i-PI format and returns it in ASE format.

    Parameters:
    filename (str): The path to the xyz file in i-PI format.
    convert_units (bool): If True, converts coordinates and cell from Bohr to Angstrom.

    Returns:
    list: A list of ASE Atoms objects representing the frames in the file.
    """

    # Open the file for reading
    file_handle = open(filename, "r")
    frames = []

    # Loop to read each frame in the file
    while True:
        try:
            # Read a frame from the file
            ret = read_file("xyz", file_handle)
            # Append the frame as an ASE Atoms object to the frames list
            frames.append(ase.Atoms(ret["atoms"].names,
                                    positions=ret["atoms"].q.reshape((-1, 3)),
                                    cell=ret["cell"].h.T, pbc=True))
        except EOFError:
            # Break the loop if end of file is reached
            break
        except:
            # Raise any other exceptions encountered
            raise

    # Close the file handle
    file_handle.close()

    # Convert units from Bohr to Angstrom if requested
    if convert_units:
        frames = convert_atom_list_bohr_to_angstrom(frames)

    return frames


def read_ipi_output(filename):
    """
    Reads an i-PI output file and returns a dictionary with the properties in a tidy order.

    Parameters:
    filename (str): The path to the i-PI output file.

    Returns:
    dict: A dictionary where keys are property names and values are the corresponding data columns.
    """

    f = open(filename, "r")

    regex = re.compile(".*column *([0-9]*) *--> ([^ {]*)")

    fields = []
    cols = []
    for line in f:
        if line[0] == "#":
            match = regex.match(line)
            if match is None:
                print("Malformed comment line: ", line)
                raise ValueError()
            fields.append(match.group(2))
            cols.append(slice(int(match.group(1)) - 1, int(match.group(1))))
        else:
            break  # done with header
    f.close()

    columns = {}
    raw = np.loadtxt(filename)
    for i, c in enumerate(fields):
        while c in columns:
            c = c + "+"
        columns[c] = raw[:, cols[i]].T
        if columns[c].shape[0] == 1:
            columns[c].shape = columns[c].shape[1]
    return columns


def write_xml(root, file):
    """
    Writes an XML tree to a file, creating any necessary directories.

    Parameters:
    root (xml.etree.ElementTree.Element): The root element of the XML tree.
    file (str): The path to the file where the XML tree will be written.

    Returns:
    None
    """
    os.makedirs(os.path.dirname(file), exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree.getroot(), space="\t", level=0)
    tree.write(file)
    return None


def write_xyz(atoms, file, vacuum=None, vacuum_min=5.0):
    """
    Writes an ASE Atoms object to an XYZ file, ensuring the directory exists and centering the atoms with a specified vacuum.
    If the atoms have no cell or a zero-sized cell, adds 5Å of vacuum by default.

    Parameters:
    atoms (ase.Atoms): The ASE Atoms object to write.
    file (str): The path to the output XYZ file.
    vacuum (float, optional): The vacuum padding to apply when centering the atoms. Default is None.
    vacuum_min (float, optional): The minimum vacuum padding to apply if the cell size is zero. Default is 5.0Å.

    Returns:
    ase.Atoms: The centered ASE Atoms object.
    """
    # Check if cell exists and is non-zero
    cell = atoms.get_cell()
    # Add vacuum if cell is zero
    if cell.volume == 0:
        print(f"Cell is zero, adding {vacuum_min}A vacuum!")
        atoms.center(vacuum=vacuum_min)

    # Add vacuum if requested
    if vacuum is not None:
        atoms.center(vacuum=vacuum)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file), exist_ok=True)
    # Write the atoms object to the XYZ file
    ase.io.write(file, atoms)
    return atoms


def remove_directory(directory):
    """
    Removes a directory if it exists, otherwise prints a message.

    Parameters:
    directory (str): The path to the directory to be removed.

    Returns:
    None
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    else:
        pass
    return None


def copy_and_rename_file(src, dst_dir, new_name):
    """
    Copies a file to a new directory and renames it.

    Parameters:
    src (str): The path to the source file.
    dst_dir (str): The path to the destination directory.
    new_name (str): The new name for the copied file.

    Returns:
    None
    """
    shutil.copy(src, os.path.join(dst_dir, new_name))
    return None


def list_files_with_pattern(directory, pattern):
    """
    Lists files in a directory that match a given pattern.

    Parameters:
    directory (str): The path to the directory to search in.
    pattern (str): The pattern to match files against.

    Returns:
    list: A list of file paths that match the given pattern.
    """
    return glob.glob(os.path.join(directory, pattern))


def get_final_xyz(directory, sub="*FINAL_*.xyz"):
    """
    Filters and returns the final XYZ file from a directory.

    Parameters:
    dir (str): The path to the directory to search in.
    sub (str, optional): The pattern to match files against. Default is "*FINAL_*.xyz".

    Returns:
    str: The path to the final XYZ file that matches the criteria.
    """
    l = list_files_with_pattern(directory, sub)
    # Only select files that end with xyz
    l_filt = [f for f in l if f.endswith(".xyz")]
    # Select the file that does not contain the string "forces"
    l_nf = [f for f in l_filt if "forces" not in f]
    # Remove the file that contains the string "_0"
    l_n0 = [f for f in l_nf if "_0.xyz" not in f]
    return l_n0[0]


def get_final_hess(directory, sub=f"*FINAL.hess*"):
    """
    Retrieves the final Hessian file from a directory.

    Parameters:
    dir (str): The path to the directory to search in.
    sub (str, optional): The pattern to match files against. Default is "*FINAL.hess*".

    Returns:
    str: The path to the final Hessian file that matches the criteria.
    """
    return list_files_with_pattern(directory, sub)[0]


def copy_xyz(file_in, new_dir, file_out="init.xyz"):
    """
    Copies an XYZ file to a new directory, renaming it if necessary.

    Parameters:
    file_in (str): The path to the input XYZ file.
    new_dir (str): The path to the destination directory.
    file_out (str, optional): The name for the output file. Default is "init.xyz".

    Returns:
    None
    """
    os.makedirs(new_dir, exist_ok=True)
    # Load the file
    atoms = ase.io.read(file_in, index=":")
    # Check if the file is a trajectory or a single structure
    if isinstance(atoms, ase.atoms.Atoms):
        ase.io.write(f"{new_dir}{file_out}", atoms)
    else:
        # Write the new file
        ase.io.write(f"{new_dir}{file_out}", atoms[-1])
    return None


def copy_hess(file_in, new_dir, file_out="hessian.dat"):
    """
    Copies a Hessian file to a new directory, renaming it if necessary.

    Parameters:
    file_in (str): The path to the input Hessian file.
    new_dir (str): The path to the destination directory.
    file_out (str, optional): The name for the output file. Default is "hessian.dat".

    Returns:
    None
    """
    os.makedirs(new_dir, exist_ok=True)
    copy_and_rename_file(file_in, new_dir, file_out)
    return None


def find_nqetools_path():
    """
    Finds the path of the nqetools package.

    Returns:
    str: The directory path where the nqetools package is located.

    Raises:
    ImportError: If the nqetools package is not found.
    """
    spec = importlib.util.find_spec('nqetools')
    if not spec:
        raise ImportError("nqetools package not found")
    return os.path.dirname(spec.origin)


def search_fes_files(target_directory: str) -> list[str]:
    """
    Searches for files in the target directory with names matching the pattern 'FES*', where '*' is an integer.

    Parameters:
    target_directory (str): The directory to search in.

    Returns:
    list[str]: A list of matching file names.
    """
    pattern = re.compile(r'^FES\d+\.dat$')
    matching_files = []

    for filename in os.listdir(target_directory):
        if pattern.match(filename):
            matching_files.append(filename)

    return matching_files


def load_fes_data(directory: str, bins: int) -> list[np.ndarray]:
    """
    Find FES files in the directory and load their data into numpy arrays.

    Parameters:
    directory (str): Directory containing the FES files.
    bins (int): Number of bins for reshaping the data.

    Returns:
    list[np.ndarray]: List of numpy arrays, each containing the transformed FES data.
    """
    fes_files = search_fes_files(directory)
    fes_arrays = []
    bins += 1

    for file in sorted(fes_files):
        file_path = os.path.join(directory, file)

        # Read first line to detect number of CVs
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            n_cv = len(first_line.split()[2:])  # Skip #! FIELDS and time
        print(f"Loading {file_path} with {n_cv} FIELDS")
        data = np.loadtxt(file_path, comments="#")
        if n_cv == 2:
            transformed_data = np.array([1.0, 1.0 / eV_to_kJpermol])[:, np.newaxis] * data[:, :2].T
        else:
            transformed_data = np.array([1.0, 1.0, 1.0 / eV_to_kJpermol])[:, np.newaxis, np.newaxis] * data[:,
                                                                                                       :3].T.reshape(3,
                                                                                                                     bins,
                                                                                                                     bins)
        fes_arrays.append(transformed_data)

    return fes_arrays
