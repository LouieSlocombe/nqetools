import glob
import os
import re
import shutil
import xml.etree.ElementTree as ET

import ase.io
import ase.io
import numpy as np
from ipi.utils.io import read_file


def read_ipi_xyz(filename):
    """ Reads a file in xyz i-PI format and returns it in ASE format. """

    file_handle = open(filename, "r")
    frames = []
    while True:
        try:
            ret = read_file("xyz", file_handle)
            frames.append(ase.Atoms(ret["atoms"].names,
                                    positions=ret["atoms"].q.reshape((-1, 3)) * 0.529177,
                                    cell=ret["cell"].h.T * 0.529177, pbc=True))
        except EOFError:
            break
        except:
            raise
    return frames


def read_ipi_output(filename):
    """ Reads an i-PI output file and returns a dictionary with the properties in a tidy order. """

    f = open(filename, "r")

    regex = re.compile(".*column *([0-9]*) *--> ([^ {]*)")

    fields = [];
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
    # Make sure the directory exists
    os.makedirs(os.path.dirname(file), exist_ok=True)
    # Write the input file
    tree = ET.ElementTree(root)
    tree.write(file)
    return None


def write_xyz(atoms, file, vacuum=20.0):
    atoms.center(vacuum=vacuum)
    # Make sure the directory exists
    os.makedirs(os.path.dirname(file), exist_ok=True)
    ase.io.write(file, atoms)
    return atoms


def remove_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    else:
        print(f"The directory {directory} does not exist.")
    return None


def copy_and_rename_file(src, dst_dir, new_name):
    dst = os.path.abspath(os.path.join(dst_dir, new_name))
    shutil.copy(src, dst)
    return None


def list_files_with_pattern(directory, pattern):
    return glob.glob(os.path.join(directory, pattern))


def get_final_xyz(dir, sub="*FINAL_*.xyz"):
    l = list_files_with_pattern(dir, sub)
    # Only select files that end with xyz
    l_filt = [f for f in l if f.endswith(".xyz")]
    # Select the file that does not contain the string "forces"
    l_nf = [f for f in l_filt if "forces" not in f]
    # Remove the file that contains the string "_0"
    l_n0 = [f for f in l_nf if "_0.xyz" not in f]
    return l_n0[0]


def get_final_hess(dir, sub=f"*FINAL.hess*"):
    return list_files_with_pattern(dir, sub)[0]


def copy_xyz(file_in, new_dir, file_out="init.xyz"):
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
    os.makedirs(new_dir, exist_ok=True)
    copy_and_rename_file(file_in, new_dir, file_out)
    return None
