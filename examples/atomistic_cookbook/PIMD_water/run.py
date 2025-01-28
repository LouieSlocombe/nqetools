import time
import os

import ase.build
from mace.calculators import mace_anicc, mace_off
from ase.io import write, read
import nqetools as nqe
from ase.visualize import view

# This follows:
# https://atomistic-cookbook.org/examples/path-integrals/path-integrals.html

if __name__ == "__main__":
    print(flush=True)
    print("PIMD of water", flush=True)
    # build the molecule
    atoms = read("water_32.pdb")
    # atoms = ase.build.molecule('H2O')
    # atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "PIMD"
    # Make sure the directory is empty
    nqe.remove_directory(directory)
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT")#, xml_in="input_pimd.xml")


