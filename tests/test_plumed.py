import time

import ase.build
from mace.calculators import mace_anicc

import nqetools as nqe

def test_plumed():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_md"
    atoms_out = nqe.run_plumed_md(directory, atoms, driver='ase-mace', md_type="NVT")
    pass