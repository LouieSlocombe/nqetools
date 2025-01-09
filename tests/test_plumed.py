import time

import ase.build
from mace.calculators import mace_anicc

import nqetools as nqe

def test_plumed_input():
    impt = """
    
    """

    # write the input file
    with open("plumed.dat", "w") as f:
        f.write(impt)

    pass