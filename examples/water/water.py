import time
import os

import ase.build
from mace.calculators import mace_anicc, mace_off
from ase.io import write, read
import nqetools as nqe

if __name__ == "__main__":
    print(flush=True)
    print("Testing MACE calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    calc = mace_anicc()
    # calc = mace_off(model="small", device="cpu", default_dtype="float32")
    # set the calculator
    atoms.calc = calc
    # run the calculation
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()
    print(f"Energy: {energy} Time: {t2 - t1}")
    pass
