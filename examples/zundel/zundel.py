from ase.io import read
from ase.visualize import view
from matplotlib import pyplot as plt
import time

import nqetools as nqe

if __name__ == "__main__":
    # load the zundel molecule
    atoms = read("h5o2+.xyz", "-1")

    # view(atoms)

    impt = nqe.write_plumed_input_coordination(atoms)
    # write the input file
    with open("plumed.dat", "w") as f:
        f.write(impt)


    # create a dictionary of presets arguments
    calc = nqe.orca_calc_preset(**nqe.orca_preset_dft_cheap,charge=1)
    atoms.calc = calc
    t0 = time.time()
    energy = atoms.get_potential_energy()
    t1 = time.time()
    print(f"Energy: {energy}, Time: {t1 - t0}", flush=True)