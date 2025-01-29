import time
import os

import ase.build
from mace.calculators import mace_anicc, mace_off
from ase.io import write, read
import nqetools as nqe
from ase.visualize import view
import matplotlib.pyplot as plt
import ipi

# This follows:
# https://atomistic-cookbook.org/examples/path-integrals/path-integrals.html

if __name__ == "__main__":
    print(flush=True)
    print("PIMD of water", flush=True)
    # build the molecule
    atoms = read("water_32.pdb")
    atoms.center(vacuum=0.0)

    f_run = False

    # Make a directory to store everything
    directory = "PIMD"

    if f_run:
        # Make sure the directory is empty
        nqe.remove_directory(directory)
        # Run the calculation
        nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT", xml_in="input_pimd.xml")

    # Read the results
    # drops first frame where all atoms overlap
    output_data, output_desc = ipi.read_output("simulation.out")
    traj_data = [ipi.read_trajectory(f"simulation.pos_{i}.xyz")[1:] for i in range(8)]

    fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    ax.plot(
        output_data["time"],
        output_data["potential"] - output_data["potential"][0],
        "b-",
        label="Potential, $V$",
    )
    ax.plot(
        output_data["time"],
        output_data["conserved"] - output_data["conserved"][0],
        "r-",
        label="Conserved, $H$",
    )
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"energy / eV")
    ax.legend()
    plt.show()