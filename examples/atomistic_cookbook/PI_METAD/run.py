import os
import subprocess
import chemiscope
import ipi
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
import nqetools as nqe
from ase.visualize import view

# This follows:
# https://atomistic-cookbook.org/examples/pi-metad/pi-metad.html


if __name__ == "__main__":
    print(flush=True)
    # Make a directory to store everything
    directory_md = "md"
    directory_metamd = "metamd"
    directory_metapimd = "metapimd"
    n_beads = 8
    timestep = 1.0
    total_steps = 5000
    stride = 10
    temperature = 298
    thermostat = 'smart_sampling_1ps_n6_w2'
    driver_code = 'zundel'
    md_type = "NVT-GLE"
    plumed_type = "mtd-coord"

    # Load the initial structure
    atoms = read("h5o2+.xyz", index=-1)
    atoms.center(vacuum=20.0)
    view(atoms)

    # Run unbiased MD
    # Make sure the directory is empty
    # Run the calculation
    nqe.remove_directory(directory_md)
    atoms = nqe.run_md(directory_md, atoms, driver=driver_code, total_steps=total_steps, temperature=temperature,
                       timestep=timestep, thermostat=thermostat, md_type=md_type, stride=stride, n_beads=1)
    view(atoms)

    # Run metadynamics
    # Make sure the directory is empty
    # Run the calculation
    nqe.remove_directory(directory_metamd)
    atoms = nqe.run_plumed_md(directory_metamd, atoms, driver=driver_code, total_steps=total_steps,
                              temperature=temperature, timestep=timestep, thermostat=thermostat, md_type=md_type,
                              stride=stride, n_beads=1, plumed_type=plumed_type)
    view(atoms)


