import matplotlib.pyplot as plt
from ase.io import read
from ase.visualize import view
from mace.calculators import mace_off

import nqetools as nqe

plt.rcParams['axes.linewidth'] = 2.0
f_neb = False
f_vib = False

fmax = 0.05
n_images = 5
calc = mace_off(model="small")
barrier = 0.2
temperature = 300.0

atoms = read("malonaldehyde.traj")
atoms.center(vacuum=5.0)

# delete the hydrogen atom
del atoms[-1]
del atoms[5]
atoms = nqe.optimise_geom(atoms, calc, fmax=fmax)

# product = swap_bonding_configuration(reactant,0, 9,1)
reactant = nqe.add_hydrogen_at_distance(atoms, 0, 1, 1.0)
product = nqe.add_hydrogen_at_distance(atoms, 1, 0, 1.0)

# reactant = nqe.optimise_geom(reactant, calc, fmax=fmax)
# view(reactant)

# product = nqe.optimise_geom(product, calc, fmax=fmax)
# view(product)

if f_neb:
    neb = nqe.prepare_neb(reactant, product, calc, n_images=n_images)
    ts_path = nqe.optimise_neb(neb, fmax=fmax, n_images=n_images)
    view(ts_path)
    nqe.plot_neb(ts_path, calc)
    ts_image = nqe.get_ts_image(ts_path, calc)
    view(ts_image)

    if f_vib:
        vib = nqe.get_vibrations(ts_image, calc)
        print(vib, flush=True)

# Minimise the reactant using ipi
# Make a directory to store everything
directory = "opti"
nqe.remove_directory(directory)
atoms_out = nqe.run_optimise(directory, reactant, driver='ase-mace')
# view(atoms_out)
atoms = atoms_out[-1]

# Conduct short MD simulation
# Run the metadynamics simulation
directory = "mtd"
# If the directory exists, remove it
nqe.remove_directory(directory)
# atoms_out = nqe.run_plumed_md(directory,
#                               atoms,
#                               driver='ase-mace',
#                               md_type="NVT",
#                               plumed_type="mtd-diff1",
#                               temperature=temperature,
#                               plumed_dict={"idx1": 1,
#                                            "idx2": 8,
#                                            "idx3": 0,
#                                            "height": barrier * 0.001,
#                                            "bias": 1,
#                                            "sigma": 0.1,
#                                            "pace": 500})

atoms_out = nqe.run_plumed_md(directory, atoms, driver='ase-mace', temperature=temperature, md_type="NVT-GLE",
                              plumed_type="opes-diff1")
view(atoms_out)
# Analyse the results

# Run the OPES simulation
# Analyse the results
