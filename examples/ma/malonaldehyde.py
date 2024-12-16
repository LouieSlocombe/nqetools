import os

from ase.data.pubchem import pubchem_atoms_search
from ase.visualize import view
import numpy as np
from ase import Atoms
from ase.build import molecule
# import mace off
from mace.calculators import mace_off, mace_anicc
from ase.optimize import BFGS

from ase.io import read, write

import nqetools as nqe
from nqetools import get_ts_image, get_neb_path

fmax = 0.05
n_images = 5
# calc = mace_off(model="large")
calc = mace_anicc()

# atoms = pubchem_atoms_search(smiles="C(C=O)C=O")
atoms = read("malonaldehyde.traj")
# delete the hydrogen atom
del atoms[-1]

del atoms[5]

atoms = nqe.optimise_geom(atoms, calc, fmax=fmax)

# product = swap_bonding_configuration(reactant,0, 9,1)
reactant = nqe.add_hydrogen_at_distance(atoms, 0, 1, 1.0)
product = nqe.add_hydrogen_at_distance(atoms, 1, 0, 1.0)

reactant = nqe.optimise_geom(reactant, calc, fmax=fmax)
view(reactant)

product = nqe.optimise_geom(product, calc, fmax=fmax)
view(product)

neb = nqe.prepare_neb(reactant, product, calc, n_images=n_images)

ts_path = nqe.optimise_neb(neb, fmax=fmax, n_images=n_images)
view(ts_path)

ts_image = nqe.get_ts_image(ts_path, calc)
view(ts_image)

nqe.plot_neb(ts_path, calc)

vib = nqe.get_vibrations(ts_image, calc)
print(vib, flush=True)
