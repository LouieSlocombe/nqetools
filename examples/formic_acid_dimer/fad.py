from ase.io import read
from ase.visualize import view
from mace.calculators import mace_off, mace_anicc

import nqetools as nqe

atoms = read('fad.xyz', -1)


fmax = 0.01
n_images = 9
calc = mace_off(model="small")  # mace_anicc()
calc = mace_anicc()
reactant = nqe.make_dimer(atoms)
reactant = nqe.optimise_geom(reactant, calc, fmax=fmax)
view(reactant)

# swap the bonding configuration
hb_1 = [0, 4, 6]
hb_2 = [5, 9, 1]
product = nqe.swap_bonding_configuration(reactant, *hb_1)
product = nqe.swap_bonding_configuration(product, *hb_2)
product = nqe.optimise_geom(product, calc, fmax=fmax)
view(product)

neb = nqe.prepare_neb(reactant, product, calc, n_images=n_images)

ts_path = nqe.optimise_neb(neb, fmax=fmax, n_images=n_images)
view(ts_path)

ts_image = nqe.get_ts_image(ts_path, calc)
view(ts_image)

nqe.plot_neb(ts_path, calc)
