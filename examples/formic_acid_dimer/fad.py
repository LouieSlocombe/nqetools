from ase.io import read
from ase.visualize import view
from mace.calculators import mace_off, mace_anicc

import nqetools as nqe


def make_dimer(file_in='fad.xyz'):
    atoms = read(file_in, -1)
    atoms.center()
    # make a copy
    atoms2 = atoms.copy()
    # rotate the second molecule
    atoms2.rotate(180, 'z', rotate_cell=False)

    # translate the second molecule
    atoms2.translate([0.0, 3.4, 0.0])
    # combine the two molecules
    combined = atoms + atoms2
    return combined


fmax = 0.01
n_images = 9
calc = mace_off(model="small")  # mace_anicc()
calc = mace_anicc()
reactant = make_dimer()
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
