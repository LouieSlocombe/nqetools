import time
import os

import ase.build
from mace.calculators import mace_anicc, mace_off
from ase.io import write, read
import nqetools as nqe
from ase.visualize import view

"""
Test the exe functions for simple cases
optimise water
md of water and confirm and output file
vibrate water
TS of water

Check that the other calculators work

"""


def test_calculate_nbeads():
    # For water omega_max ~ 3800 invcm this number is ~ 18 at 300 K. So a safe choice is 32 replicas
    n_beads = nqe.calculate_nbeads(3800.0, 300.0)
    assert n_beads == 18


def test_orca_calc():
    print(flush=True)
    print("Testing ORCA calculator", flush=True)

    # build the molecule
    atoms = ase.build.molecule('H2O')
    calc = nqe.orca_calc_preset()

    # set the calculator
    atoms.calc = calc
    # run the calculation
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()
    print(f"Energy: {energy} Time: {t2 - t1}", flush=True)
    pass


def test_orca_presets():
    print(flush=True)

    # create a dictionary of presets arguments
    calc = nqe.orca_calc_preset(**nqe.orca_preset_dft_cheap)
    atoms = ase.build.molecule('H2O')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy}")

    calc = nqe.orca_calc_preset(**nqe.orca_preset_dft_gold)
    atoms = ase.build.molecule('H2O')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy}")

    # xtb calculation
    calc = nqe.orca_calc_preset(**nqe.orca_preset_xtb)
    atoms = ase.build.molecule('H2O')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy}")

    # MP2 calculation
    calc = nqe.orca_calc_preset(**nqe.orca_preset_mp2_gold)
    atoms = ase.build.molecule('H2O')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy}")

    # CCSD(T) calculation
    calc = nqe.orca_calc_preset(**nqe.orca_preset_ccsd_gold)
    atoms = ase.build.molecule('H2O')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy}")

    pass


def test_mace_calc():
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


def test_mace_driver():
    print(flush=True)
    print("Testing MACE calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_opti"
    atoms_out = nqe.run_optimise(directory, atoms, driver='ase-mace')
    pass


def test_orca_driver():
    print(flush=True)
    print("Testing ORCA driver", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)
    # make a directory to store everything
    directory = "orca_opti"
    nqe.run_optimise(directory, atoms, driver='ase-orca')
    pass


def test_exe_md():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_opti"
    atoms_out = nqe.run_optimise(directory, atoms, driver='ase-mace')
    pass


def test_exe_phonons():
    pass


def test_exe_ts():
    pass


def test_inst():
    pass
