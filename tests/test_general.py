import time

import ase.build
from mace.calculators import mace_anicc

import nqetools as nqe

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
    orca_path = '/home/louie/orca_6_0_1/orca'

    # build the molecule
    atoms = ase.build.molecule('H2O')
    calc = nqe.orca_calc_preset(orca_path=orca_path, nprocs=1)

    # set the calculator
    atoms.calc = calc
    # run the calculation
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()
    print(f"Energy: {energy} Time: {t2 - t1}")
    pass


def test_mace_calc():
    # build the molecule
    atoms = ase.build.molecule('H2O')
    calc = mace_anicc()
    # set the calculator
    atoms.calc = calc
    # run the calculation
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()
    print(f"Energy: {energy} Time: {t2 - t1}")
    pass


def test_mace_driver():
    pass


def test_orca_driver():
    pass


def test_exe_optimise():
    pass


def test_exe_md():
    pass


def test_exe_phonons():
    pass


def test_exe_ts():
    pass


def test_inst():
    pass
