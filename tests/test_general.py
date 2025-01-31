import os
import sys
import time
from subprocess import Popen

import ase.build
from ase.build import molecule
from ase.calculators.nwchem import NWChem
from ase.io import write
from ase.optimize import BFGS
from mace.calculators import mace_anicc, mace_off
from ase.calculators.socketio import PySocketIOClient, SocketIOCalculator
import nqetools as nqe
from ase.calculators.emt import EMT


def test_calculate_nbeads():
    """
    Tests the calculate_nbeads function from the nqetools module.

    This function calculates the number of beads (replicas) needed for a
    path integral molecular dynamics (PIMD) simulation based on the
    maximum vibrational frequency (omega_max) and temperature (T).

    For water, omega_max is approximately 3800 cm^-1, which results in
    approximately 18 beads at 300 K. A safe choice is 32 replicas.

    Asserts:
        The calculated number of beads is 18.
    """
    n_beads = nqe.calculate_nbeads(3800.0, 300.0)
    assert n_beads == 18


def test_orca_calc():
    """
    Tests the ORCA calculator from the nqetools module.

    This function builds a water molecule, sets up the ORCA calculator,
    runs the calculation to get the potential energy, and prints the energy
    and the time taken for the calculation.

    Asserts:
        None
    """
    print(flush=True)
    print("Testing ORCA calculator", flush=True)

    # Build the molecule
    atoms = ase.build.molecule('H2O')
    calc = nqe.orca_calc_preset()

    # Set the calculator
    atoms.calc = calc

    # Run the calculation
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()
    print(f"Energy: {energy} Time: {t2 - t1}", flush=True)
    pass


def test_orca_presets():
    """
    Tests the ORCA calculator presets from the nqetools module.

    This function tests various ORCA calculator presets by building a water molecule,
    setting up the ORCA calculator with different presets, running the calculation to get
    the potential energy, and printing the energy.

    Asserts:
        None
    """
    print(flush=True)
    print("Testing ORCA calculator presets", flush=True)

    # create a dictionary of presets arguments
    calc = nqe.orca_calc_preset(**nqe.orca_preset_dft_cheap)
    atoms = ase.build.molecule('H2O')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy}", flush=True)

    calc = nqe.orca_calc_preset(**nqe.orca_preset_dft_gold)
    atoms = ase.build.molecule('H2O')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy}", flush=True)

    # xtb calculation
    calc = nqe.orca_calc_preset(**nqe.orca_preset_xtb)
    atoms = ase.build.molecule('H2O')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy}", flush=True)

    # MP2 calculation
    calc = nqe.orca_calc_preset(**nqe.orca_preset_mp2_gold)
    atoms = ase.build.molecule('H2O')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy}", flush=True)

    # CCSD(T) calculation
    calc = nqe.orca_calc_preset(**nqe.orca_preset_ccsd_gold)
    atoms = ase.build.molecule('H2O')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy}", flush=True)
    pass


def test_mace_calc():
    """
    Tests the MACE calculator.

    This function builds a water molecule, sets up the MACE calculator with different presets,
    runs the calculation to get the potential energy, and prints the energy and the time taken
    for the calculation.

    Asserts:
        None
    """
    print(flush=True)
    print("Testing MACE calculator", flush=True)

    # Build the molecule
    atoms = ase.build.molecule('H2O')

    print('mace_anicc', flush=True)
    calc = mace_anicc()

    # Set the calculator
    atoms.calc = calc

    # Run the calculation
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()
    print(f"Energy: {energy} Time: {t2 - t1}", flush=True)

    print('mace_off', flush=True)
    calc = mace_off(model="small", device="cpu", default_dtype="float32")

    # Set the calculator
    atoms.calc = calc

    # Run the calculation
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()
    print(f"Energy: {energy} Time: {t2 - t1}", flush=True)
    pass


def test_nwchem_calc():
    """
    Tests the NWChem calculator from the nqetools module.

    This function builds a water molecule, sets up the NWChem calculator,
    runs the calculation to get the potential energy, and prints the energy
    and the time taken for the calculation.

    Asserts:
        None
    """
    print(flush=True)
    print("Testing NWChem calculator", flush=True)

    # Build the molecule
    atoms = ase.build.molecule('H2O')

    # Set up the NWChem calculator
    calc = nqe.nwchem_calc_preset()

    # Set the calculator
    atoms.calc = calc

    # Run the calculation
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()

    # Print the energy and time taken
    print(f"Energy: {energy} Time: {t2 - t1}", flush=True)
    pass


def test_ase_mace_driver():
    """
    Tests the MACE calculator using the ASE driver.

    This function builds a water molecule, centers it with a vacuum of 5.0 Å,
    sets up the MACE calculator, runs the optimization, and stores the results
    in a specified directory.

    Asserts:
        None
    """
    print(flush=True)
    print("Testing MACE calculator", flush=True)

    # Build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_opti"
    nqe.run_optimise(directory, atoms, driver='ase-mace')
    pass


def test_ase_orca_driver():
    """
    Tests the ORCA driver using the ASE driver.

    This function builds a water molecule, centers it with a vacuum of 5.0 Å,
    sets up the ORCA driver, runs the optimization, and stores the results
    in a specified directory.

    Asserts:
        None
    """
    print(flush=True)
    print("Testing ORCA driver", flush=True)

    # Build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "orca_opti"
    nqe.run_optimise(directory, atoms, driver='ase-orca')
    pass


def test_ase_nwchem_driver():
    """
    Tests the NWChem driver using the ASE driver.

    This function builds a water molecule, centers it with a vacuum of 5.0 Å,
    sets up the NWChem driver, runs the optimization, and stores the results
    in a specified directory.

    Asserts:
        None
    """
    print(flush=True)
    print("Testing NWChem driver", flush=True)

    # Build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "nwchem_opti"
    nqe.remove_directory(directory)

    # Run the optimization
    nqe.run_optimise(directory, atoms, driver='ase-nwchem')
    pass


def test_nwchem_driver():
    """
    Tests the NWChem driver.

    This function builds a water molecule, centers it with a vacuum of 5.0 Å,
    sets up the NWChem driver, runs the optimization, and stores the results
    in a specified directory.

    Asserts:
        None
    """
    print(flush=True)
    print("Testing NWChem driver", flush=True)

    # Build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "nwchem_opti"
    nqe.remove_directory(directory)

    # Run the optimization
    nqe.run_optimise(directory, atoms, driver='nwchem')
    pass


def test_exe_md():
    """
    Tests the MACE MD calculator.

    This function builds a water molecule, centers it with a vacuum of 5.0 Å,
    sets up the MACE MD calculator, runs the molecular dynamics simulation,
    and stores the results in a specified directory.

    Asserts:
        None
    """
    print(flush=True)
    print("Testing MACE md calculator", flush=True)

    # Build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT-GLE")
    pass


def test_exe_phonons():
    pass


def test_exe_ts():
    pass


def test_inst():
    pass


def test_nwchem_socket():
    print(flush=True)
    # https://wiki.fysik.dtu.dk/ase/ase/calculators/socketio/socketio.html
    atoms = molecule('H2O')
    atoms.rattle(stdev=0.1)

    unixsocket = 'ase_nwchem'

    nwchem = NWChem(theory='scf',
                    task='optimize',
                    driver={'socket': {'unix': unixsocket}})

    opt = BFGS(atoms,
               trajectory='opt.traj',
               logfile='opt.log')

    with SocketIOCalculator(nwchem,
                            log=sys.stdout,
                            unixsocket=unixsocket) as calc:
        atoms.calc = calc
        opt.run(fmax=0.05)
    pass


def test_ase_server_socket():
    print(flush=True)
    # https://wiki.fysik.dtu.dk/ase/ase/calculators/socketio/socketio.html#run-server-and-client-manually

    unixsocket = 'driver'

    atoms = molecule('H2O', vacuum=3.0)
    atoms.rattle(stdev=0.1)
    write('initial.traj', atoms)

    opt = BFGS(atoms, trajectory='opt.driver.traj', logfile='opt.driver.log')
    # Start the server
    with SocketIOCalculator(log=sys.stdout,
                            unixsocket=unixsocket) as calc:
        # Start the client in a separate process
        Popen([sys.executable, 'example_ase_client.py'])

        atoms.calc = calc
        opt.run(fmax=0.05)
    pass


def test_py_socket():
    print(flush=True)
    atoms = molecule('H2O', vacuum=3.0)
    atoms.rattle(stdev=0.1)

    client = PySocketIOClient(EMT)
    pid = os.getpid()
    with SocketIOCalculator(launch_client=client,
                            unixsocket=f'ase-python-{pid}') as atoms.calc:
        with BFGS(atoms) as opt:
            opt.run(fmax=0.1)
