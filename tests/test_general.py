import sys
import time

import ase.build
import matplotlib.pyplot as plt
import numpy as np
import os
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.calculators.nwchem import NWChem
from ase.calculators.qmmm import SimpleQMMM
from ase.calculators.socketio import PySocketIOClient, SocketIOCalculator
from ase.io import write, read
from ase.optimize import BFGS
from ase.visualize import view
from mace.calculators import mace_anicc, mace_off, mace_omol
from subprocess import Popen

import nqetools as nqe


def test_calculate_good_nbeads():
    """
    Tests the calculate_good_nbeads function from the nqetools module.

    This function calculates the number of beads (replicas) needed for a
    path integral molecular dynamics (PIMD) simulation based on the
    maximum vibrational frequency (omega_max) and temperature (T).

    For water, omega_max is approximately 3800 cm^-1, which results in
    approximately 18 beads at 300 K. A safe choice is 32 replicas.

    Asserts:
        The calculated number of beads is 18.
    """
    n_beads = nqe.calculate_good_nbeads(3800.0, 300.0)
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
    atoms = ase.build.molecule('H2O')

    print('mace_anicc', flush=True)
    atoms.calc = mace_anicc()
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()
    print(f"Energy: {energy} Time: {t2 - t1}", flush=True)

    print('mace_off', flush=True)
    atoms.calc = mace_off(model="small",
                          device="cpu",
                          default_dtype="float32")
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()
    print(f"Energy: {energy} Time: {t2 - t1}", flush=True)

    print('mace_omol', flush=True)
    atoms.calc = mace_omol(device="cpu",
                           default_dtype="float32")
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
    calc = nqe.nwchem_calc_preset(directory='nwchem_test')

    # Set the calculator
    atoms.calc = calc

    # Run the calculation
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()

    # Print the energy and time taken
    print(f"Energy: {energy} Time: {t2 - t1}", flush=True)
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


def test_get_ipi_driver():
    print(flush=True)
    driver_path = nqe.get_ipi_driver()
    ref_path = "/home/louie/anaconda3/envs/ipi_env2/lib/python3.12/site-packages/ipi/bin/i-pi-driver"
    assert driver_path == ref_path


def test_load_xyz_with_cell():
    print(flush=True)
    atoms = nqe.read_ipi_xyz("data/h5o2+.xyz")[-1]
    target_positions = [[0.54044999, -0.97484999, -0.21658],
                        [2.92483996, -0.83925199, 0.152397],
                        [0.18386, -1.25803998, -1.07874999],
                        [1.74328998, -0.90927799, -0.100395],
                        [3.29706995, -1.59323998, 0.64725299],
                        [3.53032995, -0.60813799, -0.57607799],
                        [0.0423316, -0.19486, 0.092281]]
    target_cell = [[13.383769816511084, 0.0, 0.0],
                   [8.195195433157629e-16, 13.383769816511084, 0.0],
                   [8.195195433157629e-16, 8.195195433157629e-16, 13.383769816511084]]
    comparison = [np.allclose(a, b) for a, b in zip(atoms.get_cell(), target_cell)]
    assert all(comparison)
    comparison = [np.allclose(a, b) for a, b in zip(atoms.get_positions(), target_positions)]
    assert all(comparison)


def test_wigner_correction():
    """
    Tests the Wigner tunneling correction factor calculation.

    This function calculates the Wigner tunneling correction factor (κ)
    using the `wigner_correction` function from the `nqetools` module.
    The calculation is based on the vibrational frequency (omega) and
    temperature (T). The result is printed and compared to an expected
    value using an assertion.

    Parameters:
        None

    Asserts:
        The calculated Wigner tunneling factor for each temperature in the
        `temperature_list` matches the corresponding value in `kappa_list`
        within an absolute tolerance of 1e-2.

    Prints:
        The calculated Wigner tunneling factor (κ) for each temperature
        in the `temperature_list` to three decimal places.
    """
    print(flush=True)
    omega = -2017.96  # Frequency in cm^-1
    temperature_list = [300, 500, 1000, 1500, 2000]  # Temperatures in K
    kappa_list = [4.90263, 2.40495, 1.35124, 1.15611, 1.08781]  # Expected κ values

    for i, temperature in enumerate(temperature_list):
        kappa_wigner = nqe.wigner_correction(omega, temperature)
        print(f"Wigner tunnelling factor κ = {kappa_wigner:.3f}")
        assert np.isclose(kappa_wigner, kappa_list[i], atol=1e-2)


def test_bell_correction():
    """
    Tests the Bell tunneling correction factor calculation.

    This function calculates the Bell tunneling correction factor (κ)
    using the `bell_correction` function from the `nqetools` module.
    The calculation is based on the barrier height, width of the barrier,
    and reduced mass. The result is printed and compared to an expected
    value using an assertion.

    Parameters:
        None

    Asserts:
        The calculated Bell tunneling factor is approximately 28573.077
        with a tolerance of 1e-2.

    Prints:
        The calculated Bell tunneling factor (κ) to three decimal places.
    """
    print(flush=True)
    e_barrier = 0.7  # Barrier height in eV
    a = 0.35  # Width of the barrier in Angstroms
    mu = 1.0  # Reduced mass in atomic mass units (amu)
    kappa_bell = nqe.bell_correction(e_barrier, a, mu)
    print(f"Bell tunnelling factor κ = {kappa_bell:.3f}")
    assert np.isclose(kappa_bell, 28573.077, atol=1e-2)


def test_eckart_correction():
    """
    Tests the Eckart tunneling correction factor calculation.

    This function calculates the Eckart tunneling correction factor (κ)
    using the `eckart_correction` function from the `nqetools` module.
    The calculation is based on the temperature, vibrational frequency,
    and energy values for the reactants, transition state, and products.
    The results are printed and compared to expected values using assertions.

    Parameters:
        None

    Asserts:
        The calculated Eckart tunneling factor for each temperature in the
        `temperature_list` matches the corresponding value in `kappa_list`
        within a relative tolerance of 1e-2.

    Prints:
        The calculated Eckart tunneling factor (κ) for each temperature
        in the `temperature_list` to three decimal places.
    """
    print(flush=True)
    E_reac = 0.0  # Energy of reactants in eV
    E_TS = 2.93124318  # Energy of the transition state in eV
    E_prod = 0.329755902  # Energy of products in eV
    freq = -2017.96  # Vibrational frequency in cm^-1
    temperature_list = [300, 500, 1000, 1500, 2000]  # Temperatures in K
    kappa_list = [1623051.0, 7.69349, 1.46551, 1.18111, 1.09858]  # Expected κ values

    for i, temperature in enumerate(temperature_list):
        kappa_eckart = nqe.eckart_correction(temperature, freq, E_reac, E_TS, E_prod)
        print(f"Eckart κ({temperature} K) = {kappa_eckart:.3f}")
        assert np.isclose(kappa_eckart, kappa_list[i], rtol=1e-2)


def test_fit_exp_decay():
    print(flush=True)
    rng = np.random.default_rng(seed=42)
    x_data = np.linspace(0, 10, 60)
    true_params = (5.0, 2.0, 0.5)
    y_clean = nqe.exp_decay(x_data, *true_params)
    y_noise = y_clean + 0.3 * rng.standard_normal(size=x_data.size)

    popt = nqe.fit_exp_decay(x_data, y_noise)

    print(f"Fitted parameters (A, tau, C): {popt}", flush=True)

    plt.scatter(x_data, y_noise, label="data", marker="o")
    x_fine = np.linspace(x_data.min(), x_data.max(), 400)
    plt.plot(x_fine, nqe.exp_decay(x_fine, *popt), label="best fit", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    # Assert that the fitted parameters are close to the true parameters
    assert np.allclose(popt, true_params, rtol=0.2)


def test_extrapolate_inf_bead_limit():
    print(flush=True)
    n_beads = [10, 20, 40, 80]
    kappa = [18.486247675370873, 11.481103709870357, 10.127813329699068, 9.809437521247709]
    kappa_inf = nqe.extrapolate_inf_bead_limit(n_beads, kappa, plot=True)
    print(f"Extrapolated kappa_inf: {kappa_inf:.3f}", flush=True)
    # Assert that the extrapolated kappa_inf is close to the last value in kappa
    assert np.isclose(kappa_inf, kappa[-1], rtol=0.2)


def test_orca_onion():
    print(flush=True)
    print("Testing ORCA ONIOM calculator", flush=True)

    # Build the molecule
    atoms = read('data/onion_example.xyz')

    # Set up the ORCA ONIOM calculator
    calc = nqe.orca_calc_preset(calc_type='QM/XTB2',
                                atom_list='0:11',
                                n_procs=1)

    # Set the calculator
    atoms.calc = calc

    # Run the calculation
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()

    # Print the energy and time taken
    print(f"Energy: {energy} Time: {t2 - t1}", flush=True)
    assert np.allclose(energy, -7226.730291092625, rtol=1e-2), "Energy does not match expected value"


def test_prepare_neb():
    print(flush=True)
    n_images = 15
    f_max = 0.01
    f_climb = False
    spring_constant = 5.0
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    # Make initial state.
    reactant_init = molecule('C2H6')
    # Create final state.
    product_init = reactant_init.copy()
    product_init.positions[2:5] = product_init.positions[[3, 4, 2]]

    # Set up calculator options
    calc1 = mace_off(model="small",
                     device="cuda",
                     default_dtype="float64")
    calc2 = mace_off(model="medium",
                     device="cuda",
                     default_dtype="float64")
    calc3 = mace_off(model="large",
                     device="cuda",
                     default_dtype="float64")
    calc_a = mace_anicc()
    calc_o = mace_omol()

    calcs = [calc1, calc2, calc3, calc_a, calc_o]
    labels = ['off_sma', 'off_med', 'off_lar', 'anicc', 'omol']

    for i, calc in enumerate(calcs):
        reactant = nqe.optimise_geom(reactant_init, calc, fmax=f_max)
        product = nqe.optimise_geom(product_init, calc, fmax=f_max)
        neb = nqe.prepare_neb(reactant,
                              product,
                              calc,
                              n_images=n_images,
                              climb=f_climb,
                              k=spring_constant,
                              geo_int=True)
        neb_path = nqe.optimise_neb(neb,
                                    fmax=f_max,
                                    n_images=n_images,
                                    ts_traj='ts.traj')
        os.remove('ts.traj')
        nqe.plot_neb(neb_path, calc, fig=fig, ax=ax, label=labels[i])

    plt.legend()
    plt.show()


def test_ase_qmmm():
    print(flush=True)
    m1 = molecule('H2O')
    m2 = molecule('C2H6')
    m2.translate([3, 0, 0])
    atoms = m1 + m2
    atoms.center(vacuum=5.0)
    view(atoms)

    # Set up cheap small model for MM
    mm_calc = mace_off(model="small",
                       device="cuda",
                       default_dtype="float64")
    # Set up expensive model for QM
    qm_calc = mace_omol(device="cuda",
                        default_dtype="float64")

    qmmm_calc = SimpleQMMM([0, 1, 2],
                           qm_calc,
                           mm_calc,
                           mm_calc)
    atoms.calc = qmmm_calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy:.3f}", flush=True)
    assert np.allclose(energy, -4253.264, rtol=1e-2), "Energy does not match expected value"


def test_ase_qmmm_orca_mace():
    print(flush=True)
    m1 = molecule('H2O')
    m2 = molecule('C2H6')
    m2.translate([3, 0, 0])
    atoms = m1 + m2
    atoms.center(vacuum=5.0)

    # Set up cheap small model for MM
    mm_calc = mace_off(model="small",
                       device="cuda",
                       default_dtype="float64")
    qm_calc = nqe.orca_calc_preset()

    qmmm_calc = SimpleQMMM([0, 1, 2],
                           qm_calc,
                           mm_calc,
                           mm_calc)
    atoms.calc = qmmm_calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy:.3f}", flush=True)
    assert np.allclose(energy, -4250.656, rtol=1e-2), "Energy does not match expected value"


def test_ase_eiqmmm():
    # Broken as ORCA does not have AttributeError: 'ORCA' object has no attribute 'embed'

    from ase.calculators.qmmm import EIQMMM, Embedding, LJInteractions
    from ase.calculators.tip3p import TIP3P, epsilon0, sigma0
    from ase.data import s22

    # Create system
    atoms = s22.create_s22_system('Water_dimer')
    atoms.center(vacuum=4.0)

    # Make QM atoms selection of first water molecule:
    qm_idx = range(3)

    # Set up interaction & embedding object
    interaction = LJInteractions({('O', 'O'): (epsilon0, sigma0)})
    embedding = Embedding(rc=0.02)  # Short range analytical potential cutoff

    # Set up calculator
    atoms.calc = EIQMMM(
        qm_idx,
        nqe.orca_calc_preset(),
        TIP3P(),
        interaction,
        embedding=embedding,
        vacuum=None,  # if None, QM cell = MM cell
        output='qmmm.log',
    )

    print(atoms.get_potential_energy())
