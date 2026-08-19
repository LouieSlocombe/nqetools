"""Tests for calculators, tunnelling corrections and file handling.

Covers the quantum chemistry calculator presets, the analytic tunnelling
corrections, ASE socket and QM/MM setups, and the PDB and XYZ cleaning
routines.
"""

import os
import sys
import time
from subprocess import Popen

import ase.build
import matplotlib.pyplot as plt
import numpy as np
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.calculators.nwchem import NWChem
from ase.calculators.qmmm import SimpleQMMM
from ase.calculators.socketio import PySocketIOClient, SocketIOCalculator
from ase.io import write, read
from ase.optimize import BFGS
from ase.visualize import view
from mace.calculators import mace_anicc, mace_off, mace_omol

import nqetools as nqe
import reactiontools as rt


def test_calculate_good_nbeads():
    """Test the calculate_good_nbeads function from the nqetools module.

    Calculates the number of beads (replicas) needed for a
    path integral molecular dynamics (PIMD) simulation based on the
    maximum vibrational frequency (omega_max) and temperature (T).

    For water, omega_max is approximately 3800 cm^-1. The ratio
    hbar*omega/(kB*T) at 300 K is 18.22, so 19 beads are needed for the
    inequality to hold. A safe choice is 32 replicas.

    Notes
    -----
    Asserts that the calculated number of beads is 19.
    """
    n_beads = nqe.calculate_good_nbeads(3800.0, 300.0)
    assert n_beads == 19


def test_mace_calc():
    """Test the MACE calculator.

    Builds a water molecule, sets up the MACE calculator with different presets,
    runs the calculation to get the potential energy, and prints the energy and the time taken
    for the calculation.
    """
    print(flush=True)
    print("Testing MACE calculator", flush=True)
    atoms = ase.build.molecule("H2O")

    print("mace_anicc", flush=True)
    atoms.calc = mace_anicc()
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()
    print(f"Energy: {energy} Time: {t2 - t1}", flush=True)

    print("mace_off", flush=True)
    atoms.calc = mace_off(model="small", device="cpu", default_dtype="float32")
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()
    print(f"Energy: {energy} Time: {t2 - t1}", flush=True)

    print("mace_omol", flush=True)
    atoms.calc = mace_omol(device="cpu", default_dtype="float32")
    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()
    print(f"Energy: {energy} Time: {t2 - t1}", flush=True)

    pass


def test_nwchem_calc():
    """Test the NWChem calculator from the nqetools module.

    Builds a water molecule, sets up the NWChem calculator,
    runs the calculation to get the potential energy, and prints the energy
    and the time taken for the calculation.
    """
    print(flush=True)
    print("Testing NWChem calculator", flush=True)

    atoms = ase.build.molecule("H2O")

    calc = nqe.nwchem_calc_preset(directory="nwchem_test")

    atoms.calc = calc

    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()

    print(f"Energy: {energy} Time: {t2 - t1}", flush=True)
    pass


def test_exe_md():
    """Test the MACE MD calculator.

    Builds a water molecule, centres it with a vacuum of 5.0 Å,
    sets up the MACE MD calculator, runs the molecular dynamics simulation,
    and stores the results in a specified directory.
    """
    print(flush=True)
    print("Testing MACE md calculator", flush=True)

    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "mace_md"
    nqe.run_md(directory, atoms, driver="ase-mace", md_type="NVT-GLE")
    pass


def test_nwchem_socket():
    """Optimise water with NWChem over ASE's socket interface."""
    print(flush=True)
    # https://wiki.fysik.dtu.dk/ase/ase/calculators/socketio/socketio.html
    atoms = molecule("H2O")
    atoms.rattle(stdev=0.1)

    unixsocket = "ase_nwchem"

    nwchem = NWChem(
        theory="scf", task="optimize", driver={"socket": {"unix": unixsocket}}
    )

    opt = BFGS(atoms, trajectory="opt.traj", logfile="opt.log")

    with SocketIOCalculator(nwchem, log=sys.stdout, unixsocket=unixsocket) as calc:
        atoms.calc = calc
        opt.run(fmax=0.05)
    pass


def test_ase_server_socket():
    """Drive an optimisation with ASE as server and a separate client process."""
    print(flush=True)
    # https://wiki.fysik.dtu.dk/ase/ase/calculators/socketio/socketio.html#run-server-and-client-manually

    unixsocket = "driver"

    atoms = molecule("H2O", vacuum=3.0)
    atoms.rattle(stdev=0.1)
    write("initial.traj", atoms)

    opt = BFGS(atoms, trajectory="opt.driver.traj", logfile="opt.driver.log")
    # Start the server
    with SocketIOCalculator(log=sys.stdout, unixsocket=unixsocket) as calc:
        # Start the client in a separate process
        Popen([sys.executable, "example_ase_client.py"])

        atoms.calc = calc
        opt.run(fmax=0.05)
    pass


def test_py_socket():
    """Optimise water with ASE's in-process Python socket client."""
    print(flush=True)
    atoms = molecule("H2O", vacuum=3.0)
    atoms.rattle(stdev=0.1)

    client = PySocketIOClient(EMT)
    pid = os.getpid()
    with SocketIOCalculator(
        launch_client=client, unixsocket=f"ase-python-{pid}"
    ) as atoms.calc:
        with BFGS(atoms) as opt:
            opt.run(fmax=0.1)


def test_get_ipi_driver():
    """Check the i-PI driver executable is found on the system."""
    print(flush=True)
    driver_path = nqe.get_ipi_driver()
    ref_path = "/home/louie/anaconda3/envs/ipi_env2/lib/python3.12/site-packages/ipi/bin/i-pi-driver"
    assert driver_path == ref_path


def test_load_xyz_with_cell():
    """Read an i-PI XYZ trajectory and recover its cell."""
    print(flush=True)
    atoms = nqe.read_ipi_xyz("data/h5o2+.xyz")[-1]
    target_positions = [
        [0.54044999, -0.97484999, -0.21658],
        [2.92483996, -0.83925199, 0.152397],
        [0.18386, -1.25803998, -1.07874999],
        [1.74328998, -0.90927799, -0.100395],
        [3.29706995, -1.59323998, 0.64725299],
        [3.53032995, -0.60813799, -0.57607799],
        [0.0423316, -0.19486, 0.092281],
    ]
    target_cell = [
        [13.383769816511084, 0.0, 0.0],
        [8.195195433157629e-16, 13.383769816511084, 0.0],
        [8.195195433157629e-16, 8.195195433157629e-16, 13.383769816511084],
    ]
    comparison = [
        np.allclose(a, b) for a, b in zip(atoms.get_cell(), target_cell, strict=True)
    ]
    assert all(comparison)
    comparison = [
        np.allclose(a, b)
        for a, b in zip(atoms.get_positions(), target_positions, strict=True)
    ]
    assert all(comparison)


def test_wigner_correction():
    """Test the Wigner tunnelling correction factor calculation.

    Calculates the Wigner tunnelling correction factor (κ)
    using the `wigner_correction` function from the `nqetools` module.
    The calculation is based on the vibrational frequency (omega) and
    temperature (T). The result is printed and compared to an expected
    value using an assertion.

    Notes
    -----
    Asserts that the calculated Wigner tunnelling factor for each temperature
    in `temperature_list` matches the corresponding value in `kappa_list`
    within an absolute tolerance of 1e-2.
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
    """Test the Bell tunnelling correction factor calculation.

    Calculates the Bell tunnelling correction factor (κ)
    using the `bell_correction` function from the `nqetools` module.
    The calculation is based on the barrier height, width of the barrier,
    and reduced mass. The result is printed and compared to an expected
    value using an assertion.

    Notes
    -----
    Asserts that the calculated Bell tunnelling factor is approximately
    28573.077 with a tolerance of 1e-2.
    """
    print(flush=True)
    e_barrier = 0.7  # Barrier height in eV
    a = 0.35  # Width of the barrier in Angstroms
    mu = 1.0  # Reduced mass in atomic mass units (amu)
    kappa_bell = nqe.bell_correction(e_barrier, a, mu)
    print(f"Bell tunnelling factor κ = {kappa_bell:.3f}")
    assert np.isclose(kappa_bell, 28573.077, atol=1e-2)


def test_eckart_correction():
    """Test the Eckart tunnelling correction factor calculation.

    Calculates the Eckart tunnelling correction factor (κ)
    using the `eckart_correction` function from the `nqetools` module.
    The calculation is based on the temperature, vibrational frequency,
    and energy values for the reactants, transition state, and products.
    The results are printed and compared to expected values using assertions.

    Notes
    -----
    Asserts that the calculated Eckart tunnelling factor for each temperature
    in `temperature_list` matches the corresponding value in `kappa_list`
    within a relative tolerance of 1e-2.
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
    """Recover known parameters by fitting an exponential decay."""
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

    assert np.allclose(popt, true_params, rtol=0.2)


def test_extrapolate_inf_bead_limit():
    """Extrapolate a bead-convergence series to the infinite-bead limit."""
    print(flush=True)
    n_beads = [10, 20, 40, 80]
    kappa = [
        18.486247675370873,
        11.481103709870357,
        10.127813329699068,
        9.809437521247709,
    ]
    kappa_inf = nqe.extrapolate_inf_bead_limit(n_beads, kappa, plot=True)
    print(f"Extrapolated kappa_inf: {kappa_inf:.3f}", flush=True)
    assert np.isclose(kappa_inf, kappa[-1], rtol=0.2)


def test_orca_onion():
    """Build a layered ONIOM-style ORCA calculator."""
    print(flush=True)
    print("Testing ORCA ONIOM calculator", flush=True)

    atoms = read("data/onion_example.xyz")

    # xc/basis_set were implicit when this preset lived in nqetools; they are
    # spelled out now that it comes from reactiontools, whose defaults differ,
    # so the reference energy below still refers to the same level of theory.
    calc = rt.orca_calc_preset(
        calc_type="QM/XTB2",
        atom_list="0:11",
        n_procs=1,
        xc="wB97X",
        basis_set="def2-SVP",
    )

    atoms.calc = calc

    t1 = time.time()
    energy = atoms.get_potential_energy()
    t2 = time.time()

    print(f"Energy: {energy} Time: {t2 - t1}", flush=True)
    assert np.allclose(energy, -7226.730291092625, rtol=1e-2), (
        "Energy does not match expected value"
    )


def test_prepare_neb():
    """Run a nudged elastic band between two ethane conformers."""
    print(flush=True)
    n_images = 15
    f_max = 0.01
    f_climb = False
    spring_constant = 5.0
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    reactant_init = molecule("C2H6")
    product_init = reactant_init.copy()
    product_init.positions[2:5] = product_init.positions[[3, 4, 2]]

    calc1 = mace_off(model="small", device="cuda", default_dtype="float64")
    calc2 = mace_off(model="medium", device="cuda", default_dtype="float64")
    calc3 = mace_off(model="large", device="cuda", default_dtype="float64")
    calc_a = mace_anicc()
    calc_o = mace_omol()

    calcs = [calc1, calc2, calc3, calc_a, calc_o]
    labels = ["off_sma", "off_med", "off_lar", "anicc", "omol"]

    for i, calc in enumerate(calcs):
        reactant = rt.optimise_geom(reactant_init, calc, fmax=f_max)
        product = rt.optimise_geom(product_init, calc, fmax=f_max)
        neb = rt.prepare_neb(
            reactant,
            product,
            calc,
            n_images=n_images,
            climb=f_climb,
            k=spring_constant,
            geo_int=True,
        )
        # optimise_neb takes the image count from the band itself
        neb_path = rt.optimise_neb(neb, fmax=f_max, ts_traj="ts.traj")
        os.remove("ts.traj")
        rt.plot_neb(neb_path, calc, fig=fig, ax=ax, label=labels[i])

    plt.legend()
    plt.show()


def test_ase_qmmm():
    """Set up a QM/MM partition with ASE's SimpleQMMM."""
    print(flush=True)
    m1 = molecule("H2O")
    m2 = molecule("C2H6")
    m2.translate([3, 0, 0])
    atoms = m1 + m2
    atoms.center(vacuum=5.0)
    view(atoms)

    # Set up cheap small model for MM
    mm_calc = mace_off(model="small", device="cuda", default_dtype="float64")
    # Set up expensive model for QM
    qm_calc = mace_omol(device="cuda", default_dtype="float64")

    qmmm_calc = SimpleQMMM([0, 1, 2], qm_calc, mm_calc, mm_calc)
    atoms.calc = qmmm_calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy:.3f}", flush=True)
    assert np.allclose(energy, -4253.264, rtol=1e-2), (
        "Energy does not match expected value"
    )


def test_ase_qmmm_orca_mace():
    """Set up a QM/MM partition with ORCA as QM and MACE as MM."""
    print(flush=True)
    m1 = molecule("H2O")
    m2 = molecule("C2H6")
    m2.translate([3, 0, 0])
    atoms = m1 + m2
    atoms.center(vacuum=5.0)

    # Set up cheap small model for MM
    mm_calc = mace_off(model="small", device="cuda", default_dtype="float64")
    qm_calc = rt.orca_calc_preset(xc="wB97X", basis_set="def2-SVP")

    qmmm_calc = SimpleQMMM([0, 1, 2], qm_calc, mm_calc, mm_calc)
    atoms.calc = qmmm_calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy:.3f}", flush=True)
    assert np.allclose(energy, -4250.656, rtol=1e-2), (
        "Energy does not match expected value"
    )


def test_ase_eiqmmm():
    # Broken as ORCA does not have AttributeError: 'ORCA' object has no attribute 'embed'

    """Set up an electrostatically embedded QM/MM partition."""
    from ase.calculators.qmmm import EIQMMM, Embedding, LJInteractions
    from ase.calculators.tip3p import TIP3P, epsilon0, sigma0
    from ase.data import s22

    atoms = s22.create_s22_system("Water_dimer")
    atoms.center(vacuum=4.0)

    # Make QM atoms selection of first water molecule:
    qm_idx = range(3)

    interaction = LJInteractions({("O", "O"): (epsilon0, sigma0)})
    embedding = Embedding(rc=0.02)  # Short range analytical potential cutoff

    atoms.calc = EIQMMM(
        qm_idx,
        rt.orca_calc_preset(xc="wB97X", basis_set="def2-SVP"),
        TIP3P(),
        interaction,
        embedding=embedding,
        vacuum=None,  # if None, QM cell = MM cell
        output="qmmm.log",
    )

    print(atoms.get_potential_energy())


def test_xyz_to_sdf():
    """Convert an XYZ structure to SDF, inferring bonds."""
    print(flush=True)

    atoms = molecule("H2O")
    print(atoms.positions)
    write("water.xyz", atoms)
    nqe.xyz_to_sdf("water.xyz", "water.sdf", default_charge=0)
    # Read back the SDF file and check it has the same positions
    atoms_sdf = read("water.sdf")
    print(atoms_sdf.positions)

    comparison = [
        np.allclose(a, b, rtol=0.001)
        for a, b in zip(atoms.get_positions(), atoms_sdf.get_positions(), strict=True)
    ]
    assert all(comparison)
    os.remove("water.sdf")
    os.remove("water.xyz")


def test_extract_nonstandard_res():
    """Extract non-standard residues from a PDB file."""
    print(flush=True)

    input_pdb = "tests/data/pdb/gt_wob_solv.pdb"
    generated_files = nqe.extract_nonstandard_res(input_pdb, ".", sdf=True)
    assert len(generated_files) == 2, "Generated files do not match expected files"

    atoms_0 = read(generated_files[0])
    atoms_1 = read(generated_files[1])
    atoms_sdf = atoms_0 + atoms_1
    view(atoms_sdf)

    for file in generated_files:
        print(file, flush=True)
        os.remove(file)


def test_get_non_standard_residues():
    """Identify the non-standard residues in a PDB file."""
    print(flush=True)
    pdb_file = "tests/data/pdb/gt_wob_solv.pdb"
    non_standard_mols = nqe.get_non_standard_residues(pdb_file)
    assert len(non_standard_mols) == 2


def test_list_non_standard_residues():
    """List the non-standard residue names in a PDB file."""
    print(flush=True)
    pdb_file = "tests/data/pdb/gt_wob_solv.pdb"
    non_standard_residues = nqe.list_non_standard_residues(pdb_file)
    print(non_standard_residues, flush=True)
    assert non_standard_residues == ["DGN", "DTN"]

    pdb_file = "tests/data/pdb/gt_wob_pol.pdb"
    non_standard_residues = nqe.list_non_standard_residues(pdb_file)
    print(non_standard_residues, flush=True)
    assert non_standard_residues == ["DC3", "DC5", "DG3", "DG5", "GTP"]


def test_clean_pdb_ions():
    """Strip ions from a PDB file."""
    print(flush=True)
    input_pdb = "tests/data/pdb/gt_wob_solv.pdb"
    output_pdb = "cleaned_gt_wob_solv.pdb"
    rm_ions = ["Na+", "Cl-"]

    nqe.clean_ions_in_pdb(input_pdb, rm_ions, output_pdb)

    with open(output_pdb) as f:
        lines = f.readlines()
    ion_lines = [
        line
        for line in lines
        if line.startswith("HETATM") and line[17:20].strip() in rm_ions
    ]
    assert len(ion_lines) == 0, "Ions were not removed from the PDB file"
    os.remove(output_pdb)

    input_pdb = "tests/data/pdb/gt_wob_pol.pdb"
    output_pdb = "cleaned_gt_wob_pol.pdb"
    rm_ions = ["Na+", "Cl-", "NA"]

    nqe.clean_ions_in_pdb(input_pdb, rm_ions, output_pdb)

    with open(output_pdb) as f:
        lines = f.readlines()
    ion_lines = [
        line
        for line in lines
        if line.startswith("HETATM") and line[17:20].strip() in rm_ions
    ]
    assert len(ion_lines) == 0, "Ions were not removed from the PDB file"
    os.remove(output_pdb)


def test_relabel_residues_in_pdb():
    """Rename residues in a PDB file."""
    print(flush=True)
    input_pdb = "tests/data/pdb/gt_wob_solv.pdb"
    output_pdb = "relabelled_gt_wob_solv.pdb"
    residue_map = {"DGN": "DG", "DTN": "DT"}

    nqe.relabel_residues_in_pdb(input_pdb, residue_map, output_pdb)

    with open(output_pdb) as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("HETATM"):
            res_name = line[17:20].strip()
            assert res_name not in residue_map, f"Residue {res_name} was not relabelled"
    os.remove(output_pdb)


def test_remove_water_residues_in_pdb():
    """Strip water molecules from a PDB file."""
    print(flush=True)
    input_pdb = "tests/data/pdb/gt_wob_solv.pdb"
    output_pdb = "nowater_gt_wob_solv.pdb"

    nqe.remove_water_residues_in_pdb(input_pdb, output_pdb)

    with open(output_pdb) as f:
        lines = f.readlines()
    water_lines = [
        line
        for line in lines
        if line.startswith("HETATM") and line[17:20].strip() in ["HOH", "WAT"]
    ]
    assert len(water_lines) == 0, "Water residues were not removed from the PDB file"
    os.remove(output_pdb)
