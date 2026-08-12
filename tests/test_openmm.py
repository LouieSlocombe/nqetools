"""Tests for the OpenMM setup, equilibration and RPMD routines.

Covers ligand parameterisation, deuteration, the staged equilibration
workflow, machine-learning and mixed ML/MM potentials, and the
ring-polymer reporters.
"""

import os
import sys
from sys import stdout

import numpy as np
import openmm.app as app
import openmm.unit as unit
from matplotlib import pyplot as plt
from openmm import openmm
from openmmml import MLPotential
from openmmplumed import PlumedForce

import nqetools as nqe


def test_openmm_ml():
    """Relax a system with a pure machine-learning potential."""
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_relaxation(modeller, forcefield, platform_name='CUDA')
    os.remove('minimized.pdb')


def test_openmm_ml_mixed_system():
    """Relax a system with a mixed ML/MM potential."""
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml',
                                'amber14/tip3pfb.xml')
    potential = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    padding = 1.5
    box_shape = 'dodecahedron'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]

    nqe.run_openmm_relaxation(modeller, forcefield, platform_name='CUDA', potential=potential, ml_idx=ml_atoms)
    os.remove('minimized.pdb')


def test_prepare_ligand_ff():
    """Generate GAFF parameters for a ligand and merge them into a force field."""
    print(flush=True)
    input_pdb = "tests/data/pdb/gt_wob_pol.pdb"
    cache_name = "gaff-molecules.json"
    rm_ions = ['Na+', 'Cl-', 'NA']
    residue_map = {'DGN': 'DG', 'DTN': 'DT', 'GTP': 'LIG'}
    pdb_data, molecule = nqe.prepare_lig_system(input_pdb,
                                                rm_ions=rm_ions,
                                                residue_map=residue_map,
                                                lig_name='LIG')
    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                          molecule,
                          gen_cache=True,
                          use_cache=False,
                          cache=cache_name)

    assert os.path.exists(cache_name)

    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule,
                                       gen_cache=False,
                                       use_cache=True,
                                       cache=cache_name)
    forcefield.createSystem(modeller.topology)

    os.remove(cache_name)


def test_nonstandard_ligand():
    """Parameterise and relax a system containing a non-standard ligand."""
    print(flush=True)
    input_pdb = "tests/data/pdb/gt_wob_pol.pdb"

    rm_ions = ['Na+', 'Cl-', 'NA']
    residue_map = {'DGN': 'DG', 'DTN': 'DT', 'GTP': 'LIG'}
    pdb_data, molecule = nqe.prepare_lig_system(input_pdb,
                                                rm_ions=rm_ions,
                                                residue_map=residue_map,
                                                lig_name='LIG')
    pdb_topology = pdb_data.topology
    pdb_positions = pdb_data.positions
    modeller = app.Modeller(pdb_topology, pdb_positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"), molecule)

    nqe.run_openmm_relaxation(modeller, forcefield, platform_name='CUDA')
    os.remove('minimized.pdb')


def _get_total_mass(system):
    """Sum the particle masses of an OpenMM system.

        Parameters
        ----------
        system : openmm.System
            System to total the masses of.

        Returns
        -------
        openmm.unit.Quantity
            The total mass.
    """
    total_mass = 0.0 * unit.dalton
    for i in range(system.getNumParticles()):
        total_mass += system.getParticleMass(i)
    return total_mass


def test_deuterate_system():
    """Check deuteration raises the total mass by the expected amount."""
    print(flush=True)
    pdb = app.PDBFile('tests/data/pdb/input.pdb')
    forcefield = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield,
                        model='tip3p',
                        padding=1.0 * unit.nanometer)

    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME,
                                     constraints=app.HBonds,
                                     rigidWater=True)

    mass_before = _get_total_mass(system)

    h_count = 0
    for atom in modeller.topology.atoms():
        if atom.element.symbol == 'H':
            h_count += 1

    print(f"--- Applying Deuteration to {h_count} Hydrogens (Option='water') ---")
    nqe.deuterate_system(modeller, system, option='all')

    mass_after = _get_total_mass(system)
    print(f"{'Mass Before':<20} | {mass_before.value_in_unit(unit.dalton):.4f} Da")
    print(f"{'Mass After':<20} | {mass_after.value_in_unit(unit.dalton):.4f} Da")
    print(f"{'Number of Hydrogens':<20} | {h_count}")

    mass_H = app.element.hydrogen.mass
    mass_D = app.element.deuterium.mass
    mass_delta_per_atom = mass_D - mass_H
    expected_increase = mass_delta_per_atom * h_count

    print(f"\n{'METRIC':<20} | {'VALUE':<15}")
    print("-" * 40)
    print(f"{'Mass Before':<20} | {mass_before.value_in_unit(unit.dalton):.4f} Da")
    print(f"{'Mass After':<20} | {mass_after.value_in_unit(unit.dalton):.4f} Da")
    print("-" * 40)
    print(f"{'Actual Increase':<20} | {(mass_after - mass_before).value_in_unit(unit.dalton):.4f} Da")
    print(f"{'Expected Increase':<20} | {expected_increase.value_in_unit(unit.dalton):.4f} Da")

    tolerance = 1e-3
    diff = abs((mass_after - mass_before) - expected_increase).value_in_unit(unit.dalton)

    if diff < tolerance:
        print("\n[SUCCESS] The system mass increased exactly as expected.")
    else:
        print("\n[FAILURE] The mass change did not match theoretical expectations.")


def test_plumed():
    """Run PLUMED-biased OpenMM dynamics and reconstruct the free energy surface."""
    temperature = 300 * unit.kelvin
    timestep = 1.0 * unit.femtosecond
    friction_coeff = 1.0 / unit.picosecond
    total_steps = 100_000
    pdb_file = 'tests/data/pdb/gt_wob_pol.pdb'
    pdb_out = 'pdb_out.pdb'

    directory = 'md_plumed'
    cwd = os.getcwd()

    rm_ions = ['Na+', 'Cl-', 'NA']
    residue_map = {'DGN': 'DG', 'DTN': 'DT', 'GTP': 'LIG'}

    pdb_data, molecule = nqe.prepare_lig_system(pdb_file, rm_ions=rm_ions, residue_map=residue_map, rm_files=False)
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"), molecule, pc_methods='am1bcc')

    idx1 = nqe.get_atoms_in_residue('combined_system.pdb', 0, chain_id='F')
    idx2 = nqe.get_atoms_in_residue('combined_system.pdb', 4, chain_id='B')
    selection_str1 = ','.join([f'{i}' for i in idx1])
    selection_str2 = ','.join([f'{i}' for i in idx2])

    nqe.remove_directory(directory)

    os.makedirs(directory, exist_ok=True)
    os.chdir(directory)

    modeller = app.Modeller(pdb_data.topology, pdb_data.positions)
    modeller.addSolvent(forcefield,
                        padding=1.0 * unit.nanometer,
                        boxShape='dodecahedron')

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds
    )

    system.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, temperature, 25))

    plumed_script = f"""c1: COM ATOMS={selection_str1}
c2: COM ATOMS={selection_str2}
dist: DISTANCE ATOMS=c1,c2
wall: UPPER_WALLS ARG=dist AT=8.0 KAPPA=100.0 EXP=2
opes: METAD ARG=dist PACE=500 HEIGHT=100 SIGMA=0.05 FILE=HILLS
PRINT ARG=* STRIDE=100 FILE=COLVAR
FLUSH STRIDE=1
"""
    with open('plumed.dat', 'w') as f:
        f.write(plumed_script)

    system.addForce(PlumedForce(plumed_script))
    integrator = openmm.LangevinIntegrator(
        temperature,
        friction_coeff,
        timestep
    )

    platform = openmm.Platform.getPlatformByName('CUDA')
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()

    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      1000,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      progress=True,
                                                      remainingTime=True,
                                                      speed=True,
                                                      totalSteps=total_steps,
                                                      separator='\t'))

    simulation.reporters.append(app.PDBReporter(pdb_out, 10_000))

    simulation.reporters.append(app.DCDReporter('trajectory.dcd', 10_000))
    simulation.step(total_steps)

    os.chdir(cwd)

    n_bins = 100
    cv_limits = [None, None]
    plot_save = True
    plot_show = True

    nqe.run_plumed_hills(directory,
                         temperature=300,
                         bins=n_bins,
                         cv=cv_limits)
    fes_arrays_meta_md = nqe.load_fes_data(directory, n_bins)
    fes_times = nqe.get_fes_times(2.0, total_steps, fes_arrays_meta_md)

    nqe.plot_fes_series_1d(fes_arrays_meta_md,
                           fes_times,
                           filename='fes_md',
                           save=plot_save,
                           show=plot_show)


def test_get_atoms_in_residue():
    """Look up the atom indices belonging to one residue."""
    print(flush=True)
    input_pdb = 'tests/data/pdb/input.pdb'
    indexes = nqe.get_atoms_in_residue(input_pdb, 0)
    print(indexes)
    ref_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    assert indexes == ref_indexes


def test_run_openmm_relaxation():
    """Run the staged restrained minimisation."""
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')
    nqe.run_openmm_relaxation(modeller, forcefield)
    os.remove('minimized.pdb')


def test_run_openmm_heating():
    """Run the staged heating equilibration."""
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')
    nqe.run_openmm_heating(modeller, forcefield)
    os.remove('equilibrated.pdb')


def test_run_openmm_heating_deuterate():
    """Run the staged heating equilibration on a deuterated system."""
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')
    nqe.run_openmm_heating(modeller, forcefield, deuterate=True)
    os.remove('equilibrated.pdb')


def test_run_openmm_npt():
    """Run the constant-pressure equilibration."""
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')
    nqe.run_openmm_npt(modeller, forcefield)
    os.remove('npt_equilibrated.pdb')


def test_eq_workflow():
    """Run relaxation, heating and NPT equilibration in sequence."""
    print(flush=True)
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')

    nqe.run_openmm_relaxation(modeller, forcefield)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_heating(modeller, forcefield)

    pdb = app.PDBFile("equilibrated.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_npt(modeller, forcefield)

    os.remove('minimized.pdb')
    os.remove('equilibrated.pdb')
    os.remove('npt_equilibrated.pdb')


def test_eq_workflow_mixed():
    """Run the full equilibration sequence with a mixed ML/MM potential."""
    print(flush=True)

    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml',
                                'amber14/tip3pfb.xml')
    potential = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    padding = 2.0
    box_shape = 'dodecahedron'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]

    nqe.run_openmm_relaxation(modeller, forcefield, potential=potential, ml_idx=ml_atoms)
    nqe.run_openmm_heating(modeller, forcefield, potential=potential, ml_idx=ml_atoms)
    nqe.run_openmm_npt(modeller, forcefield, potential=potential, ml_idx=ml_atoms)

    os.remove('minimized.pdb')
    os.remove('equilibrated.pdb')
    os.remove('npt_equilibrated.pdb')


def test_openmm_rpmd():
    """Run ring-polymer dynamics on a gas-phase system."""
    n_steps = 1_000
    report_every = 100
    in_pdb = "tests/data/pdb/input_aaa.pdb"
    n_beads = 2
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond

    pdb = app.PDBFile(in_pdb)
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    platform = openmm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    nqe.init_beads(modeller, simulation, n_beads)
    simulation.step(n_steps)


def test_openmm_qtb():
    """Run quantum thermal bath dynamics as a cheaper alternative to RPMD."""
    n_steps = 1_000
    report_every = 100
    in_pdb = "tests/data/pdb/input_aaa.pdb"
    n_beads = 2
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond

    pdb = app.PDBFile(in_pdb)
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    n_atoms = modeller.topology.getNumAtoms()
    print(f"System has {n_atoms} atoms.")

    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    integrator = openmm.QTBIntegrator(n_beads, temperature, friction, dt)

    platform = openmm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    simulation.step(n_steps)


def test_openmm_rpmd_solvated():
    """Run ring-polymer dynamics on a solvated system."""
    n_steps = 200
    report_every = 100
    in_pdb = "tests/data/pdb/input_aaa.pdb"
    n_beads = 2
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond

    pdb = app.PDBFile(in_pdb)
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.0 * unit.nanometer,
                        boxShape='dodecahedron')

    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=0.5 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    platform = openmm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    nqe.init_beads(modeller, simulation, n_beads)
    simulation.step(n_steps)


def test_openmm_rpmd_ml():
    """Run ring-polymer dynamics with a machine-learning potential."""
    n_steps = 200
    report_every = 100
    in_pdb = "tests/data/pdb/input_aaa.pdb"
    n_beads = 2
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond

    pdb = app.PDBFile(in_pdb)
    potential = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = potential.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    platform = openmm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    nqe.init_beads(modeller, simulation, n_beads)
    simulation.step(n_steps)


def test_openmm_rpmd_mixed():
    """Run ring-polymer dynamics with a mixed ML/MM potential."""
    print(flush=True)
    n_steps = 200
    report_every = 100
    in_pdb = "tests/data/pdb/input_aaa.pdb"
    n_beads = 4
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond

    padding = 1.5
    box_shape = 'dodecahedron'

    pdb = app.PDBFile(in_pdb)
    potential = MLPotential('mace-off23-small')
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)

    has_box = modeller.topology.getUnitCellDimensions() is not None
    mm_system = forcefield.createSystem(modeller.topology,
                                        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
                                        nonbondedCutoff=1.0 * unit.nanometer,
                                        constraints=None,
                                        rigidWater=False,
                                        removeCMMotion=True)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]
    n_atoms = modeller.topology.getNumAtoms()
    print(f"System has {n_atoms} atoms", flush=True)
    print(f"Number of ML atoms: {len(ml_atoms)}", flush=True)
    print(f"Number of MM atoms: {n_atoms - len(ml_atoms)}", flush=True)

    system = potential.createMixedSystem(modeller.topology,
                                         mm_system,
                                         ml_atoms)

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)

    platform = openmm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    nqe.init_beads(modeller, simulation, n_beads)
    simulation.step(n_steps)


def test_rpmd_quantum_spread_reporter():
    """Check the quantum spread reporter logs a radius of gyration per atom."""
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    modeller = app.Modeller(pdb.topology, pdb.positions)
    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    n_beads = 32
    integrator = openmm.RPMDIntegrator(n_beads,
                                       300 * unit.kelvin,
                                       1.0 / unit.picosecond,
                                       0.5 * unit.femtosecond)

    simulation = app.Simulation(modeller.topology, system, integrator)
    for i in range(n_beads):
        integrator.setPositions(i, modeller.positions)
    nqe.init_beads(modeller, simulation, n_beads)

    atoms_to_watch = [0, 1]
    atom_names = ["Atom0", "Atom1"]

    simulation.reporters.append(nqe.RPMDQuantumSpreadReporter(
        file="quantum_spread.txt",
        reportInterval=1,
        atom_indices=atoms_to_watch,
        names=atom_names
    ))

    print("Running RPMD with Quantum Spread reporting...")
    simulation.step(500)
    print("Done. Check 'quantum_spread.txt'.")
    data = np.loadtxt("quantum_spread.txt", skiprows=1, delimiter='\t')

    plt.plot(data[:, 0], data[:, 1], label=atom_names[0])
    plt.plot(data[:, 0], data[:, 2], label=atom_names[1])
    plt.xlabel('Step')
    plt.ylabel('Quantum Rg (nm)')
    plt.legend()
    plt.show()

    os.remove("quantum_spread.txt")


def test_rpmd_bead_reporter():
    """Check the bead reporter writes one trajectory file per bead."""
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    modeller = app.Modeller(pdb.topology, pdb.positions)
    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    n_beads = 4
    integrator = openmm.RPMDIntegrator(n_beads,
                                       300 * unit.kelvin,
                                       1.0 / unit.picosecond,
                                       0.5 * unit.femtosecond)

    simulation = app.Simulation(modeller.topology, system, integrator)

    nqe.init_beads(modeller, simulation, n_beads)

    simulation.reporters.append(nqe.RPMDBeadReporter(
        topology=modeller.topology,
        file_base_name="out",
        reportInterval=10,
        num_beads=n_beads,
    ))

    simulation.step(100)
    for i in range(n_beads):
        os.remove(f'out_bead_{i}.pdb')


def test_rpmd_centroid_reporter():
    """Check the centroid reporter writes the bead-averaged trajectory."""
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    modeller = app.Modeller(pdb.topology, pdb.positions)
    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    n_beads = 4
    integrator = openmm.RPMDIntegrator(n_beads,
                                       300 * unit.kelvin,
                                       1.0 / unit.picosecond,
                                       0.5 * unit.femtosecond)

    simulation = app.Simulation(modeller.topology, system, integrator)

    nqe.init_beads(modeller, simulation, n_beads)

    simulation.reporters.append(nqe.RPMDCentroidReporter(
        topology=modeller.topology,
        file_name="centroid.pdb",
        reportInterval=10,
        num_beads=n_beads,
    ))

    simulation.step(100)
    os.remove('centroid.pdb')


def test_count_dna_and_estimate_charge():
    """Check the DNA charge estimate matches the nucleotide count."""
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/gt_wob_pol.pdb")
    est_charge = nqe.count_dna_and_estimate_charge(pdb.topology)
    print(f"Estimated net charge: {est_charge}")
    assert est_charge == -6


def test_fep():
    # https://openmm.github.io/openmm-cookbook/latest/notebooks/tutorials/Alchemical_free_energy_calculations.html
    """Run an alchemical free energy calculation on a host-guest system."""
    print(flush=True)
    from openmmtools import testsystems, alchemy
    import copy
    from pymbar import MBAR, timeseries

    print("Creating test system...")
    host_guest = testsystems.HostGuestVacuum()
    system = host_guest.system
    positions = host_guest.positions
    topology = host_guest.topology

    ligand_atoms = [atom.index for atom in topology.atoms() if atom.residue.name == 'B2']
    factory = alchemy.AbsoluteAlchemicalFactory(consistent_exceptions=False)
    alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=ligand_atoms)
    alchemical_system = factory.create_alchemical_system(system, alchemical_region)

    lambda_electrostatics = [1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    lambda_sterics = [1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
    n_steps = len(lambda_electrostatics)
    alchemical_state = alchemy.AlchemicalState.from_system(alchemical_system)

    temperature = 300 * unit.kelvin

    integrator = openmm.LangevinIntegrator(temperature, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds)
    context = openmm.Context(alchemical_system, integrator)
    context.setPositions(positions)

    openmm.LocalEnergyMinimizer.minimize(context)

    # Storage for energy differences (u_kln)
    # u_kln[k, l] is the reduced potential energy of snapshot from state k evaluated at state l
    u_kln = np.zeros([n_steps, n_steps])

    print("Starting Alchemical FEP...")

    for k in range(n_steps):
        print(f"Sampling Window {k + 1}/{n_steps} (Elec: {lambda_electrostatics[k]}, VdW: {lambda_sterics[k]})")
        alchemical_state.lambda_electrostatics = lambda_electrostatics[k]
        alchemical_state.lambda_sterics = lambda_sterics[k]
        alchemical_state.apply_to_context(context)
        integrator.step(500)
        # In production, you would run this much longer (e.g., ns scale) and save many frames
        production_steps = 100_000
        integrator.step(production_steps)

        # Energy evaluation (cross-calculations for MBAR): take the current
        # configuration (x_k) and calculate its energy at ALL other lambda
        # states (l=0...N).
        for l in range(n_steps):
            temp_state = copy.deepcopy(alchemical_state)
            temp_state.lambda_electrostatics = lambda_electrostatics[l]
            temp_state.lambda_sterics = lambda_sterics[l]
            temp_state.apply_to_context(context)
            energy = context.getState(getEnergy=True).getPotentialEnergy()
            kT = unit.MOLAR_GAS_CONSTANT_R * temperature

            # Strip units safely by converting both to kJ/mol
            energy_val = energy.value_in_unit(unit.kilojoules_per_mole)
            kT_val = kT.value_in_unit(unit.kilojoules_per_mole)

            # Calculate reduced potential (dimensionless float)
            u_kln[k, l] = energy_val / kT_val

        # Reset context back to state k for the next loop iteration continuity
        alchemical_state.lambda_electrostatics = lambda_electrostatics[k]
        alchemical_state.lambda_sterics = lambda_sterics[k]
        alchemical_state.apply_to_context(context)

    print("Analyzing with MBAR...")
    N_k = np.zeros([n_steps], np.int32)  # number of uncorrelated samples
    for k in range(n_steps):
        [_nequil, g, _Neff_max] = timeseries.detect_equilibration(u_kln[k, k, :])
        indices = timeseries.subsample_correlated_data(u_kln[k, k, :], g=g)
        N_k[k] = len(indices)
        u_kln[k, :, 0:N_k[k]] = u_kln[k, :, indices].T

    mbar = MBAR(u_kln, N_k)
    results = mbar.compute_free_energy_differences(compute_uncertainty=True)

    print("Free energy change to insert a particle = ", results['Delta_f'][n_steps - 1, 0])
    print("Statistical uncertainty = ", results['dDelta_f'][n_steps - 1, 0])

    mbar = MBAR(u_kln, [1] * n_steps)
    result = mbar.compute_free_energy_differences()
    delta_f = result['Delta_f'][0, -1]
    delta_f_error = result['dDelta_f'][0, -1]
    print(f"Free Energy Difference (Complex Leg): {delta_f:.3f} +/- {delta_f_error:.3f} kT")
