import os
import sys
from pathlib import Path
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
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_relaxation(modeller, forcefield, platform_name='CUDA')
    os.remove('minimized.pdb')


def test_openmm_ml_mixed_system():
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml',
                                'amber14/tip3pfb.xml')
    potential = MLPotential('mace-off23-small')

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    # Solvate
    padding = 1.5
    box_shape = 'dodecahedron'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)

    mm_system = forcefield.createSystem(modeller.topology)
    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]

    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = potential.createMixedSystem(
        modeller.topology,
        mm_system,
        ml_atoms,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    # Run langevin dynamics at 300K for 1000 steps
    integrator = openmm.LangevinIntegrator(300 * unit.kelvin,
                                           1.0 / unit.picoseconds,
                                           1.0 * unit.femtosecond)

    platform = openmm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(modeller.topology,
                                system,
                                integrator,
                                platform)

    simulation.context.setPositions(modeller.positions)
    simulation.reporters.append(
        app.StateDataReporter(
            stdout,
            100,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True
        )
    )

    # Set the velocities to 300K and run 1000 steps
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
    simulation.step(1_000)


def test_prepare_ligand_ff():
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

    # Check that the cache files were created
    assert os.path.exists(cache_name)

    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"),
                                       molecule,
                                       gen_cache=False,
                                       use_cache=True,
                                       cache=cache_name)
    forcefield.createSystem(modeller.topology)

    os.remove(cache_name)


def test_nonstandard_ligand():
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


def test_openmm_rpmd():
    # Simple run parameters
    n_steps = 1_000
    report_every = 100
    data_out = 'md_log.txt'
    in_pdb = "tests/data/pdb/input_aaa.pdb"
    out_pdb = Path("centroid_trajectory.pdb")
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

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    integrator.setApplyThermostat(True)
    integrator.setRandomNumberSeed(2025)

    platform = openmm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    simulation.reporters.append(app.StateDataReporter(data_out,
                                                      report_every,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    # Initialize each bead with the input coordinates + tiny random jiggle
    nqe.init_beads(modeller, simulation, n_beads)

    # Prepare multi-MODEL PDB (centroid coordinates)
    with open(out_pdb, "w") as fh:
        app.PDBFile.writeHeader(modeller.topology, fh)
        # write initial centroid (model 0)
        centroid = nqe.centroid_positions(simulation, n_atoms, n_beads)
        nqe.write_multimodel_pdb(modeller.topology, centroid, fh, model_index=0)

        # Integrate and save snapshots
        for step in range(1, n_steps + 1):
            simulation.step(1)

            if step % report_every == 0:
                centroid = nqe.centroid_positions(simulation, n_atoms, n_beads)
                nqe.write_multimodel_pdb(modeller.topology, centroid, fh, model_index=step // report_every)

        app.PDBFile.writeFooter(modeller.topology, fh)
    print(f"\nWrote centroid trajectory to: {out_pdb.resolve()}")
    os.remove(out_pdb)
    os.remove(data_out)


def test_openmm_rpmd_solvated():
    # Simple run parameters
    n_steps = 200
    report_every = 100
    data_out = 'md_log.txt'
    in_pdb = "tests/data/pdb/input_aaa.pdb"
    out_pdb = Path("centroid_trajectory.pdb")
    n_beads = 2
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond

    pdb = app.PDBFile(in_pdb)
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    # Solvate
    modeller.addSolvent(forcefield,
                        padding=1.0 * unit.nanometer,
                        boxShape='dodecahedron')

    n_atoms = modeller.topology.getNumAtoms()
    print(f"System has {n_atoms} atoms.")

    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=0.5 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    k = 1000.0 * unit.kilojoule_per_mole / unit.nanometer ** 2  # typical strong restraint for minimization
    restraint = openmm.CustomExternalForce("0.5 * k * periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", k)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    # Select backbone atoms (N, CA, C) and add them with their reference coordinates
    bb_indices = []
    for atom, pos in zip(modeller.topology.atoms(), modeller.positions):
        if atom.name in ("N", "CA", "C"):
            idx = atom.index
            bb_indices.append(idx)
            restraint.addParticle(idx, [pos.x, pos.y, pos.z])

    system.addForce(restraint)

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    integrator.setApplyThermostat(True)
    integrator.setRandomNumberSeed(2025)

    platform = openmm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    simulation.reporters.append(app.StateDataReporter(data_out,
                                                      report_every,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    # Initialize each bead with the input coordinates + tiny random jiggle
    nqe.init_beads(modeller, simulation, n_beads)

    # Prepare multi-MODEL PDB (centroid coordinates)
    with open(out_pdb, "w") as fh:
        app.PDBFile.writeHeader(modeller.topology, fh)
        # write initial centroid (model 0)
        centroid = nqe.centroid_positions(simulation, n_atoms, n_beads)
        nqe.write_multimodel_pdb(modeller.topology, centroid, fh, model_index=0)

        # Integrate and save snapshots
        for step in range(1, n_steps + 1):
            simulation.step(1)

            if step % report_every == 0:
                centroid = nqe.centroid_positions(simulation, n_atoms, n_beads)
                nqe.write_multimodel_pdb(modeller.topology, centroid, fh, model_index=step // report_every)

        app.PDBFile.writeFooter(modeller.topology, fh)
    print(f"\nWrote centroid trajectory to: {out_pdb.resolve()}")
    os.remove(out_pdb)
    os.remove(data_out)


def test_openmm_rpmd_ml():
    # Simple run parameters
    n_steps = 200
    report_every = 100
    data_out = 'md_log.txt'
    in_pdb = "tests/data/pdb/input_aaa.pdb"
    out_pdb = Path("centroid_trajectory.pdb")
    n_beads = 2
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond

    pdb = app.PDBFile(in_pdb)
    potential = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    n_atoms = modeller.topology.getNumAtoms()
    print(f"System has {n_atoms} atoms.")

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
    integrator.setApplyThermostat(True)
    integrator.setRandomNumberSeed(2025)

    platform = openmm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    simulation.reporters.append(app.StateDataReporter(data_out,
                                                      report_every,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    # Initialize each bead with the input coordinates + tiny random jiggle
    nqe.init_beads(modeller, simulation, n_beads)

    # Prepare multi-MODEL PDB (centroid coordinates)
    with open(out_pdb, "w") as fh:
        app.PDBFile.writeHeader(modeller.topology, fh)
        # write initial centroid (model 0)
        centroid = nqe.centroid_positions(simulation, n_atoms, n_beads)
        nqe.write_multimodel_pdb(modeller.topology, centroid, fh, model_index=0)

        # Integrate and save snapshots
        for step in range(1, n_steps + 1):
            simulation.step(1)

            if step % report_every == 0:
                centroid = nqe.centroid_positions(simulation, n_atoms, n_beads)
                nqe.write_multimodel_pdb(modeller.topology, centroid, fh, model_index=step // report_every)

        app.PDBFile.writeFooter(modeller.topology, fh)
    print(f"\nWrote centroid trajectory to: {out_pdb.resolve()}")
    os.remove(out_pdb)
    os.remove(data_out)


def test_openmm_rpmd_mixed():
    # Simple run parameters
    n_steps = 200
    report_every = 100
    data_out = 'md_log.txt'
    in_pdb = "tests/data/pdb/input_aaa.pdb"
    out_pdb = Path("centroid_trajectory.pdb")
    n_beads = 2
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

    n_atoms = modeller.topology.getNumAtoms()
    print(f"System has {n_atoms} atoms.")

    mm_system = forcefield.createSystem(modeller.topology)
    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]

    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = potential.createMixedSystem(
        modeller.topology,
        mm_system,
        ml_atoms,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )
    # system = potential.createSystem(
    #     modeller.topology,
    #     nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
    #     nonbondedCutoff=1.0 * unit.nanometer,
    #     constraints=None,
    #     rigidWater=False,
    #     removeCMMotion=True,
    # )

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    integrator.setApplyThermostat(True)
    integrator.setRandomNumberSeed(2025)

    platform = openmm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      report_every,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    simulation.reporters.append(app.StateDataReporter(data_out,
                                                      report_every,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    # Initialize each bead with the input coordinates + tiny random jiggle
    nqe.init_beads(modeller, simulation, n_beads)

    # Prepare multi-MODEL PDB (centroid coordinates)
    with open(out_pdb, "w") as fh:
        app.PDBFile.writeHeader(modeller.topology, fh)
        # write initial centroid (model 0)
        centroid = nqe.centroid_positions(simulation, n_atoms, n_beads)
        nqe.write_multimodel_pdb(modeller.topology, centroid, fh, model_index=0)

        # Integrate and save snapshots
        for step in range(1, n_steps + 1):
            simulation.step(1)

            if step % report_every == 0:
                centroid = nqe.centroid_positions(simulation, n_atoms, n_beads)
                nqe.write_multimodel_pdb(modeller.topology, centroid, fh, model_index=step // report_every)

        app.PDBFile.writeFooter(modeller.topology, fh)
    print(f"\nWrote centroid trajectory to: {out_pdb.resolve()}")
    os.remove(out_pdb)
    os.remove(data_out)


def _get_total_mass(system):
    total_mass = 0.0 * unit.dalton
    for i in range(system.getNumParticles()):
        total_mass += system.getParticleMass(i)
    return total_mass


def test_deuterate_system():
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
    # print the number of hydrogens
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

    # Assertion Check
    tolerance = 1e-3
    diff = abs((mass_after - mass_before) - expected_increase).value_in_unit(unit.dalton)

    if diff < tolerance:
        print("\n[SUCCESS] The system mass increased exactly as expected.")
    else:
        print("\n[FAILURE] The mass change did not match theoretical expectations.")


def test_plumed():
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

    # nqe.save_pdb_selection('combined_system.pdb', idx1 + idx2, 'selection.pdb')
    # Clean the directory if it exists
    nqe.remove_directory(directory)

    # Make the directory if it doesn't exist
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
    # Write the PLUMED script to a file
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

    # Run the hills command
    nqe.run_plumed_hills(directory,
                         temperature=300,
                         bins=n_bins,
                         cv=cv_limits)
    # Plot the free energy surface convergence
    fes_arrays_meta_md = nqe.load_fes_data(directory, n_bins)
    fes_times = nqe.get_fes_times(2.0, total_steps, fes_arrays_meta_md)

    nqe.plot_fes_series_1d(fes_arrays_meta_md,
                           fes_times,
                           filename='fes_md',
                           save=plot_save,
                           show=plot_show)


def test_get_atoms_in_residue():
    print(flush=True)
    input_pdb = 'tests/data/pdb/input.pdb'
    indexes = nqe.get_atoms_in_residue(input_pdb, 0)
    print(indexes)
    ref_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    assert indexes == ref_indexes


def test_run_openmm_relaxation():
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


def calculate_quantum_spread(integrator, atom_indices=None):
    """
    Calculates the root-mean-square distance of beads from the ring polymer centroid.
    This is a measure of quantum delocalization (quantum spread).

    Parameters
    ----------
    integrator : openmm.RPMDIntegrator
        The integrator running the simulation.
    atom_indices : list of int, optional
        List of atom indices to calculate the spread for.
        If None, calculates for ALL atoms (can be memory intensive).

    Returns
    -------
    spreads : openmm.unit.Quantity (numpy array)
        An array of shape (n_selected_atoms,) containing the quantum Rg
        for each selected atom in nanometers.
    """
    num_beads = integrator.getNumCopies()
    all_bead_positions = []
    for i in range(num_beads):
        state = integrator.getState(copy=i, getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        if atom_indices is not None:
            pos = pos[atom_indices]
        all_bead_positions.append(pos)

    # Convert to numpy array for vector math
    coords = np.array(all_bead_positions)
    # Calculate Centroid (average position across beads)
    centroid = np.mean(coords, axis=0)
    # Calculate Squared Distance of each bead from the Centroid
    diff = coords - centroid
    sq_dist = np.sum(diff ** 2, axis=2)  # Sum x,y,z components -> (n_beads, n_atoms)
    # Average over beads (Mean Squared Displacement from Centroid)
    mean_sq_dist = np.mean(sq_dist, axis=0)
    quantum_rg = np.sqrt(mean_sq_dist)
    return quantum_rg * unit.nanometers


class RPMDQuantumSpreadReporter(object):
    """
    A Reporter class to log the quantum spread (delocalization) of specific atoms
    during an RPMD simulation.
    """

    def __init__(self, file, reportInterval, atom_indices, names=None):
        """
        Parameters
        ----------
        file : str
            Filename to write to.
        reportInterval : int
            The interval (in steps) at which to write frames.
        atom_indices : list of int
            The indices of the atoms to monitor (e.g., the transferring proton).
        names : list of str, optional
            Names for the columns (e.g., ["Proton_H1", "Donor_N"]).
            If None, uses indices.
        """
        self._reportInterval = reportInterval
        self._atom_indices = atom_indices
        self._out = open(file, 'w')

        # Header
        if names:
            header = "Step\t" + "\t".join([f"Rg_{n}(nm)" for n in names])
        else:
            header = "Step\t" + "\t".join([f"Rg_Atom{i}(nm)" for i in atom_indices])
        self._out.write(header + "\n")

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation, state):
        # We need access to the integrator to get bead positions, not just the simulation state
        integrator = simulation.integrator

        # Calculate spreads using the helper function defined above
        # Note: This requires the helper function to be available or methodized
        spreads = calculate_quantum_spread(integrator, self._atom_indices)

        # Write to file
        step = simulation.currentStep
        spread_values = spreads.value_in_unit(unit.nanometers)

        line = f"{step}"
        for val in spread_values:
            line += f"\t{val:.6f}"
        self._out.write(line + "\n")
        self._out.flush()

    def __del__(self):
        self._out.close()


def test_rpmd_quantum_spread_reporter():
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")  # Replace with your PDB
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

    n_beads = 64
    integrator = openmm.RPMDIntegrator(n_beads,
                                       300 * unit.kelvin,
                                       1.0 / unit.picosecond,
                                       0.5 * unit.femtosecond)

    simulation = app.Simulation(pdb.topology, system, integrator)

    for i in range(n_beads):
        integrator.setPositions(i, pdb.positions)
    nqe.init_beads(modeller, simulation, n_beads)

    atoms_to_watch = [0, 1]
    atom_names = ["Atom0", "Atom1"]

    # 4. Attach the custom reporter
    simulation.reporters.append(RPMDQuantumSpreadReporter(
        file="quantum_spread.txt",
        reportInterval=1,
        atom_indices=atoms_to_watch,
        names=atom_names
    ))

    print("Running RPMD with Quantum Spread reporting...")
    simulation.step(500)
    print("Done. Check 'quantum_spread.txt'.")
    data = np.loadtxt("quantum_spread.txt", skiprows=1, delimiter='\t')

    plt.plot(data[:, 0], data[:, 1], label='Atom0')
    plt.plot(data[:, 0], data[:, 2], label='Atom1')
    plt.xlabel('Step')
    plt.ylabel('Quantum Rg (nm)')
    plt.legend()
    plt.show()
