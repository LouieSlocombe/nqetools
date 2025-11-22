import os
import sys
from pathlib import Path
from sys import stdout

import openmm.app as app
import openmm.unit as unit
from openmm import openmm
from openmmml import MLPotential
from openmmplumed import PlumedForce

import nqetools as nqe


def test_openmm_ml():
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
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
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"), molecule)

    # Solvate
    modeller.addSolvent(forcefield,
                        padding=1.0 * unit.nanometer,
                        boxShape='dodecahedron')

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds
    )

    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds
    )

    platform = openmm.Platform.getPlatformByName("CUDA")
    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    state = sim.context.getState(getEnergy=True)
    print("Initial potential energy:", state.getPotentialEnergy())
    sim.minimizeEnergy(maxIterations=500)
    # write minimized structure
    min_positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    with open("minimized.pdb", "w") as f:
        app.PDBFile.writeFile(modeller.topology, min_positions, f)
    os.remove('minimized.pdb')


def test_openmm_constraints():
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")  # must have coordinates
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")  # or your choice

    # Optional: add hydrogens, etc.
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, model='tip3p', padding=1.0 * unit.nanometer)  # if you want solvent

    # Solvate
    modeller.addSolvent(forcefield,
                        padding=1.0 * unit.nanometer,
                        boxShape='dodecahedron')

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,  # or app.NoCutoff for vacuum
        nonbondedCutoff=1.0 * unit.nanometer,  # ignored if NoCutoff
        constraints=app.HBonds  # bond-length constraints to H (not positional!)
    )

    # --- Build a harmonic positional restraint on the backbone ---
    # Energy: 0.5 * k * periodicdistance((x,y,z), (x0,y0,z0))^2
    # Use "periodicdistance" to be PBC-safe; for vacuum you can also use (x-x0)^2 + ...
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

    # --- Integrator & Simulation ---
    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin,  # temperature (not used by minimizer but fine to define)
        1.0 / unit.picosecond,  # friction
        0.002 * unit.picoseconds  # timestep
    )

    platform = openmm.Platform.getPlatformByName("CUDA")  # or "CUDA"/"OpenCL"
    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    # (Optional) quick energy/position sanity check
    state = sim.context.getState(getEnergy=True)
    print("Initial potential energy:", state.getPotentialEnergy())

    # --- Minimize while restraints are active ---
    sim.minimizeEnergy(maxIterations=500)  # increase if needed

    # # Get minimized coordinates
    # min_positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    # with open("minimized.pdb", "w") as f:
    #     app.PDBFile.writeFile(mod.topology, min_positions, f)

    # --- If you only wanted restraints during minimization, drop or soften them now ---
    # To keep but soften:
    # sim.context.setParameter("k", 100.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    # To remove entirely:
    # system.removeForce(system.getNumForces() - 1)  # if 'restraint' was added last


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


def test_deuterate():
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
    nqe.deuterate(modeller, system)

    integrator = openmm.LangevinIntegrator(300 * unit.kelvin,
                                           1.0 / unit.picosecond,
                                           0.002 * unit.picoseconds)
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()


def run_run_plumed_md(directory, pdb_file):
    pdb_file = 'tests/data/pdb/gt_wob_pol.pdb'


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


def run_relaxation(pdb_filename, output_filename='minimized.pdb'):
    print(f"Loading {pdb_filename}...")
    pdb = PDBFile(pdb_filename)

    # 1. Define ForceField (Standard Amber14 + TIP3P water)
    # You can change these XMLs to match your specific system (e.g., charmm36)
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')

    modeller = Modeller(pdb.topology, pdb.positions)

    # =========================================================================
    # STAGE 1: Relax Hydrogens Only (Heavy Atoms Fixed)
    # =========================================================================
    print("\n--- Stage 1: Relaxing Hydrogens (Heavy atoms fixed) ---")

    # Create a temporary system for Stage 1
    system_h_only = forcefield.createSystem(modeller.topology,
                                            nonbondedMethod=PME,
                                            nonbondedCutoff=1 * nanometer,
                                            constraints=HBonds)

    # Iterate through particles and set mass to 0 for non-Hydrogens
    # In OpenMM, mass=0 makes the particle immovable (infinite mass)
    for i, atom in enumerate(modeller.topology.atoms()):
        if atom.element.symbol != 'H':
            system_h_only.setParticleMass(i, 0 * dalton)

    # Create integrator (Integrator is required even for minimization)
    integrator_1 = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.004 * picoseconds)
    sim_1 = Simulation(modeller.topology, system_h_only, integrator_1)
    sim_1.context.setPositions(modeller.positions)

    # Minimize
    sim_1.minimizeEnergy(maxIterations=1000)

    # Get the new positions to carry over to the next stage
    current_positions = sim_1.context.getState(getPositions=True).getPositions()
    print("Stage 1 complete.")

    # =========================================================================
    # PREPARATION FOR STAGES 2, 3, & 4
    # =========================================================================
    # Create the REAL system with correct masses
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=PME,
                                     nonbondedCutoff=1 * nanometer,
                                     constraints=HBonds)

    # Define the harmonic restraint force
    # k * (r - r0)^2 where r0 is the starting position
    restraint = CustomExternalForce("k * periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", 0.0)  # Initialize with 0, we will change this
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    # Identify backbone atoms to restrain (Protein Backbone usually CA, C, N)
    # Note: Extend this list if you have nucleic acids (e.g., "P", "O3'")
    backbone_names = ['CA', 'C', 'N']

    atom_indices = []
    for atom in modeller.topology.atoms():
        if atom.name in backbone_names:
            # We anchor the restraint to the position resulting from Stage 1
            pos = current_positions[atom.index]
            restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
            atom_indices.append(atom.index)

    system.addForce(restraint)
    print(f"Restraints applied to {len(atom_indices)} backbone atoms.")

    # Create new Simulation context
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.004 * picoseconds)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(current_positions)

    # =========================================================================
    # STAGE 2: Relax All with STRONG Backbone Restraints
    # =========================================================================
    print("\n--- Stage 2: Strong Backbone Restraints (100 kJ/mol/nm^2) ---")
    k_strong = 100.0 * kilojoules_per_mole / (nanometer ** 2)
    simulation.context.setParameter("k", k_strong)
    simulation.minimizeEnergy(maxIterations=1000)

    # =========================================================================
    # STAGE 3: Relax All with WEAK Backbone Restraints
    # =========================================================================
    print("\n--- Stage 3: Weak Backbone Restraints (10 kJ/mol/nm^2) ---")
    k_weak = 10.0 * kilojoules_per_mole / (nanometer ** 2)
    simulation.context.setParameter("k", k_weak)
    simulation.minimizeEnergy(maxIterations=1000)

    # =========================================================================
    # STAGE 4: Relax All (Unrestrained)
    # =========================================================================
    print("\n--- Stage 4: Unrestrained Relaxation ---")
    simulation.context.setParameter("k", 0.0)
    simulation.minimizeEnergy(maxIterations=2000)  # Allow more iterations for final cleanup

    # =========================================================================
    # Save Result
    # =========================================================================
    final_state = simulation.context.getState(getPositions=True)
    with open(output_filename, 'w') as f:
        PDBFile.writeFile(simulation.topology, final_state.getPositions(), f)
    print(f"\nProcess complete. Saved to {output_filename}")


import sys
from openmm.app import *
from openmm import *
from openmm.unit import *


def run_heating(input_pdb='minimized.pdb', output_pdb='equilibrated.pdb'):
    print(f"Loading {input_pdb}...")
    pdb = PDBFile(input_pdb)

    # 1. Define ForceField (Must match the minimization step)
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')

    modeller = Modeller(pdb.topology, pdb.positions)

    # 2. Create System
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=PME,
                                     nonbondedCutoff=1 * nanometer,
                                     constraints=HBonds)

    # 3. Add Harmonic Restraints to Backbone
    # It is standard practice to restrain the backbone while heating to prevent
    # thermal shock from unfolding the protein before the solvent equilibrates.
    print("Applying backbone restraints for heating...")
    restraint = CustomExternalForce("k * periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", 100.0 * kilojoules_per_mole / (nanometer ** 2))
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    backbone_names = ['CA', 'C', 'N']
    for atom in modeller.topology.atoms():
        if atom.name in backbone_names:
            restraint.addParticle(atom.index, modeller.positions[atom.index])
    system.addForce(restraint)

    # 4. Integrator
    # Start at 0 Kelvin. We will ramp this up.
    # Time step: 2.0 femtoseconds (standard with HBonds constraints)
    current_temp = 0 * kelvin
    integrator = LangevinMiddleIntegrator(current_temp, 1 / picosecond, 0.002 * picoseconds)

    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # 5. Heating Protocol
    # We will heat from 50K to 300K in steps of 50K
    # At each step, we run for 5000 steps (10 ps)
    target_temp = 300 * kelvin
    temp_step = 50 * kelvin
    steps_per_stage = 5000  # 10 picoseconds per stage

    print(f"\n--- Starting Gentle Heating (0K -> {target_temp}) ---")

    # Add a reporter to track progress in the terminal
    simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True,
                                                  potentialEnergy=True, temperature=True))

    temp = temp_step
    while temp <= target_temp:
        print(f"\n-> Heating to {temp}...")

        # A. Set the integrator's target temperature
        integrator.setTemperature(temp)

        # B. If this is the very first step (coming from 0K), we must
        # initialize velocities. Otherwise, the thermostat handles it.
        if temp == temp_step:
            simulation.context.setVelocitiesToTemperature(temp)

        # C. Run dynamics
        simulation.step(steps_per_stage)

        temp += temp_step

    print("\n--- Heating Complete ---")

    # 6. Optional: Short equilibration at final temp
    print(f"Running final equilibration at {target_temp} for 10000 steps...")
    simulation.step(10000)

    # 7. Save Output
    state = simulation.context.getState(getPositions=True)
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(simulation.topology, state.getPositions(), f)
    print(f"Saved equilibrated structure to {output_pdb}")
