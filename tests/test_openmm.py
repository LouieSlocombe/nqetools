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


def test_plumed():
    temperature = 300 * unit.kelvin
    timestep = 2.0 * unit.femtosecond
    friction_coeff = 1.0 / unit.picosecond
    total_steps = 100_000
    pdb_file = 'tests/data/pdb/gt_wob_pol.pdb'
    pdb_out = 'pdb_out.pdb'

    rm_ions = ['Na+', 'Cl-', 'NA']
    residue_map = {'DGN': 'DG', 'DTN': 'DT', 'GTP': 'LIG'}

    pdb_data, molecule = nqe.prepare_lig_system(pdb_file, rm_ions=rm_ions, residue_map=residue_map, lig_name='LIG')
    forcefield = nqe.prepare_ligand_ff(("amber14-all.xml", "amber14/tip3pfb.xml"), molecule)

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

    plumed_script = """c1: COM ATOMS=5740
c2: COM ATOMS=5226
dist: DISTANCE ATOMS=c1,c2
wall: UPPER_WALLS ARG=dist AT=3.5 KAPPA=1000.0 EXP=2
opes: METAD ARG=dist PACE=500 HEIGHT=100 SIGMA=0.05
PRINT ARG=dist,opes.bias,wall.bias STRIDE=100 FILE=colvar.dat"""

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

    simulation.reporters.append(app.PDBReporter(pdb_out, 1_000))

    simulation.reporters.append(app.DCDReporter('trajectory.dcd', 1_000))
    simulation.step(total_steps)


def test_get_atoms_in_residue():
    print(flush=True)
    input_pdb = 'tests/data/pdb/input.pdb'
    indexes = nqe.get_atoms_in_residue(input_pdb, 0)
    print(indexes)
    ref_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    assert indexes == ref_indexes
