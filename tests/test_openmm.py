import os
from pathlib import Path
from sys import stdout

from openff.toolkit import Molecule
from openmm import openmm, app, unit
from openmmforcefields.generators import GAFFTemplateGenerator
from openmmml import MLPotential
from rdkit import Chem

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


def get_non_standard_residues(pdb_file):
    """
    Loads a PDB file and returns a list of RDKit Mol objects
    for each non-standard residue (e.g., ligands, water, ions).
    """

    # 1. Define all "standard" residue names for proteins and nucleic acids
    # We will filter *out* everything in this set.
    STANDARD_RESIDUES = {
        # Standard 20 protein residues
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
        'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        # Standard DNA residues (desoxy)
        'DA', 'DC', 'DG', 'DT',
        # Standard RNA residues (ribo)
        'A', 'C', 'G', 'U', 'RA', 'RC', 'RG', 'RU',
        # Common alternative protonation states for Histidine
        'HID', 'HIE', 'HIP',
        # Common synonyms
        'ADE', 'CYT', 'GUA', 'THY', 'URA',
        # Water
        'HOH', 'WAT', 'SOL',
        # Ions
        'NA', 'CL', 'K', 'MG', 'CA',
        'Na+', 'Cl-', 'K+', 'Mg2+', 'Ca2+'
    }

    mol = Chem.MolFromPDBFile(pdb_file, sanitize=False, removeHs=False)
    mols_by_residue = Chem.SplitMolByPDBResidues(mol)

    print(f"\n--- Found {len(mols_by_residue)} total residue fragments ---")

    non_standard_mols = []
    for residue_key, fragment_mol in mols_by_residue.items():
        res_name = residue_key.split('_')[0].strip()
        if res_name not in STANDARD_RESIDUES:
            print(f"  > Found non-standard residue: {residue_key}")
            print(Chem.MolToSmiles(fragment_mol))
            non_standard_mols.append(fragment_mol)
        else:
            print(f"  - Skipping standard residue: {residue_key}")

    return non_standard_mols


def test_get_non_standard_residues():
    pdb_file = "tests/data/pdb/gt_wob_pol_clean.pdb"
    non_standard_mols = get_non_standard_residues(pdb_file)
    assert len(non_standard_mols) == 2


def test_openmm_ff_param():
    # smi = "c1ccccc1"
    # smi = '[H]-[O]-[C](-[H])(-[H])-[C@@]1(-[H])-[O]-[C@@](-[H])(-[n]2:[c](-[H]):[n]:[c]3:[c](=[O]):[n](-[H]):[c](-[N](-[H])-[H]):[n]:[c]:3:2)-[C](-[H])(-[H])-[C@]-1(-[H])-[O]-[H]'
    # input_pdb = "benzene.pdb"
    # mol = Chem.MolFromSmiles(smi)
    # mol = Chem.AddHs(mol)
    #
    # # Write a pdb file for the molecule
    # Chem.MolToPDBFile(mol, input_pdb)
    #
    # mol = Chem.MolFromPDBFile(input_pdb)

    input_pdb = "tests/data/pdb/gt_wob_solv.pdb"
    non_standard_mols = get_non_standard_residues(input_pdb)
    for mol in non_standard_mols:
        Chem.SanitizeMol(mol)
        print(Chem.MolToSmiles(mol, isomericSmiles=True))

    # write sdf files for each non-standard residue
    for i, mol in enumerate(non_standard_mols):
        Chem.MolToMolFile(mol, f"non_standard_{i}.sdf")

    # molecule = [Molecule.from_rdkit(mol) for mol in non_standard_mols]

    # Create an OpenFF Molecule object for benzene from SMILES

    # molecule = Molecule.from_smiles(smi)
    # Create the GAFF template generator

    # molecules = Molecule.from_file("molecules.sdf")
    molecule = [Molecule.from_file(f"non_standard_{i}.sdf", allow_undefined_stereo=True) for i in
                range(len(non_standard_mols))]

    gaff = GAFFTemplateGenerator(molecules=molecule)
    # Create an OpenMM ForceField object with AMBER ff14SB and TIP3P with compatible ions
    forcefield = app.ForceField(
        "amber/protein.ff14SB.xml",
        "amber/tip3p_standard.xml",
        "amber/tip3p_HFE_multivalent.xml",
    )
    # Register the GAFF template generator
    forcefield.registerTemplateGenerator(gaff.generator)

    pdbfile = app.PDBFile(input_pdb)

    system = forcefield.createSystem(pdbfile.topology)
    os.remove(input_pdb)


def test_openmm_ff_param_gt_wobble():
    input_pdb = "tests/data/pdb/gt_wob_solv.pdb"
    from openmm.app import PDBFile
    from openff.toolkit import Molecule
    from openmmforcefields.generators import (
        SMIRNOFFTemplateGenerator,
    )

    smis = [
        '[H]-[O]-[C](-[H])(-[H])-[C@@]1(-[H])-[O]-[C@@](-[H])(-[n]2:[c](-[H]):[n]:[c]3:[c](=[O]):[n](-[H]):[c](-[N](-[H])-[H]):[n]:[c]:3:2)-[C](-[H])(-[H])-[C@]-1(-[H])-[O]-[H]',
        '[H]-[O]-[C](-[H])(-[H])-[C@@]1(-[H])-[O]-[C@@](-[H])(-[n]2:[c](-[H]):[c](-[C](-[H])(-[H])-[H]):[c](=[O]):[n](-[H]):[c]:2=[O])-[C](-[H])(-[H])-[C@]-1(-[H])-[O]-[H]']
    m0 = Molecule.from_smiles(smis[0])
    m1 = Molecule.from_smiles(smis[1])
    gaff = GAFFTemplateGenerator(molecules=[m0, m1], forcefield="gaff-2.2.20")
    smirnoff = SMIRNOFFTemplateGenerator(molecules=[m0, m1])
    forcefield = app.ForceField(
        "amber/protein.ff14SB.xml",
        "amber/tip3p_standard.xml",
        "amber/tip3p_HFE_multivalent.xml",
    )
    # Register the GAFF template generator
    # forcefield.registerTemplateGenerator(gaff.generator)
    # forcefield.registerTemplateGenerator(smirnoff.generator)
    pdbfile = PDBFile(input_pdb)

    system = forcefield.createSystem(pdbfile.topology)

    # generated_files = nqe.extract_nonstandard_res(input_pdb, '.', sdf=True)
    # molecules = [Molecule.from_file(f) for f in generated_files]
    #
    # gaff = GAFFTemplateGenerator(molecules=molecules, forcefield="gaff-2.2.20")
    # # Create an OpenMM ForceField object with AMBER ff14SB and TIP3P with compatible ions
    # forcefield = app.ForceField(
    #     "amber/protein.ff14SB.xml",
    #     "amber/tip3p_standard.xml",
    #     "amber/tip3p_HFE_multivalent.xml",
    # )
    # # Register the GAFF template generator
    # forcefield.registerTemplateGenerator(gaff.generator)
    # for f in generated_files:
    #     os.remove(f)
    #
    # pdb = app.PDBFile(input_pdb)
    # system = forcefield.createSystem(pdb.topology, ignoreExternalBonds=True)
    # modeller = app.Modeller(system.topology, system.positions)
    # modeller.deleteWater()
    # modeller.addHydrogens()
    #
    # # Solvate
    # modeller.addSolvent(forcefield,
    #                     padding=1.0 * unit.nanometer,
    #                     boxShape='dodecahedron')
    #
    # n_atoms = modeller.topology.getNumAtoms()
    # print(f"System has {n_atoms} atoms.")


def test_openmm_gt_wobble():
    input_pdb = "tests/data/pdb/gt_wob_pol.pdb"
    clean_pdb = "tests/data/pdb/gt_wob_pol_clean.pdb"

    # fix the pdb
    nqe.fix_pdb(input_pdb, clean_pdb)

    pdb = app.PDBFile(clean_pdb)
    forcefield = app.ForceField("amber14-all.xml",
                                "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
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
    # --- Integrator & Simulation ---
    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin,  # temperature (not used by minimizer but fine to define)
        1.0 / unit.picosecond,  # friction
        0.002 * unit.picoseconds  # timestep
    )

    platform = openmm.Platform.getPlatformByName("CUDA")
    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    state = sim.context.getState(getEnergy=True)
    print("Initial potential energy:", state.getPotentialEnergy())
    sim.minimizeEnergy(maxIterations=500)


def test_openff():
    from openff.toolkit import Molecule, Topology

    smis = [
        '[H]-[O]-[C](-[H])(-[H])-[C@@]1(-[H])-[O]-[C@@](-[H])(-[n]2:[c](-[H]):[n]:[c]3:[c](=[O]):[n](-[H]):[c](-[N](-[H])-[H]):[n]:[c]:3:2)-[C](-[H])(-[H])-[C@]-1(-[H])-[O]-[H]',
        '[H]-[O]-[C](-[H])(-[H])-[C@@]1(-[H])-[O]-[C@@](-[H])(-[n]2:[c](-[H]):[c](-[C](-[H])(-[H])-[H]):[c](=[O]):[n](-[H]):[c]:2=[O])-[C](-[H])(-[H])-[C@]-1(-[H])-[O]-[H]']

    mols = [Molecule.from_smiles(smi) for smi in smis]
    complex = Topology.from_pdb("tests/data/pdb/gt_wob_solv.pdb", unique_molecules=mols)

    complex.visualize()


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
