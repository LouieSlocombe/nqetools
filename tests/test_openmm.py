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


def test_openmm_ff_param():
    # Create an OpenMM ForceField object with AMBER ff14SB and TIP3P with compatible ions
    forcefield = app.ForceField(
        "amber/protein.ff14SB.xml",
        "amber/tip3p_standard.xml",
        "amber/tip3p_HFE_multivalent.xml",
        "amber/DNA.OL15.xml",
    )

    # smi = "c1ccccc1"
    smi = '[H]O[C@@]1([H])C([H])([H])[C@]([H])(N2C([H])N[C@@H]3C2N[C@H](N([H])[H])N([H])[C@H]3O)O[C@]1([H])C([H])([H])O[PH](O)(O)O[PH](O)(O)O[PH](O)(O)O'
    # input_pdb = "benzene.pdb"
    # mol = Chem.MolFromSmiles(smi)
    # mol = Chem.AddHs(mol)
    #
    # # Write a pdb file for the molecule
    # Chem.MolToPDBFile(mol, input_pdb)
    #
    # mol = Chem.MolFromPDBFile(input_pdb)

    input_pdb = "tests/data/pdb/gt_wob_pol.pdb"
    clean_pdb = "gt_wob_pol_clean.pdb"

    rm_ions = ['Na+', 'Cl-', 'NA']
    residue_map = {'DGN': 'DG', 'DTN': 'DT', 'GTP': 'LIG'}

    nqe.clean_ions_in_pdb(input_pdb, rm_ions, clean_pdb)
    nqe.relabel_residues_in_pdb(clean_pdb, residue_map, clean_pdb)
    nqe.remove_water_residues_in_pdb(clean_pdb, clean_pdb)
    nqe.fix_pdb(input_pdb, clean_pdb)

    non_standard_mols = nqe.get_non_standard_residues(clean_pdb)
    n_ns = len(non_standard_mols)
    for mol in non_standard_mols:
        Chem.SanitizeMol(mol)

    # write sdf files for each non-standard residue
    for i, mol in enumerate(non_standard_mols):
        Chem.MolToMolFile(mol, f"non_standard_{i}.sdf")
    #
    # # molecule = [Molecule.from_rdkit(mol, allow_undefined_stereo=True) for mol in non_standard_mols]
    # # molecule = Molecule.from_smiles(smi, allow_undefined_stereo=True)
    # molecule = [Molecule.from_file(f"non_standard_{i}.sdf", allow_undefined_stereo=True) for i in range(n_ns)]
    #
    # # Register the GAFF template generator
    # gaff = GAFFTemplateGenerator(molecules=molecule)
    # forcefield.registerTemplateGenerator(gaff.generator)

    pdbfile = app.PDBFile(clean_pdb)
    system = forcefield.createSystem(pdbfile.topology)
    os.remove(clean_pdb)
    # for i in range(n_ns):
    #     os.remove(f"non_standard_{i}.sdf")


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
    import openmm.app as app
    import openmm as mm
    import openmm.unit as unit
    from openff.toolkit.topology import Molecule
    from openmmforcefields.generators import SystemGenerator

    print("--- 1. Loading Input Files ---")
    # Load the PDB file containing the entire complex (protein, DNA, ligand)
    # This provides the topology and the initial positions.

    pdb = app.PDBFile('gt_wob_pol_clean_fix.pdb')

    # Load the ligand's SDF file.
    # This provides the high-quality chemical definition (e.g., bond orders)
    # for the OpenFF toolkit to parameterize.

    ligand_molecule = Molecule.from_file('non_standard_0_fix.sdf', allow_undefined_stereo=True)

    smi = '[H][O][C@@]1([H])[C]([H])([H])[C@]([H])([N]2[C]([H])([H])[N]([H])[C@]3([H])[C]2([H])[N]([H])[C@]([H])([N]([H])[H])[N]([H])[C@@]3([H])[O][H])[O][C@]1([H])[C]([H])([H])[O][P]([H])([O][H])([O][H])[O][P]([H])([O][H])([O][H])[O][P]([H])([O][H])([O][H])[O][H]'
    smi = 'c1nc2c(n1[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@@](=O)(O)O[P@](=O)(O)OP(=O)(O)O)O)O)[nH]c(nc2=O)N'
    ligand_molecule = Molecule.from_smiles(smi)

    print(f"Loaded PDB with {pdb.topology.getNumAtoms()} atoms.")
    print(f"Loaded ligand molecule: {ligand_molecule.to_smiles()}")

    print("\n--- 2. Defining Force Fields ---")
    # Define the standard force fields for protein and DNA.
    # We use AMBER's ff14SB (protein), DNA.OL15 (DNA), and TIP3P (water).
    standard_forcefields = [
        'amber/ff14SB.xml',
    ]

    # Define the OpenFF force field for the small molecule.
    # "openff-2.1.0.offxml" (Sage) is a robust, modern choice.
    openff_forcefield = 'openff-2.1.0.offxml'

    print("\n--- 3. Generating the Parameterized System ---")
    # Initialize the SystemGenerator.
    # This object will combine the force fields for us.
    system_generator = SystemGenerator(
        forcefields=standard_forcefields,
        small_molecule_forcefield=openff_forcefield
    )

    # This is the key step. We provide the *topology* from the PDB
    # and a list of *Molecule* objects we want to parameterize with OpenFF.
    #
    # The generator will:
    # 1. Parameterize Protein/DNA/Water using the AMBER/TIP3P XML files.
    # 2. Find the residue(s) in the PDB topology that are *not*
    #    protein/DNA/water (i.e., our ligand).
    # 3. Match these "unknown" residues against the `ligand_molecule`
    #    based on chemical structure.
    # 4. If a match is found, it applies the `openff_forcefield`
    #    parameters to that residue.
    system = system_generator.create_system(
        pdb.topology,
        molecules=[ligand_molecule]
    )

    print("Successfully created a combined OpenMM System.")
    print(f"System has {system.getNumParticles()} particles.")
    print(f"Using {system.getNumForces()} force groups:")
    for i in range(system.getNumForces()):
        print(f"  - Force {i}: {system.getForce(i).__class__.__name__}")

    print("\n--- 4. Setting up the OpenMM Simulation ---")
    # Set up a standard OpenMM simulation.
    # We'll use a Langevin integrator for NVT dynamics.
    integrator = mm.LangevinIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picoseconds,
        2.0 * unit.femtoseconds
    )

    # Create the Simulation object
    simulation = app.Simulation(pdb.topology, system, integrator)

    # Set the initial atomic positions from the PDB file
    simulation.context.setPositions(pdb.positions)

    print("Simulation object created successfully.")

    print("\n--- 5. Running the Simulation ---")
    # Add reporters to save data
    # Save a DCD trajectory file every 1000 steps
    simulation.reporters.append(
        app.DCDReporter('trajectory.dcd', 1000)
    )
    # Print simulation state data (like energy and temp) to the console
    simulation.reporters.append(
        app.StateDataReporter(
            'stdout',
            1000,
            step=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=5000
        )
    )

    # First, minimize the energy to relax any clashes
    print("Minimizing energy...")
    simulation.minimizeEnergy()
    print("Minimization complete.")

    # Run 5,000 steps of NVT dynamics (10 ps)
    print("Running NVT dynamics...")
    simulation.step(5000)

    print("\n--- Simulation Finished ---")
    print("Check your directory for 'trajectory.dcd'.")


def test_same():
    from openff.toolkit.topology import Molecule, Topology
    import openmm.app as app

    # --- 1. Load your ligand from the SDF file ---
    print("--- Loading from ligand.sdf ---")
    # Use the same stereo flag from our previous fix
    sdf_mol = Molecule.from_file('non_standard_0_fix.sdf', allow_undefined_stereo=True)
    # Generate a canonical, isomeric SMILES string
    sdf_smiles = sdf_mol.to_smiles(isomeric=True, explicit_hydrogens=True)

    print(f"  SDF Molecule SMILES: {sdf_smiles}")
    print(f"  SDF Molecule Atoms:  {sdf_mol.n_atoms}")

    print("\n" + "-" * 30 + "\n")

    # --- 2. Load your ligand from the PDB file ---
    print("--- Loading from gt_wob_pol_clean_fix.pdb ---")

    # Load the PDB
    pdb = app.PDBFile('gt_wob_pol_clean_fix.pdb')

    # Convert to an OpenFF Topology
    # We pass the PDB's OpenMM topology to the OpenFF toolkit
    off_topology = Topology.from_openmm(
        pdb.topology,
        unique_molecules=[]  # We don't provide any molecules here
    )

    # Find the GTP residue (Residue 344)
    # We look for residue number 344 and name "GTP"
    pdb_mol = None
    for residue in off_topology.residues:
        if residue.residue_number == 344 and residue.name == 'GTP':
            # Found it! Get the corresponding OpenFF Molecule object
            pdb_mol = off_topology.molecule_for_residue(residue)
            break

    if pdb_mol:
        # Generate a canonical, isomeric SMILES string
        pdb_smiles = pdb_mol.to_smiles(isomeric=True, explicit_hydrogens=True)

        print(f"  PDB Molecule SMILES: {pdb_smiles}")
        print(f"  PDB Molecule Atoms:  {pdb_mol.n_atoms}")
    else:
        print("  ERROR: Could not find residue 344 (GTP) in the PDB file.")


def test_split():
    from openff.toolkit.topology import Molecule
    from openmm.app import ForceField
    input_pdb = "tests/data/pdb/gt_wob_pol.pdb"
    clean_pdb = "gt_wob_pol_clean.pdb"

    rm_ions = ['Na+', 'Cl-', 'NA']
    residue_map = {'DGN': 'DG', 'DTN': 'DT', 'GTP': 'LIG'}

    smi = '[H]-[O]-[C@@]1(-[H])-[C](-[H])(-[H])-[C@](-[H])(-[N]2-[CH](-[H])-[NH]-[C@@H]3-[CH]-2-[NH]-[C@H](-[N](-[H])-[H])-[N](-[H])-[C@H]-3-[OH])-[O]-[C@]-1(-[H])-[C](-[H])(-[H])-[O]-[P-](-[O-])(-[O-])-[O]-[P-](-[O-])(-[O-])-[O]-[P-](-[O-])(-[O-])-[O-]'


    nqe.clean_ions_in_pdb(input_pdb, rm_ions, clean_pdb)
    nqe.relabel_residues_in_pdb(clean_pdb, residue_map, clean_pdb)
    nqe.remove_water_residues_in_pdb(clean_pdb, clean_pdb)

    non_standard_mols = nqe.get_non_standard_residues(clean_pdb)
    # write sdf files for each non-standard residue
    for i, mol in enumerate(non_standard_mols):
        # print the number of atoms and bonds
        #Chem.SanitizeMol(mol)
        print(f"Non-standard residue {i}: {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds")


        print(Chem.MolToSmiles(mol, isomericSmiles=True, allBondsExplicit=True, allHsExplicit=True,canonical=True))
        print(Chem.MolToInchi(mol))
        Chem.MolToMolFile(mol, f"non_standard_{i}.sdf", kekulize=False)

    # nqe.fix_pdb(clean_pdb, clean_pdb)
    lig_mol = Molecule.from_file('non_standard_0.sdf', allow_undefined_stereo=True, file_format="SDF")
    print(lig_mol.n_bonds, lig_mol.n_atoms, lig_mol.properties)

    lig_mol = Molecule.from_rdkit(non_standard_mols[0], allow_undefined_stereo=True)
    print(lig_mol.n_bonds, lig_mol.n_atoms, lig_mol.properties)


    lig_mol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
    print(lig_mol.n_bonds, lig_mol.n_atoms, lig_mol.properties)

    lig_mol = Molecule.from_inchi('InChI=1S/C10H30N5O13P3/c11-10-13-8-7(9(17)14-10)12-3-15(8)6-1-4(16)5(26-6)2-25-30(21,22)28-31(23,24)27-29(18,19)20/h4-10,12-14,16-24,29-31H,1-3,11H2/t4-,5+,6+,7+,8?,9?,10-/m0/s1', allow_undefined_stereo=True)
    print(lig_mol.n_bonds, lig_mol.n_atoms, lig_mol.properties)


    gaff = GAFFTemplateGenerator(molecules=lig_mol)
    forcefield = ForceField(
        'amber/ff14SB.xml',
    )
    forcefield.registerTemplateGenerator(gaff.generator)
    from openmm.app import PDBFile

    pdbfile = PDBFile("gt_wob_pol_clean.pdb")
    system = forcefield.createSystem(pdbfile.topology)


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
