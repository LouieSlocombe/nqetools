import os
import sys

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
import openmm.unit as unit
from openff.toolkit import Molecule
from openmm import openmm, app
from openmmforcefields.generators import GAFFTemplateGenerator
from openmmtools.integrators import GeodesicBAOABIntegrator
from pdbfixer import PDBFixer
from rdkit import Chem
from scipy import constants

from .io import remove_water_residues_in_pdb, clean_ions_in_pdb, relabel_residues_in_pdb, remove_residues_in_pdb
from .plotting import n_plot


def fix_pdb(file_in, file_out, ph=7.0, rm_heterogens=True):
    fixer = PDBFixer(filename=file_in)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    if rm_heterogens:
        fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    app.PDBFile.writeFile(fixer.topology, fixer.positions, open(file_out, 'w'))
    return None


def zero_velocities(n_atoms):
    return [openmm.Vec3(0, 0, 0) for _ in range(n_atoms)] * (unit.nanometer / unit.picosecond)


def write_multimodel_pdb(topology, positions, fh, model_index):
    app.PDBFile.writeModel(topology, positions, fh, modelIndex=model_index)


def centroid_positions(simulation, n_atoms, n_beads):
    acc = np.zeros((n_atoms, 3), dtype=float)
    for b in range(n_beads):
        state = simulation.integrator.getState(b, getPositions=True)
        r = state.getPositions(asNumpy=True)
        acc += r.value_in_unit(unit.nanometer)
    acc /= n_beads
    return [openmm.Vec3(*acc[i]) for i in range(n_atoms)] * unit.nanometer


def get_thermal_de_broglie_wavelength(mass, temperature):
    if unit.is_quantity(mass):
        mass_amu = mass.value_in_unit(unit.dalton)
    else:
        mass_amu = mass

    if unit.is_quantity(temperature):
        temp_k = temperature.value_in_unit(unit.kelvin)
    else:
        temp_k = temperature

    mass_kg = mass_amu * constants.atomic_mass

    h = constants.h
    k_b = constants.k
    lambda_meters = h / np.sqrt(2 * np.pi * mass_kg * k_b * temp_k)
    return lambda_meters * unit.meter


def init_beads_scaled(simulation, positions, n_beads, temperature, scale_factor=0.1):
    system = simulation.system
    n_atoms = system.getNumParticles()

    masses_val = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                           for i in range(n_atoms)])
    masses_quantity = masses_val * unit.dalton

    lambdas = get_thermal_de_broglie_wavelength(masses_quantity, temperature)
    lambdas_nm = lambdas.value_in_unit(unit.nanometer)

    if not unit.is_quantity(positions):
        positions = positions * unit.nanometer
    pos0 = positions.value_in_unit(unit.nanometer)

    rng = np.random.default_rng(0)

    # 4. Initialize Beads
    print(f"Initializing {n_beads} beads scaled by thermal wavelengths...")
    print(f"Max Lambda (lightest atom): {np.max(lambdas_nm):.4f} nm")
    print(f"Min Lambda (heaviest atom): {np.min(lambdas_nm):.4f} nm")

    for b in range(n_beads):
        noise = rng.normal(size=(n_atoms, 3)) * lambdas_nm[:, np.newaxis] * scale_factor
        bead_pos = pos0 + noise
        simulation.integrator.setPositions(b, bead_pos * unit.nanometer)

    simulation.context.setVelocitiesToTemperature(temperature)


def init_beads(modeller, simulation, n_beads, perturb=0.002):
    rng = np.random.default_rng(0)
    pos0 = modeller.positions
    n_atoms = len(pos0)
    for b in range(n_beads):
        jiggle = perturb * rng.normal(size=(n_atoms, 3))
        bead_pos = [openmm.Vec3(p.x + dx, p.y + dy, p.z + dz)
                    for p, (dx, dy, dz) in zip(pos0, jiggle)]
        simulation.integrator.setPositions(b, bead_pos * unit.nanometer)
        simulation.integrator.setVelocities(b, zero_velocities(n_atoms))


def md_workflow(file_in,
                ff='amber19-all.xml',  # charmm36_2024.xml amber19-all.xml
                water_model='amber19/opc3.xml',  # charmm36_2024/tip5p.xml amber19/opc3.xml
                padding=1.0,
                temperature=300.0,
                pressure=1.0,
                friction_coeff=1.0,
                time_step=0.004,
                report_pdb=1_000,
                report_std=1_000,
                report_data=100,
                file_out='output.pdb',
                data_out='md_log.txt',
                n_nvt=10_000,
                n_npt=50_000,
                box_shape='dodecahedron',
                gbaoab=True,
                platform='CPU',
                ):
    # Prepare system
    pdb = app.PDBFile(file_in)
    forcefield = app.ForceField(ff, water_model)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens(forcefield)

    # Solvate
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)

    # Setup system and integrator
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME,
                                     nonbondedCutoff=1.0 * unit.nanometer,
                                     constraints=app.HBonds)

    if gbaoab:
        integrator = GeodesicBAOABIntegrator(temperature * unit.kelvin,
                                             friction_coeff / unit.picosecond,
                                             time_step * unit.picoseconds)
    else:
        integrator = openmm.LangevinIntegrator(temperature * unit.kelvin,
                                               friction_coeff / unit.picosecond,
                                               time_step * unit.picoseconds)

    platform = openmm.Platform.getPlatformByName(platform)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    # Local energy minimization
    print("Minimizing energy", flush=True)
    simulation.minimizeEnergy()

    # Setup reporting
    simulation.reporters.append(app.PDBReporter(file_out, report_pdb))

    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      report_std,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True))

    simulation.reporters.append(app.StateDataReporter(data_out,
                                                      report_data,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    # NVT equilibration
    print("Running NVT", flush=True)
    simulation.step(n_nvt)

    # NPT production MD
    system.addForce(openmm.MonteCarloBarostat(pressure * unit.bar,
                                              temperature * unit.kelvin))
    simulation.context.reinitialize(preserveState=True)
    print("Running NPT", flush=True)
    simulation.step(n_npt)

    return None


def md_analysis(file_in='md_log.txt'):
    # Analysis
    data = np.loadtxt(file_in, delimiter=',')

    step = data[:, 0]
    time = data[:, 1]
    potential_energy = data[:, 2]
    kinetic_energy = data[:, 3]
    total_energy = data[:, 4]
    temperature = data[:, 5]
    volume = data[:, 6]

    plt.plot(time, potential_energy, lw=2)
    n_plot('Time (ps)', 'Potential Energy (kJ/mol)')
    plt.show()

    plt.plot(time, kinetic_energy, lw=2)
    n_plot('Time (ps)', 'Kinetic Energy (kJ/mol)')
    plt.show()

    plt.plot(time, total_energy, lw=2)
    n_plot('Time (ps)', 'Total Energy (kJ/mol)')
    plt.show()

    plt.plot(time, temperature, lw=2)
    n_plot('Time (ps)', 'Temperature (K)')
    plt.show()

    plt.plot(time, volume, lw=2)
    n_plot('Time (ps)', 'Volume (nm^3)')
    plt.show()


def make_sdf(pdb_file, lig_name='LIG'):
    u = mda.Universe(pdb_file)
    elements = mda.topology.guessers.guess_types(u.atoms.names)
    u.add_TopologyAttr('elements', elements)
    lig = u.select_atoms(f"resname {lig_name}")
    mol = lig.convert_to("RDKIT")
    # write to sdf file
    Chem.MolToMolFile(mol, f"{lig_name}.sdf", kekulize=False)
    return None


def pdb_patcher(pdb_file, lig_name='LIG'):
    with open(pdb_file, 'r') as f:
        pdb_data = f.read()
    pdb_data = pdb_data.replace('x', ' ')
    pdb_data = pdb_data.replace('UNK', lig_name)
    with open(pdb_file, 'w') as f:
        f.write(pdb_data)
    return None


def combine_sdf_pdb(input_pdb, lig_name='LIG', patch=True):
    # Combine ligand and receptor into one pdb
    pdb = app.PDBFile(input_pdb)
    molecule = Molecule.from_file(f'{lig_name}.sdf')
    ligand_ff_topology = molecule.to_topology()
    ligand_omm_topology = ligand_ff_topology.to_openmm()
    ligand_positions = ligand_ff_topology.get_positions().to_openmm()
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.add(ligand_omm_topology, ligand_positions)
    with open(input_pdb, 'w') as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
    if patch:
        pdb_patcher(input_pdb, lig_name=lig_name)
    return None


def prepare_lig_system(input_pdb,
                       combined_pdb='combined_system.pdb',
                       clean_pdb='cleaned.pdb',
                       rm_ions=None,
                       residue_map=None,
                       rm_files=True,
                       lig_name='LIG'):
    remove_water_residues_in_pdb(input_pdb, clean_pdb)

    if rm_ions is not None:
        clean_ions_in_pdb(clean_pdb, rm_ions, clean_pdb)
    if residue_map is not None:
        relabel_residues_in_pdb(clean_pdb, residue_map, clean_pdb)

    # Save ligand as sdf
    make_sdf(clean_pdb, lig_name=lig_name)

    # Strip out the ligand and fix the pdb
    fix_pdb(clean_pdb, combined_pdb, rm_heterogens=False)
    # Remove the ligand
    remove_residues_in_pdb(combined_pdb, combined_pdb, names={lig_name})

    combine_sdf_pdb(combined_pdb, lig_name=lig_name, patch=True)

    pdb_data = app.PDBFile(combined_pdb)
    molecule = Molecule.from_file(f'{lig_name}.sdf')

    if rm_files:
        os.remove(clean_pdb)
        os.remove(combined_pdb)
        os.remove(f'{lig_name}.sdf')
    return pdb_data, molecule


def prepare_ligand_ff(standard_ff,
                      molecule,
                      use_cache=False,
                      cache="gaff-molecules.json",
                      n_conf=10,
                      pc_methods='mmff94',
                      gaff_ver='gaff-2.11'):  # gaff-2.2.20
    # mmff94 am1bcc am1-mulliken
    if use_cache:
        if cache is None:
            cache = "gaff-molecules.json"
        molecule.generate_conformers(n_conformers=n_conf)
        molecule.assign_partial_charges(partial_charge_method=pc_methods,
                                        use_conformers=molecule.conformers)
        gaff = GAFFTemplateGenerator(molecules=molecule,
                                     cache=cache,
                                     forcefield=gaff_ver)
    else:
        molecule.generate_conformers(n_conformers=n_conf)
        molecule.assign_partial_charges(partial_charge_method=pc_methods,  # mmff94 am1bcc
                                        use_conformers=molecule.conformers)
        gaff = GAFFTemplateGenerator(molecules=molecule, forcefield=gaff_ver)

    forcefield = app.ForceField(*standard_ff)
    forcefield.registerTemplateGenerator(gaff.generator)
    return forcefield


def deuterate(modeller, system, option='all'):
    deuterium_mass = app.element.deuterium.mass
    if option == 'all':
        for atom in modeller.topology.atoms():
            if atom.element.symbol == 'H':
                system.setParticleMass(atom.index, deuterium_mass)
    elif option == 'water':
        for residue in modeller.topology.residues():
            if residue.name in ['HOH', 'H2O', 'TIP3', 'WAT']:
                for atom in residue.atoms():
                    if atom.element.symbol == 'H':
                        system.setParticleMass(atom.index, deuterium_mass)
    else:
        raise ValueError("Option must be 'all' or 'water'")


def get_atoms_in_residue(pdb_file_path, residue_index, chain_id=None):
    """
    Loads a PDB file and returns a list of atom indices for the specified residue index.
    Optionally filters by chain ID.

    Args:
        pdb_file_path (str): Path to the .pdb file.
        residue_index (int): The 0-based index of the residue.
                             If chain_id is None: index in the entire topology.
                             If chain_id is set: index within that specific chain.
        chain_id (str, optional): The chain ID (e.g., 'A', 'B') to search within.

    Returns:
        list: A list of integers representing the indices of atoms in the residue.
              Returns None if the file/chain is not found or index is out of bounds.
    """

    # 1. Check if file exists
    if not os.path.exists(pdb_file_path):
        print(f"Error: File '{pdb_file_path}' not found.")
        return None

    try:
        # 2. Load the PDB file
        print(f"Loading {pdb_file_path}...")
        pdb = app.PDBFile(pdb_file_path)

        # 3. Get the topology
        topology = pdb.topology

        target_residue = None

        if chain_id is not None:
            # Filter by chain
            found_chain = None
            for chain in topology.chains():
                if chain.id == chain_id:
                    found_chain = chain
                    break

            if found_chain is None:
                available_chains = [c.id for c in topology.chains()]
                print(f"Error: Chain '{chain_id}' not found. Available chains: {available_chains}")
                return None

            residues = list(found_chain.residues())
            if residue_index < 0 or residue_index >= len(residues):
                print(f"Error: Residue index {residue_index} is out of bounds for Chain {chain_id}.")
                print(f"Chain {chain_id} contains {len(residues)} residues.")
                return None

            target_residue = residues[residue_index]
            print(f"Looking in Chain {chain_id}, Residue Index {residue_index}...")

        else:
            # Global index behavior
            residues = list(topology.residues())

            # 5. Validate index
            if residue_index < 0 or residue_index >= len(residues):
                print(f"Error: Residue index {residue_index} is out of bounds.")
                print(f"The file contains {len(residues)} residues (indices 0 to {len(residues) - 1}).")
                return None

            # 6. Get the target residue
            target_residue = residues[residue_index]
            print(f"Looking in global topology, Residue Index {residue_index}...")

        # 7. Extract atom indices
        # residue.atoms() returns a generator
        atom_indices = [atom.index for atom in target_residue.atoms()]

        print(
            f"Successfully retrieved residue: {target_residue.name} (Chain: {target_residue.chain.id}, Index: {target_residue.index}, PDB ID: {target_residue.id})")
        return atom_indices

    except Exception as e:
        print(f"An error occurred processing the PDB: {e}")
        return None


def save_pdb_selection(input_pdb_path, atom_indices, output_pdb_path):
    """
    Loads a PDB, keeps only the atoms specified in atom_indices, and saves to a new file.

    Args:
        input_pdb_path (str): Path to the source .pdb file.
        atom_indices (list[int]): List of 0-based atom indices to KEEP.
        output_pdb_path (str): Path where the new .pdb file will be saved.
    """
    if not os.path.exists(input_pdb_path):
        print(f"Error: Input file '{input_pdb_path}' not found.")
        return

    try:
        print(f"Loading {input_pdb_path} for selection...")
        pdb = app.PDBFile(input_pdb_path)

        # We use Modeller to edit the topology
        modeller = app.Modeller(pdb.topology, pdb.positions)

        # Create a set for faster lookup
        keep_indices = set(atom_indices)

        # Identify atoms to DELETE (Modeller deletes, so we invert the selection)
        atoms_to_delete = []
        all_atoms = list(modeller.topology.atoms())

        for atom in all_atoms:
            if atom.index not in keep_indices:
                atoms_to_delete.append(atom)

        # Perform the deletion
        num_deleted = len(atoms_to_delete)
        if num_deleted == len(all_atoms):
            print("Warning: Your selection is empty! The output PDB will be empty.")

        modeller.delete(atoms_to_delete)

        # Save the result
        print(f"Writing selection ({len(all_atoms) - num_deleted} atoms) to {output_pdb_path}...")
        with open(output_pdb_path, 'w') as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

        print("Done.")

    except Exception as e:
        print(f"Error saving selection: {e}")


def run_openmm_relaxation(modeller,
                          forcefield,
                          output_filename='minimized.pdb',
                          temperature=300.0 * unit.kelvin,
                          gamma=1.0 / unit.picosecond,
                          time_step=1.0 * unit.femtoseconds,
                          n_1=1_000,
                          n_2=1_000,
                          n_3=1_000,
                          n_4=2_000,
                          backbone_names=None,
                          ks_2=100.0,
                          ks_3=10.0,
                          ks_4=0.0,
                          platform_name='CPU',
                          ):
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    platform = openmm.Platform.getPlatformByName(platform_name)
    has_box = modeller.topology.getUnitCellDimensions() is not None

    print("\n--- Stage 1: Relaxing Hydrogens (Heavy atoms fixed) ---", flush=True)
    system_h_only = forcefield.createSystem(modeller.topology,
                                            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
                                            nonbondedCutoff=1.0 * unit.nanometer,
                                            constraints=None,
                                            rigidWater=False,
                                            removeCMMotion=True)

    # In OpenMM, mass=0 makes the particle immovable (infinite mass)
    for i, atom in enumerate(modeller.topology.atoms()):
        if atom.element.symbol != 'H':
            system_h_only.setParticleMass(i, 0.0 * unit.dalton)

    integrator_1 = openmm.LangevinMiddleIntegrator(temperature,
                                                   gamma,
                                                   time_step)
    sim_1 = app.Simulation(modeller.topology, system_h_only, integrator_1, platform)
    sim_1.context.setPositions(modeller.positions)
    sim_1.minimizeEnergy(maxIterations=n_1)

    current_positions = sim_1.context.getState(getPositions=True).getPositions()
    print("Stage 1 complete.", flush=True)

    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
                                     nonbondedCutoff=1.0 * unit.nanometer,
                                     constraints=None,
                                     rigidWater=False,
                                     removeCMMotion=True)

    # Define the harmonic restraint force
    restraint = openmm.CustomExternalForce("k * periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", 0.0)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    atom_indices = []
    for atom in modeller.topology.atoms():
        if atom.name in backbone_names:
            pos = current_positions[atom.index]
            restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
            atom_indices.append(atom.index)

    system.addForce(restraint)
    print(f"Restraints applied to {len(atom_indices)} backbone atoms.", flush=True)
    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(current_positions)

    print(f"\n--- Stage 2: Strong Backbone Restraints ({ks_2} kJ/mol/nm^2) ---", flush=True)
    k_strong = ks_2 * unit.kilojoules_per_mole / (unit.nanometer ** 2)
    simulation.context.setParameter("k", k_strong)
    simulation.minimizeEnergy(maxIterations=n_2)

    print("\n--- Stage 3: Weak Backbone Restraints (10 kJ/mol/nm^2) ---", flush=True)
    k_weak = ks_3 * unit.kilojoules_per_mole / (unit.nanometer ** 2)
    simulation.context.setParameter("k", k_weak)
    simulation.minimizeEnergy(maxIterations=n_3)

    print("\n--- Stage 4: Unrestrained Relaxation ---", flush=True)
    k_vweak = ks_4 * unit.kilojoules_per_mole / (unit.nanometer ** 2)
    simulation.context.setParameter("k", k_vweak)
    simulation.minimizeEnergy(maxIterations=n_4)

    final_state = simulation.context.getState(getPositions=True)
    with open(output_filename, 'w') as f:
        app.PDBFile.writeFile(simulation.topology, final_state.getPositions(), f)
    print(f"\nProcess complete. Saved to {output_filename}", flush=True)


def run_openmm_heating(modeller,
                       forcefield,
                       output_pdb='equilibrated.pdb',
                       k1=100.0,
                       backbone_names=None,
                       target_temp=300.0 * unit.kelvin,
                       temp_step=50.0 * unit.kelvin,
                       gamma=1.0 / unit.picosecond,
                       time_step=1.0 * unit.femtoseconds,
                       n_report=1_000,
                       steps_per_stage=5_000,
                       steps_final=10_000,
                       platform_name='CPU',
                       ):
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    platform = openmm.Platform.getPlatformByName(platform_name)
    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
                                     nonbondedCutoff=1.0 * unit.nanometer,
                                     constraints=None,
                                     rigidWater=False,
                                     removeCMMotion=True)

    print("Applying backbone restraints for heating...", flush=True)
    restraint = openmm.CustomExternalForce("k * periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", k1 * unit.kilojoules_per_mole / (unit.nanometer ** 2))
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    for atom in modeller.topology.atoms():
        if atom.name in backbone_names:
            restraint.addParticle(atom.index, modeller.positions[atom.index])
    system.addForce(restraint)

    current_temp = 0 * unit.kelvin
    integrator = openmm.LangevinMiddleIntegrator(current_temp,
                                                 gamma,
                                                 time_step)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    print(f"\n--- Starting Gentle Heating (0K -> {target_temp}) ---", flush=True)
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True))
    temp = temp_step
    while temp <= target_temp:
        print(f"\n-> Heating to {temp}...", flush=True)
        integrator.setTemperature(temp)
        if temp == temp_step:
            simulation.context.setVelocitiesToTemperature(temp)
        simulation.step(steps_per_stage)
        temp += temp_step
    print("\n--- Heating Complete ---", flush=True)
    print(f"Running final equilibration at {target_temp} for {steps_final} steps...", flush=True)
    simulation.step(steps_final)

    state = simulation.context.getState(getPositions=True)
    with open(output_pdb, 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
    print(f"Saved equilibrated structure to {output_pdb}", flush=True)


def run_openmm_npt(modeller,
                   forcefield,
                   output_pdb='npt_equilibrated.pdb',
                   pressure=1.0 * unit.bar,
                   temperature=300.0 * unit.kelvin,
                   gamma=1.0 / unit.picosecond,
                   time_step=2.0 * unit.femtoseconds,
                   barostat_freq=25,
                   backbone_names=None,
                   k=10.0,
                   n_report=500,
                   n_1=5_000,
                   n_2=25_000,
                   platform_name='CPU',
                   ):
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    platform = openmm.Platform.getPlatformByName(platform_name)
    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
                                     nonbondedCutoff=1.0 * unit.nanometer,
                                     constraints=None,
                                     rigidWater=False,
                                     removeCMMotion=True)

    print("Adding MonteCarloBarostat...", flush=True)
    system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_freq))
    restraint = openmm.CustomExternalForce("k * periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", k * unit.kilojoules_per_mole / (unit.nanometer ** 2))
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    atom_indices = []
    for atom in modeller.topology.atoms():
        if atom.name in backbone_names:
            restraint.addParticle(atom.index, modeller.positions[atom.index])
            atom_indices.append(atom.index)
    system.addForce(restraint)

    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      volume=True,
                                                      density=True))

    print("\n--- Phase 1: Restrained NPT (Relaxing Density) ---", flush=True)
    simulation.step(n_1)

    print("\n--- Phase 2: Removing Restraints (Unrestrained NPT) ---", flush=True)
    simulation.context.setParameter("k", 0.0)
    simulation.step(n_2)

    state = simulation.context.getState(getPositions=True, getVelocities=True)

    with open(output_pdb, 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)

    print(f"\nDensity equilibration complete. Saved to {output_pdb}", flush=True)


def run_rpmd_equilibration(modeller,
                           forcefield,
                           output_prefix='rpmd_ready',
                           num_beads=32,
                           temperature=300 * unit.kelvin,
                           pressure=1 * unit.bar,
                           barostat_freq=25,
                           friction=1.0 / unit.picosecond,
                           safe_timestep=0.5 * unit.femtoseconds,
                           production_timestep=1.0 * unit.femtoseconds,
                           n_report=1_000,
                           n_1=2_000,
                           n_2=10_000,
                           platform_name='CPU'):
    platform = openmm.Platform.getPlatformByName(platform_name)
    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
                                     nonbondedCutoff=1.0 * unit.nanometer,
                                     constraints=None,
                                     rigidWater=False,
                                     removeCMMotion=True)

    system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_freq))
    integrator = openmm.RPMDIntegrator(num_beads, temperature, friction, safe_timestep)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      temperature=True,
                                                      volume=True,
                                                      speed=True))

    print("\n--- Stage 1: Bead Expansion  ---", flush=True)
    simulation.step(n_1)

    print(f"\n--- Stage 2: Relaxation at full timestep ({production_timestep}) ---", flush=True)
    integrator.setStepSize(production_timestep)
    simulation.step(n_2)

    print("\n--- Saving State ---", flush=True)
    simulation.saveCheckpoint(f'{output_prefix}.chk')  # simulation.loadCheckpoint('rpmd_ready.chk')
    state = simulation.context.getState(getPositions=True)
    with open(f'{output_prefix}_centroid.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)

    print(f"Saved checkpoint to {output_prefix}.chk (Use this to start production)", flush=True)
    print(f"Saved centroid visualization to {output_prefix}_centroid.pdb", flush=True)


def run_contracted_rpmd(modeller,
                        forcefield,
                        num_beads=32,
                        temperature=300 * unit.kelvin,
                        pressure=1 * unit.bar,
                        barostat_freq=25,
                        friction=1.0 / unit.picosecond,
                        timestep=0.5 * unit.femtoseconds,
                        checkpoint_file='rpmd_ready.chk',
                        steps=100_000,
                        n_report=1_000,
                        contractions=None,
                        platform_name='CPU'):
    platform = openmm.Platform.getPlatformByName(platform_name)

    if contractions is None:
        # Note: NumCopies must be a divisor of num_beads (32).
        # Valid divisors for 32: 1, 2, 4, 8, 16, 32.
        contractions = {
            1: 8,  # Nonbonded Direct Space (calculate on every 4th bead)
            2: 1  # PME Reciprocal Space (calculate only on centroid)
        }
        # Group 0 is not in the dict, so it defaults to num_beads (32)

    has_box = modeller.topology.getUnitCellDimensions() is not None
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
                                     nonbondedCutoff=1.0 * unit.nanometer,
                                     constraints=None,
                                     rigidWater=False,
                                     removeCMMotion=True)

    system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_freq))

    # We must assign specific forces to specific integer groups (0-31).
    # Group 0: Bonded Forces (Cheap) -> 32 copies (Implicit default)
    # Group 1: Nonbonded Direct Space (Expensive) -> 8 copies
    # Group 2: PME Reciprocal Space (Very Expensive) -> 1 copy

    print("Assigning force groups for contraction...")

    for force in system.getForces():
        # Check for NonbondedForce (Handles VdW + Coulomb)
        if isinstance(force, openmm.NonbondedForce):
            # Set the Direct Space calculation to Group 1
            force.setForceGroup(1)
            # Set the Reciprocal Space (PME) calculation to Group 2
            force.setReciprocalSpaceForceGroup(2)
            print(f"  - {force.__class__.__name__}: Direct->Group 1, Reciprocal->Group 2")

        # Check for Bonded Forces (HarmonicBond, Angle, Torsion, etc.)
        elif isinstance(force, (openmm.HarmonicBondForce,
                                openmm.HarmonicAngleForce,
                                openmm.PeriodicTorsionForce,
                                openmm.RBTorsionForce,
                                openmm.CMAPTorsionForce)):
            force.setForceGroup(0)
            print(f"  - {force.__class__.__name__}: Group 0")

        # Barostat and others
        else:
            force.setForceGroup(0)

    print(f"\nInitializing RPMDIntegrator with contractions: {contractions}")
    integrator = openmm.RPMDIntegrator(num_beads, temperature, friction, timestep, contractions)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    if not os.path.exists(checkpoint_file):
        print(f"Error: Checkpoint {checkpoint_file} not found. Run equilibration first.")
        return

    print(f"Loading state from {checkpoint_file}...")
    simulation.loadCheckpoint(checkpoint_file)
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True,
                                                      remainingTime=True,
                                                      totalSteps=steps))

    simulation.reporters.append(app.DCDReporter('rpmd_production.dcd', n_report))
    simulation.reporters.append(app.CheckpointReporter('rpmd_production.chk', n_report))

    print(f"\nStarting Production Run ({steps} steps)...")
    simulation.step(steps)
    print("Done.")
