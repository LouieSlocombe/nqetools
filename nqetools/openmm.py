"""OpenMM simulation setup, equilibration and ring-polymer dynamics.

Covers the path from a raw PDB file to a production trajectory: repairing
structures, parameterising non-standard ligands through GAFF, solvating,
and then the staged relaxation, heating and pressure equilibration that
a condensed-phase system needs before production.

Nuclear quantum effects enter two ways. Deuteration via
:func:`deuterate_system` gives the cheap mass-only estimate, while the
``run_openmm_rpmd_*`` functions run full ring-polymer dynamics, where each
nucleus is represented by a closed chain of beads. The reporter classes at
the end of the module exist because OpenMM's own reporters only see the
centroid, not the individual beads.

Notes
-----
Every ``run_openmm_*`` function accepts a `potential` and `ml_idx` pair;
supplying both builds a mixed ML/MM system and forces the CUDA platform.
"""

import os
import sys

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
import openmm.unit as unit
from openff.toolkit import Molecule
from openmm import openmm, app
from openmmforcefields.generators import GAFFTemplateGenerator
from openmmplumed import PlumedForce
from openmmtools.integrators import GeodesicBAOABIntegrator
from pdbfixer import PDBFixer
from rdkit import Chem
from reactiontools import n_plot
from scipy import constants

from .io import remove_water_residues_in_pdb, clean_ions_in_pdb, relabel_residues_in_pdb, remove_residues_in_pdb


def fix_pdb(file_in, file_out, ph=7.0, rm_heterogens=True):
    """Fixes a PDB file using PDBFixer.

    This function processes a PDB file to correct common issues like
    missing residues, non-standard residues, missing atoms, and missing
    hydrogens.

    Parameters
    ----------
    file_in : str
        Path to the input PDB file.
    file_out : str
        Path to write the fixed PDB file.
    ph : float, optional
        The pH to use when adding missing hydrogens. Default is 7.0.
    rm_heterogens : bool, optional
        If True, remove heterogen atoms like water, ions, and ligands.
        Default is True.

    Returns
    -------
    None
    """
    fixer = PDBFixer(filename=file_in)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    if rm_heterogens:
        fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    with open(file_out, 'w') as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
    return None


def zero_velocities(n_atoms):
    """Generates a list of zero velocity vectors for a given number of atoms.

    Parameters
    ----------
    n_atoms : int
        The number of atoms for which zero velocity vectors are to be created.

    Returns
    -------
    list of openmm.Vec3
        A list of zero velocity vectors, each scaled by the unit of nanometer/picosecond.
    """
    return [openmm.Vec3(0, 0, 0) for _ in range(n_atoms)] * (unit.nanometer / unit.picosecond)


def write_multimodel_pdb(topology, positions, fh, model_index):
    """Writes a single model to a multi-model PDB file.

    This function appends a model to an existing PDB file, allowing the creation
    of a multi-model PDB file. Each model is identified by a unique index.

    Parameters
    ----------
    topology : openmm.app.Topology
        The topology of the system to be written.
    positions : openmm.unit.Quantity
        The atomic positions to be written, with units of length.
    fh : file-like object
        An open file handle where the PDB model will be written.
    model_index : int
        The index of the model to be written, used to distinguish models in the PDB file.

    Returns
    -------
    None
    """
    app.PDBFile.writeModel(topology, positions, fh, modelIndex=model_index)


def centroid_positions(simulation, n_atoms, n_beads):
    """Computes the centroid positions of atoms across multiple beads in a simulation.

    This function calculates the average positions of atoms over a specified number
    of beads in a ring-polymer molecular dynamics (RPMD) simulation.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        The OpenMM simulation object containing the integrator and system state.
    n_atoms : int
        The number of atoms in the system.
    n_beads : int
        The number of beads in the RPMD simulation.

    Returns
    -------
    list of openmm.Vec3
        A list of centroid positions for each atom, with units of nanometers.
    """
    acc = np.zeros((n_atoms, 3), dtype=float)
    for b in range(n_beads):
        state = simulation.integrator.getState(b, getPositions=True)
        r = state.getPositions(asNumpy=True)
        acc += r.value_in_unit(unit.nanometer)
    acc /= n_beads
    return [openmm.Vec3(*acc[i]) for i in range(n_atoms)] * unit.nanometer


def get_thermal_de_broglie_wavelength(mass, temperature):
    """Calculates the thermal de Broglie wavelength for a given mass and temperature.

    The thermal de Broglie wavelength is a quantum mechanical property that
    characterizes the wave-like behavior of particles at a given temperature.

    Parameters
    ----------
    mass : openmm.unit.Quantity or float
        The mass of the particle. If a `Quantity`, it should have units of daltons.
        If a float, it is assumed to be in atomic mass units (amu).
    temperature : openmm.unit.Quantity or float
        The temperature of the system. If a `Quantity`, it should have units of kelvin.
        If a float, it is assumed to be in kelvin.

    Returns
    -------
    openmm.unit.Quantity
        The thermal de Broglie wavelength with units of meters.
    """
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
    """Initializes bead positions for a ring-polymer molecular dynamics (RPMD) simulation.

    This function perturbs the initial positions of atoms in the system to create
    multiple beads, scaled by the thermal de Broglie wavelength of each atom.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        The OpenMM simulation object containing the system and integrator.
    positions : openmm.unit.Quantity or np.ndarray
        The initial atomic positions. If not a Quantity, it is assumed to be in nanometers.
    n_beads : int
        The number of beads to initialize for the RPMD simulation.
    temperature : openmm.unit.Quantity
        The temperature of the system, used to calculate the thermal de Broglie wavelength.
    scale_factor : float, optional
        A scaling factor applied to the thermal wavelength perturbation. Default is 0.1.

    Returns
    -------
    None
    """
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

    rng = np.random.default_rng(0)  # fixed seed for reproducibility

    print(f"Initializing {n_beads} beads scaled by thermal wavelengths...")
    print(f"Max Lambda (lightest atom): {np.max(lambdas_nm):.4f} nm")
    print(f"Min Lambda (heaviest atom): {np.min(lambdas_nm):.4f} nm")

    for b in range(n_beads):
        noise = rng.normal(size=(n_atoms, 3)) * lambdas_nm[:, np.newaxis] * scale_factor
        bead_pos = pos0 + noise
        simulation.integrator.setPositions(b, bead_pos * unit.nanometer)

    simulation.context.setVelocitiesToTemperature(temperature)


def init_beads(modeller, simulation, n_beads, perturb=0.002):
    """Initializes bead positions and velocities for a ring-polymer molecular dynamics (RPMD) simulation.

    This function perturbs the initial positions of atoms to create multiple beads
    and sets their velocities to zero.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM modeller object containing the system topology and positions.
    simulation : openmm.app.Simulation
        The OpenMM simulation object containing the integrator and system state.
    n_beads : int
        The number of beads to initialize for the RPMD simulation.
    perturb : float, optional
        The magnitude of the random perturbation applied to the initial positions.
        Default is 0.002.

    Returns
    -------
    None
    """
    rng = np.random.default_rng(0)
    pos0 = modeller.positions
    n_atoms = len(pos0)
    for b in range(n_beads):
        jiggle = perturb * rng.normal(size=(n_atoms, 3))
        bead_pos = [openmm.Vec3(p.x + dx, p.y + dy, p.z + dz)
                    for p, (dx, dy, dz) in zip(pos0, jiggle, strict=True)]
        simulation.integrator.setPositions(b, bead_pos * unit.nanometer)
        simulation.integrator.setVelocities(b, zero_velocities(n_atoms))


def md_workflow(file_in,
                ff='amber19-all.xml',
                water_model='amber19/opc3.xml',
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
    """Solvate, equilibrate and run MD on a PDB structure in one pass.

    Builds the system from a PDB file, adds hydrogens and solvent,
    minimises, then runs NVT equilibration followed by NPT production.

    Parameters
    ----------
    file_in : str
        Path to the input PDB file.
    ff : str, optional
        Force field XML for the solute. Default is "amber19-all.xml".
    water_model : str, optional
        Force field XML for the water model, which must match the solute
        force field family. Default is "amber19/opc3.xml".
    padding : float, optional
        Minimum solvent padding around the solute in nm. Default is 1.0.
    temperature : float, optional
        Target temperature in K. Default is 300.0.
    pressure : float, optional
        Target pressure in bar for the NPT stage. Default is 1.0.
    friction_coeff : float, optional
        Langevin friction coefficient in ps^-1. Default is 1.0.
    time_step : float, optional
        Integration timestep in ps. Default is 0.004.
    report_pdb : int, optional
        Steps between PDB frames. Default is 1000.
    report_std : int, optional
        Steps between progress lines on stdout. Default is 1000.
    report_data : int, optional
        Steps between rows in the data log. Default is 100.
    file_out : str, optional
        Trajectory output path. Default is "output.pdb".
    data_out : str, optional
        Scalar data log path, readable by :func:`md_analysis`. Default is
        "md_log.txt".
    n_nvt : int, optional
        Number of NVT equilibration steps. Default is 10000.
    n_npt : int, optional
        Number of NPT production steps. Default is 50000.
    box_shape : str, optional
        Solvent box shape. Default is "dodecahedron".
    gbaoab : bool, optional
        If True, use the geodesic BAOAB integrator, which tolerates the
        larger 4 fs timestep better than plain Langevin. Default is True.
    platform : str, optional
        OpenMM platform to run on. Default is "CPU".

    Returns
    -------
    None

    Notes
    -----
    The 4 fs default timestep relies on the HBonds constraints applied
        here; loosening those requires a shorter step.
    """
    pdb = app.PDBFile(file_in)
    forcefield = app.ForceField(ff, water_model)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens(forcefield)

    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)

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

    print("Minimizing energy", flush=True)
    simulation.minimizeEnergy()

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

    print("Running NVT", flush=True)
    simulation.step(n_nvt)

    system.addForce(openmm.MonteCarloBarostat(pressure * unit.bar,
                                              temperature * unit.kelvin))
    simulation.context.reinitialize(preserveState=True)
    print("Running NPT", flush=True)
    simulation.step(n_npt)

    return None


def md_analysis(file_in='md_log.txt'):
    """Plot the energy, temperature and volume traces from an MD log.

    Reads the comma-separated log written by :func:`md_workflow` and
    shows one figure per quantity, as a quick equilibration check.

    Parameters
    ----------
    file_in : str, optional
        Path to the state data log. Default is "md_log.txt".

    Returns
    -------
    None

    Notes
    -----
    Column order is assumed to match the reporter configured in
        :func:`md_workflow`: step, time, potential, kinetic, total,
        temperature, volume.
    """
    data = np.loadtxt(file_in, delimiter=',')

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
    """Converts a ligand from a PDB file to an SDF file.

    This function reads a PDB file, extracts the ligand specified by its residue name,
    and writes it to an SDF file. The ligand's atomic elements are guessed and added
    to the topology before conversion.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file.
    lig_name : str, optional
        Residue name of the ligand to extract. Default is 'LIG'.

    Returns
    -------
    None
    """
    u = mda.Universe(pdb_file)
    elements = mda.topology.guessers.guess_types(u.atoms.names)
    u.add_TopologyAttr('elements', elements)
    lig = u.select_atoms(f"resname {lig_name}")
    mol = lig.convert_to("RDKIT")
    Chem.MolToMolFile(mol, f"{lig_name}.sdf", kekulize=False)
    return None


def pdb_patcher(pdb_file, lig_name='LIG'):
    """Modifies a PDB file to replace placeholder residue names and characters.

    This function reads a PDB file, replaces occurrences of the character 'x' with a space,
    and changes the residue name 'UNK' to the specified ligand name. The modified PDB
    content is then written back to the same file.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file to be modified.
    lig_name : str, optional
        The new residue name to replace 'UNK'. Default is 'LIG'.

    Returns
    -------
    None
    """
    with open(pdb_file) as f:
        pdb_data = f.read()
    pdb_data = pdb_data.replace('x', ' ')
    pdb_data = pdb_data.replace('UNK', lig_name)
    with open(pdb_file, 'w') as f:
        f.write(pdb_data)
    return None


def combine_sdf_pdb(input_pdb, lig_name='LIG', patch=True):
    """Combines a ligand from an SDF file with a receptor from a PDB file into a single PDB file.

    This function reads a receptor structure from a PDB file and a ligand structure from an SDF file,
    then combines them into a single PDB file. Optionally, it can patch the resulting PDB file to
    replace placeholder residue names and characters.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file containing the receptor structure.
    lig_name : str, optional
        Residue name of the ligand to be added. Default is 'LIG'.
    patch : bool, optional
        If True, applies the `pdb_patcher` function to the combined PDB file. Default is True.

    Returns
    -------
    None
    """
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
                       save_lig_sdf=False,
                       lig_name='LIG'):
    """Prepares a ligand-receptor system for molecular simulations.

    This function processes a PDB file to clean up water residues, optionally remove ions,
    relabel residues, and extract the ligand. The ligand is saved as an SDF file, and the
    cleaned receptor and ligand are combined into a single PDB file. Temporary files can
    optionally be removed after processing.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file containing the ligand-receptor system.
    combined_pdb : str, optional
        Path to save the combined ligand-receptor PDB file. Default is 'combined_system.pdb'.
    clean_pdb : str, optional
        Path to save the cleaned PDB file. Default is 'cleaned.pdb'.
    rm_ions : set of str, optional
        A set of ion residue names to remove from the PDB file. Default is None.
    residue_map : dict, optional
        A mapping of residue names to relabel in the PDB file. Default is None.
    rm_files : bool, optional
        If True, removes intermediate files generated during processing. Default is True.
    save_lig_sdf : bool, optional
        If True, saves the ligand as an SDF file. Default is False.
    lig_name : str, optional
        Residue name of the ligand to extract. Default is 'LIG'.

    Returns
    -------
    tuple
        A tuple containing:
        - pdb_data (openmm.app.PDBFile): The combined ligand-receptor PDB data.
        - molecule (openff.toolkit.Molecule): The ligand molecule object.
    """
    remove_water_residues_in_pdb(input_pdb, clean_pdb)

    if rm_ions is not None:
        clean_ions_in_pdb(clean_pdb, rm_ions, clean_pdb)
    if residue_map is not None:
        relabel_residues_in_pdb(clean_pdb, residue_map, clean_pdb)

    make_sdf(clean_pdb, lig_name=lig_name)

    fix_pdb(clean_pdb, combined_pdb, rm_heterogens=False)
    remove_residues_in_pdb(combined_pdb, combined_pdb, names=[lig_name])  # drop the ligand, kept separately as SDF

    combine_sdf_pdb(combined_pdb, lig_name=lig_name, patch=True)

    pdb_data = app.PDBFile(combined_pdb)
    molecule = Molecule.from_file(f'{lig_name}.sdf')

    if rm_files:
        os.remove(clean_pdb)
        os.remove(combined_pdb)
    if not save_lig_sdf:
        os.remove(f'{lig_name}.sdf')
    return pdb_data, molecule


def prepare_ligand_ff(standard_ff,
                      molecule,
                      gen_cache=False,
                      use_cache=False,
                      cache="gaff-molecules.json",
                      n_conf=10,
                      pc_methods='mmff94',
                      gaff_ver='gaff-2.11'):
    """Prepares a ligand-specific force field using the General Amber Force Field (GAFF).

    This function generates or loads GAFF parameters for a given molecule and integrates
    them into a standard force field. It supports caching of GAFF parameters for faster
    reuse and allows the generation of conformers and assignment of partial charges.

    Parameters
    ----------
    standard_ff : list of str
        A list of file paths or names of the standard force field XML files.
    molecule : openff.toolkit.Molecule
        The molecule for which GAFF parameters are to be prepared.
    gen_cache : bool, optional
        If True, generates a cache file for the GAFF parameters. Default is False.
    use_cache : bool, optional
        If True, loads GAFF parameters from the specified cache file. Default is False.
    cache : str, optional
        Path to the cache file for GAFF parameters. Default is 'gaff-molecules.json'.
    n_conf : int, optional
        The number of conformers to generate for the molecule. Default is 10.
    pc_methods : str, optional
        The method to use for assigning partial charges. Default is 'mmff94'.
        Other options include 'am1bcc' and 'am1-mulliken'.
    gaff_ver : str, optional
        The version of the General Amber Force Field to use. Default is 'gaff-2.11'.

    Returns
    -------
    openmm.app.ForceField
        The prepared force field object with GAFF parameters integrated.
    """
    if use_cache:
        print('Using cached GAFF parameters...', flush=True)
        gaff = GAFFTemplateGenerator(molecules=molecule, cache=cache, forcefield=gaff_ver)
        forcefield = app.ForceField(*standard_ff)
        forcefield.registerTemplateGenerator(gaff.generator)
    else:
        print('Generating GAFF parameters...', flush=True)
        molecule.generate_conformers(n_conformers=n_conf)
        molecule.assign_partial_charges(partial_charge_method=pc_methods,
                                        use_conformers=molecule.conformers)
        if gen_cache:
            print('Generating GAFF cache file...', flush=True)
            gaff = GAFFTemplateGenerator(molecules=molecule, cache=cache, forcefield=gaff_ver)
            forcefield = app.ForceField(*standard_ff)
            forcefield.registerTemplateGenerator(gaff.generator)
            forcefield.createSystem(topology=molecule.to_topology().to_openmm())
        else:
            gaff = GAFFTemplateGenerator(molecules=molecule, forcefield=gaff_ver)
            forcefield = app.ForceField(*standard_ff)
            forcefield.registerTemplateGenerator(gaff.generator)

    return forcefield


def deuterate_system(modeller, system, option='all', target_resname=None):
    """Replaces hydrogen atoms with deuterium in a molecular system.

    This function modifies the masses of hydrogen atoms in the system to the mass of deuterium
    based on the specified option. It supports deuteration of all hydrogens, or specific subsets
    such as water, protein, DNA, RNA, nucleic acids, or a specific ligand.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        The OpenMM modeller object containing the system topology and positions.
    system : openmm.System
        The OpenMM system object to be modified.
    option : str, optional
        Specifies the subset of the system to deuterate. Options include:
        'all', 'water', 'protein', 'dna', 'rna', 'nucleic', or 'ligand'. Default is 'all'.
    target_resname : str, optional
        The residue name of the ligand to deuterate. Required if `option` is 'ligand'.

    Returns
    -------
    None
        The system's particle masses are modified in place.

    Raises
    ------
    ValueError
        If `option` is 'ligand' and `target_resname` is not provided, or
        if `option` is not one of the supported values.

    Notes
    -----
    Only masses change, not the force field, so this captures the mass
    dependence of nuclear quantum effects but not zero-point energy. Use
    the ``run_openmm_rpmd_*`` functions when that matters.

    A warning is printed, rather than an error raised, when no residues
    match the selection.
    """
    deuterium_mass = app.element.deuterium.mass

    protein_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
        'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
        'TYR', 'VAL', 'HID', 'HIE', 'HIP', 'CYX', 'LYN', 'ASH', 'GLH',
        'ACE', 'NME', 'NAC'
    }

    dna_residues = {
        'DA', 'DC', 'DG', 'DT',
        'DA5', 'DC5', 'DG5', 'DT5',  # 5' terminals
        'DA3', 'DC3', 'DG3', 'DT3'  # 3' terminals
    }

    rna_residues = {
        'A', 'C', 'G', 'U',
        'RA', 'RC', 'RG', 'RU',  # Common in Amber force fields
        'A5', 'C5', 'G5', 'U5',  # 5' terminals
        'A3', 'C3', 'G3', 'U3'  # 3' terminals
    }

    water_residues = {'HOH', 'H2O', 'TIP3', 'WAT', 'SOL'}

    nucleic_residues = dna_residues.union(rna_residues)

    target_residues = set()
    if option == 'all':
        pass
    elif option == 'water':
        target_residues = water_residues
    elif option == 'protein':
        target_residues = protein_residues
    elif option == 'dna':
        target_residues = dna_residues
    elif option == 'rna':
        target_residues = rna_residues
    elif option == 'nucleic':
        target_residues = nucleic_residues
    elif option == 'ligand':
        if target_resname is None:
            raise ValueError("If option is 'ligand', you must provide a 'target_resname'.")
        target_residues = {target_resname}
    else:
        raise ValueError("Option must be 'all', 'water', 'protein', 'dna', 'rna', 'nucleic', or 'ligand'")

    if option == 'all':
        for atom in modeller.topology.atoms():
            if atom.element and atom.element.symbol == 'H':
                system.setParticleMass(atom.index, deuterium_mass)
    else:
        found_target = False
        for residue in modeller.topology.residues():
            if residue.name in target_residues:
                found_target = True
                for atom in residue.atoms():
                    if atom.element and atom.element.symbol == 'H':
                        system.setParticleMass(atom.index, deuterium_mass)

        if not found_target and option == 'ligand':
            print(f"Warning: No ligand named '{target_resname}' was found.")
        elif not found_target and option != 'all':
            print(f"Warning: No residues matching option '{option}' were found.")


def get_atoms_in_residue(pdb_file_path, residue_index, chain_id=None):
    """Retrieves the atom indices of a specific residue in a PDB file.

    This function reads a PDB file, identifies the specified residue by its index
    and optionally its chain ID, and returns the indices of all atoms in that residue.

    Parameters
    ----------
    pdb_file_path : str
        Path to the PDB file to be read.
    residue_index : int
        The index of the residue whose atom indices are to be retrieved.
    chain_id : str, optional
        The ID of the chain containing the residue. If None, the residue is
        searched in the global topology. Default is None.

    Returns
    -------
    list of int or None
        A list of atom indices in the specified residue. Returns None if the
        residue or chain is not found, or if the residue index is out of bounds.

    Notes
    -----
    - If `chain_id` is provided, the residue is searched within the specified chain.
    - If `chain_id` is None, the residue is searched in the global topology.
    - Prints error messages if the chain or residue index is invalid.
    """
    pdb = app.PDBFile(pdb_file_path)
    topology = pdb.topology
    if chain_id is not None:
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
        residues = list(topology.residues())

        if residue_index < 0 or residue_index >= len(residues):
            print(f"Error: Residue index {residue_index} is out of bounds.")
            print(f"The file contains {len(residues)} residues (indices 0 to {len(residues) - 1}).")
            return None

        target_residue = residues[residue_index]
        print(f"Looking in global topology, Residue Index {residue_index}...")

    atom_indices = [atom.index for atom in target_residue.atoms()]

    print(
        f"Successfully retrieved residue: {target_residue.name} (Chain: {target_residue.chain.id}, Index: {target_residue.index}, PDB ID: {target_residue.id})")
    return atom_indices


def save_pdb_selection(input_pdb_path, atom_indices, output_pdb_path):
    """Saves a subset of atoms from a PDB file to a new PDB file.

    This function reads a PDB file, selects a subset of atoms based on their indices,
    and writes the selected atoms to a new PDB file. Atoms not in the specified indices
    are removed from the output.

    Parameters
    ----------
    input_pdb_path : str
        Path to the input PDB file.
    atom_indices : list of int
        A list of atom indices to keep in the output PDB file.
    output_pdb_path : str
        Path to save the output PDB file containing the selected atoms.

    Returns
    -------
    None

    Notes
    -----
    - If the selection is empty, a warning is printed, and the output PDB file will be empty.
    - The atom indices should correspond to the indices in the input PDB file.
    """
    pdb = app.PDBFile(input_pdb_path)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    keep_indices = set(atom_indices)
    atoms_to_delete = []
    all_atoms = list(modeller.topology.atoms())

    for atom in all_atoms:
        if atom.index not in keep_indices:
            atoms_to_delete.append(atom)

    num_deleted = len(atoms_to_delete)
    if num_deleted == len(all_atoms):
        print("Warning: Your selection is empty! The output PDB will be empty.")

    modeller.delete(atoms_to_delete)

    print(f"Writing selection ({len(all_atoms) - num_deleted} atoms) to {output_pdb_path}...")
    with open(output_pdb_path, 'w') as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)


def run_openmm_relaxation(modeller,
                          forcefield,
                          output_prefix='minimized',
                          temperature=300.0 * unit.kelvin,
                          gamma=1.0 / unit.picosecond,
                          time_step=1.0 * unit.femtoseconds,
                          n_1=1_000,
                          n_2=1_000,
                          n_3=2_000,
                          backbone_names=None,
                          ks_1=100.0,
                          ks_2=10.0,
                          ks_3=0.0,
                          platform_name='CPU',
                          potential=None,
                          ml_idx=None,
                          ):
    """Minimise a structure in three stages of decreasing restraint.

    Backbone atoms are held by a harmonic restraint that is relaxed
    between stages, so the solvent and side chains settle before the
    backbone is allowed to move. Releasing everything at once tends to
    distort an experimental starting structure.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Prepared topology and positions for the system.
    forcefield : openmm.app.ForceField
        Force field used to parameterise the system.
    output_prefix : str, optional
        Stem for the PDB, log and checkpoint files written. Default is
        "minimized".
    temperature : openmm.unit.Quantity, optional
        Temperature for the integrator. Default is 300 K.
    gamma : openmm.unit.Quantity, optional
        Langevin friction coefficient. Default is 1/ps.
    time_step : openmm.unit.Quantity, optional
        Integration timestep. Default is 1 fs.
    n_1 : int, optional
        Maximum minimisation iterations in stage 1. Default is 1000.
    n_2 : int, optional
        Maximum minimisation iterations in stage 2. Default is 1000.
    n_3 : int, optional
        Maximum minimisation iterations in stage 3. Default is 2000.
    backbone_names : list of str, optional
        Atom names treated as backbone and restrained. Default is
        ['CA', 'C', 'N', 'P', 'O3'], which covers both protein and nucleic
        acid backbones.
    ks_1 : float, optional
        Stage 1 restraint constant in kJ/mol/nm^2. Default is 100.0.
    ks_2 : float, optional
        Stage 2 restraint constant in kJ/mol/nm^2. Default is 10.0.
    ks_3 : float, optional
        Stage 3 restraint constant in kJ/mol/nm^2. Default is 0.0, which
        removes the restraint entirely.
    platform_name : str, optional
        OpenMM platform to run on. Overridden to 'CUDA' whenever a mixed
        ML potential is in use. Default is "CPU".
    potential : openmmml.MLPotential, optional
        Machine-learning potential for the subsystem named by `ml_idx`. Has
        no effect unless `ml_idx` is also given. Default is None.
    ml_idx : list of int, optional
        Indices of the atoms to treat with `potential`. Has no effect unless
        `potential` is also given. Default is None.

    Returns
    -------
    None

    Notes
    -----
    Passing both `potential` and `ml_idx` builds a mixed ML/MM system and
    forces the CUDA platform.

        The system is built without constraints and with flexible water, so
        that the minimisation can relax bond lengths too.
    """
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )

    current_positions = modeller.positions
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

    print(f"\n--- Stage 1: Strong Backbone Restraints ({ks_1} kJ/mol/nm^2) ---", flush=True)
    k_strong = ks_1 * unit.kilojoules_per_mole / (unit.nanometer ** 2)
    simulation.context.setParameter("k", k_strong)
    simulation.minimizeEnergy(maxIterations=n_1)

    print("\n--- Stage 2: Weak Backbone Restraints (10 kJ/mol/nm^2) ---", flush=True)
    k_weak = ks_2 * unit.kilojoules_per_mole / (unit.nanometer ** 2)
    simulation.context.setParameter("k", k_weak)
    simulation.minimizeEnergy(maxIterations=n_2)

    print("\n--- Stage 3: Unrestrained Relaxation ---", flush=True)
    k_vweak = ks_3 * unit.kilojoules_per_mole / (unit.nanometer ** 2)
    simulation.context.setParameter("k", k_vweak)
    simulation.minimizeEnergy(maxIterations=n_3)

    final_state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, final_state.getPositions(), f)
    print(f"\nProcess complete. Saved to {output_prefix}", flush=True)


def run_openmm_relaxation_simple(modeller,
                                 forcefield,
                                 output_prefix='minimized',
                                 temperature=300.0 * unit.kelvin,
                                 gamma=1.0 / unit.picosecond,
                                 time_step=1.0 * unit.femtoseconds,
                                 n_report=1,
                                 platform_name='CUDA',
                                 potential=None,
                                 ml_idx=None,
                                 ):
    """Minimise a structure in a single unrestrained pass.

    The plain counterpart to :func:`run_openmm_relaxation`, for systems
    that do not need staged restraints.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Prepared topology and positions for the system.
    forcefield : openmm.app.ForceField
        Force field used to parameterise the system.
    output_prefix : str, optional
        Stem for the PDB, log and checkpoint files written. Default is
        "minimized".
    temperature : openmm.unit.Quantity, optional
        Temperature for the integrator. Default is 300 K.
    gamma : openmm.unit.Quantity, optional
        Langevin friction coefficient. Default is 1/ps.
    time_step : openmm.unit.Quantity, optional
        Integration timestep. Default is 1 fs.
    n_report : int, optional
        Steps between reporter writes. Default is 1.
    platform_name : str, optional
        OpenMM platform to run on. Overridden to 'CUDA' whenever a mixed
        ML potential is in use. Default is "CUDA".
    potential : openmmml.MLPotential, optional
        Machine-learning potential for the subsystem named by `ml_idx`. Has
        no effect unless `ml_idx` is also given. Default is None.
    ml_idx : list of int, optional
        Indices of the atoms to treat with `potential`. Has no effect unless
        `potential` is also given. Default is None.

    Returns
    -------
    None

    Notes
    -----
    Passing both `potential` and `ml_idx` builds a mixed ML/MM system and
    forces the CUDA platform.
    """
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )

    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)

    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    simulation.reporters.append(app.PDBReporter(f'{output_prefix}_steps.pdb', n_report))
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    print("Minimizing energy", flush=True)
    simulation.minimizeEnergy()

    simulation.saveCheckpoint(f'{output_prefix}.chk')
    final_state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, final_state.getPositions(), f)
    print(f"\nProcess complete. Saved to {output_prefix}", flush=True)


def run_openmm_heating(modeller,
                       forcefield,
                       output_prefix='equilibrate',
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
                       deuterate=False,
                       deuterate_option='water',
                       potential=None,
                       ml_idx=None,
                       ):
    """Heat a system to its target temperature in stages.

    Temperature is raised in increments of `temp_step`, with the backbone
    restrained throughout, then a final unrestrained run is performed at
    the target temperature. Heating gradually avoids the structural
    damage that setting the full temperature at once can cause.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Prepared topology and positions for the system.
    forcefield : openmm.app.ForceField
        Force field used to parameterise the system.
    output_prefix : str, optional
        Stem for the PDB, log and checkpoint files written. Default is
        "equilibrate".
    k1 : float, optional
        Backbone restraint constant in kJ/mol/nm^2. Default is 100.0.
    backbone_names : list of str, optional
        Atom names treated as backbone and restrained. Default is
        ['CA', 'C', 'N', 'P', 'O3'].
    target_temp : openmm.unit.Quantity, optional
        Final temperature. Default is 300 K.
    temp_step : openmm.unit.Quantity, optional
        Temperature increment per stage. Default is 50 K.
    gamma : openmm.unit.Quantity, optional
        Langevin friction coefficient. Default is 1/ps.
    time_step : openmm.unit.Quantity, optional
        Integration timestep. Default is 1 fs.
    n_report : int, optional
        Steps between reporter writes. Default is 1000.
    steps_per_stage : int, optional
        Steps run at each intermediate temperature. Default is 5000.
    steps_final : int, optional
        Steps run at the target temperature. Default is 10000.
    platform_name : str, optional
        OpenMM platform to run on. Overridden to 'CUDA' whenever a mixed
        ML potential is in use. Default is "CPU".
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    deuterate_option : str, optional
        Which hydrogens to deuterate, for example 'water' or 'all'. Default
        is "water".
    potential : openmmml.MLPotential, optional
        Machine-learning potential for the subsystem named by `ml_idx`. Has
        no effect unless `ml_idx` is also given. Default is None.
    ml_idx : list of int, optional
        Indices of the atoms to treat with `potential`. Has no effect unless
        `potential` is also given. Default is None.

    Returns
    -------
    None

    Notes
    -----
    Passing both `potential` and `ml_idx` builds a mixed ML/MM system and
    forces the CUDA platform.
    """
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
    has_box = modeller.topology.getUnitCellDimensions() is not None
    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

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
    simulation.reporters.append(app.PDBReporter(f'{output_prefix}_steps.pdb', n_report))
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

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

    simulation.saveCheckpoint(f'{output_prefix}.chk')
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
    print(f"Saved equilibrated structure to {output_prefix}", flush=True)


def run_openmm_npt(modeller,
                   forcefield,
                   output_prefix='npt_equilibrated',
                   pressure=1.0 * unit.bar,
                   temperature=300.0 * unit.kelvin,
                   gamma=1.0 / unit.picosecond,
                   time_step=1.0 * unit.femtoseconds,
                   barostat_freq=50,
                   backbone_names=None,
                   k=10.0,
                   n_report=500,
                   n_1=5_000,
                   n_2=25_000,
                   platform_name='CPU',
                   deuterate=False,
                   deuterate_option='water',
                   potential=None,
                   ml_idx=None,
                   ):
    """Equilibrate a system at constant pressure.

    Adds a Monte Carlo barostat and runs a restrained stage followed by
    an unrestrained one, letting the box relax to the correct density
    before production.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Prepared topology and positions for the system.
    forcefield : openmm.app.ForceField
        Force field used to parameterise the system.
    output_prefix : str, optional
        Stem for the PDB, log and checkpoint files written. Default is
        "npt_equilibrated".
    pressure : openmm.unit.Quantity, optional
        Target pressure. Default is 1 bar.
    temperature : openmm.unit.Quantity, optional
        Target temperature. Default is 300 K.
    gamma : openmm.unit.Quantity, optional
        Langevin friction coefficient. Default is 1/ps.
    time_step : openmm.unit.Quantity, optional
        Integration timestep. Default is 1 fs.
    barostat_freq : int, optional
        Steps between barostat volume moves. Default is 50.
    backbone_names : list of str, optional
        Atom names treated as backbone and restrained. Default is
        ['CA', 'C', 'N', 'P', 'O3'].
    k : float, optional
        Backbone restraint constant in kJ/mol/nm^2 for the first stage.
        Default is 10.0.
    n_report : int, optional
        Steps between reporter writes. Default is 500.
    n_1 : int, optional
        Steps in the restrained stage. Default is 5000.
    n_2 : int, optional
        Steps in the unrestrained stage. Default is 25000.
    platform_name : str, optional
        OpenMM platform to run on. Overridden to 'CUDA' whenever a mixed
        ML potential is in use. Default is "CPU".
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    deuterate_option : str, optional
        Which hydrogens to deuterate, for example 'water' or 'all'. Default
        is "water".
    potential : openmmml.MLPotential, optional
        Machine-learning potential for the subsystem named by `ml_idx`. Has
        no effect unless `ml_idx` is also given. Default is None.
    ml_idx : list of int, optional
        Indices of the atoms to treat with `potential`. Has no effect unless
        `potential` is also given. Default is None.

    Returns
    -------
    None

    Notes
    -----
    Passing both `potential` and `ml_idx` builds a mixed ML/MM system and
    forces the CUDA platform.
    """
    if backbone_names is None:
        backbone_names = ['CA', 'C', 'N', 'P', 'O3']

    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

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

    simulation.reporters.append(app.PDBReporter(f'{output_prefix}_steps.pdb', n_report))
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    print("\n--- Phase 1: Restrained NPT (Relaxing Density) ---", flush=True)
    simulation.step(n_1)

    print("\n--- Phase 2: Removing Restraints (Unrestrained NPT) ---", flush=True)
    simulation.context.setParameter("k", 0.0)
    simulation.step(n_2)

    simulation.saveCheckpoint(f'{output_prefix}.chk')
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)

    print(f"\nDensity equilibration complete. Saved to {output_prefix}", flush=True)


def run_openmm_prod(modeller,
                    forcefield,
                    plumed_script_path=None,
                    pressure=1.0 * unit.bar,
                    temperature=300.0 * unit.kelvin,
                    gamma=1.0 / unit.picosecond,
                    time_step=1.0 * unit.femtoseconds,
                    barostat_freq=50,
                    n_report=1_000,
                    steps=500_000,
                    output_prefix='prod',
                    platform_name='CPU',
                    deuterate=False,
                    deuterate_option='water',
                    potential=None,
                    ml_idx=None,
                    ):
    """Run classical production MD, optionally biased with PLUMED.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Prepared topology and positions for the system.
    forcefield : openmm.app.ForceField
        Force field used to parameterise the system.
    plumed_script_path : str, optional
        Path to a PLUMED input file to apply as a bias. Default is None,
        which runs unbiased.
    pressure : openmm.unit.Quantity, optional
        Target pressure. Default is 1 bar.
    temperature : openmm.unit.Quantity, optional
        Target temperature. Default is 300 K.
    gamma : openmm.unit.Quantity, optional
        Langevin friction coefficient. Default is 1/ps.
    time_step : openmm.unit.Quantity, optional
        Integration timestep. Default is 1 fs.
    barostat_freq : int, optional
        Steps between barostat volume moves. Default is 50.
    n_report : int, optional
        Steps between reporter writes. Default is 1000.
    steps : int, optional
        Number of production steps. Default is 500000.
    output_prefix : str, optional
        Stem for the PDB, log and checkpoint files written. Default is
        "prod".
    platform_name : str, optional
        OpenMM platform to run on. Overridden to 'CUDA' whenever a mixed
        ML potential is in use. Default is "CPU".
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    deuterate_option : str, optional
        Which hydrogens to deuterate, for example 'water' or 'all'. Default
        is "water".
    potential : openmmml.MLPotential, optional
        Machine-learning potential for the subsystem named by `ml_idx`. Has
        no effect unless `ml_idx` is also given. Default is None.
    ml_idx : list of int, optional
        Indices of the atoms to treat with `potential`. Has no effect unless
        `potential` is also given. Default is None.

    Returns
    -------
    None

    Notes
    -----
    Passing both `potential` and `ml_idx` builds a mixed ML/MM system and
    forces the CUDA platform.
    """
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

    system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_freq))

    if plumed_script_path is not None:
        print(f"Adding PLUMED bias from {plumed_script_path}...", flush=True)

        with open(plumed_script_path) as f:
            script_content = f.read()

        plumed_force = PlumedForce(script_content)
        system.addForce(plumed_force)
    integrator = openmm.LangevinMiddleIntegrator(temperature,
                                                 gamma,
                                                 time_step)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    simulation.reporters.append(app.PDBReporter(f'{output_prefix}_steps.pdb', n_report))
    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    simulation.reporters.append(app.CheckpointReporter(f'{output_prefix}.chk', n_report * 10))
    print(f"Starting production run for {steps} steps...", flush=True)
    simulation.step(steps)
    print("Production run complete.", flush=True)

    simulation.saveCheckpoint(f'{output_prefix}.chk')
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)


def run_openmm_rpmd_equilibration(modeller,
                                  forcefield,
                                  output_prefix='rpmd_ready',
                                  n_beads=32,
                                  temperature=300 * unit.kelvin,
                                  pressure=1.0 * unit.bar,
                                  barostat_freq=50,
                                  friction=1.0 / unit.picosecond,
                                  timestep=0.5 * unit.femtoseconds,
                                  n_report=1_000,
                                  n_1=2_000,
                                  n_2=10_000,
                                  platform_name='CPU',
                                  deuterate=False,
                                  deuterate_option='water',
                                  potential=None,
                                  ml_idx=None,
                                  atoms_to_watch=None):
    """Equilibrate a ring-polymer MD system and save a checkpoint.

    Runs in two stages: a first at half the requested timestep, letting
    the beads spread out from their collapsed starting positions without
    the stiff spring forces destabilising the integrator, then a second
    at the full timestep. The checkpoint written here is what the
    production runs restart from.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Prepared topology and positions for the system.
    forcefield : openmm.app.ForceField
        Force field used to parameterise the system.
    output_prefix : str, optional
        Stem for the PDB, log and checkpoint files written. Default is
        "rpmd_ready".
    n_beads : int, optional
        Number of ring-polymer beads. Default is 32.
    temperature : openmm.unit.Quantity, optional
        Target temperature. Default is 300 K.
    pressure : openmm.unit.Quantity, optional
        Target pressure. Default is 1 bar.
    barostat_freq : int, optional
        Steps between barostat volume moves. Default is 50.
    friction : openmm.unit.Quantity, optional
        Langevin friction coefficient. Default is 1/ps.
    timestep : openmm.unit.Quantity, optional
        Integration timestep at full speed. Default is 0.5 fs.
    n_report : int, optional
        Steps between reporter writes. Default is 1000.
    n_1 : int, optional
        Steps in the bead expansion stage, run at half `timestep`. Default
        is 2000.
    n_2 : int, optional
        Steps in the relaxation stage at full `timestep`. Default is 10000.
    platform_name : str, optional
        OpenMM platform to run on. Overridden to 'CUDA' whenever a mixed
        ML potential is in use. Default is "CPU".
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    deuterate_option : str, optional
        Which hydrogens to deuterate, for example 'water' or 'all'. Default
        is "water".
    potential : openmmml.MLPotential, optional
        Machine-learning potential for the subsystem named by `ml_idx`. Has
        no effect unless `ml_idx` is also given. Default is None.
    ml_idx : list of int, optional
        Indices of the atoms to treat with `potential`. Has no effect unless
        `potential` is also given. Default is None.
    atoms_to_watch : list of int, optional
        Indices of atoms whose quantum spread is logged each report. Default
        is None, which disables the spread reporter.

    Returns
    -------
    None

    Notes
    -----
    Passing both `potential` and `ml_idx` builds a mixed ML/MM system and
    forces the CUDA platform.

        Ring-polymer dynamics needs a shorter timestep than classical MD
        because the internal spring frequencies scale with the bead count.
    """
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

    system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_freq))
    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, timestep)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    if atoms_to_watch is not None:
        simulation.reporters.append(RPMDQuantumSpreadReporter(
            file=f'{output_prefix}_spread.log',
            reportInterval=n_report,
            atom_indices=atoms_to_watch,
        ))

    simulation.reporters.append(RPMDCentroidReporter(
        topology=modeller.topology,
        file_name=f"{output_prefix}_centroid.pdb",
        reportInterval=n_report,
        num_beads=n_beads,
    ))

    simulation.reporters.append(RPMDBeadReporter(
        topology=modeller.topology,
        file_base_name=output_prefix,
        reportInterval=n_report,
        num_beads=n_beads,
    ))

    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    print("\n--- Stage 1: Bead Expansion  ---", flush=True)
    integrator.setStepSize(timestep * 0.5)
    simulation.step(n_1)

    print(f"\n--- Stage 2: Relaxation at full timestep ({timestep}) ---", flush=True)
    integrator.setStepSize(timestep)
    simulation.step(n_2)

    print("\n--- Saving State ---", flush=True)
    simulation.saveCheckpoint(f'{output_prefix}.chk')
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}_centroid.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
    print(f"Saved centroid visualization to {output_prefix}_centroid.pdb", flush=True)


def run_openmm_rpmd_contracted(modeller,
                               forcefield,
                               plumed_script_path=None,
                               checkpoint_file='rpmd_ready.chk',
                               output_prefix='prod_contracted',
                               n_beads=32,
                               temperature=300 * unit.kelvin,
                               pressure=1.0 * unit.bar,
                               barostat_freq=50,
                               friction=1.0 / unit.picosecond,
                               timestep=0.5 * unit.femtoseconds,
                               steps=100_000,
                               n_report=1_000,
                               contractions=None,
                               platform_name='CPU',
                               deuterate=False,
                               deuterate_option='water',
                               potential=None,
                               ml_idx=None,
                               atoms_to_watch=None):
    """Run ring-polymer production MD with ring-polymer contraction.

    Contraction evaluates the slowly varying parts of the force field on
    fewer beads than the full ring, which is where most of the cost of
    RPMD otherwise goes. Restarts from an equilibration checkpoint.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Prepared topology and positions for the system.
    forcefield : openmm.app.ForceField
        Force field used to parameterise the system.
    plumed_script_path : str, optional
        Path to a PLUMED input file to apply as a bias. Default is None.
    checkpoint_file : str, optional
        Checkpoint to restart from. Default is "rpmd_ready.chk".
    output_prefix : str, optional
        Stem for the PDB, log and checkpoint files written. Default is
        "prod_contracted".
    n_beads : int, optional
        Number of ring-polymer beads. Default is 32.
    temperature : openmm.unit.Quantity, optional
        Target temperature. Default is 300 K.
    pressure : openmm.unit.Quantity, optional
        Target pressure. Default is 1 bar.
    barostat_freq : int, optional
        Steps between barostat volume moves. Default is 50.
    friction : openmm.unit.Quantity, optional
        Langevin friction coefficient. Default is 1/ps.
    timestep : openmm.unit.Quantity, optional
        Integration timestep. Default is 0.5 fs.
    steps : int, optional
        Number of production steps. Default is 100000.
    n_report : int, optional
        Steps between reporter writes. Default is 1000.
    contractions : dict, optional
        Maps force group index to the number of bead copies it is evaluated
        on. Each count must divide `n_beads`, and groups left out default to
        the full ring. Default contracts direct-space nonbonded onto 8
        copies and PME reciprocal space onto the centroid alone.
    platform_name : str, optional
        OpenMM platform to run on. Overridden to 'CUDA' whenever a mixed
        ML potential is in use. Default is "CPU".
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    deuterate_option : str, optional
        Which hydrogens to deuterate, for example 'water' or 'all'. Default
        is "water".
    potential : openmmml.MLPotential, optional
        Machine-learning potential for the subsystem named by `ml_idx`. Has
        no effect unless `ml_idx` is also given. Default is None.
    ml_idx : list of int, optional
        Indices of the atoms to treat with `potential`. Has no effect unless
        `potential` is also given. Default is None.
    atoms_to_watch : list of int, optional
        Indices of atoms whose quantum spread is logged each report. Default
        is None, which disables the spread reporter.

    Returns
    -------
    None

    Notes
    -----
    Passing both `potential` and `ml_idx` builds a mixed ML/MM system and
    forces the CUDA platform.
    """
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if contractions is None:
        contractions = {
            1: 8,  # Nonbonded Direct Space (calculate on every 4th bead)
            2: 1  # PME Reciprocal Space (calculate only on centroid)
        }

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

    system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_freq))

    if plumed_script_path is not None:
        print(f"Adding PLUMED bias from {plumed_script_path}...", flush=True)

        with open(plumed_script_path) as f:
            script_content = f.read()

        plumed_force = PlumedForce(script_content)
        system.addForce(plumed_force)

    # Contraction is keyed on force group, so forces are sorted into groups by
    # cost: 0 for cheap bonded terms evaluated on every bead, 1 for nonbonded
    # direct space, 2 for the expensive PME reciprocal sum on the centroid alone
    print("Assigning force groups for contraction...", flush=True)

    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setForceGroup(1)
            force.setReciprocalSpaceForceGroup(2)
            print(f"  - {force.__class__.__name__}: Direct->Group 1, Reciprocal->Group 2")

        elif isinstance(force, (openmm.HarmonicBondForce,
                                openmm.HarmonicAngleForce,
                                openmm.PeriodicTorsionForce,
                                openmm.RBTorsionForce,
                                openmm.CMAPTorsionForce)):
            force.setForceGroup(0)
            print(f"  - {force.__class__.__name__}: Group 0")

        else:  # Barostat and anything else, evaluated on every bead
            force.setForceGroup(0)

    print(f"\nInitializing RPMDIntegrator with contractions: {contractions}", flush=True)
    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, timestep, contractions)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)

    if not os.path.exists(checkpoint_file):
        print(f"Error: Checkpoint {checkpoint_file} not found. Run equilibration first.", flush=True)
        return

    print(f"Loading state from {checkpoint_file}...", flush=True)
    simulation.loadCheckpoint(checkpoint_file)

    if atoms_to_watch is not None:
        simulation.reporters.append(RPMDQuantumSpreadReporter(
            file=f'{output_prefix}_spread.log',
            reportInterval=n_report,
            atom_indices=atoms_to_watch,
        ))

    simulation.reporters.append(RPMDCentroidReporter(
        topology=modeller.topology,
        file_name=f"{output_prefix}_centroid.pdb",
        reportInterval=n_report,
        num_beads=n_beads,
    ))

    simulation.reporters.append(RPMDBeadReporter(
        topology=modeller.topology,
        file_base_name=output_prefix,
        reportInterval=n_report,
        num_beads=n_beads,
    ))

    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    print(f"\nStarting Production Run ({steps} steps)...")
    simulation.step(steps)
    print("Done.", flush=True)

    print("\n--- Saving State ---", flush=True)
    simulation.saveCheckpoint(f'{output_prefix}.chk')
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}_centroid.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
    print(f"Saved centroid visualization to {output_prefix}_centroid.pdb", flush=True)


def run_openmm_rpmd_prod(modeller,
                         forcefield,
                         plumed_script_path=None,
                         checkpoint_file='rpmd_ready.chk',
                         output_prefix='prod',
                         n_beads=32,
                         pressure=1.0 * unit.bar,
                         temperature=300.0 * unit.kelvin,
                         gamma=1.0 / unit.picosecond,
                         time_step=1.0 * unit.femtoseconds,
                         barostat_freq=50,
                         n_report=1_000,
                         steps=500_000,
                         platform_name='CPU',
                         deuterate=False,
                         deuterate_option='water',
                         potential=None,
                         ml_idx=None,
                         atoms_to_watch=None):
    """Run ring-polymer production MD on the full ring polymer.

    The uncontracted counterpart of
    :func:`run_openmm_rpmd_contracted`: every force is evaluated on
    every bead. More expensive, but free of any contraction
    approximation. Restarts from an equilibration checkpoint.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Prepared topology and positions for the system.
    forcefield : openmm.app.ForceField
        Force field used to parameterise the system.
    plumed_script_path : str, optional
        Path to a PLUMED input file to apply as a bias. Default is None.
    checkpoint_file : str, optional
        Checkpoint to restart from. Default is "rpmd_ready.chk".
    output_prefix : str, optional
        Stem for the PDB, log and checkpoint files written. Default is
        "prod".
    n_beads : int, optional
        Number of ring-polymer beads. Default is 32.
    pressure : openmm.unit.Quantity, optional
        Target pressure. Default is 1 bar.
    temperature : openmm.unit.Quantity, optional
        Target temperature. Default is 300 K.
    gamma : openmm.unit.Quantity, optional
        Langevin friction coefficient. Default is 1/ps.
    time_step : openmm.unit.Quantity, optional
        Integration timestep. Default is 1 fs.
    barostat_freq : int, optional
        Steps between barostat volume moves. Default is 50.
    n_report : int, optional
        Steps between reporter writes. Default is 1000.
    steps : int, optional
        Number of production steps. Default is 500000.
    platform_name : str, optional
        OpenMM platform to run on. Overridden to 'CUDA' whenever a mixed
        ML potential is in use. Default is "CPU".
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    deuterate_option : str, optional
        Which hydrogens to deuterate, for example 'water' or 'all'. Default
        is "water".
    potential : openmmml.MLPotential, optional
        Machine-learning potential for the subsystem named by `ml_idx`. Has
        no effect unless `ml_idx` is also given. Default is None.
    ml_idx : list of int, optional
        Indices of the atoms to treat with `potential`. Has no effect unless
        `potential` is also given. Default is None.
    atoms_to_watch : list of int, optional
        Indices of atoms whose quantum spread is logged each report. Default
        is None, which disables the spread reporter.

    Returns
    -------
    None

    Notes
    -----
    Passing both `potential` and `ml_idx` builds a mixed ML/MM system and
    forces the CUDA platform.
    """
    if potential is not None and ml_idx is not None:
        print("Adding ML potential to the system...", flush=True)
        run_mixed = True
        platform_name = 'CUDA'
    else:
        run_mixed = False

    platform = openmm.Platform.getPlatformByName(platform_name)
    has_box = modeller.topology.getUnitCellDimensions() is not None

    if run_mixed:
        mm_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        system = potential.createMixedSystem(
            modeller.topology,
            mm_system,
            ml_idx,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )

    if deuterate:
        print("Deuterating system...", flush=True)
        deuterate_system(modeller, system, option=deuterate_option)

    system.addForce(openmm.MonteCarloBarostat(pressure, temperature, barostat_freq))

    if plumed_script_path is not None:
        print(f"Adding PLUMED bias from {plumed_script_path}...", flush=True)

        with open(plumed_script_path) as f:
            script_content = f.read()

        plumed_force = PlumedForce(script_content)
        system.addForce(plumed_force)
    integrator = openmm.RPMDIntegrator(temperature,
                                       gamma,
                                       time_step)
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    if not os.path.exists(checkpoint_file):
        print(f"Error: Checkpoint {checkpoint_file} not found. Run equilibration first.", flush=True)
        return

    print(f"Loading state from {checkpoint_file}...", flush=True)
    simulation.loadCheckpoint(checkpoint_file)

    if atoms_to_watch is not None:
        simulation.reporters.append(RPMDQuantumSpreadReporter(
            file=f'{output_prefix}_spread.log',
            reportInterval=n_report,
            atom_indices=atoms_to_watch,
        ))

    simulation.reporters.append(RPMDCentroidReporter(
        topology=modeller.topology,
        file_name=f"{output_prefix}_centroid.pdb",
        reportInterval=n_report,
        num_beads=n_beads,
    ))

    simulation.reporters.append(RPMDBeadReporter(
        topology=modeller.topology,
        file_base_name=output_prefix,
        reportInterval=n_report,
        num_beads=n_beads,
    ))

    simulation.reporters.append(app.StateDataReporter(sys.stdout,
                                                      n_report,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))
    simulation.reporters.append(app.StateDataReporter(f'{output_prefix}.log',
                                                      n_report,
                                                      step=True,
                                                      time=True,
                                                      potentialEnergy=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True,
                                                      volume=True))

    print(f"Starting production run for {steps} steps...", flush=True)
    simulation.step(steps)
    print("Production run complete.", flush=True)

    simulation.saveCheckpoint(f'{output_prefix}.chk')
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(f'{output_prefix}.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)


def _calculate_quantum_spread(integrator, atom_indices=None):
    """Calculates the root-mean-square distance of beads from the ring polymer centroid.
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

    coords = np.array(all_bead_positions)
    centroid = np.mean(coords, axis=0)
    diff = coords - centroid
    sq_dist = np.sum(diff ** 2, axis=2)  # sum x,y,z components -> (n_beads, n_atoms)
    mean_sq_dist = np.mean(sq_dist, axis=0)
    quantum_rg = np.sqrt(mean_sq_dist)
    return quantum_rg * unit.nanometers


def _close_pdb_files(topology, files):
    """Write the PDB footer to each open file and close it.

    Errors are swallowed because interpreter shutdown may already have torn
    down what this needs, and a failure here would mask the real reason the
    run ended. Each file is handled separately so one failure does not leave
    the rest unterminated.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology the footer is written for.
    files : iterable of file object
        Open PDB files to terminate and close.
    """
    for handle in files:
        try:
            app.PDBFile.writeFooter(topology, handle)
            handle.close()
        except Exception:
            pass


class _RPMDReporterBase:
    """Shared scheduling for the ring-polymer reporters.

    OpenMM asks a reporter how many steps remain until it next wants data,
    and what to assemble for it. All three reporters below answer the same
    way, and none of them want anything assembled: the bead positions they
    report on come from the RPMD integrator directly, since the state OpenMM
    would hand them holds only the centroid.

    Subclasses set ``_reportInterval`` in their own ``__init__``.
    """

    def describeNextReport(self, simulation):
        """Report how many steps remain until the next report is due.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation requesting the report.

        Returns
        -------
        tuple
            Steps until the next report, followed by flags for whether
            positions, velocities, forces and energies are needed.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)


class RPMDQuantumSpreadReporter(_RPMDReporterBase):
    """Log the quantum delocalisation of selected atoms during RPMD.

    Writes the radius of gyration of each atom's ring polymer, which
    measures how far that nucleus is spread in imaginary time. A proton
    with a large spread relative to its bond lengths is one where
    tunnelling is likely to matter.

    Attributes
    ----------
    _atom_indices : list of int
        Indices of the atoms being monitored.
    """

    def __init__(self, file, reportInterval, atom_indices, names=None):
        """Initialize the reporter.

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

        if names:
            header = "Step\t" + "\t".join([f"Rg_{n}(nm)" for n in names])
        else:
            header = "Step\t" + "\t".join([f"Rg_Atom{i}(nm)" for i in atom_indices])
        self._out.write(header + "\n")

    def report(self, simulation, state):
        """Write the radius of gyration of each watched atom.

        The spread of a ring polymer measures how delocalised that nucleus
        is, so a proton with a large radius of gyration is one where
        tunnelling matters.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation requesting the report.
        state : openmm.State
            Current state of the simulation. Unused; bead positions are read
            from the RPMD integrator instead, since the state holds only the
            centroid.

        Returns
        -------
        None
        """
        integrator = simulation.integrator

        spreads = _calculate_quantum_spread(integrator, self._atom_indices)

        step = simulation.currentStep
        spread_values = spreads.value_in_unit(unit.nanometers)

        line = f"{step}"
        for val in spread_values:
            line += f"\t{val:.6f}"
        self._out.write(line + "\n")
        self._out.flush()

    def __del__(self):
        """Close the log file when the reporter is destroyed.

        Returns
        -------
        None
        """
        self._out.close()


class RPMDBeadReporter(_RPMDReporterBase):
    """Write the trajectory of every individual bead to its own PDB file.

    OpenMM's built-in reporters see only the centroid, so the bead
    positions have to be pulled from the RPMD integrator directly. Use
    when the spread of the ring polymer itself is the object of interest;
    :class:`RPMDCentroidReporter` is cheaper when it is not.

    Attributes
    ----------
    _files : list of file object
        One open PDB file per bead.
    """

    def __init__(self, file_base_name, reportInterval, num_beads, topology):
        """Initialize the reporter.

        Parameters
        ----------
        file_base_name : str
            Prefix for files (e.g., 'output' -> 'output_bead_0.pdb').
        reportInterval : int
            How often to write frames (steps).
        num_beads : int
            Number of beads in the RPMD integrator.
        topology : Topology
            The system topology.
        """
        self._reportInterval = reportInterval
        self._num_beads = num_beads
        self._topology = topology
        self._next_frame_index = 0

        self._files = []
        for i in range(num_beads):
            filename = f"{file_base_name}_bead_{i}.pdb"
            f = open(filename, 'w')
            app.PDBFile.writeHeader(topology, f)
            self._files.append(f)

    def report(self, simulation, state):
        """Write the current positions of every bead, one file each.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation requesting the report.
        state : openmm.State
            Current state of the simulation. Unused; bead positions are read
            from the RPMD integrator instead, since the state holds only the
            centroid.

        Returns
        -------
        None

        Notes
        -----
        Files are flushed every tenth frame so an interrupted run still
            leaves readable trajectories.
        """
        integrator = simulation.integrator

        for i in range(self._num_beads):
            bead_state = integrator.getState(i, getPositions=True, enforcePeriodicBox=True)
            positions = bead_state.getPositions()

            app.PDBFile.writeModel(self._topology, positions, self._files[i], self._next_frame_index)

            if self._next_frame_index % 10 == 0:
                self._files[i].flush()

        self._next_frame_index += 1

    def __del__(self):
        """Write the PDB footer to every bead file and close them.

        Returns
        -------
        None
        """
        _close_pdb_files(self._topology, self._files)


class RPMDCentroidReporter(_RPMDReporterBase):
    """Write the bead-averaged positions to a single PDB file.

    The centroid is the closest thing a ring polymer has to a classical
    configuration, which makes it the sensible trajectory to visualise or
    to feed to analysis tools that expect one particle per atom.

    Attributes
    ----------
    _num_beads : int
        Number of beads averaged over at each report.
    """

    def __init__(self, file_name, reportInterval, num_beads, topology):
        """Initialise the reporter and open the output file.

        Parameters
        ----------
        file_name : str
            Path of the PDB file to write.
        reportInterval : int
            Steps between frames.
        num_beads : int
            Number of beads in the RPMD integrator, used to average over.
        topology : openmm.app.Topology
            Topology written into the PDB header.

        Returns
        -------
        None
        """
        self._reportInterval = reportInterval
        self._num_beads = num_beads
        self._topology = topology
        self._next_frame_index = 0
        self._out = open(file_name, 'w')
        app.PDBFile.writeHeader(topology, self._out)

    def report(self, simulation, state):
        """Write the bead-averaged positions as one PDB model.

        The centroid is the closest thing the ring polymer has to a
        classical configuration, so it is what gets visualised.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The simulation requesting the report.
        state : openmm.State
            Current state of the simulation. Unused; bead positions are read
            from the RPMD integrator instead, since the state holds only the
            centroid.

        Returns
        -------
        None
        """
        integrator = simulation.integrator

        # asNumpy=True for vector efficiency, though OpenMM Quantities also support math
        sum_pos = integrator.getState(0, getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)

        for i in range(1, self._num_beads):
            pos = integrator.getState(i, getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
            sum_pos += pos

        centroid_pos = sum_pos / self._num_beads

        app.PDBFile.writeModel(self._topology, centroid_pos, self._out, self._next_frame_index)
        self._next_frame_index += 1

        if self._next_frame_index % 10 == 0:
            self._out.flush()

    def __del__(self):
        """Write the PDB footer and close the file.

        Returns
        -------
        None
        """
        _close_pdb_files(self._topology, [self._out])


def count_dna_and_estimate_charge(topology):
    """Estimate the net charge of the DNA in a topology.

    Counts nucleotide residues and assumes one negative charge each,
    which is the charge of the phosphate backbone at neutral pH. Used to
    work out how many counter-ions a system needs.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology to scan.

    Returns
    -------
    int
        The estimated net charge in units of the elementary charge.

    Notes
    -----
    Recognises internal, 5'-terminal and 3'-terminal residue names. Only
        DNA is counted, so RNA or a phosphorylated protein will not be
        accounted for.
    """
    dna_residue_names = {
        "DA", "DC", "DG", "DT",  # internal
        "DA5", "DC5", "DG5", "DT5",  # 5'-terminal
        "DA3", "DC3", "DG3", "DT3",  # 3'-terminal
    }

    num_dna_residues = 0

    for residue in topology.residues():
        if residue.name.strip() in dna_residue_names:
            num_dna_residues += 1

    estimated_charge = -num_dna_residues

    return estimated_charge
