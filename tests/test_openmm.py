import math
import os
import re
from collections import defaultdict
from pathlib import Path
from sys import stdout
import numpy as np
import MDAnalysis as mda
import openmm.app as app
import openmm.unit as unit
from openff.toolkit import Molecule, ForceField, Topology
from openff.units import Quantity, unit as off_unit
from openmm import openmm
from openmmforcefields.generators import SMIRNOFFTemplateGenerator, GAFFTemplateGenerator, SystemGenerator
from openmmml import MLPotential
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem

import nqetools as nqe


def create_ligand_xml(mol2_path, frcmod_path, xml_out_path="ligand.xml"):
    """
    Parses AMBER .mol2 and .frcmod files to create a custom OpenMM XML
    force field file for a new ligand.

    This script is generalized to handle MASS, BOND, ANGLE, DIHE,
    IMPROPER, and NONBON sections from the .frcmod file,
    performing the necessary unit conversions for OpenMM.
    """

    # --- Constants for Unit Conversion ---
    KCAL_TO_KJ = 4.184
    DEG_TO_RAD = math.pi / 180.0
    ANGSTROM_TO_NM = 0.1
    # 1/(Å^2) to 1/(nm^2) -> 1 / (0.1*nm)^2 = 1 / (0.01 * nm^2) = 100 / nm^2
    KCAL_A2_TO_KJ_NM2 = KCAL_TO_KJ * 100.0
    # AMBER Rmin/2 -> OpenMM sigma
    # Rmin = 2 * (Rmin/2)
    # sigma = Rmin / (2^(1/6))
    # sigma_nm = (2 * Rmin_half_A / (2**(1/6))) * 0.1
    RMIN_HALF_A_TO_SIGMA_NM = (2.0 / (2.0 ** (1.0 / 6.0))) * ANGSTROM_TO_NM

    # --- 1. Parse .mol2 file ---
    atoms = []
    bonds = []
    residue_name = "LIG"  # Default
    try:
        with open(mol2_path, 'r') as f:
            mol2_lines = f.readlines()

        in_atom_section = False
        in_bond_section = False
        atom_id_map = {}  # Maps 1-based mol2 ID to 0-based XML index

        line_iter = iter(mol2_lines)  # Create an iterator

        while True:
            try:
                line = next(line_iter)
            except StopIteration:
                break  # End of list

            if line.startswith("@<TRIPOS>MOLECULE"):
                # Try to get residue name from the line after
                try:
                    residue_name = next(line_iter).strip()  # Get next line from iterator
                except StopIteration:
                    pass  # Reached end of file, just use default name
                continue

            if line.startswith("@<TRIPOS>BOND"):
                in_atom_section = False
                in_bond_section = True
                continue

            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_section = True
                continue

            if line.startswith("@<TRIPOS>"):
                # End of a section
                in_atom_section = False
                in_bond_section = False
                continue

            if in_atom_section:
                parts = line.split()
                if len(parts) < 6:
                    continue

                atom_id = int(parts[0])
                atom_name = parts[1]
                atom_type = parts[5]
                # Use residue name from mol2 if available
                if len(parts) > 7 and residue_name == "LIG":
                    residue_name = parts[7]
                charge = float(parts[8])

                # Store atom info and create mapping
                atom_index = len(atoms)  # 0-based index
                atoms.append({
                    "name": atom_name,
                    "type": atom_type,
                    "charge": charge
                })
                atom_id_map[atom_id] = atom_index

            if in_bond_section:
                parts = line.split()
                if len(parts) < 4:
                    continue

                atom1_id = int(parts[1])
                atom2_id = int(parts[2])

                # Convert 1-based mol2 IDs to 0-based XML indices
                bonds.append({
                    "from": atom_id_map[atom1_id],
                    "to": atom_id_map[atom2_id]
                })

    except Exception as e:
        print(f"Error parsing {mol2_path}: {e}")
        return

    print(f"Parsed {len(atoms)} atoms and {len(bonds)} bonds from {mol2_path} for residue '{residue_name}'.")

    # --- 2. Parse .frcmod file ---
    mass_params = []
    bond_params = []
    angle_params = []
    # Use defaultdict to group dihedrals by atom types
    dihe_params = defaultdict(list)
    improper_params = []
    nonbon_params = []

    # Define which .frcmod sections are supported by this parser
    SUPPORTED_SECTIONS = {"MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON"}

    try:
        with open(frcmod_path, 'r') as f:
            frcmod_lines = f.readlines()

        current_section = None
        for line_number, line in enumerate(frcmod_lines, 1):
            line = line.split('#')[0].strip()  # Remove comments and strip whitespace
            if not line:
                continue

            # Check for section headers
            is_header = False
            if line.startswith("MASS"):
                current_section = "MASS"
                is_header = True
            elif line.startswith("BOND"):
                current_section = "BOND"
                is_header = True
            elif line.startswith("ANGLE"):
                current_section = "ANGLE"
                is_header = True
            elif line.startswith("DIHE"):
                current_section = "DIHE"
                is_header = True
            elif line.startswith("IMPROPER"):
                current_section = "IMPROPER"
                is_header = True
            elif line.startswith("NONBON"):
                current_section = "NONBON"
                is_header = True
            elif re.match(r'^[A-Z_]+$', line):  # Check for other potential headers
                # This is a potential header line that we don't recognize
                if line not in SUPPORTED_SECTIONS:
                    raise ValueError(
                        f"Error: Unsupported .frcmod section '{line}' found "
                        f"at line {line_number} in {frcmod_path}.\n"
                        "This script only supports: "
                        f"{', '.join(SUPPORTED_SECTIONS)}"
                    )
                else:
                    # It's a supported section we just didn't list above
                    current_section = line
                    is_header = True

            if is_header:
                continue

            # Skip empty lines that might follow a header
            if not current_section:
                continue

            # Parse lines based on the current section
            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                if current_section == "MASS":
                    # Ex: ca 12.01
                    mass_params.append({
                        "type": parts[0],
                        "mass": float(parts[1])
                    })

                elif current_section == "BOND":
                    # Ex: ca-ca 620.0 1.390
                    atom_types = parts[0].split('-')
                    bond_params.append({
                        "t1": atom_types[0],
                        "t2": atom_types[1],
                        "k_kj_nm2": float(parts[1]) * KCAL_A2_TO_KJ_NM2,
                        "len_nm": float(parts[2]) * ANGSTROM_TO_NM
                    })

                elif current_section == "ANGLE":
                    # Ex: ca-ca-ca 80.0 120.00
                    atom_types = parts[0].split('-')
                    angle_params.append({
                        "t1": atom_types[0],
                        "t2": atom_types[1],
                        "t3": atom_types[2],
                        "k_kj_rad2": float(parts[1]) * KCAL_TO_KJ,  # Amber k is kcal/mol/rad^2
                        "angle_rad": float(parts[2]) * DEG_TO_RAD
                    })

                elif current_section == "DIHE":
                    # Ex: X -c -c -X 4 14.500 180.0 2.0
                    # OpenMM combines terms, so we group them
                    atom_types = parts[0].split('-')
                    if len(atom_types) != 4:
                        atom_types = [parts[0], parts[1], parts[2], parts[3]]

                    num_terms = int(parts[4])  # AMBER IDIVF
                    k = float(parts[5])  # AMBER PK / IDIVF
                    phase_deg = float(parts[6])
                    period = abs(float(parts[7]))  # AMBER PN

                    # This is a simplified parser. AMBER's IDIVF
                    # means divide PK by this. OpenMM wants k per term.
                    # This logic assumes PK is already the per-term k.
                    # A more robust parser would handle multi-line dihedrals.
                    # For `parmchk` output, this is usually fine.

                    key = (atom_types[0], atom_types[1], atom_types[2], atom_types[3])
                    dihe_params[key].append({
                        "k_kj": k * KCAL_TO_KJ,  # AMBER Vn/2 is in kcal/mol
                        "phase_rad": phase_deg * DEG_TO_RAD,
                        "period": int(period)
                    })

                elif current_section == "IMPROPER":
                    # Ex: c -cc-n -hn 1.1 180.0 2.0
                    atom_types = parts[0].split('-')
                    if len(atom_types) != 4:
                        atom_types = [parts[0], parts[1], parts[2], parts[3]]

                    improper_params.append({
                        "t1": atom_types[0],
                        "t2": atom_types[1],
                        "t3": atom_types[2],
                        "t4": atom_types[3],
                        "k_kj": float(parts[-3]) * KCAL_TO_KJ,  # AMBER Vn/2 is in kcal/mol
                        "phase_rad": float(parts[-2]) * DEG_TO_RAD,
                        "period": int(float(parts[-1]))
                    })

                elif current_section == "NONBON":
                    # Ex: ca 1.9080 0.0860
                    nonbon_params.append({
                        "type": parts[0],
                        "sigma_nm": float(parts[1]) * RMIN_HALF_A_TO_SIGMA_NM,
                        "epsilon_kj": float(parts[2]) * KCAL_TO_KJ
                    })

            except Exception as e:
                print(f"Skipping unparsed line in section {current_section}: {line} (Error: {e})")
                continue

    except Exception as e:
        print(f"Error parsing {frcmod_path}: {e}")
        return

    print(f"Parsed parameters from {frcmod_path}:")
    print(f"  MASS: {len(mass_params)}, NONBON: {len(nonbon_params)}")
    print(f"  BOND: {len(bond_params)}, ANGLE: {len(angle_params)}")
    print(f"  DIHE: {len(dihe_params)}, IMPROPER: {len(improper_params)}")

    # --- 3. Write .xml file ---
    try:
        with open(xml_out_path, 'w') as f:
            f.write("<ForceField>\n\n")

            # --- AtomTypes ---
            f.write(" <AtomTypes>\n")
            if not mass_params and not nonbon_params:
                f.write("  <!-- All atom types are assumed to be in the base force field -->\n")

            # Write new mass definitions
            for p in mass_params:
                # Find matching nonbon to write a complete <Atom> tag
                nb = next((nb for nb in nonbon_params if nb["type"] == p["type"]), None)
                if nb:
                    f.write(
                        f'  <Type name="{p["type"]}" class="{p["type"]}" element="{_guess_element(p["type"])}" mass="{p["mass"]:.4f}"/>\n')
                else:
                    # Mass only (less common, but possible)
                    f.write(f'  <!-- WARNING: MASS found for {p["type"]} but no NONBON. -->\n')
                    f.write(
                        f'  <Type name="{p["type"]}" class="{p["type"]}" element="{_guess_element(p["type"])}" mass="{p["mass"]:.4f}"/>\n')

            # Write nonbon-only definitions (if mass was in base FF)
            for p in nonbon_params:
                mass = next((m for m in mass_params if m["type"] == p["type"]), None)
                if not mass:
                    f.write(
                        f'  <!-- WARNING: NONBON found for {p["type"]} but no MASS. Assuming mass is in base FF. -->\n')
                    f.write(
                        f'  <Type name="{p["type"]}" class="{p["type"]}" element="{_guess_element(p["type"])}" mass="?.????"/>\n')

            f.write(" </AtomTypes>\n\n")

            # --- NonbondedForce (LJ parameters) ---
            f.write(" <NonbondedForce coulomb14scale=\"0.833333\" lj14scale=\"0.5\">\n")
            if not nonbon_params:
                f.write("  <!-- All LJ parameters are assumed to be in the base force field -->\n")
            for p in nonbon_params:
                f.write(f'  <Atom type="{p["type"]}" sigma="{p["sigma_nm"]:.6f}" epsilon="{p["epsilon_kj"]:.6f}"/>\n')
            f.write(" </NonbondedForce>\n\n")

            # --- Residues ---
            f.write(" <Residues>\n")
            f.write(f'  <Residue name="{residue_name}">\n')

            # Write atoms
            for atom in atoms:
                f.write(f'   <Atom name="{atom["name"]}" type="{atom["type"]}" charge="{atom["charge"]:.6f}" />\n')

            # Write bonds
            for bond in bonds:
                f.write(f'   <Bond from="{bond["from"]}" to="{bond["to"]}" />\n')

            f.write("  </Residue>\n")
            f.write(" </Residues>\n\n")

            # --- Bond Parameters ---
            f.write(" <HarmonicBondForce>\n")
            if not bond_params:
                f.write("  <!-- All bond parameters are assumed to be in the base force field -->\n")
            for p in bond_params:
                f.write(
                    f'  <Bond type1="{p["t1"]}" type2="{p["t2"]}" length="{p["len_nm"]:.6f}" k="{p["k_kj_nm2"]:.2f}"/>\n')
            f.write(" </HarmonicBondForce>\n\n")

            # --- Angle Parameters ---
            f.write(" <HarmonicAngleForce>\n")
            if not angle_params:
                f.write("  <!-- All angle parameters are assumed to be in the base force field -->\n")
            for p in angle_params:
                f.write(
                    f'  <Angle type1="{p["t1"]}" type2="{p["t2"]}" type3="{p["t3"]}" angle="{p["angle_rad"]:.6f}" k="{p["k_kj_rad2"]:.2f}"/>\n')
            f.write(" </HarmonicAngleForce>\n\n")

            # --- Torsion Parameters (Proper and Improper) ---
            f.write(" <PeriodicTorsionForce>\n")

            # Write Proper Dihedrals
            if not dihe_params:
                f.write("  <!-- All proper torsions are assumed to be in the base force field -->\n")
            for (t1, t2, t3, t4), terms in dihe_params.items():
                line = f'  <Proper type1="{t1}" type2="{t2}" type3="{t3}" type4="{t4}"'
                for i, term in enumerate(terms, 1):
                    line += f' periodicity{i}="{term["period"]}" phase{i}="{term["phase_rad"]:.6f}" k{i}="{term["k_kj"]:.6f}"'
                line += " />\n"
                f.write(line)

            # Write Improper Dihedrals
            if not improper_params:
                f.write("  <!-- All improper torsions are assumed to be in the base force field -->\n")
            for imp in improper_params:
                f.write(f'  <Improper type1="{imp["t1"]}" type2="{imp["t2"]}" type3="{imp["t3"]}" type4="{imp["t4"]}"')
                f.write(f' periodicity1="{imp["period"]}" phase1="{imp["phase_rad"]:.6f}" k1="{imp["k_kj"]:.6f}" />\n')

            f.write(" </PeriodicTorsionForce>\n\n")

            f.write("</ForceField>\n")

    except Exception as e:
        print(f"Error writing {xml_out_path}: {e}")
        return

    print(f"\nSuccessfully created {xml_out_path}!")
    print("You can now load this in OpenMM with your base force field:")
    print(f"forcefield = ForceField('amber/gaff.xml', '{xml_out_path}')")


def _guess_element(atom_type):
    """A simple heuristic to guess element from atom type for the XML."""
    # This is just for XML completeness; OpenMM uses mass.
    element = atom_type[0].upper()
    if len(atom_type) > 1 and atom_type[1].islower():
        element = atom_type[0:2].upper()  # e.g., 'cl' -> 'CL'

    # Handle common ambiguities
    if element == 'H' and atom_type.lower().startswith('hc'):
        element = 'H'
    elif element == 'C' and atom_type.lower().startswith('ca'):
        element = 'C'  # Not calcium

    # Strip numbers
    element = re.sub(r'\d.*', '', element)
    return element


def rdkit_to_openff_manual(rdmol: Chem.Mol) -> Molecule:
    rdkit_bond_type_map = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 1.5,
    }
    rdkit_bond_stereo_map = {
        Chem.BondStereo.STEREOCIS: 'Cis',
        Chem.BondStereo.STEREOE: 'E',
        Chem.BondStereo.STEREOTRANS: 'Trans',
        Chem.BondStereo.STEREOZ: 'Z',
        Chem.BondStereo.STEREOANY: None,
        Chem.BondStereo.STEREONONE: None,
    }
    rdkit_chiral_tag_map = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 'S',
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: 'R',
        Chem.ChiralType.CHI_UNSPECIFIED: None,
    }

    Chem.AssignStereochemistryFrom3D(rdmol)
    AllChem.DetectBondStereochemistry(rdmol)

    offmol = Molecule()
    for rdatom in rdmol.GetAtoms():
        atomic_num = rdatom.GetAtomicNum()
        formal_charge = rdatom.GetFormalCharge()
        is_aromatic = rdatom.GetIsAromatic()
        chiral_tag = rdatom.GetChiralTag()
        atom_stereo = rdkit_chiral_tag_map.get(chiral_tag, None)
        offmol.add_atom(
            atomic_number=atomic_num,
            formal_charge=formal_charge,
            is_aromatic=is_aromatic,
            stereochemistry=atom_stereo
        )
    for rdbond in rdmol.GetBonds():
        idx1 = rdbond.GetBeginAtomIdx()
        idx2 = rdbond.GetEndAtomIdx()
        is_aromatic = rdbond.GetIsAromatic()
        rd_bond_type = rdbond.GetBondType()
        bond_order = rdkit_bond_type_map.get(rd_bond_type)
        if bond_order is None:
            raise ValueError(f"Unsupported RDKit bond type: {rd_bond_type}")
        rd_bond_stereo = rdbond.GetStereo()
        bond_stereo = rdkit_bond_stereo_map.get(rd_bond_stereo, None)
        offmol.add_bond(
            atom1=idx1,
            atom2=idx2,
            bond_order=bond_order,
            is_aromatic=is_aromatic,
            stereochemistry=bond_stereo
        )
    if rdmol.GetNumConformers() > 0:
        conformer = rdmol.GetConformer(0)
        offmol.add_conformer(conformer.GetPositions() * 1.0)
    return offmol


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
    pdbfile = app.PDBFile(input_pdb)

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
    integrator = openmm.LangevinIntegrator(
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
    print(flush=True)
    input_pdb = "tests/data/pdb/gt_wob_pol.pdb"
    clean_pdb = "gt_wob_pol_clean.pdb"

    rm_ions = ['Na+', 'Cl-', 'NA']
    residue_map = {'DGN': 'DG', 'DTN': 'DT', 'GTP': 'LIG'}

    smi = '[H]-[O]-[C@@]1(-[H])-[C](-[H])(-[H])-[C@](-[H])(-[N]2-[CH](-[H])-[NH]-[C@@H]3-[CH]-2-[NH]-[C@H](-[N](-[H])-[H])-[N](-[H])-[C@H]-3-[OH])-[O]-[C@]-1(-[H])-[C](-[H])(-[H])-[O]-[P-](-[O-])(-[O-])-[O]-[P-](-[O-])(-[O-])-[O]-[P-](-[O-])(-[O-])-[O-]'
    smi = 'C1=NC2=C(N1[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])O)O)N=C(NC2=O)N'
    mol = Chem.MolFromSmiles(smi)
    lig_mol = rdkit_to_openff_manual(mol)
    print(lig_mol.n_bonds, lig_mol.n_atoms)

    nqe.clean_ions_in_pdb(input_pdb, rm_ions, clean_pdb)
    nqe.relabel_residues_in_pdb(clean_pdb, residue_map, clean_pdb)
    nqe.remove_water_residues_in_pdb(clean_pdb, clean_pdb)

    non_standard_mols = nqe.get_non_standard_residues(clean_pdb)
    # write sdf files for each non-standard residue
    for i, mol in enumerate(non_standard_mols):
        print(f"Non-standard residue {i}: {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds")
        print(Chem.MolToSmiles(mol, isomericSmiles=True, allBondsExplicit=True, allHsExplicit=True, canonical=True))
        Chem.MolToMolFile(mol, f"non_standard_{i}.sdf", kekulize=False)
    #
    # # nqe.fix_pdb(clean_pdb, clean_pdb)
    # lig_mol = Molecule.from_file('non_standard_0.sdf', allow_undefined_stereo=True, file_format="SDF")
    # print(lig_mol.n_bonds, lig_mol.n_atoms, lig_mol.properties)
    #
    # lig_mol = Molecule.from_rdkit(non_standard_mols[0], allow_undefined_stereo=True)
    # print(lig_mol.n_bonds, lig_mol.n_atoms, lig_mol.properties)
    #
    #
    # lig_mol = Molecule.from_smiles(smi, allow_undefined_stereo=False)
    # print(lig_mol.n_bonds, lig_mol.n_atoms)
    #
    # gaff = GAFFTemplateGenerator(molecules=lig_mol)
    # forcefield = app.ForceField(
    #     'amber/ff14SB.xml',
    # )
    # forcefield.registerTemplateGenerator(gaff.generator)
    #
    # pdbfile = app.PDBFile("gt_wob_pol_clean.pdb")
    # system = forcefield.createSystem(pdbfile.topology)


def fix_pdb(file_in, file_out, ph=7.0):
    fixer = PDBFixer(filename=file_in)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    app.PDBFile.writeFile(fixer.topology, fixer.positions, open(file_out, 'w'))
    return None


def make_sdf(pdb_file, lig_name='LIG'):
    u = mda.Universe(pdb_file)
    elements = mda.topology.guessers.guess_types(u.atoms.names)
    u.add_TopologyAttr('elements', elements)
    lig = u.select_atoms(f"resname {lig_name}")
    mol = lig.convert_to("RDKIT")
    # write to sdf file
    Chem.MolToMolFile(mol, f"{lig_name}.sdf", kekulize=False)


def insert_molecule_and_remove_clashes(
        topology: Topology,
        insert: Molecule,
        ligand: Molecule,
        radius: Quantity = 1.5 * off_unit.angstrom,
        keep: list[Molecule] = [],
) -> Topology:
    # We'll collect the molecules for the output topology into a list
    new_top_mols = []
    # A molecule's positions in a topology are stored as its zeroth conformer
    insert_coordinates = insert.conformers[0][:, None, :]
    for molecule in topology.molecules:
        if any(keep_mol.is_isomorphic_with(molecule) for keep_mol in keep):
            new_top_mols.append(molecule)
            continue
        molecule_coordinates = molecule.conformers[0][None, :, :]
        diff_matrix = molecule_coordinates - insert_coordinates

        # np.linalg.norm doesn't work on Pint quantities 😢
        working_unit = off_unit.nanometer
        distance_matrix = np.linalg.norm(diff_matrix.m_as(working_unit), axis=-1) * working_unit

        if distance_matrix.min() > radius:
            # This molecule is not clashing, so add it to the topology
            new_top_mols.append(molecule)
        else:
            print(f"Removed {molecule.to_smiles()} molecule")

    # Insert the ligand at the end
    new_top_mols.append(ligand)

    # This pattern of assembling a topology from a list of molecules
    # ends up being much more efficient than adding each molecule
    # to a new topology one at a time
    new_top = Topology.from_molecules(new_top_mols)

    # Don't forget the box vectors!
    new_top.box_vectors = topology.box_vectors
    return new_top


def test_mdanalysis():
    print(flush=True)
    input_pdb = "tests/data/pdb/gt_wob_pol.pdb"
    clean_pdb = "gt_wob_pol_clean.pdb"
    stripped_pdb = "gt_wob_pol_stripped.pdb"

    rm_ions = ['Na+', 'Cl-', 'NA']
    residue_map = {'DGN': 'DG', 'DTN': 'DT', 'GTP': 'LIG'}

    nqe.remove_water_residues_in_pdb(input_pdb, clean_pdb)
    nqe.clean_ions_in_pdb(clean_pdb, rm_ions, clean_pdb)
    nqe.relabel_residues_in_pdb(clean_pdb, residue_map, clean_pdb)
    fix_pdb(clean_pdb, stripped_pdb)

    print("Loading protein...")
    pdb = app.PDBFile(stripped_pdb)
    pdb_topology = pdb.topology
    pdb_positions = pdb.positions
    print(f"Loaded PDB with {pdb_topology.getNumAtoms()} atoms.")

    # make the ligand sdf
    make_sdf(clean_pdb, lig_name='LIG')
    molecule = Molecule.from_file('LIG.sdf')
    molecule.generate_conformers(n_conformers=1)
    molecule.assign_partial_charges(partial_charge_method='am1bcc',
                                    use_conformers=molecule.conformers)


    ligand_ff_topology = molecule.to_topology()
    ligand_omm_topology = ligand_ff_topology.to_openmm()
    ligand_positions = ligand_ff_topology.get_positions().to_openmm()
    print(f"Loaded SDF with {ligand_omm_topology.getNumAtoms()} atoms.")

    modeller = app.Modeller(pdb_topology, pdb_positions)
    modeller.add(ligand_omm_topology, ligand_positions)
    combined_topology = modeller.topology
    combined_positions = modeller.positions
    print(f"Combined system has {combined_topology.getNumAtoms()} total atoms.")

    print("Saving combined PDB file...")
    with open('combined_system.pdb', 'w') as f:
        app.PDBFile.writeFile(combined_topology, combined_positions, f)


    with open('combined_system.pdb', 'r') as f:
        pdb_data = f.read()
    # open the file and replace 'x' with ' ' to fix formatting issues
    pdb_data = pdb_data.replace('x', ' ')
    # replace UNK with LIG
    pdb_data = pdb_data.replace('UNK', 'LIG')
    with open('combined_system_fixed.pdb', 'w') as f:
        f.write(pdb_data)

    forcefield = app.ForceField("amber/ff14SB.xml")
    gaff = GAFFTemplateGenerator(molecules=molecule, cache="gaff-molecules.json", forcefield='gaff-2.2.20')
    forcefield.registerTemplateGenerator(gaff.generator)
    pdbfile = app.PDBFile('combined_system_fixed.pdb')
    system = forcefield.createSystem(pdbfile.topology)



    # molecule = Molecule.from_file('LIG.sdf')
    #
    # # Create an OpenFF Topology object from the molecule
    #
    # topology = Topology.from_molecules(molecule)
    #
    # # Load the latest OpenFF force field release: version 2.1.0, codename "Sage"
    #
    # forcefield = ForceField('openff-2.1.0.offxml')
    #
    # # Create an OpenMM system representing the molecule with SMIRNOFF-applied parameters
    # openmm_system = forcefield.create_openmm_system(topology)
    #
    # # Create an Interchange object for representations in other formats
    # interchange = forcefield.create_interchange(topology)

    # ### 2. Load the Ligand (from SDF)
    # print("Loading ligand...")
    # # This loads the ligand and its 3D coordinates from the SDF
    # ligand_molecule = Molecule.from_file('LIG.sdf')
    #
    #
    # protein_off_topology = OpenFFTopology.from_openmm(
    #     protein_topology,
    #     unique_molecules=[ligand_molecule]
    # )

    # u = mda.Universe(clean_pdb)
    # elements = mda.topology.guessers.guess_types(u.atoms.names)
    # u.add_TopologyAttr('elements', elements)
    # lig = u.select_atoms("resname LIG")
    # mol = lig.convert_to("RDKIT")
    # # get the charge of the molecule
    # print(Chem.GetFormalCharge(mol))
    # # get the number of atoms and bonds
    # print(mol.GetNumAtoms(), mol.GetNumBonds())
    # # print the formula of the molecule
    # print(Chem.rdMolDescriptors.CalcMolFormula(mol))
    # # Get the smiles of the molecule
    # print(Chem.MolToSmiles(mol, isomericSmiles=True, allHsExplicit=True))
    #
    # # write to sdf file
    # Chem.MolToMolFile(mol, "gtp.sdf", kekulize=False)
    #
    # img = Draw.MolToImage(mol, size=(700, 500))
    # img.save("ligand_rdkit.png")
    # # draw 2d structure of the molecule
    # rdDepictor.Compute2DCoords(mol)
    # rdDepictor.StraightenDepiction(mol)
    # img = Draw.MolToImage(mol, size=(700, 500))
    # img.save("ligand_rdkit_2d.png")
    #
    # lig_mol = Molecule.from_rdkit(mol)
    # print(lig_mol.n_atoms, lig_mol.n_bonds)
    # # get the smiles of the molecule
    # print(lig_mol.to_smiles(isomeric=True, explicit_hydrogens=True))
    #
    # gaff = GAFFTemplateGenerator(molecules=lig_mol)
    # forcefield = app.ForceField(
    #     'amber/ff14SB.xml',
    # )
    # forcefield.registerTemplateGenerator(gaff.generator)
    #
    # pdbfile = app.PDBFile("gt_wob_pol_clean.pdb")
    # system = forcefield.createSystem(pdbfile.topology)


def test_create_ligand_xml():
    mol2_file = 'gtp.mol2'
    frcmod_file = 'gtp.frcmod'
    output_xml = 'gtp.xml'

    create_ligand_xml(mol2_file, frcmod_file, output_xml)

    # 1. Load the PDB file containing the protein AND the GTP
    # Note: The GTP residue must be named "GTP" in this PDB
    pdb = app.PDBFile('sqm.pdb')

    # 2. Load the force fields
    forcefield = app.ForceField('gtp.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,  # or app.NoCutoff for vacuum
        nonbondedCutoff=1.0 * unit.nanometer,  # ignored if NoCutoff
        constraints=app.HBonds  # bond-length constraints to H (not positional!)
    )


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
