from sys import stdout
import openmm
import openmm.app as app
import openmm.unit as unit

from openmmml import MLPotential
from openff.toolkit import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator

from rdkit import Chem
from rdkit.Chem import AllChem

def test_openmm_ml():
    # Load toluene structure
    pdb = app.PDBFile("data/pdb/input_aaa.pdb")

    potential = MLPotential(
        'mace-off23-large'
    )

    system = potential.createSystem(pdb.topology)

    # Run langevin dynamics at 300K for 1000 steps
    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin, 1.0 / unit.picoseconds, 1.0 * unit.femtosecond
    )
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.reporters.append(app.DCDReporter("output.dcd", 100))
    simulation.reporters.append(
        app.StateDataReporter(
            stdout, 100, step=True, potentialEnergy=True, temperature=True, speed=True
        )
    )

    # Minimize the energy
    simulation.minimizeEnergy()

    # Set the velocities to 300K and run 1000 steps
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
    simulation.step(1_000)


def test_openmm_ml_mixed_system():
    pass





def test_openmm_ff_param():
    # Create an OpenFF Molecule object for benzene from SMILES


    molecule = Molecule.from_smiles("c1ccccc1")
    # Create the GAFF template generator

    molecules = Molecule.from_file("molecules.sdf")


    gaff = GAFFTemplateGenerator(molecules=molecule)
    # Create an OpenMM ForceField object with AMBER ff14SB and TIP3P with compatible ions


    forcefield = app.ForceField(
        "amber/protein.ff14SB.xml",
        "amber/tip3p_standard.xml",
        "amber/tip3p_HFE_multivalent.xml",
    )
    # Register the GAFF template generator
    forcefield.registerTemplateGenerator(gaff.generator)
    # You can now parameterize an OpenMM Topology object that contains the specified molecule.
    # forcefield will load the appropriate GAFF parameters when needed, and antechamber
    # will be used to generate small molecule parameters on the fly.
    from openmm.app import PDBFile

    pdbfile = PDBFile("t4-lysozyme-L99A-with-benzene.pdb")
    system = forcefield.createSystem(pdbfile.topology)


def test_openmm_rpmd():
    pass
