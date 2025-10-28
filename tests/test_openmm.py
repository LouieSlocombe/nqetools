from sys import stdout

from openff.toolkit import Molecule
from openmm import openmm, app, unit
from openmmforcefields.generators import GAFFTemplateGenerator
from openmmml import MLPotential


def test_openmm_ml():
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")

    potential = MLPotential('mace-off23-small')

    system = potential.createSystem(pdb.topology)

    # Run langevin dynamics at 300K for 1000 steps
    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin, 1.0 / unit.picoseconds, 1.0 * unit.femtosecond
    )
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
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
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    # Solvate
    padding = 1.5
    box_shape = 'dodecahedron'  # 'dodecahedron' 'cubic'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)


    mm_system = forcefield.createSystem(modeller.topology)
    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]
    potential = MLPotential('mace-off23-small')
    system = potential.createMixedSystem(modeller.topology, mm_system, ml_atoms)

    # Run langevin dynamics at 300K for 1000 steps
    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin, 1.0 / unit.picoseconds, 1.0 * unit.femtosecond
    )
    simulation = app.Simulation(modeller.topology, system, integrator)
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
    # Create an OpenFF Molecule object for benzene from SMILES

    molecule = Molecule.from_smiles("c1ccccc1")
    # Create the GAFF template generator

    # molecules = Molecule.from_file("molecules.sdf")

    gaff = GAFFTemplateGenerator(molecules=molecule)
    # Create an OpenMM ForceField object with AMBER ff14SB and TIP3P with compatible ions
    forcefield = app.ForceField(
        "amber/protein.ff14SB.xml",
        "amber/tip3p_standard.xml",
        "amber/tip3p_HFE_multivalent.xml",
    )
    # Register the GAFF template generator
    forcefield.registerTemplateGenerator(gaff.generator)


def test_openmm_rpmd():
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens(forcefield)

    modeller.addSolvent(forcefield,
                        padding=2.0 * unit.nanometer)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
    )

    # # Run langevin dynamics at 300K for 1000 steps
    # integrator = openmm.LangevinIntegrator(300 * unit.kelvin,
    #                                        1.0 / unit.picoseconds,
    #                                        1.0 * unit.femtosecond)

    integrator = openmm.RPMDIntegrator(2,
                                       300 * unit.kelvin,
                                       1.0 / unit.picoseconds,
                                       1.0 * unit.femtosecond)

    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
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
