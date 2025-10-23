from sys import stdout
import openmm
import openmm.app as app
import openmm.unit as unit

from openmmml import MLPotential


if __name__ == "__main__":
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
