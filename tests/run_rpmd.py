#!/usr/bin/env python3
from pathlib import Path
from sys import stdout

from openmm import openmm, app, unit
from openmmml import MLPotential

import nqetools as nqe

if __name__ == "__main__":
    # Simple run parameters
    n_steps = 1_000
    report_every = 100
    in_pdb = "data/pdb/input_aaa.pdb"
    out_pdb = Path("centroid_trajectory.pdb")

    # --- RPMD integrator ---
    n_beads = 2
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond

    padding = 1.5
    box_shape = 'dodecahedron'  # 'dodecahedron' 'cubic'

    pdb = app.PDBFile(in_pdb)
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens(forcefield)

    # Solvate
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)

    n_atoms = modeller.topology.getNumAtoms()
    print(f"System has {n_atoms} atoms.")

    # Determine whether we have a periodic box (PME) or vacuum (CutoffNonPeriodic)
    has_box = modeller.topology.getUnitCellDimensions() is not None

    # system = forcefield.createSystem(
    #     modeller.topology,
    #     nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
    #     nonbondedCutoff=1.0 * unit.nanometer,
    #     constraints=None,  # <-- no bond/angle constraints app.HBonds
    #     rigidWater=False,  # <-- make waters flexible (otherwise they stay rigid)
    #     removeCMMotion=True,
    # )

    potential = MLPotential('mace-off23-small')
    system = potential.createSystem(modeller.topology)

    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    integrator.setApplyThermostat(True)
    integrator.setRandomNumberSeed(2025)

    # Platform, CPU/GPU selection
    platform = openmm.Platform.getPlatformByName("CUDA")  # CUDA CPU
    # props = {
    #     # Use both GPU 0 and GPU 1 for a single Context
    #     "DeviceIndex": "0,1",
    #     # Commonly fastest/accurate tradeoff
    #     "Precision": "mixed",
    #     # Keep PME on the GPU when using CUDA (generally best)
    #     "UseCpuPme": "True",
    # }
    simulation = app.Simulation(modeller.topology, system, integrator, platform)  # props

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      100,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      speed=True))

    data_out = 'md_log.txt'
    simulation.reporters.append(app.StateDataReporter(data_out,
                                                      100,
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
