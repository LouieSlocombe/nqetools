#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from openmm import openmm, app, unit
from pdbfixer import PDBFixer
from openmmml import MLPotential

from sys import stdout

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


def zero_velocities(n_atoms):
    return [openmm.Vec3(0, 0, 0) for _ in range(n_atoms)] * (unit.nanometer / unit.picosecond)


def write_multimodel_pdb(topology, positions, fh, model_index):
    app.PDBFile.writeModel(topology, positions, fh, modelIndex=model_index)


def centroid_positions(integrator, n_atoms):
    n_beads = integrator.getNumCopies()
    acc = np.zeros((n_atoms, 3), dtype=float)
    for b in range(n_beads):
        state = integrator.getState(b, getPositions=True)
        r = state.getPositions(asNumpy=True)
        acc += r.value_in_unit(unit.nanometer)
    acc /= n_beads
    return [openmm.Vec3(*acc[i]) for i in range(n_atoms)] * unit.nanometer


def init_beads(modeller, integrator, perturb=0.002):
    rng = np.random.default_rng(0)
    pos0 = modeller.positions
    n_atoms = len(pos0)
    n_beads = integrator.getNumCopies()
    for b in range(n_beads):
        jiggle = perturb * rng.normal(size=(n_atoms, 3))
        bead_pos = [openmm.Vec3(p.x + dx, p.y + dy, p.z + dz)
                    for p, (dx, dy, dz) in zip(pos0, jiggle)]
        integrator.setPositions(b, bead_pos * unit.nanometer)
        integrator.setVelocities(b, zero_velocities(n_atoms))


if __name__ == "__main__":
    # Simple run parameters
    n_steps = 1_000
    report_every = 100
    in_pdb = "data/pdb/input_aaa.pdb"
    out_pdb = Path("centroid_trajectory.pdb")

    # --- RPMD integrator ---
    n_beads = 4
    temperature = 300.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond

    padding = 2.0
    box_shape = 'dodecahedron'  # 'dodecahedron' 'cubic'

    pdb = app.PDBFile(in_pdb)
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens(forcefield)

    # # Solvate
    # modeller.addSolvent(forcefield,
    #                     padding=padding * unit.nanometer,
    #                     boxShape=box_shape)

    n_atoms = modeller.topology.getNumAtoms()
    print(f"System has {n_atoms} atoms.")

    # Determine whether we have a periodic box (PME) or vacuum (CutoffNonPeriodic)
    has_box = modeller.topology.getUnitCellDimensions() is not None
    print(has_box)
    # system = forcefield.createSystem(
    #     modeller.topology,
    #     nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
    #     nonbondedCutoff=1.0 * unit.nanometer,
    #     constraints=None,  # <-- no bond/angle constraints app.HBonds
    #     rigidWater=False,  # <-- make waters flexible (otherwise they stay rigid)
    #     removeCMMotion=True,
    # )

    potential = MLPotential(
        'mace-off23-small'
    )
    system = potential.createSystem(pdb.topology)


    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    integrator.setApplyThermostat(True)
    integrator.setRandomNumberSeed(2025)

    # # Platform, CPU/GPU selection
    # platform = openmm.Platform.getPlatformByName("CUDA")  # CUDA CPU
    # context = openmm.Context(system, integrator, platform)
    simulation = app.Simulation(modeller.topology, system, integrator)

    simulation.reporters.append(app.StateDataReporter(stdout,
                                                      100,
                                                      step=True,
                                                      potentialEnergy=True,
                                                      temperature=True))

    # Initialize each bead with the input coordinates + tiny random jiggle
    init_beads(modeller, integrator)

    # Prepare multi-MODEL PDB (centroid coordinates)
    with open(out_pdb, "w") as fh:
        app.PDBFile.writeHeader(modeller.topology, fh)
        # write initial centroid (model 0)
        centroid = centroid_positions(integrator, n_atoms)
        write_multimodel_pdb(modeller.topology, centroid, fh, model_index=0)

        # Integrate and save snapshots
        for step in range(1, n_steps + 1):
            integrator.step(1)

            if step % report_every == 0:
                # E = integrator.getTotalEnergy().value_in_unit(unit.kilojoule_per_mole)
                # print(f"step {step:6d} | total RPMD energy: {E:12.3f} kJ/mol")

                centroid = centroid_positions(integrator, n_atoms)
                write_multimodel_pdb(modeller.topology, centroid, fh, model_index=step // report_every)

        app.PDBFile.writeFooter(modeller.topology, fh)
    print(f"\nWrote centroid trajectory to: {out_pdb.resolve()}")
