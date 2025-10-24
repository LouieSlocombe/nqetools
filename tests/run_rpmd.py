#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from openmm import openmm, app, unit


def gaussian_velocities(n_atoms, temperature):
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA  # kJ/mol/K
    return [openmm.Vec3(0, 0, 0) for _ in range(n_atoms)] * (unit.nanometer / unit.picosecond)


def write_multimodel_pdb(topology, positions, fh, model_index):
    app.PDBFile.writeModel(topology, positions, fh, modelIndex=model_index)


def centroid_positions(integrator, n_beads, n_atoms):
    acc = np.zeros((n_atoms, 3), dtype=float)
    for b in range(n_beads):
        state = integrator.getState(b, getPositions=True)
        r = state.getPositions(asNumpy=True)
        acc += r.value_in_unit(unit.nanometer)
    acc /= n_beads
    return [openmm.Vec3(*acc[i]) for i in range(n_atoms)] * unit.nanometer


def main(pdb_path):
    pdb = app.PDBFile(pdb_path)
    topo = pdb.topology
    pos0 = pdb.positions

    # Try a commonly available force field (standard AA + water/ions).
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    # Determine whether we have a periodic box (PME) or vacuum (CutoffNonPeriodic)
    has_box = topo.getUnitCellDimensions() is not None
    system = ff.createSystem(
        topo,
        nonbondedMethod=app.PME if has_box else app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,  # <-- no bond/angle constraints
        rigidWater=False,  # <-- make waters flexible (otherwise they stay rigid)
        removeCMMotion=True,
    )

    # --- RPMD integrator ---
    n_beads = 64
    temperature = 300 * unit.kelvin
    friction = 1.0 / unit.picosecond
    dt = 0.5 * unit.femtosecond
    integrator = openmm.RPMDIntegrator(n_beads, temperature, friction, dt)
    integrator.setApplyThermostat(True)
    integrator.setRandomNumberSeed(2025)

    # Platform (pick what you have; CPU works everywhere)
    platform = openmm.Platform.getPlatformByName("CUDA")
    context = openmm.Context(system, integrator, platform)

    # Initialize each bead with the input coordinates + tiny random jiggle
    n_atoms = topo.getNumAtoms()
    rng = np.random.default_rng(0)
    for b in range(n_beads):
        jiggle = 0.002 * rng.normal(size=(n_atoms, 3))  # ~0.002 nm perturbation
        bead_pos = []
        for i, p in enumerate(pos0):
            bead_pos.append(openmm.Vec3(p.x + jiggle[i, 0],
                                        p.y + jiggle[i, 1],
                                        p.z + jiggle[i, 2]))
        integrator.setPositions(b, bead_pos * unit.nanometer)
        integrator.setVelocities(b, gaussian_velocities(n_atoms, temperature))

    # Simple run parameters
    n_steps = 20000
    report_every = 100
    out_pdb = Path("centroid_trajectory.pdb")

    # Prepare multi-MODEL PDB (centroid coordinates)
    with open(out_pdb, "w") as fh:
        app.PDBFile.writeHeader(topo, fh)
        # write initial centroid (model 0)
        centroid = centroid_positions(integrator, n_beads, n_atoms)
        write_multimodel_pdb(topo, centroid, fh, model_index=0)

        # Integrate and save snapshots
        for step in range(1, n_steps + 1):
            integrator.step(1)

            if step % report_every == 0:
                # energies: total ring-polymer energy
                E = integrator.getTotalEnergy().value_in_unit(unit.kilojoule_per_mole)
                print(f"step {step:6d} | total RPMD energy: {E:12.3f} kJ/mol")

                centroid = centroid_positions(integrator, n_beads, n_atoms)
                write_multimodel_pdb(topo, centroid, fh, model_index=step // report_every)

        app.PDBFile.writeFooter(topo, fh)

    print(f"\nWrote centroid trajectory to: {out_pdb.resolve()}")


if __name__ == "__main__":
    main("data/pdb/input_aaa.pdb")
