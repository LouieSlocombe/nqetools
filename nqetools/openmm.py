import numpy as np
from openmm import openmm, app, unit
from pdbfixer import PDBFixer


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


def centroid_positions(simulation, n_atoms, n_beads):
    acc = np.zeros((n_atoms, 3), dtype=float)
    for b in range(n_beads):
        state = simulation.integrator.getState(b, getPositions=True)
        r = state.getPositions(asNumpy=True)
        acc += r.value_in_unit(unit.nanometer)
    acc /= n_beads
    return [openmm.Vec3(*acc[i]) for i in range(n_atoms)] * unit.nanometer


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
