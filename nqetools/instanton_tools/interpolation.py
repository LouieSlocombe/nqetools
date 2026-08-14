"""Interpolate an instanton geometry and Hessian onto more beads.

Instanton optimisations are normally converged by stepping up the number
of ring-polymer beads. This script takes a converged low-bead result and
resamples it onto a finer discretisation, so the next optimisation starts
close to its solution rather than from scratch.

Input comes either from an i-PI checkpoint, or, in manual mode, from an
explicit XYZ geometry and Hessian file. Two files are written:
``new_instanton.xyz`` and ``new_hessian.dat``.

This is a command-line script, executed on import rather than imported for
its functions. It relies on i-PI's normal-mode machinery, so the ``ipi``
package must be importable.

Notes
-----
Bead counts throughout refer to the half polymer. The instanton orbit is
symmetric about its turning points, so only half of it is stored; the full
ring is reconstructed by mirroring before any normal-mode transform.

Examples
--------
::

    python interpolation.py -chk RESTART -n 30
    python interpolation.py -m -xyz INSTANTON.xyz -hess INSTANTON.hess -n 30
"""

import argparse
import os
import sys

import numpy as np
from ipi.utils.io import read_file, print_file
from ipi.utils.nmtransform import nm_rescale
from ipi.utils.units import unit_to_internal

parser = argparse.ArgumentParser(
    description="""Script for interpolate hessian and/or instanton geometry"""
)
parser.add_argument(
    "-m",
    "--manual",
    action="store_true",
    default=False,
    help="Boolean which decides between a checkpoint file or a manual entry.",
)
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="None",
    help="Name of the instanton checkpoint file.",
)
parser.add_argument(
    "-xyz",
    "--xyz",
    type=str,
    default="None",
    help="Name of the instanton geometry file.",
)
parser.add_argument(
    "-hess", "--hessian", type=str, default="None", help="Name of the hessian file."
)
parser.add_argument(
    "-n",
    "--nbeadsNew",
    required=True,
    default=0,
    help="New number of beads (half polymer)",
    type=int,
)

args = parser.parse_args()
chk = args.checkpoint
input_geo = args.xyz
input_hess = args.hessian
nbeadsNew = args.nbeadsNew
manual = args.manual

if not manual:
    if chk == "None":
        print("Manual mode not specified and checkpoint file name not provided")
        sys.exit()
else:
    if input_geo == "None":
        print("Manual mode  specified and geometry file name not provided")
        sys.exit()

if input_geo != "None" or chk != "None":
    if manual:
        if os.path.exists(input_geo):
            ipos = open(input_geo)
        else:
            print(f"We can't find {input_geo}", flush=True)
            sys.exit()

        pos = []
        nbeads = 0
        while True:
            try:
                ret = read_file("xyz", ipos)
                pos.append(ret["atoms"])
                cell = ret["cell"]
                nbeads += 1
            except EOFError:  # finished reading files
                break
        ipos.close()

        natoms = pos[0].natoms
        atom = pos[0]
        q = np.vstack([i.q for i in pos])
    else:
        from ipi.engine.simulation import Simulation

        if os.path.exists(chk):
            simulation = Simulation.load_from_xml(
                open(chk), custom_verbosity="low", request_banner=False, read_only=True
            )
        else:
            print(f"We can't find {chk}", flush=True)
            sys.exit()
        cell = simulation.syslist[0].cell
        beads = simulation.syslist[0].motion.beads.clone()
        natoms = simulation.syslist[0].motion.beads.natoms
        nbeads = beads.nbeads
        q = beads.q
        atom = beads._blist[0]

    print(" ", flush=True)
    print(
        f"We have a half ring polymer made of {nbeads} beads and {natoms} atoms.",
        flush=True,
    )
    print(
        f"We will expand the ring polymer to get a half polymer of {nbeadsNew} beads.",
        flush=True,
    )

    # Mirror to the full ring before rescaling: the closed-path transform is
    # better behaved than the open-path one in corner cases
    q2 = np.concatenate((q, np.flipud(q)), axis=0)
    rpc = nm_rescale(2 * nbeads, 2 * nbeadsNew)
    new_q = rpc.b1tob2(q2)[0:nbeadsNew]

    out = open("new_instanton.xyz", "w")
    for i in range(nbeadsNew):
        atom.q = new_q[i] / unit_to_internal(
            "length", "angstrom", 1.0
        )  # i-PI works in atomic units; XYZ is written in angstrom
        print_file(
            "xyz", atom, cell, out, title="cell{atomic_unit}  Traj: positions{angstrom}"
        )
    out.close()

    print("The new Instanton geometry (half polymer) was generated", flush=True)
    print("Check new_instanton.xyz", flush=True)
    print("", flush=True)
    print(
        f"Don't forget to change the number of beads to the new value ({nbeadsNew}) in your input file",
        flush=True,
    )
    print(
        "when starting your new simulation with an increased number of beads.",
        flush=True,
    )
    print("", flush=True)

if input_hess != "None" or chk != "None":
    if manual:
        try:
            hess = open(input_hess)
        except OSError:
            print(f"We can't find {input_hess}", flush=True)
            sys.exit()
        h = np.zeros((natoms * 3) ** 2 * nbeads)
        aux = hess.readline().split()

        for i in range((natoms * 3) ** 2 * nbeads):
            h[i] = float(aux[i])
        h = h.reshape((natoms * 3, natoms * 3 * nbeads))
        hess.close()

    else:
        from ipi.engine.simulation import Simulation

        try:
            h = simulation.syslist[0].motion.optarrays["hessian"].copy()
        except (KeyError, AttributeError):
            print("We don't have a hessian so there is nothing more to do", flush=True)
            sys.exit()
        if np.linalg.norm(h) < 1e-13:
            print("We don't have a hessian so there is nothing more to do", flush=True)
            sys.exit()

    print(f"The new hessian is {3 * natoms} x {natoms * 3 * nbeadsNew}.", flush=True)
    out = open("new_hessian.dat", "w")

    print("Creating matrix... ", flush=True)

    hessian = h
    size0 = natoms * 3

    size1 = size0 * (2 * nbeads)
    size2 = size0 * (2 * nbeadsNew)
    new_h = np.zeros([size0, size2])
    q2 = np.concatenate((q, np.flipud(q)), axis=0)
    rpc = nm_rescale(2 * nbeads, 2 * nbeadsNew)
    new_q = rpc.b1tob2(q2)[0:nbeadsNew]

    # Each (i, j) element is interpolated independently along the bead index,
    # mirrored to the full ring in the same way as the positions above
    for i in range(size0):
        for j in range(size0):
            h = np.array([])
            for n in range(nbeads):
                h = np.append(h, hessian[i, j + size0 * n])
            h2 = np.concatenate((h, np.flipud(h)), axis=0)
            diag = rpc.b1tob2(h2)
            new_h[i, j:size2:size0] += diag

    new_h_half = new_h[:, 0 : size2 // 2]
    np.savetxt(out, new_h_half.reshape(1, new_h_half.size))

    print("The new physical Hessian (half polymer) was generated", flush=True)
    print("Check new_hessian.dat", flush=True)
    print("", flush=True)
    print("Remeber to adapt/add the following line in your input file:", flush=True)
    print("", flush=True)
    print(
        f" <hessian mode='file' shape='({3 * natoms}, {natoms * 3 * nbeadsNew})' >hessian.dat</hessian>",
        flush=True,
    )
    print("", flush=True)

sys.exit()
