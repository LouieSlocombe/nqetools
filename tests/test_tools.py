import os
import sys
import time
from subprocess import Popen

import ase.build
from ase.build import molecule
from ase.calculators.nwchem import NWChem
from ase.io import write
from ase.optimize import BFGS
from mace.calculators import mace_anicc, mace_off
from ase.calculators.socketio import PySocketIOClient, SocketIOCalculator
import nqetools as nqe
from ase.calculators.emt import EMT
from ase.build import molecule

def test_cluster_atoms():
    print(flush=True)
    # Create an example Atoms object (for instance, two separate water molecules)

    water = molecule("H2O")
    # Translate the second water molecule so it does not overlap with the first.
    water2 = water.copy()
    water2.translate([5, 0, 0])

    # Combine into a single Atoms object.
    combined = water + water2

    # Cluster the atoms.
    clusters = nqe.cluster_atoms(combined)
    print(f"Found {len(clusters)} clusters:", flush=True)
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1} with {len(cluster)} atoms", flush=True)
    assert len(clusters[0]) == 3
    assert len(clusters[1]) == 3
    assert len(clusters) == 2