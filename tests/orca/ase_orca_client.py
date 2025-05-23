#!/usr/bin/env python3
"""
Tiny ASE client for i-PI that evaluates energies + forces with ORCA.
Usage:  python ase_orca_client.py [socket_name]
If you omit the argument it defaults to "orca_ase" so that it matches
the <address> tag in the XML.
"""
import os

from ase.build import molecule  # only to make a placeholder Atoms object
from ase.calculators.orca import ORCA  # ASE ↔ ORCA interface
from ase.calculators.orca import OrcaProfile
from ase.calculators.socketio import SocketClient  # i-PI protocol client


unixsocket = "orca_ase"


orca_path = os.environ.get('ORCA_PATH')
profile = OrcaProfile(command=orca_path)
orca_calc = ORCA(
    profile=profile,
    label="orca_run",
    #orcasimpleinput="PBE def2-SVP",
    charge=0,
    mult=1,
)

atoms = molecule("H2O")  # any three-atom placeholder is fine
atoms.calc = orca_calc

client = SocketClient(unixsocket=unixsocket)
client.run(atoms)  # blocks; i-PI will control the loop
