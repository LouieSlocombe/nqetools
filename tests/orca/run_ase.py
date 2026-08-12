"""Standalone ORCA i-PI client script.

Run as a driver alongside an i-PI server; connects over a socket and
returns ORCA energies and forces.
"""

import os

from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile
from ase.calculators.socketio import SocketClient
from ase.io import read

atoms = read("init.xyz", 0)

orca_path = os.environ.get('ORCA_PATH')
profile = OrcaProfile(command=orca_path)

calc = ORCA(profile=profile, orcasimpleinput="B3LYP D3BJ def2-SVP TightSCF EnGrad")

atoms.calc = calc

port = 10200
host = "localhost"
client = SocketClient(host=host, port=port)

client.run(atoms)
