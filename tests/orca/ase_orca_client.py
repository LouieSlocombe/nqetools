#!/usr/bin/env python3
import os
from ase.calculators.orca import ORCA, OrcaProfile
from ase.calculators.socketio import SocketClient
from ase.io import read

unixsocket = "orca_ase"

orca_path = os.environ.get('ORCA_PATH')
profile = OrcaProfile(command=orca_path)
orca_calc = ORCA(
    profile=profile,
    orcasimpleinput="PBE def2-SVP EnGrad",
    charge=0,
    mult=1,
)
atoms = read("water.xyz", index=0)
atoms.calc = orca_calc
client = SocketClient(unixsocket=unixsocket)
client.irun(atoms, use_stress=True)
