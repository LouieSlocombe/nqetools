import os
import sys

from ase.calculators.socketio import SocketClient
from ase.io import read
from ase.calculators.cp2k import CP2K
from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile
from ase.calculators.emt import EMT

# Define atoms object

atoms = read("init.xyz", 0)

# Set CP2K calculator #################

workdir = "CP2Ktest"
aux_settings = {"label": workdir}


orca_path = os.environ.get('ORCA_PATH')
# Create an ORCA profile with the specified command
profile = OrcaProfile(command=orca_path)

calc = ORCA(profile=profile,orcasimpleinput="B3LYP D3BJ def2-SVP TightSCF EnGrad")
# calc = CP2K(**aux_settings)

atoms.calc = calc#EMT()#calc

print(atoms.get_potential_energy())
print(atoms.get_forces())

# Create Client
# inet
port = 10200
host = "localhost"
client = SocketClient(host=host, port=port)

client.run(atoms)
