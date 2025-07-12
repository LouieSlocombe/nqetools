import os

from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile
from ase.calculators.socketio import SocketClient
from ase.io import read

# Define atoms object

atoms = read("init.xyz", 0)

orca_path = os.environ.get('ORCA_PATH')
# Create an ORCA profile with the specified command
profile = OrcaProfile(command=orca_path)

calc = ORCA(profile=profile, orcasimpleinput="B3LYP D3BJ def2-SVP TightSCF EnGrad")

atoms.calc = calc

# Create Client
port = 10200
host = "localhost"
client = SocketClient(host=host, port=port)

client.run(atoms)
