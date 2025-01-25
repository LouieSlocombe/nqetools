from ase.calculators.emt import EMT

from ase.calculators.socketio import SocketClient
from ase.io import read

# The atomic numbers are not transferred over the socket, so we have to
# read the file
atoms = read('initial.traj')
unixsocket = 'ase_server_socket'


calc = EMT()

atoms.calc = calc

client = SocketClient(unixsocket=unixsocket)

# Each step of the loop changes the atomic positions, but the generator
# yields None.
for i, _ in enumerate(client.irun(atoms, use_stress=False)):
    print('step:', i)