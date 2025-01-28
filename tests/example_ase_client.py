from ase.calculators.emt import EMT

from ase.calculators.socketio import SocketClient
from ase.io import read

import nqetools as nqe

# The atomic numbers are not transferred over the socket, so we have to
# read the file
atoms = read('initial.traj')
unixsocket = 'driver'


calc = EMT()
# calc = nqe.orca_calc_preset()
atoms.calc = calc

client = SocketClient(unixsocket=unixsocket)

# Each step of the loop changes the atomic positions, but the generator
# yields None.
for i, _ in enumerate(client.irun(atoms, use_stress=False)):
    print('step:', i)