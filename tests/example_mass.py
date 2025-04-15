from ase.build import molecule

from ase.io import write

# build H2
h2 = molecule('H2')
# Set the mass
h2.set_masses([1.0, 2.0])
# write the molecule to a file
write('h2.xyz', h2)
