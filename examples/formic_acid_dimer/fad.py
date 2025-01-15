from ase.visualize import view
from ase.io import read

atoms = read('fad.xyz', -1)
view(atoms)

