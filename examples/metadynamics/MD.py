from ase import units
from ase.calculators.lj import LennardJones
from ase.calculators.plumed import Plumed
from ase.constraints import FixedPlane
from ase.io import read
from ase.md.langevin import Langevin

# Define the timestep for the molecular dynamics simulation in femtoseconds
timestep = 0.005  # in fs

# Define the conversion factor for picoseconds
ps = 1000 * units.fs

# Input and output file names
file_input = 'isomer.xyz'
file_output = 'UnbiasMD.traj'

# Define the temperature for the simulation in Kelvin
temperature = 0.1 / units.kB

# Define the friction coefficient for the Langevin dynamics
friction = 1.0

# Number of timesteps for the simulation
nt = 100000

# Set up the Lennard-Jones calculator with specified cutoff and equilibrium distance
calc = LennardJones(rc=2.5, r0=3.0)

# Define the PLUMED input setup
setup = [f"UNITS LENGTH=A TIME={1 / ps} ENERGY={units.mol / units.kJ}",
         "c1: COORDINATIONNUMBER SPECIES=1-7 MOMENTS=2-3" +
         " SWITCH={RATIONAL R_0=1.5 NN=8 MM=16}",
         "PRINT ARG=c1.* STRIDE=100 FILE=COLVAR",
         "FLUSH STRIDE=1000"]

# Read the atomic structure from the input file
atoms = read(file_input)

# Apply a fixed plane constraint to the first seven atoms
cons = [FixedPlane(i, [0, 0, 1]) for i in range(7)]
atoms.set_constraint(cons)

# Set the masses of the atoms
atoms.set_masses([1, 1, 1, 1, 1, 1, 1])

# Set up the PLUMED calculator with the defined setup and parameters
atoms.calc = Plumed(calc=calc,
                    input=setup,
                    timestep=timestep,
                    atoms=atoms,
                    kT=temperature * units.kB)

# Set up the Langevin dynamics with the defined parameters
dyn = Langevin(atoms,
               timestep,
               temperature_K=temperature,
               friction=friction,
               fixcm=False,
               trajectory=file_output)

# Run the molecular dynamics simulation for the specified number of timesteps
dyn.run(nt)
