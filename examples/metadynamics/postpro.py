from ase import units
from ase.calculators.idealgas import IdealGas
from ase.calculators.plumed import Plumed
from ase.io import read


def plumed_postpro(file_traj, setup, timestep, temperature):
    """
    Perform post-processing using PLUMED on a given trajectory file.

    Parameters:
    file_traj (str): The name of the trajectory file to read.
    setup (list): The PLUMED input setup.
    timestep (float): The timestep for the simulation in femtoseconds.
    temperature (float): The temperature for the simulation in Kelvin.

    Returns:
    None
    """
    # Read the trajectory from the file
    traj = read(file_traj, index=':')
    # Get the first frame of the trajectory
    atoms = traj[0]
    # Set up the PLUMED calculator with the IdealGas model and the provided parameters
    calc = Plumed(calc=IdealGas(),
                  input=setup,
                  timestep=timestep,
                  atoms=atoms,
                  kT=temperature * units.kB)
    # Write the PLUMED files for the trajectory
    calc.write_plumed_files(traj)
    return None


traj = read('UnbiasMD.traj', index=':')

atoms = traj[0]

timestep = 0.005
temperature = 0.1 / units.kB
ps = 1000 * units.fs
setup = [f"UNITS LENGTH=A TIME={1 / ps} ENERGY={units.mol / units.kJ}",
         "c1: COORDINATIONNUMBER SPECIES=1-7 MOMENTS=2-3" +
         " SWITCH={RATIONAL R_0=1.5 NN=8 MM=16}",
         "PRINT ARG=c1.* STRIDE=100 FILE=COLVAR_postpro1",
         "FLUSH STRIDE=1000"]

plumed_postpro('UnbiasMD.traj', setup, timestep, temperature)
