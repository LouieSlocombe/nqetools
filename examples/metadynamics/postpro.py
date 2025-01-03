from ase import units
from ase.calculators.idealgas import IdealGas
from ase.calculators.plumed import Plumed
from ase.io import read


def plumed_postpro(file_traj, setup, timestep, temperature):
    traj = read(file_traj, index=':')
    atoms = traj[0]
    calc = Plumed(calc=IdealGas(),
                  input=setup,
                  timestep=timestep,
                  atoms=atoms,
                  kT=temperature * units.kB)
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
