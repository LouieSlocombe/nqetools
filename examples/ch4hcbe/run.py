#%%
import os

import chemiscope
import ipi
import numpy as np
from ase.visualize import view
from matplotlib import pyplot as plt
import nqetools as nqe
# This follows:
# https://github.com/i-pi/piqm2023-tutorial/blob/main/05-RPI/tutorial-4.ipynb
#%%
#UNITS:
invcm2au = 4.5563353e-06  #in i-PI unit_to_internal("frequency", "inversecm", 1.0)
kelvin2au = 3.1668152e-06


def get_freq_from_eigvals(eigvals):
    """Transforms eigenvalues in atomic units to frequencies in cm^-1"""
    freq_invcm = np.zeros(eigvals.shape)
    for i, eig in enumerate(eigvals):
        freq_invcm[i] = np.sign(eig) * np.absolute(eig) ** 0.5 / invcm2au
    return freq_invcm


# Paths
directory_opti = 'opti'
directory_phonon_react = 'phonon_react'
directory_ts = 'ts'
directory_phonon_ts = 'phonon_ts'
directory_instanton = 'instanton'

# Driver
driver_code = 'cbe'

# Values
temperature = 300.0
n_beads = 10
tol_energy = 5.0e-6
tol_force = 5.0e-6
tol_position = 1.0e-6
total_steps = 1000
optimizer = "cg"

#%%
atoms = nqe.read_ipi_xyz("react.xyz")[-1]
n_atoms = len(atoms)
#%%
# Run minimization
output = nqe.run_optimise(directory_opti,
                          atoms,
                          driver=driver_code)
atoms_opti, output_data_opti, output_desc_opti = output
#%%
# Plot the energy of the minimisation
nqe.plot_step_energy(output_data_opti, save=False)
#%%
nqe.run_phonons(directory_phonon_react,
                atoms_opti,
                driver=driver_code)
#%%
eigvals = np.genfromtxt(os.path.join(directory_phonon_react, 'phonon.phonons.eigval'))
fr = get_freq_from_eigvals(eigvals)

plt.plot(fr, 'o')
plt.xlabel('Vibrational Mode  Index')
plt.ylabel('Frequency (cm$^{-1}$)')
plt.show()
#%%
nqe.run_instanton_post_process(directory_phonon_react,
                               process_type='reactant',
                               temperature=temperature,
                               filter_list=n_atoms-1)
#%%
data=nqe.parse_react_thermo_data(directory_phonon_react)
print(data)
#%%
output = nqe.run_ts(directory_ts,
                    atoms_opti,
                    driver=driver_code)
atoms_ts, output_data_ts, output_desc_ts = output
#%%
# Plot the energy of the minimisation
nqe.plot_step_energy(output_data_ts, save=False)
#%%
rate = nqe.calc_forward_rate(n_atoms, directory_ts, directory_phonon_react, temperature)
print(rate)
#%%
nqe.run_phonons(directory_phonon_ts,
                atoms_ts,
                driver=driver_code)
#%%
eigvals = np.genfromtxt(os.path.join(directory_phonon_ts, 'phonon.phonons.eigval'))
fr = get_freq_from_eigvals(eigvals)

plt.plot(fr, 'o')
plt.xlabel('Vibrational Mode  Index')
plt.ylabel('Frequency (cm$^{-1}$)')
plt.show()
#%%
# Run the instanton
nqe.run_instanton(directory_instanton,
                  atoms_ts,
                  directory_ts,
                  driver=driver_code,
                  n_beads=n_beads,
                  temperature=temperature)
#%%
nqe.calc_kappa_full(directory_ts,
                    directory_instanton,
                    temperature,
                    n_beads)
#%%
# Try and converge over beads
list_n_beads = [10, 20, 40, 60, 80]
list_kappa = []

for n_beads in list_n_beads:
    print(f"Running with {n_beads} beads")
    nqe.run_instanton(directory_instanton,
                      atoms_ts,
                      directory_ts,
                      driver=driver_code,
                      n_beads=n_beads,
                      temperature=temperature)
    kappa = nqe.calc_kappa_full(directory_ts,
                                directory_instanton,
                                temperature,
                                n_beads)
    list_kappa.append(kappa)


#%%
nqe.plot_bead_convergence(list_n_beads, list_kappa)
#%%
# Converge over temperature
list_temperature = [100, 200, 300, 400, 500]
list_kappa = []
n_beads = 20

for temperature in list_temperature:
    print(f"Running with {temperature} K")
    nqe.run_instanton(directory_instanton,
                      atoms_ts,
                      directory_ts,
                      driver=driver_code,
                      n_beads=n_beads,
                      temperature=temperature)
    kappa = nqe.calc_kappa_full(directory_ts,
                                directory_instanton,
                                temperature,
                                n_beads)
    list_kappa.append(kappa)
#%%
print(list_kappa)
plt.plot(list_temperature, list_kappa, 'o-')
plt.xlabel('Temperature (K)')
plt.ylabel('Kappa')
plt.show()
#%%
nqe.plot_kappa_temperature(list_temperature, list_kappa)
#%%
