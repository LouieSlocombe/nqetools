import os

import ipi
import matplotlib.pyplot as plt
import numpy as np
from ase.units import kB
from scipy.constants import k, h
from scipy.optimize import curve_fit

from .execution import run_instanton_post_process
from .plotting import n_plot


def calc_kappa(ts_path, instanton_path, temperature, n_beads):
    """
    Calculates the tunnelling factor (kappa) for a given transition state and instanton path.

    Parameters:
    ts_path (str): Path to the transition state data.
    instanton_path (str): Path to the instanton data.
    temperature (float): Temperature in Kelvin.
    n_beads (int): Number of beads.

    Returns:
    float: The computed tunnelling factor (kappa).
    """
    # Use run_instanton_post_process to get TS data
    ts_data = run_instanton_post_process(
        ts_path,
        process_type="TS",
        temperature=temperature,
        n_beads=n_beads
    )

    # Use run_instanton_post_process to get instanton data
    instanton_data = run_instanton_post_process(
        instanton_path,
        process_type="instanton",
        temperature=temperature
    )

    # Extract required values from the processed data
    q_trn_ts = ts_data["Q_trn"]
    q_rot_ts = ts_data["Q_rot"]
    q_vib_ts = ts_data["Q_vib"]
    beta_times_v = ts_data["Beta_times_V"]

    q_trn_inst = instanton_data["Q_trn"]
    q_rot_inst = instanton_data["Q_rot"]
    q_vib_inst = instanton_data["Q_vib"]
    bn = instanton_data["BN"]
    s_over_hbar = instanton_data["S_over_hbar"]

    # Compute kappa directly
    kelvin2au = 3.1668152e-06
    beta = 1.0 / (temperature * kelvin2au)
    f_trn = q_trn_inst / q_trn_ts
    f_rot = q_rot_inst / q_rot_ts
    f_vib = np.sqrt((2. * np.pi * n_beads * bn) / (beta * 1.0 ** 2)) * q_vib_inst / q_vib_ts

    kappa = f_trn * f_rot * f_vib * np.exp(-s_over_hbar + beta_times_v)

    # Printing out the transmission factor and the relevant contributions
    print('f_tra               = {:5.3f}'.format(f_trn), flush=True)
    print('f_rot               = {:5.3f}'.format(f_rot), flush=True)
    print('f_vib               = {:5.3f}'.format(f_vib), flush=True)
    print('exp(-S/hbar+V/beta) = {:5.3f}'.format(np.exp(-s_over_hbar + beta_times_v)), flush=True)
    print('=============================', flush=True)
    print('Tunnelling factor   = {:5.3f}'.format(kappa), flush=True)

    return kappa


def parse_react_thermo_data(directory,
                            filename='thermo_data.out',
                            ref_filename='phonon.out'):
    data = {}  # Initialize an empty dictionary to store the extracted data.
    temp = None  # Initialize temperature variable.
    # Extract the energy from the output data
    output_data, output_desc = ipi.read_output(os.path.join(directory, ref_filename))
    data['energy'] = output_data.get('potential', None)[-1]
    if data['energy'] is None:
        raise ValueError(f"Could not find potential energy in {ref_filename} in {directory}")

    filepath = os.path.join(directory, filename)  # Construct the full file path.

    # Open the file and read all lines.
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Iterate through the lines to find and extract the required data.
    for i, line in enumerate(lines):
        if 'Qtras(bohr^-3) | Qrot     | logQvib_rp' in line:
            # Extract the next line containing the values.
            data_line = lines[i + 1].strip()
            values = data_line.split('|')
            # Parse and store the values in the dictionary.
            data['Qtras'] = float(values[0].strip())
            data['Qrot'] = float(values[1].strip())
            data['logQvib_rp'] = float(values[2].strip())
        elif 'Temperature:' in line:
            # Extract the temperature from the line.
            temp = float(line.split()[1])
    if temp is None:
        raise ValueError(f"Could not find temperature in {filepath}")

    data['V/kBT'] = float(data['energy'] / (kB * temp))  # Calculate V/kBT.
    # Raise an error if no data was extracted.
    if not data:
        raise ValueError(f"Could not find reactant thermodynamic data in {filepath}")

    return data  # Return the extracted data as a dictionary.


def parse_ts_thermo_data(directory, filename='thermo_data.out'):
    """
    Parse the TS thermo_data.out file and extract thermodynamic values.

    Args:
        directory (str): Directory containing the thermo_data file
        filename (str): Name of the thermo data file (default: 'thermo_data.out')

    Returns:
        dict: Dictionary containing Qtras, Qrot, logQvib, and V/kBT values
    """
    filepath = os.path.join(directory, filename)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {}

    for line in lines:
        if 'Qtras:' in line:
            data['Qtras'] = float(line.split(':')[1].strip())
        elif 'Qrot:' in line:
            data['Qrot'] = float(line.split(':')[1].strip())
        elif 'logQvib:' in line:
            data['logQvib'] = float(line.split(':')[1].strip())
        elif 'V/kBT' in line:
            data['V/kBT'] = float(line.split()[-1].strip())

    if not data:
        raise ValueError(f"Could not find thermodynamic data in {filepath}")

    return data


def parse_inst_thermo_data(directory, filename='thermo_data.out'):
    """
    Parse the thermo_data.out file and extract thermodynamic values.

    Args:
        directory (str): Directory containing the thermo_data file
        filename (str): Name of the thermo data file (default: 'thermo_data.out')

    Returns:
        dict: Dictionary containing the thermodynamic values including Temperature, NBEADS, and 1/(betaP*hbar)
    """
    filepath = os.path.join(directory, filename)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {}

    # Find the needed values from the file
    for i, line in enumerate(lines):
        if 'Temperature:' in line:
            data['Temperature'] = float(line.split()[1])
        elif 'NBEADS:' in line:
            data['NBEADS'] = int(line.split()[1])
        elif '1/(betaP*hbar)' in line:
            data['1/(betaP*hbar)'] = float(line.split('=')[1].strip())
        elif 'BN' in line and 'Qt' in line:
            # Get the next line which contains the values
            data_line = lines[i + 1]
            # Split the line and clean up the values
            values = data_line.split('|')
            bn_parts = values[0].strip().replace('(', '').replace(')', '').split()

            # Add the remaining data
            data.update({
                'BN': float(bn_parts[0]),
                'Qt': float(values[1].strip()),
                'Qrot': float(values[2].strip()),
                'log(Qvib*N)': float(values[3].strip()),
                'S/hbar': float(values[4].split('(')[0].strip()),
            })
            return data

    raise ValueError(f"Could not find thermodynamic data in {filepath}")


def calc_instanton_kappa(ts_data, inst_data):
    """
    Calculate the tunnelling factor (kappa) from TS and instanton thermodynamic data.

    Args:
        ts_data (dict): Dictionary containing TS thermodynamic data
        inst_data (dict): Dictionary containing instanton thermodynamic data

    Returns:
        float: Tunnelling factor (kappa)
    """
    # Extract partition functions and other parameters
    f_trn = inst_data['Qt'] / ts_data['Qtras']
    f_rot = inst_data['Qrot'] / ts_data['Qrot']

    # Calculate f_vib using BN and temperature
    beta = 1.0 / (inst_data['Temperature'] * 3.1668152e-06)  # kelvin2au = 3.1668152e-06
    f_vib = np.sqrt((2.0 * np.pi * inst_data['NBEADS'] * inst_data['BN']) / (beta * 1.0 ** 2)) * \
            np.exp(inst_data['log(Qvib*N)'] - ts_data['logQvib'])

    # Calculate kappa
    kappa = f_trn * f_rot * f_vib * np.exp(-inst_data['S/hbar'] + ts_data['V/kBT'])

    return kappa


def calc_kappa_full(
        directory_ts,
        directory_instanton,
        temperature,
        react_energy=0.0,
        n_beads=4,
        filter_list=None):
    run_instanton_post_process(directory_ts,
                               process_type='TS',
                               temperature=temperature,
                               filter_list=filter_list,
                               ref_energy=react_energy)
    data_ts = parse_ts_thermo_data(directory_ts)

    run_instanton_post_process(directory_instanton,
                               process_type='instanton',
                               temperature=temperature,
                               n_beads=n_beads,
                               filter_list=filter_list,
                               ref_energy=react_energy)
    data_inst = parse_inst_thermo_data(directory_instanton)
    return calc_instanton_kappa(data_ts, data_inst)


def calc_forward_rate(ts_directory, react_directory, temperature):
    run_instanton_post_process(react_directory,
                               process_type='reactant',
                               temperature=temperature)
    # Get reactant data
    react_data = parse_react_thermo_data(react_directory)

    run_instanton_post_process(ts_directory,
                               process_type='TS',
                               temperature=temperature,
                               ref_energy=react_data['energy'])

    # Get TS data
    ts_data = parse_ts_thermo_data(ts_directory)

    # Extract partition functions and calculate the forward rate
    q_vib_react = np.exp(react_data['logQvib_rp'])
    q_vib_ts = np.exp(ts_data['logQvib'])

    partition_function_ratio = (ts_data['Qtras'] * ts_data['Qrot'] * q_vib_ts) / \
                               (react_data['Qtras'] * react_data['Qrot'] * q_vib_react)

    boltzmann_factor = np.exp(react_data['V/kBT'] - ts_data['V/kBT'])
    boltzmann_factor = np.exp(- ts_data['V/kBT'])

    k_f = (k * temperature / h) * partition_function_ratio * boltzmann_factor

    return k_f


def exp_decay(t, a, tau, c) -> np.ndarray:
    """
    Models an exponential decay function.

    This function calculates the value of an exponential decay at time `t`
    based on the given amplitude (`a`), decay constant (`tau`), and offset (`c`).

    Parameters:
        t (float or np.ndarray): Time or array of time values.
        a (float): Amplitude of the decay.
        tau (float): Decay constant, which determines the rate of decay.
        c (float): Offset value added to the decay.

    Returns:
        np.ndarray: The calculated exponential decay values.
    """
    return a * np.exp(-t / tau) + c


def fit_exp_decay(x, y, p0=None, bounds=None):
    """
    Fits an exponential decay function to the given data.

    This function uses non-linear least squares to fit the `exp_decay` function
    to the provided data points `(x, y)`. It allows for optional initial parameter
    guesses and bounds for the fitting process.

    Parameters:
        x (array-like): The independent variable data (e.g., time values).
        y (array-like): The dependent variable data (e.g., observed values).
        p0 (tuple, optional): Initial guesses for the parameters (a, tau, c).
                              Defaults to automatically calculated values.
        bounds (tuple, optional): Lower and upper bounds for the parameters.
                                  Defaults to unbounded fitting.

    Returns:
        tuple: Optimal values for the parameters (a, tau, c) that minimize
               the squared residuals between the observed and fitted data.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Rough automatic starting guesses if user gives none
    if p0 is None:
        a0 = y.max() - y.min()
        tau0 = 0.3 * (x[-1] - x[0]) if x[-1] != x[0] else 1.0
        c0 = y.min()
        p0 = (a0, tau0, c0)

    # Default to unbounded fit unless user supplies bounds
    if bounds is None:
        bounds = (-np.inf, np.inf)

    popt, _ = curve_fit(exp_decay, x, y, p0=p0, bounds=bounds)
    return popt


def extrapolate_inf_bead_limit(x, y, plot=False):
    """
    Extrapolates the infinite bead limit for a given dataset.

    This function fits an exponential decay model to the provided data points `(x, y)`
    and optionally plots the data along with the fitted curve. It returns the extrapolated
    value at the infinite bead limit.

    Parameters:
        x (array-like): The independent variable data (e.g., number of beads).
        y (array-like): The dependent variable data (e.g., tunneling factor values).
        plot (bool, optional): Whether to plot the data and the fitted curve. Defaults to False.

    Returns:
        float: The extrapolated value at the infinite bead limit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    popt = fit_exp_decay(x, y)

    if plot:
        plt.scatter(x, y, marker="o")
        t_fine = np.linspace(x.min(), x.max(), 400)
        plt.plot(t_fine, exp_decay(t_fine, *popt), color='black', linewidth=2, linestyle='--')
        n_plot("Number of instanton beads", r"$\kappa$")
        plt.show()

    return popt[-1]  # Return the fitted value at infinity
