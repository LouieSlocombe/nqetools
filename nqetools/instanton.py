"""Instanton rate theory: tunnelling factors and reaction rates.

Ring-polymer instanton theory recovers the deep-tunnelling rate by locating
the dominant periodic orbit on the inverted potential. This module parses
the partition functions that i-PI's post-processing writes out for the
reactant, transition state and instanton, combines them into a tunnelling
enhancement factor kappa, and computes classical Eyring rates for
comparison.

Because kappa converges only slowly with ring-polymer size,
:func:`extrapolate_inf_bead_limit` fits the bead dependence to recover the
continuum value.
"""

import os

import ipi
import matplotlib.pyplot as plt
import numpy as np
from ase.units import kB
from scipy.constants import k, h
from scipy.optimize import curve_fit
from ipi.utils.units import unit_to_internal
from reactiontools import n_plot
from .execution import run_instanton_post_process
from .calculators import calculate_free_energy

K2au = unit_to_internal("temperature", "kelvin", 1.0)


def parse_react_thermo_data(
    directory, filename="thermo_data.out", ref_filename="phonon.out"
):
    """Parse reactant partition functions and reference energy.

    Parameters
    ----------
    directory : str
        Directory containing the reactant post-processing output.
    filename : str, optional
        Name of the thermo data file. Default is 'thermo_data.out'.
    ref_filename : str, optional
        i-PI output file supplying the reference potential energy. Default
        is 'phonon.out'.

    Returns
    -------
    dict
        Translational and rotational partition functions ('Qtras',
        'Qrot'), the log vibrational partition function ('logQvib_rp'),
        the potential energy ('energy') and the reduced energy 'V/kBT'.

    Raises
    ------
    ValueError
        If the reference energy or temperature is missing, or if the
        phonon calculation produced a negative frequency, which means the
        reactant is not a genuine minimum.
    """
    data = {}
    temp = None
    output_data, _output_desc = ipi.read_output(os.path.join(directory, ref_filename))
    if output_data.get("potential") is None:
        raise ValueError(
            f"Could not find potential energy in {ref_filename} in {directory}"
        )
    data["energy"] = output_data["potential"][-1]

    filepath = os.path.join(directory, filename)

    with open(filepath) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "We have a negative frequency" in line:
            raise ValueError(
                f"Negative frequency found in {filepath}. Check the optimisation/phonon calculation."
            )
        if "Qtras(bohr^-3) | Qrot     | logQvib_rp" in line:
            data_line = lines[i + 1].strip()
            values = data_line.split("|")
            data["Qtras"] = float(values[0].strip())
            data["Qrot"] = float(values[1].strip())
            data["logQvib_rp"] = float(values[2].strip())
        elif "Temperature:" in line:
            temp = float(line.split()[1])
    if temp is None:
        raise ValueError(f"Could not find temperature in {filepath}")

    data["V/kBT"] = float(data["energy"] / (kB * temp))

    return data


def parse_ts_thermo_data(directory, filename="thermo_data.out"):
    """Parse the TS thermo_data.out file and extract thermodynamic values.

    Parameters
    ----------
    directory : str
        Directory containing the thermo_data file.
    filename : str, optional
        Name of the thermo data file. Default is 'thermo_data.out'.

    Returns
    -------
    dict
        Dictionary containing Qtras, Qrot, logQvib, and V/kBT values.
    """
    filepath = os.path.join(directory, filename)

    with open(filepath) as f:
        lines = f.readlines()

    data = {}

    for line in lines:
        if "Qtras:" in line:
            data["Qtras"] = float(line.split(":")[1].strip())
        elif "Qrot:" in line:
            data["Qrot"] = float(line.split(":")[1].strip())
        elif "logQvib:" in line:
            data["logQvib"] = float(line.split(":")[1].strip())
        elif "V/kBT" in line:
            data["V/kBT"] = float(line.split()[-1].strip())

    if not data:
        raise ValueError(f"Could not find thermodynamic data in {filepath}")

    return data


def parse_inst_thermo_data(directory, filename="thermo_data.out"):
    """Parse the thermo_data.out file and extract thermodynamic values.

    Parameters
    ----------
    directory : str
        Directory containing the thermo_data file.
    filename : str, optional
        Name of the thermo data file. Default is 'thermo_data.out'.

    Returns
    -------
    dict
        Dictionary containing the thermodynamic values including Temperature, NBEADS, and 1/(betaP*hbar).
    """
    filepath = os.path.join(directory, filename)

    with open(filepath) as f:
        lines = f.readlines()

    data = {}

    for i, line in enumerate(lines):
        if "Temperature:" in line:
            data["Temperature"] = float(line.split()[1])
        elif "NBEADS:" in line:
            data["NBEADS"] = int(line.split()[1])
        elif "1/(betaP*hbar)" in line:
            data["1/(betaP*hbar)"] = float(line.split("=")[1].strip())
        elif "BN" in line and "Qt" in line:
            data_line = lines[i + 1]
            values = data_line.split("|")
            bn_parts = values[0].strip().replace("(", "").replace(")", "").split()

            data.update(
                {
                    "BN": float(bn_parts[0]),
                    "Qt": float(values[1].strip()),
                    "Qrot": float(values[2].strip()),
                    "log(Qvib*N)": float(values[3].strip()),
                    "S/hbar": float(values[4].split("(")[0].strip()),
                }
            )
            return data

    raise ValueError(f"Could not find thermodynamic data in {filepath}")


def calc_instanton_kappa(ts_data, inst_data):
    """Calculate the tunnelling factor (kappa) from TS and instanton thermodynamic data.

    Parameters
    ----------
    ts_data : dict
        Dictionary containing TS thermodynamic data.
    inst_data : dict
        Dictionary containing instanton thermodynamic data.

    Returns
    -------
    float
        Tunnelling factor (kappa).
    """
    f_trn = inst_data["Qt"] / ts_data["Qtras"]
    f_rot = inst_data["Qrot"] / ts_data["Qrot"]

    beta = 1.0 / (inst_data["Temperature"] * K2au)
    prefactor = np.sqrt(
        (2.0 * np.pi * inst_data["NBEADS"] * inst_data["BN"]) / (beta * 1.0**2)
    )
    f_vib = prefactor * np.exp(inst_data["log(Qvib*N)"]) / np.exp(ts_data["logQvib"])
    exp_factor = np.exp(-inst_data["S/hbar"] + ts_data["V/kBT"])

    print(
        f"f_trn: {f_trn:.3f}, f_rot: {f_rot:.3f}, f_vib: {f_vib:.3f}, exp: {exp_factor:.3f}",
        flush=True,
    )

    kappa = f_trn * f_rot * f_vib * exp_factor
    if kappa < 1.0:
        print(f"Warning: kappa < 1.0, {kappa}", flush=True)
        kappa = 1.0

    return kappa


def calc_kappa_full(
    dir_react,
    dir_ts,
    dir_instanton,
    temperature,
    n_beads,
    filter_list=None,
    ref_energy=True,
):
    """Calculates the tunnelling factor (kappa) using reactant, transition state, and instanton data.

    This function processes the thermodynamic data for the reactant, transition state (TS),
    and instanton, and calculates the tunnelling factor (kappa) based on the provided parameters.

    Parameters
    ----------
    dir_react : str
        Directory containing the reactant data.
    dir_ts : str
        Directory containing the transition state data.
    dir_instanton : str
        Directory containing the instanton data.
    temperature : float
        Temperature in Kelvin.
    n_beads : int
        Number of beads used in the instanton calculation.
    filter_list : list, optional
        List of filters to apply during data processing. Defaults to None.
    ref_energy : bool, optional
        Whether to use the reactant energy as a reference. Defaults to True.

    Returns
    -------
    float
        The calculated tunnelling factor (kappa).
    """
    run_instanton_post_process(
        dir_react,
        process_type="reactant",
        temperature=temperature,
        filter_list=filter_list,
    )
    react_data = parse_react_thermo_data(dir_react)

    if ref_energy:
        ref_energy = react_data["energy"]
    else:
        ref_energy = None

    run_instanton_post_process(
        dir_ts,
        process_type="TS",
        temperature=temperature,
        filter_list=filter_list,
        ref_energy=ref_energy,
    )
    data_ts = parse_ts_thermo_data(dir_ts)

    run_instanton_post_process(
        dir_instanton,
        process_type="instanton",
        temperature=temperature,
        n_beads=n_beads,
        filter_list=filter_list,
        ref_energy=ref_energy,
    )
    data_inst = parse_inst_thermo_data(dir_instanton)

    return calc_instanton_kappa(data_ts, data_inst)


def calc_forward_rate(
    dir_react,
    dir_ts,
    temperature,
    filter_list=None,
    use_part_funcs=True,
    debug=False,
    barrier=None,
):
    """Calculate the classical TST forward rate from partition functions.

    Evaluates the Eyring expression using partition functions extracted
    from i-PI post-processing. The result carries no tunnelling
    correction, so it serves as the classical baseline that
    :func:`calc_kappa_full` multiplies.

    Parameters
    ----------
    dir_react : str
        Directory containing the reactant data.
    dir_ts : str
        Directory containing the transition state data.
    temperature : float
        Temperature in K.
    filter_list : list, optional
        Modes to discard during post-processing. Default is None.
    use_part_funcs : bool, optional
        If True, include the partition function ratio. If False, the ratio
        is fixed at 1.0, leaving only the prefactor and Boltzmann factor.
        Default is True.
    debug : bool, optional
        If True, print the individual contributions to the rate. Default
        is False.
    barrier : float, optional
        Barrier height in eV, overriding the value derived from the
        transition state data. Default is None.

    Returns
    -------
    float
        The forward rate constant in s^-1.

    Notes
    -----
    Reactant energies are used as the reference for the transition state,
    so the barrier is measured relative to the reactant minimum.
    """
    run_instanton_post_process(
        dir_react,
        process_type="reactant",
        temperature=temperature,
        filter_list=filter_list,
    )
    react_data = parse_react_thermo_data(dir_react)

    run_instanton_post_process(
        dir_ts,
        process_type="TS",
        temperature=temperature,
        filter_list=filter_list,
        ref_energy=react_data["energy"],
    )
    ts_data = parse_ts_thermo_data(dir_ts)

    q_vib_react = np.exp(react_data["logQvib_rp"])
    q_vib_ts = np.exp(ts_data["logQvib"])

    react_part = react_data["Qtras"] * react_data["Qrot"] * q_vib_react
    ts_part = ts_data["Qtras"] * ts_data["Qrot"] * q_vib_ts
    if use_part_funcs:
        partition_function_ratio = ts_part / react_part
    else:
        partition_function_ratio = 1.0

    if barrier is None:
        boltzmann_factor = np.exp(-ts_data["V/kBT"])
    else:
        boltzmann_factor = np.exp(-barrier / (kB * temperature))

    prefactor = k * temperature / h

    if debug:
        print(f"react_part: {react_part}, ts_part: {ts_part}")
        print(f"Prefactor: {prefactor}")
        print(f"Partition function ratio: {partition_function_ratio}")
        print(f"boltzmann_factor: {boltzmann_factor}")

    return prefactor * partition_function_ratio * boltzmann_factor


def calc_forward_rate_orca(
    atoms_react, atoms_ts, temperature=300.0, calc_settings=None
):
    """Calculate the Eyring TST forward rate from ORCA Gibbs free energies.

    Parameters
    ----------
    atoms_react : ase.Atoms
        The reactant structure.
    atoms_ts : ase.Atoms
        The transition state structure.
    temperature : float, optional
        Temperature in kelvin (K). Default is 300.0.
    calc_settings : dict, optional
        Extra keyword arguments forwarded to `calculate_free_energy`.

    Returns
    -------
    float
        The forward rate constant in s^-1.
    """
    if calc_settings is None:
        calc_settings = {}

    # calculate_free_energy returns (gibbs, enthalpy, entropy); only Gibbs is needed here
    e_react, _, _ = calculate_free_energy(
        atoms_react, temperature=temperature, **calc_settings
    )
    e_ts, _, _ = calculate_free_energy(
        atoms_ts, temperature=temperature, **calc_settings
    )

    # The activation free energy is the TS energy relative to the reactant
    boltzmann_factor = np.exp(-(e_ts - e_react) / (kB * temperature))

    return (k * temperature / h) * boltzmann_factor


def exp_decay(t, a, tau, c) -> np.ndarray:
    """Models an exponential decay function.

    This function calculates the value of an exponential decay at time `t`
    based on the given amplitude (`a`), decay constant (`tau`), and offset (`c`).

    Parameters
    ----------
    t : float or np.ndarray
        Time or array of time values.
    a : float
        Amplitude of the decay.
    tau : float
        Decay constant, which determines the rate of decay.
    c : float
        Offset value added to the decay.

    Returns
    -------
    np.ndarray
        The calculated exponential decay values.
    """
    return a * np.exp(-t / tau) + c


def fit_exp_decay(x, y, p0=None, bounds=None):
    """Fits an exponential decay function to the given data.

    This function uses non-linear least squares to fit the `exp_decay` function
    to the provided data points `(x, y)`. It allows for optional initial parameter
    guesses and bounds for the fitting process.

    Parameters
    ----------
    x : array-like
        The independent variable data (e.g., time values).
    y : array-like
        The dependent variable data (e.g., observed values).
    p0 : tuple, optional
        Initial guesses for the parameters (a, tau, c). Defaults to
        automatically calculated values.
    bounds : tuple, optional
        Lower and upper bounds for the parameters. Defaults to unbounded
        fitting.

    Returns
    -------
    tuple
        Optimal values for the parameters (a, tau, c) that minimize the
        squared residuals between the observed and fitted data.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Rough automatic starting guesses if user gives none
    if p0 is None:
        a0 = y.max() - y.min()
        tau0 = 0.3 * (x[-1] - x[0]) if x[-1] != x[0] else 1.0
        c0 = y.min()
        p0 = (a0, tau0, c0)

    if bounds is None:
        bounds = (-np.inf, np.inf)

    popt, _ = curve_fit(exp_decay, x, y, p0=p0, bounds=bounds)
    return popt


def extrapolate_inf_bead_limit(x, y, plot=False):
    """Extrapolates the infinite bead limit for a given dataset.

    This function fits an exponential decay model to the provided data points `(x, y)`
    and optionally plots the data along with the fitted curve. It returns the extrapolated
    value at the infinite bead limit.

    Parameters
    ----------
    x : array-like
        The independent variable data (e.g., number of beads).
    y : array-like
        The dependent variable data (e.g., tunneling factor values).
    plot : bool, optional
        Whether to plot the data and the fitted curve. Defaults to False.

    Returns
    -------
    float
        The extrapolated value at the infinite bead limit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    popt = fit_exp_decay(x, y)

    if plot:
        plt.scatter(x, y, marker="o")
        t_fine = np.linspace(x.min(), x.max(), 400)
        plt.plot(
            t_fine, exp_decay(t_fine, *popt), color="black", linewidth=2, linestyle="--"
        )
        n_plot("Number of instanton beads", r"$\kappa$")
        plt.show()

    return popt[-1]  # The offset c is the value the decay approaches as beads -> inf
