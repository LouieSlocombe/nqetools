import math
from typing import List, Dict

import numpy as np
from scipy import constants

from .conversions import eV_to_kJpermol


def correlate(x: np.ndarray,
              y: np.ndarray,
              xbar: float = None,
              ybar: float = None,
              normalise: bool = True) -> np.ndarray:
    """
    Computes the correlation function of two quantities.

    Parameters:
    x (numpy.ndarray): The first quantity.
    y (numpy.ndarray): The second quantity.
    xbar (float, optional): The mean of the first quantity. If None, it is computed from x.
    ybar (float, optional): The mean of the second quantity. If None, it is computed from y.
    normalise (bool, optional): Whether to normalise the correlation function. Default is True.

    Returns:
    numpy.ndarray: The correlation function of the two quantities.
    """
    if xbar is None:
        xbar = x.mean()
    if ybar is None:
        ybar = y.mean()

    cf = np.correlate(x - xbar, y - ybar, mode='same')
    return cf[len(x) // 2:] / (((x - xbar) * (y - ybar)).sum() if normalise else 1)


def autocorrelate(x: np.ndarray,
                  xbar: float = None,
                  normalise: bool = True) -> np.ndarray:
    """
    Computes the autocorrelation function of a trajectory.

    Parameters:
    x (numpy.ndarray): The input trajectory.
    xbar (float, optional): The mean of the trajectory. If None, it is computed from x.
    normalise (bool, optional): Whether to normalise the autocorrelation function. Default is True.

    Returns:
    numpy.ndarray: The autocorrelation function of the trajectory.
    """
    if xbar is None:
        xbar = x.mean()
    acf = np.correlate(x - xbar, x - xbar, mode='same')
    return acf[len(x) // 2:] / (((x - xbar) * (x - xbar)).sum() if normalise else 1)


def moving_average(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Computes the moving average of a given array using a specified window size.

    Parameters:
    arr (numpy.ndarray): The input array.
    window_size (int): The size of the moving window.

    Returns:
    numpy.ndarray: The array of moving averages.
    """
    # Create a window of the specified size with equal weights
    window = np.ones(window_size) / window_size
    # Use the 'valid' mode to only return elements where the window fully
    # overlaps with the data
    return np.convolve(arr, window, mode="valid")


def freq_from_eigvals(eigvals: np.ndarray) -> np.ndarray:
    """
    Converts eigenvalues to frequencies in inverse centimeters (cm^-1).

    Parameters:
    eigvals (numpy.ndarray): Array of eigenvalues.

    Returns:
    numpy.ndarray: Array of frequencies in inverse centimeters (cm^-1).
    """
    cm2hartree = 1. / (constants.physical_constants['hartree-inverse meter relationship'][0] / 100)
    freq_invcm = np.zeros(eigvals.shape)
    for i, eig in enumerate(eigvals):
        freq_invcm[i] = np.sign(eig) * np.absolute(eig) ** 0.5 / cm2hartree
    return freq_invcm


def calculate_temperature_crossover(omega: float) -> float:
    """
    Calculates the temperature crossover value for a given frequency.

    Parameters:
    omega (float): Frequency in inverse centimeters (cm^-1).

    Returns:
    float: Temperature crossover value.
    """
    cm2hartree = 1. / (constants.physical_constants['hartree-inverse meter relationship'][0] / 100)
    boltzmann_au = constants.physical_constants['Boltzmann constant in eV/K'][0] * \
                   constants.physical_constants['electron volt-hartree relationship'][0]
    return 1. / (2 * np.pi / (omega * cm2hartree) * boltzmann_au)


def calculate_good_nbeads(omega_max: float, temperature: float) -> int:
    """
    Calculate the number of beads where nbeads > (hbar * omega_max) / (kB * T).

    Parameters:
    omega_max (float): Maximum frequency in inverse centimeters (cm^-1).
    T (float): Temperature in Kelvin (K).

    Returns:
    int: Number of beads.
    """
    cm2hartree = 1. / (constants.physical_constants['hartree-inverse meter relationship'][0] / 100)
    boltzmann_au = constants.physical_constants['Boltzmann constant in eV/K'][0] * \
                   constants.physical_constants['electron volt-hartree relationship'][0]
    # Convert omega_max to atomic units
    omega_max = omega_max * cm2hartree
    # nbeads > (hbar * omega_max) / (kB * T)
    nbeads = (1.0 * omega_max) / (boltzmann_au * temperature)
    return int(nbeads)


def wigner_correction(omega_cm: float, temperature: float) -> float:
    """
    Calculate the Wigner tunnelling correction factor (κ).

    The Wigner tunnelling factor is given by the formula:
    κ = 1 + (ħ ω)² / (24 kB² T²),
    where ω is the angular frequency derived from the input wavenumber (in cm⁻¹).

    Parameters
    ----------
    omega_cm : float
        The imaginary transition state (TS) frequency, |ω‡|, in cm⁻¹.
    temperature : float
        The temperature in kelvin (K).

    Returns
    -------
    float
        The Wigner tunnelling correction factor (κ), which is dimensionless and always ≥ 1.

    Notes
    -----
    - The correction factor accounts for quantum tunnelling effects in reaction rates.
    - The input frequency (ω) is converted from wavenumber (cm⁻¹) to angular frequency (rad/s).
    """
    # Pre-compute speed of light in cm s⁻¹ so we can keep ω in cm⁻¹
    _c_cm_s = constants.c * 100.0  # 2.997 924 58 × 10¹⁰ cm s⁻¹
    # Convert ω from cm⁻¹ to angular frequency (rad s⁻¹)
    omega_rad_s = 2.0 * constants.pi * _c_cm_s * abs(omega_cm)
    # Apply Wigner formula
    x = (constants.hbar * omega_rad_s) / (constants.k * temperature)
    return 1.0 + (x * x) / 24.0


def bell_correction(e_barrier: float, a: float, mu: float) -> float:
    """
    Return the Bell tunnelling factor κ for a symmetric 1-D parabolic barrier.

    The Bell tunnelling factor is calculated using the formula:
    κ_Bell = (e^α / α) · [1 − e^(−α)]
    where:
    α = 2 a √(2 μ E_a) / ħ

    Parameters:
    e_barrier (float): Barrier height in electron volts (eV).
    a (float): Width of the barrier in Angstroms (Å).
    mu (float): Reduced mass in atomic mass units (amu).

    Returns:
    float: The Bell tunnelling factor κ (dimensionless, κ ≥ 1).

    Notes:
    - κ ≥ 1 by definition (tunnelling always enhances the rate).
    - The factor is independent of temperature; multiply any conventional
      TST rate constant by κ to obtain the tunnelling-corrected rate.
    """
    angstrom_to_m = 1.0e-10  # Å → m
    amu_to_kg = constants.physical_constants["atomic mass constant"][0]  # amu → kg
    ev_to_j = constants.e  # eV → J

    # --- convert inputs to SI ----------------------------
    e_a_j = e_barrier * ev_to_j
    a_m = a * angstrom_to_m
    mu_kg = mu * amu_to_kg

    alpha = (2.0 * a_m / constants.hbar) * np.sqrt(2.0 * mu_kg * e_a_j)

    # No barrier or zero half-width → classical, κ = 1
    if alpha == 0.0:
        return 1.0

    return (np.exp(alpha) / alpha) * (1.0 - np.exp(-alpha))


def _eckart_inner(
        e_list: np.ndarray,
        frequency: float,
        e_reac: float,
        e_ts: float,
        e_prod: float,
) -> np.ndarray:
    """
    Computes the micro-canonical Eckart tunnelling correction factor (κ) for a given energy grid.

    This function calculates the tunnelling correction factor based on the Eckart model,
    which accounts for quantum mechanical tunnelling effects in reaction rates.

    Parameters
    ----------
    e_list : np.ndarray
        Array of energies (in the same units as the input barriers).
    frequency : float
        Transition state frequency in the same units as the energy barriers.
    e_reac : float
        Energy of the reactants.
    e_ts : float
        Energy of the transition state.
    e_prod : float
        Energy of the products.

    Returns
    -------
    np.ndarray
        Array of tunnelling correction factors (κ) for each energy in `e_list`.

    Raises
    ------
    ValueError
        If the forward barrier height exceeds the reverse barrier height.

    Notes
    -----
    - The function assumes that the forward barrier (d_v1) is less than or equal
      to the reverse barrier (d_v2). If this condition is violated, an exception
      is raised.
    - The correction factor is computed using the Eckart tunnelling model, which
      is based on a one-dimensional potential energy surface.
    """
    # Determine reference energy and barriers
    e0 = max(e_reac, e_prod)  # Reference energy (highest of reactants or products)
    d_v1, d_v2 = sorted([e_ts - e_reac, e_ts - e_prod])  # Forward and reverse barriers

    # Ensure forward barrier does not exceed reverse barrier
    if d_v1 > d_v2:
        raise ValueError("Forward barrier must not exceed reverse barrier")

    # Compute dimensionless Eckart parameters
    alpha_1 = 2.0 * constants.pi * d_v1 / frequency
    alpha_2 = 2.0 * constants.pi * d_v2 / frequency
    denom = 1.0 / np.sqrt(alpha_1) + 1.0 / np.sqrt(alpha_2)  # Denominator for scaling
    two_pi_d = 2.0 * np.sqrt(abs(alpha_1 * alpha_2 - (constants.pi ** 2) / 4.0))  # Parameter for tunnelling

    # Initialize the correction factor array
    kappa = np.zeros_like(e_list, dtype=float)
    r0 = np.searchsorted(e_list, e0, side="left")  # Index of the reference energy in `e_list`

    # Loop over energies greater than or equal to the reference energy
    for r in range(r0, len(e_list)):
        x_i = (e_list[r] - e0) / d_v1  # Dimensionless energy relative to the forward barrier
        two_pi_a = 2.0 * np.sqrt(alpha_1 * x_i) / denom  # Parameter for forward tunnelling
        two_pi_b = 2.0 * np.sqrt(abs((x_i - 1.0) * alpha_1 + alpha_2)) / denom  # Parameter for reverse tunnelling

        # Compute the correction factor based on the tunnelling parameters
        if max(two_pi_a, two_pi_b, two_pi_d) < 200.0:
            # Use hyperbolic functions for small values
            num = np.cosh(two_pi_a - two_pi_b) + np.cosh(two_pi_d)
            den = np.cosh(two_pi_a + two_pi_b) + np.cosh(two_pi_d)
        elif any(x > 10.0 for x in
                 [two_pi_a - two_pi_b - two_pi_d, two_pi_b - two_pi_a - two_pi_d, two_pi_a + two_pi_b - two_pi_d]):
            # Approximation for large values to avoid overflow
            kappa[r] = 1.0 - sum(np.exp(-x) for x in [2.0 * two_pi_a, 2.0 * two_pi_b, two_pi_a + two_pi_b - two_pi_d,
                                                      two_pi_a + two_pi_b + two_pi_d])
            continue
        else:
            # Use exponential functions for intermediate values
            num = sum(np.exp(x) for x in
                      [two_pi_a - two_pi_b - two_pi_d, -two_pi_a + two_pi_b - two_pi_d, -2.0 * two_pi_d]) + 1.0
            den = sum(np.exp(x) for x in
                      [two_pi_a + two_pi_b - two_pi_d, -two_pi_a - two_pi_b - two_pi_d, -2.0 * two_pi_d]) + 1.0

        # Compute the tunnelling correction factor
        kappa[r] = 1.0 - num / den

    return kappa


def eckart_correction(
        temperature: float,
        frequency: float,
        e_reac: float,
        e_ts: float,
        e_prod: float,
) -> float:
    """
    Computes the Eckart tunnelling correction factor (κ) for a reaction at a given temperature.

    This function calculates the tunnelling correction factor using the Eckart model,
    which accounts for quantum mechanical tunnelling effects in reaction rates.

    Parameters
    ----------
    temperature : float
        Temperature in kelvin (K).
    frequency : float
        Transition state frequency in inverse centimeters (cm⁻¹).
    e_reac : float
        Energy of the reactants in electron volts (eV).
    e_ts : float
        Energy of the transition state in electron volts (eV).
    e_prod : float
        Energy of the products in electron volts (eV).

    Returns
    -------
    float
        The Eckart tunnelling correction factor (κ), which is dimensionless.

    Raises
    ------
    ValueError
        If the forward barrier height (d_v1) is negative, the reverse barrier height (d_v2) is negative,
        or if the forward barrier height exceeds the reverse barrier height.

    Notes
    -----
    - The function converts input energies from eV to kJ/mol and then to J/mol for calculations.
    - The frequency is converted from cm⁻¹ to energy in J/mol.
    - The correction factor is computed by integrating the micro-canonical correction factor (κ(E))
      over an energy grid using the `_eckart_inner` function.
    """
    beta = 1.0 / (constants.R * temperature)  # mol · J⁻¹

    # Convert energy from eV to kJ mol⁻¹
    e_reac *= eV_to_kJpermol
    e_ts *= eV_to_kJpermol
    e_prod *= eV_to_kJpermol

    # Convert energy from kJ mol⁻¹ to (J mol⁻¹)
    e_reac *= 1e3
    e_ts *= 1e3
    e_prod *= 1e3

    # Convert frequency cm⁻¹ to energy (J mol⁻¹)
    frequency = constants.h * abs(frequency) * constants.c * 100.0 * constants.N_A

    # Determine reference energy and barriers
    e0 = max(e_reac, e_prod)
    d_v1, d_v2 = sorted([e_ts - e_reac, e_ts - e_prod])

    # Sanity checks
    if d_v1 < 0 or d_v2 < 0:
        raise ValueError(f"Invalid barrier heights: d_v1={d_v1 / 1e3:.3f} kJ/mol, d_v2={d_v2 / 1e3:.3f} kJ/mol")
    if d_v1 > d_v2:
        raise ValueError("Eckart requirement d_v1 ≤ d_v2 violated.")

    # Build energy grid and compute micro-canonical κ(E)
    d_e = 100.0
    upper = e0 + 2.0 * (e_ts - e0) + 40.0 * constants.R * temperature
    e_list = np.arange(e0, upper, d_e)
    kappa_e = _eckart_inner(e_list, frequency, e_reac, e_ts, e_prod)
    return np.exp(d_v1 * beta) * np.sum(kappa_e * np.exp(-beta * (e_list - e0))) * d_e * beta


def analyze_opes_free_energy(
        free_energy: List[float],
        temperature: float = 300.0,
        unit: str = "kJ/mol",
        dx: float = 1.0,
) -> Dict[str, object]:
    """
    # Example free-energy profile in eV (per molecule)
    F_ev = [0.35, 0.12, 0.02, 0.05, 0.22, 0.11, 0.01, 0.04, 0.30]
    out = analyze_opes_free_energy(F_ev, temperature=300.0, unit="eV")
    print(out["deltaG_forward_dagger"], out["k_forward"])


    Analyze a 1D free-energy profile (e.g., from OPES) to extract:
      - two lowest metastable minima and the barrier (maximum) between them,
      - energy difference between states,
      - forward/reverse barrier heights,
      - forward/reverse Eyring TST rates.

    Parameters
    ----------
    free_energy : list of float
        Free energy values along a reaction coordinate (uniform spacing).
        Units given by `unit`.
    temperature : float
        Temperature in K.
    unit : {'kJ/mol','kcal/mol','kBT','eV'}
        Units of the provided energies. For 'eV', values are assumed PER MOLECULE
        and converted to J/mol using Avogadro's number.
    dx : float
        Spacing of the reaction coordinate (for reporting only).

    Returns
    -------
    dict with keys as described above.
    """
    F = np.asarray(free_energy, dtype=float)
    n = F.size
    if n < 3:
        raise ValueError("Need at least 3 points to define minima and a barrier.")

    # --- find local minima/maxima ---
    minima = []
    for i in range(1, n - 1):
        if (F[i] <= F[i - 1] and F[i] <= F[i + 1]) and (F[i] < F[i - 1] or F[i] < F[i + 1]):
            minima.append(i)
    # consider endpoints as possible minima
    if F[0] <= F[1]:
        minima.append(0)
    if F[-1] <= F[-2]:
        minima.append(n - 1)

    if len(minima) < 2:
        raise ValueError("Fewer than two metastable minima found. Consider smoothing your profile.")

    # pick two lowest minima (keep left-to-right order)
    i1, i2 = sorted(sorted(minima, key=lambda i: F[i])[:2])
    if i2 - i1 < 2:
        raise ValueError("Two lowest minima are adjacent; no interior barrier point.")
    ib = int(np.argmax(F[i1 + 1: i2]) + i1 + 1)

    F_min1, F_min2, F_bar = float(F[i1]), float(F[i2]), float(F[ib])
    dG12 = F_min2 - F_min1
    dGf = F_bar - F_min1
    dGr = F_bar - F_min2
    if dGf <= 0 or dGr <= 0:
        raise ValueError("Barrier not above both minima. Profile may be noisy or multimodal.")

    # --- convert to ΔG/(RT) for Eyring using SciPy constants ---
    R = constants.R  # J/mol/K
    k_B = constants.k  # J/K
    h = constants.h  # J*s

    u = unit.lower()
    if u in ["kj/mol", "kjmol", "kj"]:
        to_J_per_mol = constants.kilo  # 1000.0
        dGf_over_RT = (dGf * to_J_per_mol) / (R * temperature)
        dGr_over_RT = (dGr * to_J_per_mol) / (R * temperature)
    elif u in ["kcal/mol", "kcalmol", "kcal"]:
        to_J_per_mol = constants.kilo * constants.calorie  # 4184 J
        dGf_over_RT = (dGf * to_J_per_mol) / (R * temperature)
        dGr_over_RT = (dGr * to_J_per_mol) / (R * temperature)
    elif u in ["eV".lower(), "ev"]:
        # 1 eV per particle = const.electron_volt J; per mole multiply by Avogadro
        eV_to_J_per_mol = constants.electron_volt * constants.N_A
        dGf_over_RT = (dGf * eV_to_J_per_mol) / (R * temperature)
        dGr_over_RT = (dGr * eV_to_J_per_mol) / (R * temperature)
    elif u in ["kbt", "rt"]:
        dGf_over_RT = float(dGf)
        dGr_over_RT = float(dGr)
    else:
        raise ValueError("unit must be 'kJ/mol', 'kcal/mol', 'eV', or 'kBT'.")

    # --- Eyring TST ---
    prefactor = (k_B * temperature) / h  # s^-1
    k_forward = prefactor * math.exp(-dGf_over_RT)
    k_reverse = prefactor * math.exp(-dGr_over_RT)

    return {
        "minima_indices": (int(i1), int(i2)),
        "barrier_index": int(ib),
        "F_min1": F_min1,
        "F_min2": F_min2,
        "F_barrier": F_bar,
        "deltaG_2_minus_1": dG12,
        "deltaG_forward_dagger": dGf,
        "deltaG_reverse_dagger": dGr,
        "k_forward": float(k_forward),
        "k_reverse": float(k_reverse),
        "method": "Eyring (SciPy constants)",
        "notes": (
            "Two lowest minima chosen as metastable states; barrier is the maximum between them. "
            "Rates from Eyring TST; assumes a single dominant barrier and well-defined basins. "
            "For unit='eV', inputs are interpreted as per-molecule energies."
        ),
        "units": {
            "energy": unit,
            "rate": "s^-1",
            "temperature_K": temperature,
            "dx": dx,
            "prefactor_Eyring_kBT_over_h": prefactor,
        },
    }
