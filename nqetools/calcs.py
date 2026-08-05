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
    """Compute the correlation function of two quantities.

    Parameters
    ----------
    x : numpy.ndarray
        The first quantity.
    y : numpy.ndarray
        The second quantity.
    xbar : float, optional
        Mean of the first quantity. Computed from `x` if None.
    ybar : float, optional
        Mean of the second quantity. Computed from `y` if None.
    normalise : bool, optional
        Whether to normalise the correlation function. Default is True.

    Returns
    -------
    numpy.ndarray
        The correlation function of the two quantities.
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
    """Compute the autocorrelation function of a trajectory.

    Parameters
    ----------
    x : numpy.ndarray
        The input trajectory.
    xbar : float, optional
        Mean of the trajectory. Computed from `x` if None.
    normalise : bool, optional
        Whether to normalise the autocorrelation function. Default is True.

    Returns
    -------
    numpy.ndarray
        The autocorrelation function of the trajectory.
    """
    if xbar is None:
        xbar = x.mean()
    acf = np.correlate(x - xbar, x - xbar, mode='same')
    return acf[len(x) // 2:] / (((x - xbar) * (x - xbar)).sum() if normalise else 1)


def moving_average(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Compute the moving average of an array using a specified window size.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array.
    window_size : int
        The size of the moving window.

    Returns
    -------
    numpy.ndarray
        The array of moving averages.
    """
    window = np.ones(window_size) / window_size
    return np.convolve(arr, window, mode="valid")


def freq_from_eigvals(eigvals: np.ndarray) -> np.ndarray:
    """Convert eigenvalues to frequencies in inverse centimeters (cm^-1).

    Parameters
    ----------
    eigvals : numpy.ndarray
        Array of eigenvalues.

    Returns
    -------
    numpy.ndarray
        Array of frequencies in inverse centimeters (cm^-1).
    """
    cm2hartree = 1. / (constants.physical_constants['hartree-inverse meter relationship'][0] / 100)
    freq_invcm = np.zeros(eigvals.shape)
    for i, eig in enumerate(eigvals):
        freq_invcm[i] = np.sign(eig) * np.absolute(eig) ** 0.5 / cm2hartree
    return freq_invcm


def calculate_temperature_crossover(omega: float) -> float:
    """Calculate the temperature crossover value for a given frequency.

    Parameters
    ----------
    omega : float
        Frequency in inverse centimeters (cm^-1).

    Returns
    -------
    float
        Temperature crossover value.
    """
    cm2hartree = 1. / (constants.physical_constants['hartree-inverse meter relationship'][0] / 100)
    boltzmann_au = constants.physical_constants['Boltzmann constant in eV/K'][0] * \
                   constants.physical_constants['electron volt-hartree relationship'][0]
    return 1. / (2 * np.pi / (omega * cm2hartree) * boltzmann_au)


def calculate_good_nbeads(omega_max: float, temperature: float) -> int:
    """Calculate the number of beads where nbeads > (hbar * omega_max) / (kB * T).

    Parameters
    ----------
    omega_max : float
        Maximum frequency in inverse centimeters (cm^-1).
    temperature : float
        Temperature in kelvin (K).

    Returns
    -------
    int
        Number of beads.
    """
    cm2hartree = 1. / (constants.physical_constants['hartree-inverse meter relationship'][0] / 100)
    boltzmann_au = constants.physical_constants['Boltzmann constant in eV/K'][0] * \
                   constants.physical_constants['electron volt-hartree relationship'][0]
    omega_max = omega_max * cm2hartree
    nbeads = (1.0 * omega_max) / (boltzmann_au * temperature)
    return int(nbeads)


def wigner_correction(omega_cm: float, temperature: float) -> float:
    """Calculate the Wigner tunnelling correction factor (κ).

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
    The input frequency (ω) is converted from wavenumber (cm⁻¹) to angular frequency (rad/s)
    before applying the formula.
    """
    _c_cm_s = constants.c * 100.0  # speed of light in cm s⁻¹, so ω can stay in cm⁻¹
    omega_rad_s = 2.0 * constants.pi * _c_cm_s * abs(omega_cm)
    x = (constants.hbar * omega_rad_s) / (constants.k * temperature)
    return 1.0 + (x * x) / 24.0


def bell_correction(e_barrier: float, a: float, mu: float) -> float:
    """Return the Bell tunnelling factor κ for a symmetric 1-D parabolic barrier.

    The Bell tunnelling factor is calculated using the formula:
    κ_Bell = (e^α / α) · [1 − e^(−α)]
    where:
    α = 2 a √(2 μ E_a) / ħ

    Parameters
    ----------
    e_barrier : float
        Barrier height in electron volts (eV).
    a : float
        Width of the barrier in Angstroms (Å).
    mu : float
        Reduced mass in atomic mass units (amu).

    Returns
    -------
    float
        The Bell tunnelling factor κ (dimensionless, κ ≥ 1).

    Notes
    -----
    The factor is independent of temperature; multiply any conventional TST
    rate constant by κ to obtain the tunnelling-corrected rate.
    """
    angstrom_to_m = 1.0e-10
    amu_to_kg = constants.physical_constants["atomic mass constant"][0]
    ev_to_j = constants.e

    e_a_j = e_barrier * ev_to_j
    a_m = a * angstrom_to_m
    mu_kg = mu * amu_to_kg

    alpha = (2.0 * a_m / constants.hbar) * np.sqrt(2.0 * mu_kg * e_a_j)

    if alpha == 0.0:  # no barrier or zero half-width -> classical, κ = 1
        return 1.0

    return (np.exp(alpha) / alpha) * (1.0 - np.exp(-alpha))


def _eckart_inner(
        e_list: np.ndarray,
        frequency: float,
        e_reac: float,
        e_ts: float,
        e_prod: float,
) -> np.ndarray:
    """Compute the micro-canonical Eckart tunnelling correction factor (κ) for an energy grid.

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
    Follows the Eckart tunnelling model for a one-dimensional potential energy
    surface, switching between hyperbolic, exponential, and large-argument
    approximations of the correction factor to avoid overflow.
    """
    e0 = max(e_reac, e_prod)
    d_v1, d_v2 = sorted([e_ts - e_reac, e_ts - e_prod])

    if d_v1 > d_v2:
        raise ValueError("Forward barrier must not exceed reverse barrier")

    alpha_1 = 2.0 * constants.pi * d_v1 / frequency
    alpha_2 = 2.0 * constants.pi * d_v2 / frequency
    denom = 1.0 / np.sqrt(alpha_1) + 1.0 / np.sqrt(alpha_2)
    two_pi_d = 2.0 * np.sqrt(abs(alpha_1 * alpha_2 - (constants.pi ** 2) / 4.0))

    kappa = np.zeros_like(e_list, dtype=float)
    r0 = np.searchsorted(e_list, e0, side="left")  # first index at or above the reference energy

    for r in range(r0, len(e_list)):
        x_i = (e_list[r] - e0) / d_v1  # dimensionless energy relative to the forward barrier
        two_pi_a = 2.0 * np.sqrt(alpha_1 * x_i) / denom
        two_pi_b = 2.0 * np.sqrt(abs((x_i - 1.0) * alpha_1 + alpha_2)) / denom

        if max(two_pi_a, two_pi_b, two_pi_d) < 200.0:
            num = np.cosh(two_pi_a - two_pi_b) + np.cosh(two_pi_d)
            den = np.cosh(two_pi_a + two_pi_b) + np.cosh(two_pi_d)
        elif any(x > 10.0 for x in
                 [two_pi_a - two_pi_b - two_pi_d, two_pi_b - two_pi_a - two_pi_d, two_pi_a + two_pi_b - two_pi_d]):
            # large-argument approximation to avoid overflow in the exponentials below
            kappa[r] = 1.0 - sum(np.exp(-x) for x in [2.0 * two_pi_a, 2.0 * two_pi_b, two_pi_a + two_pi_b - two_pi_d,
                                                      two_pi_a + two_pi_b + two_pi_d])
            continue
        else:
            num = sum(np.exp(x) for x in
                      [two_pi_a - two_pi_b - two_pi_d, -two_pi_a + two_pi_b - two_pi_d, -2.0 * two_pi_d]) + 1.0
            den = sum(np.exp(x) for x in
                      [two_pi_a + two_pi_b - two_pi_d, -two_pi_a - two_pi_b - two_pi_d, -2.0 * two_pi_d]) + 1.0

        kappa[r] = 1.0 - num / den

    return kappa


def eckart_correction(
        temperature: float,
        frequency: float,
        e_reac: float,
        e_ts: float,
        e_prod: float,
) -> float:
    """Compute the Eckart tunnelling correction factor (κ) for a reaction at a given temperature.

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
    The correction factor is obtained by integrating the micro-canonical
    correction factor κ(E) from `_eckart_inner` over an energy grid.
    """
    beta = 1.0 / (constants.R * temperature)  # mol · J⁻¹

    # eV -> kJ/mol -> J/mol
    e_reac *= eV_to_kJpermol
    e_ts *= eV_to_kJpermol
    e_prod *= eV_to_kJpermol
    e_reac *= 1e3
    e_ts *= 1e3
    e_prod *= 1e3

    frequency = constants.h * abs(frequency) * constants.c * 100.0 * constants.N_A  # cm⁻¹ -> J/mol

    e0 = max(e_reac, e_prod)
    d_v1, d_v2 = sorted([e_ts - e_reac, e_ts - e_prod])

    if d_v1 < 0 or d_v2 < 0:
        raise ValueError(f"Invalid barrier heights: d_v1={d_v1 / 1e3:.3f} kJ/mol, d_v2={d_v2 / 1e3:.3f} kJ/mol")
    if d_v1 > d_v2:
        raise ValueError("Eckart requirement d_v1 ≤ d_v2 violated.")

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
    """Analyze a 1D free-energy profile (e.g., from OPES) to extract kinetics.

    Locates the two lowest metastable minima and the barrier (maximum)
    between them, then reports the energy difference between states and the
    forward/reverse barrier heights and Eyring TST rates.

    Parameters
    ----------
    free_energy : list of float
        Free energy values along a reaction coordinate (uniform spacing).
        Units given by `unit`.
    temperature : float, optional
        Temperature in K. Default is 300.0.
    unit : {'kJ/mol', 'kcal/mol', 'kBT', 'eV'}, optional
        Units of the provided energies. For 'eV', values are assumed PER
        MOLECULE and converted to J/mol using Avogadro's number. Default is
        'kJ/mol'.
    dx : float, optional
        Spacing of the reaction coordinate (for reporting only). Default is 1.0.

    Returns
    -------
    dict
        Dictionary with keys ``minima_indices``, ``barrier_index``,
        ``F_min1``, ``F_min2``, ``F_barrier``, ``deltaG_2_minus_1``,
        ``deltaG_forward_dagger``, ``deltaG_reverse_dagger``, ``k_forward``,
        ``k_reverse``, ``method``, ``notes``, and ``units``.

    Raises
    ------
    ValueError
        If fewer than 3 points are given, fewer than two metastable minima
        are found, the two lowest minima are adjacent, the barrier does not
        lie above both minima, or `unit` is not recognised.

    Examples
    --------
    >>> F_ev = [0.35, 0.12, 0.02, 0.05, 0.22, 0.11, 0.01, 0.04, 0.30]
    >>> out = analyze_opes_free_energy(F_ev, temperature=300.0, unit="eV")
    >>> out["deltaG_forward_dagger"], out["k_forward"]
    """
    F = np.asarray(free_energy, dtype=float)
    n = F.size
    if n < 3:
        raise ValueError("Need at least 3 points to define minima and a barrier.")

    minima = []
    for i in range(1, n - 1):
        if (F[i] <= F[i - 1] and F[i] <= F[i + 1]) and (F[i] < F[i - 1] or F[i] < F[i + 1]):
            minima.append(i)
    # endpoints can also be minima
    if F[0] <= F[1]:
        minima.append(0)
    if F[-1] <= F[-2]:
        minima.append(n - 1)

    if len(minima) < 2:
        raise ValueError("Fewer than two metastable minima found. Consider smoothing your profile.")

    i1, i2 = sorted(sorted(minima, key=lambda i: F[i])[:2])  # two lowest minima, left-to-right
    if i2 - i1 < 2:
        raise ValueError("Two lowest minima are adjacent; no interior barrier point.")
    ib = int(np.argmax(F[i1 + 1: i2]) + i1 + 1)

    F_min1, F_min2, F_bar = float(F[i1]), float(F[i2]), float(F[ib])
    dG12 = F_min2 - F_min1
    dGf = F_bar - F_min1
    dGr = F_bar - F_min2
    if dGf <= 0 or dGr <= 0:
        raise ValueError("Barrier not above both minima. Profile may be noisy or multimodal.")

    R = constants.R  # J/mol/K
    k_B = constants.k  # J/K
    h = constants.h  # J*s

    u = unit.lower()
    if u in ["kj/mol", "kjmol", "kj"]:
        to_J_per_mol = constants.kilo
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

    prefactor = (k_B * temperature) / h  # s^-1, Eyring TST prefactor
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
