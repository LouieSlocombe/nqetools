import numpy as np
from scipy import constants

from .postproc import instanton_postproc


def correlate(x, y, xbar=None, ybar=None, normalize=True):
    """
    Computes the correlation function of two quantities.

    Parameters:
    x (numpy.ndarray): The first quantity.
    y (numpy.ndarray): The second quantity.
    xbar (float, optional): The mean of the first quantity. If None, it is computed from x.
    ybar (float, optional): The mean of the second quantity. If None, it is computed from y.
    normalize (bool, optional): Whether to normalize the correlation function. Default is True.

    Returns:
    numpy.ndarray: The correlation function of the two quantities.
    """
    if xbar is None:
        xbar = x.mean()
    if ybar is None:
        ybar = y.mean()

    cf = np.correlate(x - xbar, y - ybar, mode='same')
    return cf[len(x) // 2:] / (((x - xbar) * (y - ybar)).sum() if normalize else 1)


def autocorrelate(x, xbar=None, normalize=True):
    """
    Computes the autocorrelation function of a trajectory.

    Parameters:
    x (numpy.ndarray): The input trajectory.
    xbar (float, optional): The mean of the trajectory. If None, it is computed from x.
    normalize (bool, optional): Whether to normalize the autocorrelation function. Default is True.

    Returns:
    numpy.ndarray: The autocorrelation function of the trajectory.
    """
    if xbar is None:
        xbar = x.mean()
    acf = np.correlate(x - xbar, x - xbar, mode='same')
    return acf[len(x) // 2:] / (((x - xbar) * (x - xbar)).sum() if normalize else 1)


def moving_average(arr, window_size):
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


def freq_from_eigvals(eigvals):
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


def temp_cross(omega):
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


def calculate_nbeads(omega_max, temperature):
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


def kappa_core(
        Q_trn_TS,
        Q_rot_TS,
        Q_vib_TS,
        Q_trn_inst,
        Q_rot_inst,
        Q_vib_inst,
        BN,
        temperature,
        N,
        S_over_hbar,
        Beta_times_V,
        hbar=1.0):
    """
    Computes the tunneling factor (kappa) for a given set of parameters.

    Parameters:
    Q_trn_TS (float): Translational partition function at the transition state.
    Q_rot_TS (float): Rotational partition function at the transition state.
    Q_vib_TS (float): Vibrational partition function at the transition state.
    Q_trn_inst (float): Translational partition function at the instanton.
    Q_rot_inst (float): Rotational partition function at the instanton.
    Q_vib_inst (float): Vibrational partition function at the instanton.
    BN (float): A parameter related to the instanton.
    temperature (float): Temperature in Kelvin.
    N (int): Number of beads.
    S_over_hbar (float): Action divided by reduced Planck's constant.
    Beta_times_V (float): Beta times potential energy.
    hbar (float, optional): Reduced Planck's constant. Default is 1.0.

    Returns:
    float: The computed tunneling factor (kappa).
    """
    kelvin2au = 3.1668152e-06
    beta = 1.0 / (temperature * kelvin2au)
    f_trn = Q_trn_inst / Q_trn_TS
    f_rot = Q_rot_inst / Q_rot_TS
    f_vib = np.sqrt((2. * np.pi * N * BN) / (beta * hbar ** 2)) * Q_vib_inst / Q_vib_TS

    kappa = f_trn * f_rot * f_vib * np.exp(-S_over_hbar + Beta_times_V)

    # printing out the transmission factor and the relevant contributions.
    print('f_tra               = {:5.3f}'.format(f_trn), flush=True)
    print('f_rot               = {:5.3f}'.format(f_rot), flush=True)
    print('f_vib               = {:5.3f}'.format(f_vib), flush=True)
    print('exp(-S/hbar+V/beta) = {:5.3f}'.format(np.exp(-S_over_hbar + Beta_times_V)), flush=True)
    print('=============================', flush=True)
    print('Tunneling factor    = {:5.3f}'.format(kappa), flush=True)
    return kappa


def calc_kappa(ts_path, instanton_path, temperature, n_beads):
    """
    Calculates the tunneling factor (kappa) for a given transition state and instanton path.

    Parameters:
    ts_path (str): Path to the transition state data.
    instanton_path (str): Path to the instanton data.
    temperature (float): Temperature in Kelvin.
    n_beads (int): Number of beads.

    Returns:
    float: The computed tunneling factor (kappa).
    """
    Q_trn_TS, Q_rot_TS, Q_vib_TS, Beta_times_V = instanton_postproc(ts_path,
                                                                    case="TS",
                                                                    temperature=temperature,
                                                                    n_beads_r=n_beads)

    Q_trn_inst, Q_rot_inst, Q_vib_inst, BN, S_over_hbar = instanton_postproc(instanton_path,
                                                                             case="instanton",
                                                                             temperature=temperature)

    kappa = kappa_core(Q_trn_TS,
                       Q_rot_TS,
                       Q_vib_TS,
                       Q_trn_inst,
                       Q_rot_inst,
                       Q_vib_inst,
                       BN,
                       temperature,
                       n_beads,
                       S_over_hbar,
                       Beta_times_V)
    return kappa
