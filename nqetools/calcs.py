import numpy as np
from scipy import constants


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
