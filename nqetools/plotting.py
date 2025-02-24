import copy

import matplotlib.pyplot as plt
import numpy as np
from ase.visualize.plot import plot_atoms

from .calcs import moving_average
from .pathway import get_neb_path

# Setting plot aesthetics for better visibility
plt.rcParams['axes.linewidth'] = 2.0


def plot_time_potential_bias(data):
    """
    Plots the potential and ensemble bias over time.

    Parameters:
    data (dict): A dictionary containing the time series data with keys:
        - "time": A list or array of time points.
        - "potential": A list or array of potential energy values.
        - "ensemble_bias": A list or array of ensemble bias values.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    time = data["time"]

    ax.plot(time, data["potential"], c="r", label="potential", lw=2)
    ax.plot(time, data["ensemble_bias"], c="b", label="bias", lw=2)

    ax.legend(loc="upper left", ncols=1)
    ax_plot(fig, ax, r"$t$, ps", r"Energy, eV")
    plt.show()
    return None


def plot_time_temperature(data, window_size=100):
    """
    Plots the temperature over time with a moving average.

    Parameters:
    data (dict): A dictionary containing the time series data with keys:
        - "time": A list or array of time points.
        - "temperature": A list or array of temperature values.
    window_size (int, optional): The window size for the moving average. Default is 100.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    min_val = int(window_size / 2)
    max_val = -int(window_size / 2 - 1)

    time = data["time"][min_val:max_val]
    ave_temperature = moving_average(data["temperature"], window_size)

    ax.plot(time, ave_temperature, c="r", lw=2)

    ax_plot(fig, ax, r"$t$, ps", r"Temperature, K")
    plt.show()
    return None


def plot_energy_conservation(data: dict) -> None:
    """
    Plot energy conservation (total energy) as a function of time.

    Parameters:
    data (dict): A dictionary containing time series data with keys:
        - "time": Array of time points in picoseconds
        - "potential": Array of potential energy values
        - "conserved": Array of total conserved energy values

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    time = data["time"]

    # Plot individual energy components
    ax.plot(time, data["potential"], c="b", label="Potential", lw=2)
    ax.plot(time, data["conserved"], c="r", label="Total", lw=2)

    ax.legend(loc="upper left", ncols=1)
    ax_plot(fig, ax, r"$t$ (ps)", r"Energy (eV)")
    plt.show()
    return None


def plot_energy_contour_series(fes_arrays: list[np.ndarray], times: list[float]) -> None:
    """
    Plot a series of energy contour plots from FES arrays.

    Parameters:
    fes_arrays (list[np.ndarray]): List of FES arrays, each with shape (3, bins, bins).
    times (list[float]): List of time points corresponding to each FES array.

    Returns:
    None
    """
    fig, ax = plt.subplots(
        1, len(fes_arrays), figsize=(8, 3), sharex=True, sharey=True, constrained_layout=True
    )

    contours = []
    for i, xyz in enumerate(fes_arrays):
        cf = ax[i].contourf(xyz[1], xyz[0], xyz[2])
        contours.append(cf)
        ax[i].set_xlabel(r"$d_\mathrm{OO}$ / Å")
        ax[i].set_title(fr"$t={times[i]}$ ps")

    ax[0].set_ylabel(r"$\Delta C_\mathrm{H}$")
    fig.colorbar(contours[-1], ax=ax, orientation="vertical", label=r"$F$, eV")
    plt.show()
    return None


def plot_energy_contour_compare(xyz_a, xyz_b):
    """
    Plot a comparison of energy contour plots for two sets of data.

    Parameters:
    xyz_a (tuple): Data for the first contour plot.
    xyz_b (tuple): Data for the second contour plot.

    Returns:
    None
    """
    fig, ax = plt.subplots(
        1, 1, figsize=(4, 3), sharex=True, sharey=True, constrained_layout=True
    )

    levels = np.linspace(0, 0.5, 6)
    cp1 = ax.contour(*xyz_a, colors="b", levels=levels)
    cp2 = ax.contour(*xyz_b, colors="r", levels=levels)
    ax.set_ylabel(r"$\Delta C_\mathrm{H}$")
    ax.set_xlabel(r"$d_\mathrm{OO}$ / Å")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="b", label="MD"),
            plt.Line2D([0], [0], color="r", label="PIMD"),
        ]
    )
    plt.show()
    return None


def plot_energy_sep(xyz_a, xyz_b):
    """
    Plot the energy separation for two sets of data.

    Parameters:
    xyz_a (numpy.ndarray): Data for the first plot.
    xyz_b (numpy.ndarray): Data for the second plot.

    Returns:
    None
    """
    fig, ax = plt.subplots(
        1, 1, figsize=(4, 3), sharex=True, sharey=True, constrained_layout=True
    )

    ax.plot(
        xyz_a[1, :, 50],
        xyz_a[2, :, 50],
        "b",
        label=r"MD, $d_\mathrm{OO}=2.6 $Å"
    )
    ax.plot(
        xyz_b[1, :, 50],
        xyz_b[2, :, 50],
        "r",
        label=r"PIMD, $d_\mathrm{OO}=2.6 $Å",
    )
    ax.plot(
        xyz_a[1, :, 60],
        xyz_a[2, :, 60],
        "b--",
        label=r"MD, $d_\mathrm{OO}=2.7 $Å",
    )
    ax.plot(
        xyz_b[1, :, 60],
        xyz_b[2, :, 60],
        "r--",
        label=r"PIMD, $d_\mathrm{OO}=2.7 $Å",
    )
    ax.set_ylim(0.08, 0.6)
    ax.legend(ncols=2, loc="upper right", fontsize=9)
    ax.set_ylabel(r"$F$ / eV")
    ax.set_xlabel(r"$\Delta C_\mathrm{H}$")
    plt.show()
    return None


def n_plot(xlab, ylab, xs=14, ys=14):
    """
    Add labels and formatting to a plot.

    Use with
    plt.rcParams['axes.linewidth'] = 2.0

    Parameters:
    xlab (str): Label for the x-axis.
    ylab (str): Label for the y-axis.
    xs (int, optional): Font size for the x-axis label. Default is 14.
    ys (int, optional): Font size for the y-axis label. Default is 14.

    Returns:
    None
    """
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None


def ax_plot(fig, ax, xlab, ylab, xs=14, ys=14):
    """
    Configure and style the plot with specified labels and tick parameters for a given axis.

    Use with
    plt.rcParams['axes.linewidth'] = 2.0

    Args:
        fig (matplotlib.figure.Figure): The figure object containing the plot.
        ax (matplotlib.axes.Axes): The axes object to be styled.
        xlab (str): The label for the x-axis.
        ylab (str): The label for the y-axis.
        xs (int, optional): Font size for the x-axis label. Default is 14.
        ys (int, optional): Font size for the y-axis label. Default is 14.

    Returns:
        None
    """
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    ax.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.set_xlabel(xlab, fontsize=xs)
    ax.set_ylabel(ylab, fontsize=ys)
    fig.tight_layout()
    return None


def plot_neb(images, calc, filename="neb", show=True):
    """
    Plot the Nudged Elastic Band (NEB) path for a series of images.

    Parameters:
    images (list of ase.Atoms): List of ASE Atoms objects representing the images along the NEB path.
    calc (ase.Calculator): Calculator to be used for the energy calculations.
    filename (str, optional): The base name for the saved plot files. Default is "neb".
    show (bool, optional): Whether to display the plot. Default is True.

    Returns:
    None
    """
    # Attach the calculator to the images
    for image in images:
        image.calc = copy.copy(calc)

    # Get the energy
    energies = np.array([i.get_potential_energy() for i in images])

    # Shift the graph
    energies -= min(energies)

    # Get the path
    path = get_neb_path(images)

    # Plot the energy profile
    plt.plot(path, energies, 'o-', c='k', lw=2)

    # Add labels and formatting
    n_plot("Path (Å)", "Energy (eV)")

    # Save the plot
    plt.savefig(f"{filename}.png", dpi=600)
    plt.savefig(f"{filename}.pdf")

    # Display the plot
    if show:
        plt.show()
    plt.close()
    return None


def plot_sella(images, calc, filename="irc", show=True):
    """
    Plot the energy profile along the NEB path for a series of images using the Sella method.

    Parameters:
    images (list of ase.Atoms): List of ASE Atoms objects representing the images along the NEB path.
    calc (ase.Calculator): Calculator to be used for the energy calculations.
    filename (str, optional): The base name for the saved plot files. Default is "irc".
    show (bool, optional): Whether to display the plot. Default is True.

    Returns:
    None
    """

    # Attach the calculator to the images
    for image in images:
        image.calc = copy.copy(calc)

    # Get the energy
    energies = np.array([i.get_potential_energy() for i in images])

    # Shift the graph so the minimum energy is zero
    energies -= min(energies)

    # Get the path
    path = get_neb_path(images)

    # Plot the energy profile
    plt.plot(path, energies, 'o-', c='k', lw=2)

    # Add labels and formatting
    n_plot("Path (Å)", "Energy (eV)")

    # Save the plot
    plt.savefig(f"{filename}.png", dpi=600)
    plt.savefig(f"{filename}.pdf")

    # Display the plot
    if show:
        plt.show()
    plt.close()
    return None


def show_atoms(atoms):
    """
    Plot the atomic structure using matplotlib.

    Parameters:
    atoms (ase.Atoms): ASE Atoms object representing the atomic structure.

    Returns:
    None
    """
    if isinstance(atoms, list):
        fig, ax = plt.subplots()
        for atom in atoms:
            plot_atoms(atom, ax)
        plt.show()
    else:
        fig, ax = plt.subplots()
        plot_atoms(atoms, ax)
        plt.show()


def plot_arrhenius(temperatures: list[float], rates: list[float]) -> None:
    """
    Create an Arrhenius plot of ln(k) vs 1/T.

    Parameters:
    temperatures (list[float]): List of temperatures in Kelvin.
    rates (list[float]): List of rate constants corresponding to temperatures.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    # Convert temperatures to 1/T (in K^-1)
    inv_temp = [1000.0 / t for t in temperatures]  # Multiply by 1000 for better scale

    # Calculate ln(k)
    ln_rates = np.log(rates)

    # Plot data
    ax.plot(inv_temp, ln_rates, 'o-', c='black', lw=2)

    # Labels and formatting
    ax_plot(fig, ax, r"$1000/T$ (K$^{-1}$)", r"ln($k$)")
    plt.show()
    return None


def plot_kappa_temperature(temperatures: list[float], kappa: list[float]) -> None:
    """
    Plot the temperature dependence of the kappa value.

    Parameters:
    temperatures (list[float]): List of temperatures in Kelvin.
    kappa (list[float]): List of kappa values corresponding to the temperatures.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    # Convert temperatures to 1/T (in K^-1)
    inv_temp = [1000.0 / t for t in temperatures]  # Multiply by 1000 for better scale

    # Plot data
    ax.plot(inv_temp, kappa, 'o-', c='black', lw=2)

    # Labels and formatting
    ax_plot(fig, ax, r"$1000/T$ (K$^{-1}$)", r"$\kappa$")
    plt.show()
    return None


def plot_kie_temperature(temperatures: list[float], kie: list[float]) -> None:
    """
    Plot the temperature dependence of the kinetic isotope effect (KIE).

    Parameters:
    temperatures (list[float]): List of temperatures in Kelvin.
    kie (list[float]): List of KIE values corresponding to the temperatures.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    # Convert temperatures to 1/T (in K^-1)
    inv_temp = [1000.0 / t for t in temperatures]  # Multiply by 1000 for better scale

    # Plot data
    ax.plot(inv_temp, kie, 'o-', c='black', lw=2)

    # Labels and formatting
    ax_plot(fig, ax, r"$1000/T$ (K$^{-1}$)", r"KIE")
    plt.show()
    return None


def plot_bead_convergence(n_beads: list[float], kappa: list[float]) -> None:
    """
    Plot the convergence of kappa values with respect to the number of beads.

    Parameters:
    n_beads (list[float]): List of the number of beads.
    kappa (list[float]): List of kappa values corresponding to the number of beads.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    # Plot data
    ax.plot(n_beads, kappa, 'o-', c='black', lw=2)

    # Labels and formatting
    ax_plot(fig, ax, r"Number of Beads", r"$\kappa$")
    plt.show()
    return None
