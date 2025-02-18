import copy

import matplotlib.pyplot as plt
import numpy as np
from ase.visualize.plot import plot_atoms

from .calcs import moving_average
from .pathway import get_neb_path


def plot_time_energy(output_data):
    """
    Plot the potential and ensemble bias energy over time.

    Parameters:
    output_data (dict): Dictionary containing the time, potential, and ensemble bias energy data.

    Returns:
    None
    """
    # convert time to ps and energy to eV
    time_conv = 2.4188843e-05
    energy_conv = 27.211386
    time = time_conv * output_data["time"]
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    ax.plot(
        time,
        energy_conv * output_data["potential"],
        "r",
        label="potential",
    )
    ax.plot(
        time,
        energy_conv * output_data["ensemble_bias"],
        "b",
        label="bias",
    )

    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"energy / eV")
    ax.legend(loc="upper left", ncols=1)
    plt.show()


def plot_time_temperature(output_data):
    """
    Plot the temperature over time for different components.

    Parameters:
    output_data (dict): Dictionary containing the time and temperature data for different components.

    Returns:
    None
    """
    # convert time to ps
    time_conv = 2.4188843e-05
    time = time_conv * output_data["time"]
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    ax.plot(
        time[50:-49],
        moving_average(output_data["temperature(O)"], 100),
        "r",
        label=r"$T_\mathrm{O}$",
    )
    ax.plot(
        time[50:-49],
        moving_average(output_data["temperature(H)"], 100),
        "gray",
        label=r"$T_\mathrm{H}$",
    )
    ax.plot(
        time[50:-49],
        moving_average(output_data["temperature"], 100),
        "b",
        label="T",
    )

    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"temperature / K")
    ax.legend(loc="upper left", ncols=2)
    plt.show()


def plot_energy_contour_series(xyz_1, xyz_2, xyz_3):
    """
    Plot a series of energy contour plots for three sets of data.

    Parameters:
    xyz_1 (tuple): Data for the first contour plot.
    xyz_2 (tuple): Data for the second contour plot.
    xyz_3 (tuple): Data for the third contour plot.

    Returns:
    None
    """
    fig, ax = plt.subplots(
        1, 3, figsize=(8, 3), sharex=True, sharey=True, constrained_layout=True
    )

    cf_1 = ax[0].contourf(*xyz_1)
    cf_2 = ax[1].contourf(*xyz_2)
    cf_3 = ax[2].contourf(*xyz_3)
    fig.colorbar(cf_3, ax=ax, orientation="vertical", label=r"$F$ / eV")
    # Set the y-axis label
    ax[0].set_ylabel(r"$\Delta C_\mathrm{H}$")
    # Set the x-axis label
    ax[0].set_xlabel(r"$d_\mathrm{OO}$ / Å")
    ax[1].set_xlabel(r"$d_\mathrm{OO}$ / Å")
    ax[2].set_xlabel(r"$d_\mathrm{OO}$ / Å")
    # Set the title of the plot
    ax[0].set_title(r"$t=0.8$ ps")
    ax[1].set_title(r"$t=2.5$ ps")
    ax[2].set_title(r"$t=5.0$ ps")
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
    n_plot("Path, (Å)", "Energy, (eV)")

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
    n_plot("Path, (Å)", "Energy eV")

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
