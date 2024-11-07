import copy

import matplotlib.pyplot as plt
import numpy as np

from .calcs import moving_average
from .pathway import get_neb_path


def plot_time_energy(output_data):
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
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None


def plot_neb(images, calc):
    # Attach the calculator to the images
    for image in images:
        image.calc = copy.copy(calc)
    # Get the energy
    energies = np.array([i.get_potential_energy() for i in images])
    # shift the graph
    energies -= min(energies)
    # Get the path
    path = get_neb_path(images)
    plt.plot(path, energies)
    n_plot("Path A", "Energy eV")
    plt.show()
