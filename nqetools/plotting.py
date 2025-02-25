import copy

import matplotlib.pyplot as plt
import numpy as np
from ase.visualize.plot import plot_atoms

from .calcs import moving_average
from .pathway import get_neb_path

# Setting plot aesthetics for better visibility
plt.rcParams['axes.linewidth'] = 2.0


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


def show_atoms(atoms,
               save=True,
               show=True,
               filename="atoms"):
    if isinstance(atoms, list):
        fig, ax = plt.subplots()
        for atom in atoms:
            plot_atoms(atom, ax)
    else:
        fig, ax = plt.subplots()
        plot_atoms(atoms, ax)

    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()


def plot_step_energy(data,
                     save=True,
                     show=True,
                     filename="step_energy"):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    step = data["step"]

    ax.plot(step, np.abs(data["potential"]), c="black", label="potential", lw=2)
    ax.set_yscale("log")
    ax_plot(fig, ax, r"Optimiser step", r"Energy (eV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_time_potential_bias(data,
                             save=True,
                             show=True,
                             filename="time_potential_bias"):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    time = data["time"]

    ax.plot(time, data["potential"], c="black", label="potential", lw=2)
    ax.plot(time, data["ensemble_bias"], c="red", label="bias", lw=2)

    ax.legend(loc="upper left", ncols=1)
    ax_plot(fig, ax, r"$t$ (ps)", r"Energy (eV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_time_temperature(data,
                          window_size=100,
                          save=True,
                          show=True,
                          filename="time_temperature"):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    min_val = int(window_size / 2)
    max_val = -int(window_size / 2 - 1)

    time = data["time"][min_val:max_val]
    ave_temperature = moving_average(data["temperature"], window_size)

    ax.plot(time, ave_temperature, c="blue", lw=2)

    ax_plot(fig, ax, r"$t$ (ps)", r"Temperature (K)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_time_energy_conservation(data: dict,
                                  save=True,
                                  show=True,
                                  filename="time_conservation") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    time = data["time"]

    # Plot individual energy components
    ax.plot(time, data["potential"], c="black", label="Potential", lw=2)
    ax.plot(time, data["conserved"], c="red", label="Total", lw=2)

    ax.legend(loc="upper left", ncols=1)
    ax_plot(fig, ax, r"$t$ (ps)", r"Energy (eV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_fes_contourf_series(fes_arrays: list[np.ndarray],
                             times: list[float] = None,
                             save=True,
                             show=True,
                             filename="fes_contourf") -> None:
    if times is None:
        times = np.arange(len(fes_arrays))

    fig, ax = plt.subplots(
        1,
        len(fes_arrays),
        figsize=(8, 3),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    contours = []
    for i, xyz in enumerate(fes_arrays):
        cf = ax[i].contourf(*xyz)
        contours.append(cf)
        ax[i].set_xlabel(r"$d_\mathrm{OO}$  (Å)")
        ax[i].set_title(fr"$t={times[i]}$ ps")

    ax[0].set_ylabel(r"$\Delta C_\mathrm{H}$")
    fig.colorbar(contours[-1], ax=ax, orientation="vertical", label=r"$F$ (eV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_fes_contourf_compare(fes_a,
                              fes_b,
                              titles=None,
                              save=True,
                              show=True,
                              filename="fes_contourf_compare") -> None:
    fes_arrays = [fes_a, fes_b]
    if titles is None:
        titles = ["MD", "PIMD"]
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(8, 3),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    contours = []
    for i, xyz in enumerate(fes_arrays):
        cf = ax[i].contourf(*xyz)
        contours.append(cf)
        ax[i].set_xlabel(r"$d_\mathrm{OO}$  (Å)")
        ax[i].set_title(fr"{titles[i]}")

    ax[0].set_ylabel(r"$\Delta C_\mathrm{H}$")
    fig.colorbar(contours[-1], ax=ax, orientation="vertical", label=r"$F$ (eV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_fes_contour_compare(fes_a,
                             fes_b,
                             save=True,
                             show=True,
                             filename="fes_contour_compare"):
    fig, ax = plt.subplots(
        1, 1, figsize=(4, 3), sharex=True, sharey=True, constrained_layout=True
    )

    levels = np.linspace(0, 0.5, 6)
    cp1 = ax.contour(*fes_a, colors="b", levels=levels)
    cp2 = ax.contour(*fes_b, colors="r", levels=levels)
    ax.set_ylabel(r"$\Delta C_\mathrm{H}$")
    ax.set_xlabel(r"$d_\mathrm{OO}$ (Å)")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="b", label="MD"),
            plt.Line2D([0], [0], color="r", label="PIMD"),
        ]
    )
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_fes_sep(fes_a,
                 fes_b,
                 save=True,
                 show=True,
                 filename="energy_sep"):
    fig, ax = plt.subplots(1,
                           1,
                           figsize=(4, 3),
                           sharex=True,
                           sharey=True,
                           constrained_layout=True
                           )

    ax.plot(
        fes_a[1, :, 50],
        fes_a[2, :, 50],
        "b",
        label=r"MD, $d_\mathrm{OO}=2.6 $Å"
    )
    ax.plot(
        fes_b[1, :, 50],
        fes_b[2, :, 50],
        "r",
        label=r"PIMD, $d_\mathrm{OO}=2.6 $Å",
    )
    ax.plot(
        fes_a[1, :, 60],
        fes_a[2, :, 60],
        "b--",
        label=r"MD, $d_\mathrm{OO}=2.7 $Å",
    )
    ax.plot(
        fes_b[1, :, 60],
        fes_b[2, :, 60],
        "r--",
        label=r"PIMD, $d_\mathrm{OO}=2.7 $Å",
    )
    ax.set_ylim(0.08, 0.6)
    ax.legend(ncols=2, loc="upper right", fontsize=9)
    ax.set_ylabel(r"$F$ (eV)")
    ax.set_xlabel(r"$\Delta C_\mathrm{H}$")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_neb(images, calc, save=True, show=True, filename="neb"):
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

    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_sella(images, calc, save=True, show=True, filename="irc"):
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

    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_arrhenius(temperatures: list[float],
                   rates: list[float],
                   save=True,
                   show=True,
                   filename="arrhenius") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    # Convert temperatures to 1/T (in K^-1)
    inv_temp = [1000.0 / t for t in temperatures]  # Multiply by 1000 for better scale

    # Calculate ln(k)
    ln_rates = np.log(rates)

    # Plot data
    ax.plot(inv_temp, ln_rates, 'o-', c='black', lw=2)

    # Labels and formatting
    ax_plot(fig, ax, r"$1000/T$ (K$^{-1}$)", r"ln($k$)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_kappa_temperature(temperatures: list[float],
                           kappa: list[float],
                           save=True,
                           show=True,
                           filename="kappa_temperature") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    # Convert temperatures to 1/T (in K^-1)
    inv_temp = [1000.0 / t for t in temperatures]  # Multiply by 1000 for better scale

    # Plot data
    ax.plot(inv_temp, kappa, 'o-', c='black', lw=2)

    # Labels and formatting
    ax_plot(fig, ax, r"$1000/T$ (K$^{-1}$)", r"$\kappa$")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_kie_temperature(temperatures: list[float],
                         kie: list[float],
                         save=True,
                         show=True,
                         filename="kie_temperature") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    # Convert temperatures to 1/T (in K^-1)
    inv_temp = [1000.0 / t for t in temperatures]  # Multiply by 1000 for better scale

    # Plot data
    ax.plot(inv_temp, kie, 'o-', c='black', lw=2)

    # Labels and formatting
    ax_plot(fig, ax, r"$1000/T$ (K$^{-1}$)", r"KIE")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_bead_convergence(n_beads: list[float],
                          kappa: list[float],
                          save=True,
                          show=True,
                          filename="kie_temperature"
                          ) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    # Plot data
    ax.plot(n_beads, kappa, 'o-', c='black', lw=2)

    # Labels and formatting
    ax_plot(fig, ax, r"Number of Beads", r"$\kappa$")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None
