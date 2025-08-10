import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.visualize.plot import plot_atoms
from scipy.interpolate import make_interp_spline

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
                     diff=True,
                     save=True,
                     show=True,
                     filename="step_energy"):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    step = data["step"]
    energy = data["potential"]

    if diff:
        energy = np.abs(np.diff(energy))
        step = step[1:]
    else:
        energy = np.abs(energy)

    ax.plot(step, energy, c="black", label="potential", lw=2)
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
                             zero=True,
                             save=True,
                             show=True,
                             filename="time_potential_bias"):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    time = data["time"]

    ax.plot(time, np.subtract(data["potential"], data["potential"][0]), c="black", label="potential", lw=2)
    ax.plot(time, np.subtract(data["ensemble_bias"], data["ensemble_bias"][0]), c="red", label="bias", lw=2)

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


def plot_fes_series_1d(fes_arrays: list[np.ndarray],
                       times: list[float] = None,
                       max_times: int = 5,
                       save: bool = True,
                       show: bool = True,
                       filename: str = "fes_1d",
                       x_lab: str = r"CV1",
                       y_lab: str = r"$F$ (eV)") -> None:
    if times is None:
        times = np.arange(len(fes_arrays))

    # If there are more than max_times, select only the last max_times
    if len(times) > max_times:
        fes_arrays = fes_arrays[-max_times:]
        times = times[-max_times:]

    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)

    for i, xy in enumerate(fes_arrays):
        ax.plot(xy[0], xy[1], label=fr"$t={times[i]}$ ps")

    ax.legend()
    ax_plot(fig, ax, x_lab, y_lab)
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_fes_series_1d_compare(fes_arrays_a: list[np.ndarray],
                               fes_arrays_b: list[np.ndarray],
                               labels: list[str] = None,
                               save: bool = True,
                               show: bool = True,
                               filename: str = "fes_1d_compare",
                               x_lab: str = r"CV1",
                               y_lab: str = r"$F$ (eV)") -> None:
    if labels is None:
        labels = ["MD", "PIMD"]

    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)

    ax.plot(*fes_arrays_a, '-', label=labels[0], lw=2)
    ax.plot(*fes_arrays_b, '--', label=labels[1], lw=2)

    ax.legend()
    ax_plot(fig, ax, x_lab, y_lab)
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
                             max_times=5,
                             save=True,
                             show=True,
                             filename="fes_contourf",
                             x_lab=r"$d_\mathrm{OO}$  (Å)",
                             y_lab=r"$\Delta C_\mathrm{H}$") -> None:
    if times is None:
        times = np.arange(len(fes_arrays))

    # If there are more than 5 times, select only the last 5
    if len(times) > max_times:
        fes_arrays = fes_arrays[-max_times:]
        times = times[-max_times:]

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
        ax[i].set_xlabel(x_lab)
        ax[i].set_title(fr"$t={times[i]}$ ps")

    ax[0].set_ylabel(y_lab)
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
                              filename="fes_contourf_compare",
                              x_lab=r"$d_\mathrm{OO}$  (Å)",
                              y_lab=r"$\Delta C_\mathrm{H}$") -> None:
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
        ax[i].set_xlabel(x_lab)
        ax[i].set_title(fr"{titles[i]}")

    ax[0].set_ylabel(y_lab)
    fig.colorbar(contours[-1], ax=ax, orientation="vertical", label=r"$F$ (eV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_fes_contourf(fes,
                      save=True,
                      show=True,
                      filename="fes_contourf",
                      x_lab=r"$d_\mathrm{OO}$  (Å)",
                      y_lab=r"$\Delta C_\mathrm{H}$"
                      ) -> None:
    fig, ax = plt.subplots(1, 1,
                           figsize=(4, 3),
                           constrained_layout=True
                           )

    cf = ax.contourf(*fes)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    fig.colorbar(cf, ax=ax, orientation="vertical", label=r"$F$ (eV)")
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
                             filename="fes_contour_compare",
                             x_lab=r"$d_\mathrm{OO}$  (Å)",
                             y_lab=r"$\Delta C_\mathrm{H}$"
                             ):
    fig, ax = plt.subplots(
        1, 1, figsize=(4, 3), sharex=True, sharey=True, constrained_layout=True
    )

    levels = np.linspace(0, 0.5, 6)
    ax.contour(*fes_a, colors="b", levels=levels)
    ax.contour(*fes_b, colors="r", levels=levels)

    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)

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


def plot_neb(images, calc, save=True, show=True, filename="neb", smooth=True, k=2):
    # Attach the calculator to the images
    for image in images:
        image.calc = copy.copy(calc)

    # Get the energy
    energies = np.array([i.get_potential_energy() for i in images])
    energies -= min(energies)

    # Get the path
    path = get_neb_path(images)

    if smooth:
        spl = make_interp_spline(path, energies, k=k)
        path_smooth = np.linspace(min(path), max(path), 100)
        energies_smooth = spl(path_smooth)
        plt.scatter(path, energies, c='k')

        # Plot both spline and scatter points
        plt.plot(path_smooth, energies_smooth, '-', c='k', lw=2)
    else:
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


def plot_sella(images, calc, save=True, show=True, filename="irc", smooth=True, k=2):
    # Attach the calculator to the images
    for image in images:
        image.calc = copy.copy(calc)

    # Get the energy
    energies = np.array([i.get_potential_energy() for i in images])

    # Shift the graph so the minimum energy is zero
    energies -= min(energies)

    # Get the path
    path = get_neb_path(images)

    if smooth:
        spl = make_interp_spline(path, energies, k=k)
        path_smooth = np.linspace(min(path), max(path), 100)
        energies_smooth = spl(path_smooth)
        plt.scatter(path, energies, c='k')

        # Plot both spline and scatter points
        plt.plot(path_smooth, energies_smooth, '-', c='k', lw=2)
    else:
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


def plot_arrhenius_2(temperatures: list[float],
                     rates_c: list[float],
                     rates_q: list[float],
                     save=True,
                     show=True,
                     filename="arrhenius") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    # Convert temperatures to 1/T (in K^-1)
    inv_temp = [1000.0 / t for t in temperatures]  # Multiply by 1000 for better scale

    # Plot data
    ax.plot(inv_temp, np.log(rates_c), 'o-', c='black', lw=2, label="Classical")
    ax.plot(inv_temp, np.log(rates_q), 'o-', c='red', lw=2, label="Quantum")
    ax.legend()
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

    # Plot data
    ax.plot(temperatures, kappa, 'o-', c='black', lw=2)

    # Labels and formatting
    ax_plot(fig, ax, r"Temperature (K)", r"$\kappa$")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_kappa_temperature_inv(temperatures: list[float],
                               kappa: list[float],
                               save=True,
                               show=True,
                               filename="kappa_temperature_inv") -> None:
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
                          filename="bead_temperature"
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


def _load_plumed_colvar(path, field, derivative=False, x="time"):
    """
    Load and process data from a PLUMED colvar file.

    This function reads a PLUMED colvar file, extracts the specified columns,
    and optionally computes the derivative of the selected field.

    Parameters:
    path (str or pathlib.Path): Path to the PLUMED colvar file.
    field (str): Name of the field to extract from the file.
    derivative (bool, optional): Whether to compute the derivative of the field. Default is False.
    x (str, optional): Name of the column to use as the x-axis. Default is "time".

    Returns:
    tuple:
        - x_vals (numpy.ndarray): Values of the x-axis column.
        - y_vals (numpy.ndarray): Values of the field column (or its derivative if `derivative=True`).

    Raises:
    ValueError: If the file does not start with the expected header, or if the requested columns are not found.
    ValueError: If there are insufficient data points to compute the derivative.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()

    prefix = "#! FIELDS"
    if not header.startswith(prefix):
        raise ValueError("First line must start with '#! FIELDS'.")

    # Extract column names after '#! FIELDS'
    names = header[len(prefix):].strip().split()
    if not names:
        raise ValueError("No column names found after '#! FIELDS'.")

    # Validate requested columns
    if x not in names:
        raise ValueError(f"x-axis column '{x}' not found. Available: {names}")
    if field not in names:
        raise ValueError(f"Field '{field}' not found. Available: {names}")

    data = np.loadtxt(path, comments="#")
    name_to_idx = {name: i for i, name in enumerate(names)}
    x_idx = name_to_idx[x]
    y_idx = name_to_idx[field]

    x_vals = data[:, x_idx]
    y_vals = data[:, y_idx]

    dt = x_vals[1] - x_vals[0]
    if derivative:
        if len(x_vals) < 2:
            raise ValueError("Not enough data points to compute derivative.")
        dy = np.gradient(y_vals, dt)
        y_vals = dy
    return x_vals, y_vals


def plot_plumed_field(path,
                      field,
                      x="time",
                      save=True,
                      show=True,
                      filename="bead_temperature",
                      derivative=False):
    """
    Plot a specific field from a PLUMED colvar file.

    things one might want to plot:
    rct, (estimate of c(t)) should flatten (no drift) once the bias is stationary.
    zed, (normalization Zn ) should stop changing when no new CV region is being explored.
    neff, (effective sample size) should keep growing; a long plateau too early often means you’re not visiting enough of CV space.

    This function loads data from a PLUMED colvar file, extracts the specified field,
    and plots it against the x-axis column. Optionally, it computes the derivative of the field.

    Parameters:
    path (str or pathlib.Path): Path to the PLUMED colvar file.
    field (str): Name of the field to plot.
    x (str, optional): Name of the column to use as the x-axis. Default is "time".
    save (bool, optional): Whether to save the plot as a file. Default is True.
    show (bool, optional): Whether to display the plot. Default is True.
    filename (str, optional): Filename for saving the plot. Default is "bead_temperature".
    derivative (bool, optional): Whether to compute the derivative of the field. Default is False.

    Returns:
    None
    """
    x_vals, y_vals = _load_plumed_colvar(path, field, derivative=derivative, x=x)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    # Plot data
    ax.plot(x_vals, y_vals, 'o-', c='black', lw=2)

    # Labels and formatting
    ax_plot(fig, ax, x, field)
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    else:
        plt.close()
    return None
