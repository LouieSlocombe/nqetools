import copy
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize.plot import plot_atoms
from pathlib import Path
from scipy.interpolate import make_interp_spline

from .calcs import moving_average
from .pathway import get_neb_path

# Setting plot aesthetics for better visibility
plt.rcParams['axes.linewidth'] = 2.0


def n_plot(xlab,
           ylab,
           xs=14,
           ys=14):
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None


def ax_plot(fig,
            ax,
            xlab,
            ylab,
            xs=14,
            ys=14):
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
                     fig=None,
                     ax=None,
                     diff=True,
                     save=True,
                     show=True,
                     filename="step_energy",
                     fig_size=(8, 3),
                     y_scale='log'):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    step = data["step"]
    energy = data["potential"]

    if diff:
        energy = np.abs(np.diff(energy))
        step = step[1:]
    else:
        energy = np.abs(energy)

    ax.plot(step, energy, c="black", label="potential", lw=2)
    ax.set_yscale(y_scale)
    ax_plot(fig, ax, r"Optimiser step", r"Energy (eV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_time_potential_bias(data,
                             fig=None,
                             ax=None,
                             save=True,
                             show=True,
                             filename="time_potential_bias",
                             fig_size=(8, 3)):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    time = data["time"]

    ax.plot(time, np.subtract(data["potential"], data["potential"][0]), c="black", label="potential", lw=2)
    ax.plot(time, np.subtract(data["ensemble_bias"], data["ensemble_bias"][0]), c="red", label="bias", lw=2)

    ax.legend(loc="best", ncols=1)
    ax_plot(fig, ax, r"$t$ (ps)", r"Energy (eV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_time_temperature(data,
                          fig=None,
                          ax=None,
                          window_size=100,
                          mov_ave=True,
                          save=True,
                          show=True,
                          filename="time_temperature",
                          fig_size=(8, 3)):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    if mov_ave:
        min_val = int(window_size / 2)
        max_val = -int(window_size / 2 - 1)
        time = data["time"][min_val:max_val]
        ave_temperature = moving_average(data["temperature"], window_size)
        ax.plot(time, ave_temperature, c="black", lw=2)
    else:
        ax.plot(data["time"], data["temperature"], c="black", lw=2)

    ax_plot(fig, ax, r"$t$ (ps)", r"Temperature (K)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_time_energy_conservation(data,
                                  fig=None,
                                  ax=None,
                                  save=True,
                                  show=True,
                                  filename="time_conservation",
                                  fig_size=(8, 3)):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    time = data["time"]

    # Plot individual energy components
    ax.plot(time, np.subtract(data["potential"], data["potential"][0]), c="black", label="Potential", lw=2)
    ax.plot(time, np.subtract(data["conserved"], data["conserved"][0]), c="red", label="Total", lw=2)

    ax.legend(loc="upper left", ncols=1)
    ax_plot(fig, ax, r"$t$ (ps)", r"Energy (eV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_fes_series_1d(fes_arrays: list[np.ndarray],
                       fig=None,
                       ax=None,
                       slices: list[float] = None,
                       labels: list[str] = None,
                       max_slices: int = 5,
                       save: bool = True,
                       show: bool = True,
                       filename: str = "fes_1d",
                       x_lab: str = r"CV1",
                       y_lab: str = r"$F$ (eV)",
                       fig_size: tuple = (8, 3)):
    if slices is None:
        slices = np.arange(len(fes_arrays))

    if len(slices) > max_slices:
        fes_arrays = fes_arrays[-max_slices:]
        slices = slices[-max_slices:]
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    for i, xy in enumerate(fes_arrays):
        if labels is not None:
            label = labels[i]
        else:
            label = fr"$t={slices[i]}$ ps"
        ax.plot(xy[0], xy[1], label=label)

    ax.legend()
    ax_plot(fig, ax, x_lab, y_lab)
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_fes_series_1d_compare(fes_arrays_a: list[np.ndarray],
                               fes_arrays_b: list[np.ndarray],
                               fig=None,
                               ax=None,
                               labels: list[str] = None,
                               save: bool = True,
                               show: bool = True,
                               filename: str = "fes_1d_compare",
                               x_lab: str = r"CV1",
                               y_lab: str = r"$F$ (eV)",
                               fig_size: tuple = (8, 3)):
    if labels is None:
        labels = ["MD", "PIMD"]
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    ax.plot(*fes_arrays_a, '-', label=labels[0], lw=2)
    ax.plot(*fes_arrays_b, '--', label=labels[1], lw=2)

    ax.legend(loc="best")
    ax_plot(fig, ax, x_lab, y_lab)
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_fes_contourf_series(fes_arrays: list[np.ndarray],
                             fig=None,
                             ax=None,
                             times: list[float] = None,
                             max_times=5,
                             save=True,
                             show=True,
                             filename="fes_contourf",
                             x_lab="CV1",
                             y_lab="CV2",
                             fig_size=(8, 3)):
    if times is None:
        times = np.arange(len(fes_arrays))

    # If there are more than 5 times, select only the last 5
    if len(times) > max_times:
        fes_arrays = fes_arrays[-max_times:]
        times = times[-max_times:]
    if fig is None or ax is None:
        fig, ax = plt.subplots(
            nrows=1,
            ncols=len(fes_arrays),
            figsize=fig_size,
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
    return fig, ax


def plot_fes_contourf_compare(fes_a,
                              fes_b,
                              fig=None,
                              ax=None,
                              labels=None,
                              save=True,
                              show=True,
                              filename="fes_contourf_compare",
                              x_lab="CV1",
                              y_lab="CV2",
                              fig_size=(8, 3)):
    fes_arrays = [fes_a, fes_b]
    if labels is None:
        labels = ["MD", "PIMD"]
    if fig is None or ax is None:
        fig, ax = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=fig_size,
            sharex=True,
            sharey=True,
            constrained_layout=True
        )

    contours = []
    for i, xyz in enumerate(fes_arrays):
        cf = ax[i].contourf(*xyz)
        contours.append(cf)
        ax[i].set_xlabel(x_lab)
        ax[i].set_title(fr"{labels[i]}")

    ax[0].set_ylabel(y_lab)
    fig.colorbar(contours[-1], ax=ax, orientation="vertical", label=r"$F$ (eV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_fes_contourf(fes,
                      fig=None,
                      ax=None,
                      save=True,
                      show=True,
                      filename="fes_contourf",
                      x_lab="CV1",
                      y_lab="CV2",
                      fig_size=(8, 3),
                      ):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    cf = ax.contourf(*fes)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    fig.colorbar(cf, ax=ax, orientation="vertical", label=r"$F$ (eV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_fes_contour_compare(fes_a,
                             fes_b,
                             fig=None,
                             ax=None,
                             labels=None,
                             save=True,
                             show=True,
                             filename="fes_contour_compare",
                             x_lab="CV1",
                             y_lab="CV2",
                             fig_size=(8, 3),
                             ):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    if labels is None:
        labels = ["MD", "PIMD"]

    levels = np.linspace(0, 0.5, 6)
    ax.contour(*fes_a, colors="b", levels=levels)
    ax.contour(*fes_b, colors="r", levels=levels)

    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="b", label=labels[0]),
            plt.Line2D([0], [0], color="r", label=labels[1]),
        ]
    )
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_fes_sep(fes_a,
                 fes_b,
                 fig=None,
                 ax=None,
                 save=True,
                 show=True,
                 filename="energy_sep",
                 fig_size=(8, 3)):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

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
    return fig, ax


def plot_neb(images,
             calc,
             fig=None,
             ax=None,
             save=True,
             show=True,
             smooth=True,
             k=2,
             fig_size=(8, 3),
             filename="neb",
             label=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    for image in images:
        image.calc = copy.copy(calc)

    energies = np.array([i.get_potential_energy() for i in images])
    energies -= min(energies)

    path = get_neb_path(images)

    if smooth:
        spl = make_interp_spline(path, energies, k=k)
        path_smooth = np.linspace(min(path), max(path), 100)
        energies_smooth = spl(path_smooth)
        ax.scatter(path, energies)

        ax.plot(path_smooth, energies_smooth, '-', lw=2, label=label)
    else:
        ax.plot(path, energies, 'o-', lw=2, label=label)

    ax_plot(fig, ax, "Path (Å)", "Energy (eV)")

    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_sella(images,
               calc,
               fig=None,
               ax=None,
               save=True,
               show=True,
               filename="irc",
               smooth=True,
               k=2,
               fig_size=(8, 3)):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    for image in images:
        image.calc = copy.copy(calc)

    energies = np.array([i.get_potential_energy() for i in images])

    # Shift the graph so the minimum energy is zero
    energies -= min(energies)

    path = get_neb_path(images)

    if smooth:
        spl = make_interp_spline(path, energies, k=k)
        path_smooth = np.linspace(min(path), max(path), 100)
        energies_smooth = spl(path_smooth)
        plt.scatter(path, energies, c='k')

        ax.plot(path_smooth, energies_smooth, '-', c='k', lw=2)
    else:
        ax.plot(path, energies, 'o-', c='k', lw=2)

    ax_plot(fig, ax, "Path (Å)", "Energy (eV)")

    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_arrhenius(temperatures: list[float],
                   rates: list[float],
                   fig=None,
                   ax=None,
                   save=True,
                   show=True,
                   filename="arrhenius",
                   fig_size=(8, 3)) -> None:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    inv_temp = [1000.0 / t for t in temperatures]  # Multiply by 1000 for better scale

    ln_rates = np.log(rates)

    ax.plot(inv_temp, ln_rates, 'o-', c='black', lw=2)

    ax_plot(fig, ax, r"$1000/T$ (K$^{-1}$)", r"ln($k$)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_arrhenius_2(temperatures: list[float],
                     rates_c: list[float],
                     rates_q: list[float],
                     fig=None,
                     ax=None,
                     save=True,
                     show=True,
                     filename="arrhenius",
                     fig_size=(8, 3)) -> None:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    inv_temp = [1000.0 / t for t in temperatures]  # Multiply by 1000 for better scale

    ax.plot(inv_temp, np.log(rates_c), 'o-', c='black', lw=2, label="Classical")
    ax.plot(inv_temp, np.log(rates_q), 'o-', c='red', lw=2, label="Quantum")
    ax.legend()
    ax_plot(fig, ax, r"$1000/T$ (K$^{-1}$)", r"ln($k$)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_kappa_temperature(temperatures: list[float],
                           kappa: list[float],
                           fig=None,
                           ax=None,
                           save=True,
                           show=True,
                           filename="kappa_temperature",
                           fig_size=(8, 3)) -> None:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    ax.plot(temperatures, kappa, 'o-', c='black', lw=2)

    ax_plot(fig, ax, r"Temperature (K)", r"$\kappa$")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_kappa_temperature_inv(temperatures: list[float],
                               kappa: list[float],
                               fig=None,
                               ax=None,
                               save=True,
                               show=True,
                               filename="kappa_temperature_inv",
                               fig_size=(8, 3)) -> None:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    inv_temp = [1000.0 / t for t in temperatures]  # Multiply by 1000 for better scale

    ax.plot(inv_temp, kappa, 'o-', c='black', lw=2)

    ax_plot(fig, ax, r"$1000/T$ (K$^{-1}$)", r"$\kappa$")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_kie_temperature(temperatures: list[float],
                         kie: list[float],
                         fig=None,
                         ax=None,
                         save=True,
                         show=True,
                         filename="kie_temperature",
                         fig_size=(8, 3)) -> None:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    inv_temp = [1000.0 / t for t in temperatures]  # Multiply by 1000 for better scale

    ax.plot(inv_temp, kie, 'o-', c='black', lw=2)

    ax_plot(fig, ax, r"$1000/T$ (K$^{-1}$)", r"KIE")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_bead_convergence(n_beads: list[float],
                          kappa: list[float],
                          fig=None,
                          ax=None,
                          save=True,
                          show=True,
                          filename="bead_temperature",
                          fig_size=(8, 3)
                          ) -> None:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    ax.plot(n_beads, kappa, 'o-', c='black', lw=2)

    ax_plot(fig, ax, r"Number of Beads", r"$\kappa$")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def _load_plumed_colvar(path,
                        field,
                        derivative=False,
                        x="time"):
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()

    prefix = "#! FIELDS"
    if not header.startswith(prefix):
        raise ValueError("First line must start with '#! FIELDS'.")

    names = header[len(prefix):].strip().split()
    if not names:
        raise ValueError("No column names found after '#! FIELDS'.")

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
                      fig=None,
                      ax=None,
                      x="time",
                      save=True,
                      show=True,
                      filename="bead_temperature",
                      derivative=False,
                      fig_size=(8, 3)):
    x_vals, y_vals = _load_plumed_colvar(path,
                                         field,
                                         derivative=derivative,
                                         x=x)

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    ax.plot(x_vals, y_vals, 'o-', c='black', lw=2)

    ax_plot(fig, ax, x, field)
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax
