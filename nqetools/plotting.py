"""Plotting helpers for i-PI and PLUMED simulation output.

Covers optimiser and MD diagnostics (energy, temperature, conservation)
and rate-theory summaries (Arrhenius, transmission coefficients, kinetic
isotope effects).

Every plotting routine follows the same contract: it accepts an optional
``fig``/``ax`` pair to draw into, otherwise creating its own, and returns
that pair so plots can be composed or further customised by the caller.

Free energy surfaces are not plotted here. Reading a PLUMED surface and
drawing it in one or two dimensions lives in :mod:`reactiontools.tools_fes`,
which also supplies the house styling used below: importing
:func:`~reactiontools.ax_plot` is what sets the figure frame width, so every
plot in both packages comes out looking the same.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from reactiontools import ax_plot

from .calcs import moving_average


def plot_step_energy(data,
                     fig=None,
                     ax=None,
                     diff=True,
                     save=True,
                     show=True,
                     filename="step_energy",
                     fig_size=(8, 3),
                     y_scale='log') -> tuple[Figure, Axes]:
    """Plot potential energy against optimiser step.

    Parameters
    ----------
    data : dict
        Simulation output containing 'step' and 'potential' columns.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new one is created if either this or `fig`
        is None.
    diff : bool, optional
        If True, plot the absolute step-to-step energy change, which shows
        convergence more clearly than the raw energy. Default is True.
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "step_energy".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).
    y_scale : str, optional
        Matplotlib y-axis scale. Default is 'log'.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.
    """
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
                             fig_size=(8, 3)) -> tuple[Figure, Axes]:
    """Plot potential energy and enhanced-sampling bias against time.

    Both traces are offset to start at zero so their relative drift can be
    compared on a single axis.

    Parameters
    ----------
    data : dict
        Simulation output containing 'time', 'potential' and
        'ensemble_bias' columns.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new one is created if either this or `fig`
        is None.
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "time_potential_bias".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.
    """
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
                          fig_size=(8, 3)) -> tuple[Figure, Axes]:
    """Plot instantaneous temperature against time.

    Parameters
    ----------
    data : dict
        Simulation output containing 'time' and 'temperature' columns.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new one is created if either this or `fig`
        is None.
    window_size : int, optional
        Width of the moving-average window in frames. Default is 100.
    mov_ave : bool, optional
        If True, smooth the trace with a moving average, which makes the
        thermostat's target temperature visible through the fluctuations.
        Default is True.
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "time_temperature".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    if mov_ave:
        ave_temperature = moving_average(data["temperature"], window_size)
        # Offsets come from the actual output length so that odd window sizes,
        # which drop an uneven number of points from each end, still line up.
        lead = (window_size - 1) // 2
        time = data["time"][lead:lead + len(ave_temperature)]
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
                                  fig_size=(8, 3)) -> tuple[Figure, Axes]:
    """Plot potential and conserved energy against time.

    Drift in the conserved quantity is the standard diagnostic for an
    unstable integrator or too large a timestep. Both traces are offset to
    start at zero.

    Parameters
    ----------
    data : dict
        Simulation output containing 'time', 'potential' and 'conserved'
        columns.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new one is created if either this or `fig`
        is None.
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "time_conservation".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    time = data["time"]

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


def plot_arrhenius(temperatures: list[float],
                   rates: list[float],
                   fig=None,
                   ax=None,
                   save=True,
                   show=True,
                   filename="arrhenius",
                   fig_size=(8, 3)) -> tuple[Figure, Axes]:
    """Plot an Arrhenius plot of ln(k) against inverse temperature.

    Curvature away from a straight line is the signature of quantum
    tunnelling contributing to the rate.

    Parameters
    ----------
    temperatures : list of float
        Temperatures in K.
    rates : list of float
        Rate constants, one per temperature.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new one is created if either this or `fig`
        is None.
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "arrhenius".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    inv_temp = [1000.0 / t for t in temperatures]  # Scaled by 1000 to keep the axis readable

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
                     fig_size=(8, 3)) -> tuple[Figure, Axes]:
    """Plot classical and quantum rates on a single Arrhenius plot.

    The gap between the two traces widens as temperature falls, showing
    where tunnelling begins to dominate.

    Parameters
    ----------
    temperatures : list of float
        Temperatures in K.
    rates_c : list of float
        Classical rate constants, one per temperature.
    rates_q : list of float
        Quantum rate constants, one per temperature.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new one is created if either this or `fig`
        is None.
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "arrhenius".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    inv_temp = [1000.0 / t for t in temperatures]  # Scaled by 1000 to keep the axis readable

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
                           fig_size=(8, 3)) -> tuple[Figure, Axes]:
    """Plot the tunnelling enhancement factor against temperature.

    Parameters
    ----------
    temperatures : list of float
        Temperatures in K.
    kappa : list of float
        Tunnelling enhancement factors, one per temperature.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new one is created if either this or `fig`
        is None.
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "kappa_temperature".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.
    """
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
                               fig_size=(8, 3)) -> tuple[Figure, Axes]:
    """Plot the tunnelling enhancement factor against inverse temperature.

    Uses the same abscissa as the Arrhenius plots, so the two can be read
    side by side.

    Parameters
    ----------
    temperatures : list of float
        Temperatures in K.
    kappa : list of float
        Tunnelling enhancement factors, one per temperature.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new one is created if either this or `fig`
        is None.
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "kappa_temperature_inv".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    inv_temp = [1000.0 / t for t in temperatures]  # Scaled by 1000 to keep the axis readable

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
                         fig_size=(8, 3)) -> tuple[Figure, Axes]:
    """Plot the kinetic isotope effect against inverse temperature.

    Parameters
    ----------
    temperatures : list of float
        Temperatures in K.
    kie : list of float
        Kinetic isotope effects, one per temperature.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new one is created if either this or `fig`
        is None.
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "kie_temperature".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    inv_temp = [1000.0 / t for t in temperatures]  # Scaled by 1000 to keep the axis readable

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
                          ) -> tuple[Figure, Axes]:
    """Plot the tunnelling enhancement factor against ring-polymer size.

    Used to choose the smallest bead count at which the result has
    plateaued, since cost scales linearly with the number of beads.

    Parameters
    ----------
    n_beads : list of float
        Number of ring-polymer beads.
    kappa : list of float
        Tunnelling enhancement factors, one per bead count.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new one is created if either this or `fig`
        is None.
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "bead_temperature".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.
    """
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
    """Read one named column from a PLUMED COLVAR file.

    Column names are taken from the '#! FIELDS' header line, so fields can
    be requested by name rather than by position.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the COLVAR file.
    field : str
        Name of the column to return as the ordinate.
    derivative : bool, optional
        If True, return the numerical time derivative of the field instead
        of its raw values. Default is False.
    x : str, optional
        Name of the column to return as the abscissa. Default is "time".

    Returns
    -------
    tuple of numpy.ndarray
        The abscissa values and the requested field values.

    Raises
    ------
    ValueError
        If the header is missing, either column name is absent, or a
        derivative is requested from fewer than two rows.
    """
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

    # loadtxt squeezes a single-row file to 1-D, which would break the column indexing
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    name_to_idx = {name: i for i, name in enumerate(names)}
    x_idx = name_to_idx[x]
    y_idx = name_to_idx[field]

    x_vals = data[:, x_idx]
    y_vals = data[:, y_idx]

    if derivative:
        if len(x_vals) < 2:
            raise ValueError("Not enough data points to compute derivative.")
        dt = x_vals[1] - x_vals[0]
        y_vals = np.gradient(y_vals, dt)
    return x_vals, y_vals


def plot_plumed_field(path,
                      field,
                      fig=None,
                      ax=None,
                      x="time",
                      save=True,
                      show=True,
                      filename="plumed_field",
                      derivative=False,
                      fig_size=(8, 3)) -> tuple[Figure, Axes]:
    """Plot a named column from a PLUMED COLVAR file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the COLVAR file.
    field : str
        Name of the column to plot, also used as the y-axis label.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new one is created if either this or `fig`
        is None.
    x : str, optional
        Name of the column to use as the abscissa. Default is "time".
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "plumed_field".
    derivative : bool, optional
        If True, plot the numerical time derivative of the field. Default
        is False.
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.

    Raises
    ------
    ValueError
        Propagated from :func:`_load_plumed_colvar` if the file header or
        the requested columns are unusable.
    """
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
