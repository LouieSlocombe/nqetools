"""Plotting helpers for i-PI and PLUMED simulation output.

Covers optimiser and MD diagnostics (energy, temperature, conservation),
free energy surfaces in one and two dimensions, and rate-theory summaries
(Arrhenius, transmission coefficients, kinetic isotope effects).

Every plotting routine follows the same contract: it accepts an optional
``fig``/``ax`` pair to draw into, otherwise creating its own, and returns
that pair so plots can be composed or further customised by the caller.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .calcs import moving_average

plt.rcParams['axes.linewidth'] = 2.0


def n_plot(xlab,
           ylab,
           xs=14,
           ys=14):
    """Apply the house tick, label and layout style to the current axes.

    Parameters
    ----------
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    xs : int, optional
        Font size for the x-axis label. Default is 14.
    ys : int, optional
        Font size for the y-axis label. Tick labels are drawn two points
        smaller. Default is 14.

    Returns
    -------
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


def ax_plot(fig,
            ax,
            xlab,
            ylab,
            xs=14,
            ys=14):
    """Apply the house tick, label and layout style to a given axes.

    Object-oriented counterpart to :func:`n_plot`, for use with explicit
    figure and axes handles.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to lay out.
    ax : matplotlib.axes.Axes
        Axes to style.
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    xs : int, optional
        Font size for the x-axis label. Default is 14.
    ys : int, optional
        Font size for the y-axis label. Tick labels are drawn two points
        smaller. Default is 14.

    Returns
    -------
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


def plot_step_energy(data,
                     fig=None,
                     ax=None,
                     diff=True,
                     save=True,
                     show=True,
                     filename="step_energy",
                     fig_size=(8, 3),
                     y_scale='log'):
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
                             fig_size=(8, 3)):
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
                          fig_size=(8, 3)):
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
                                  fig_size=(8, 3)):
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


def plot_fes_series_1d(fes_arrays: list[np.ndarray],
                       fig=None,
                       ax=None,
                       slices: list[float] | None = None,
                       labels: list[str] | None = None,
                       max_slices: int = 5,
                       save: bool = True,
                       show: bool = True,
                       filename: str = "fes_1d",
                       x_lab: str = r"CV1",
                       y_lab: str = r"$F$ (eV)",
                       fig_size: tuple = (8, 3)):
    """Overlay a series of one-dimensional free energy surfaces.

    Plotting successive time slices on one axis shows whether the surface
    has stopped evolving, the usual convergence check for metadynamics and
    OPES runs.

    Parameters
    ----------
    fes_arrays : list of numpy.ndarray
        One array per slice, each of shape (2, n_points) holding the
        collective variable grid and the free energy.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new one is created if either this or `fig`
        is None.
    slices : list of float, optional
        Times in ps labelling each surface. Defaults to the array indices.
    labels : list of str, optional
        Explicit legend entries, overriding the times.
    max_slices : int, optional
        Keep only this many of the most recent slices. Default is 5.
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "fes_1d".
    x_lab : str, optional
        Label for the x-axis. Default is "CV1".
    y_lab : str, optional
        Label for the y-axis. Default is "$F$ (eV)".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.
    """
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
                               labels: list[str] | None = None,
                               save: bool = True,
                               show: bool = True,
                               filename: str = "fes_1d_compare",
                               x_lab: str = r"CV1",
                               y_lab: str = r"$F$ (eV)",
                               fig_size: tuple = (8, 3)):
    """Compare two one-dimensional free energy surfaces on shared axes.

    Typically used to contrast a classical run against its path-integral
    counterpart, exposing the nuclear quantum contribution to the barrier.

    Parameters
    ----------
    fes_arrays_a : list of numpy.ndarray
        First surface as (grid, free energy), unpacked directly into
        ``ax.plot``.
    fes_arrays_b : list of numpy.ndarray
        Second surface, in the same layout, drawn dashed.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new one is created if either this or `fig`
        is None.
    labels : list of str, optional
        Legend entries for the two surfaces. Default is ["MD", "PIMD"].
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "fes_1d_compare".
    x_lab : str, optional
        Label for the x-axis. Default is "CV1".
    y_lab : str, optional
        Label for the y-axis. Default is "$F$ (eV)".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.
    """
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
                             times: list[float] | None = None,
                             max_times=5,
                             save=True,
                             show=True,
                             filename="fes_contourf",
                             x_lab="CV1",
                             y_lab="CV2",
                             fig_size=(8, 3)):
    """Plot a series of two-dimensional free energy surfaces side by side.

    The panels share axes and a single colour bar, so successive time
    slices can be compared directly to judge convergence.

    Parameters
    ----------
    fes_arrays : list of numpy.ndarray
        One array per slice, each holding (CV1 grid, CV2 grid, free
        energy) as accepted by ``ax.contourf``.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes or numpy.ndarray of Axes, optional
        Axes to draw into, one per slice. Created if either this or `fig`
        is None.
    times : list of float, optional
        Times in ps titling each panel. Defaults to the array indices.
    max_times : int, optional
        Keep only this many of the most recent slices. Default is 5.
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "fes_contourf".
    x_lab : str, optional
        Label for the x-axis. Default is "CV1".
    y_lab : str, optional
        Label for the shared y-axis. Default is "CV2".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, numpy.ndarray of Axes)
        The figure and the array of axes that were drawn into.
    """
    if times is None:
        times = np.arange(len(fes_arrays))

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

    # subplots() returns a bare Axes for a single column, so normalise to an array
    ax = np.atleast_1d(ax)

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
    """Plot two two-dimensional free energy surfaces as adjacent panels.

    Parameters
    ----------
    fes_a : numpy.ndarray
        First surface as (CV1 grid, CV2 grid, free energy).
    fes_b : numpy.ndarray
        Second surface, in the same layout.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : numpy.ndarray of Axes, optional
        Pair of axes to draw into. Created if either this or `fig` is None.
    labels : list of str, optional
        Titles for the two panels. Default is ["MD", "PIMD"].
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "fes_contourf_compare".
    x_lab : str, optional
        Label for the x-axis. Default is "CV1".
    y_lab : str, optional
        Label for the shared y-axis. Default is "CV2".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, numpy.ndarray of Axes)
        The figure and the pair of axes that were drawn into.
    """
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
    """Plot a single two-dimensional free energy surface as filled contours.

    Parameters
    ----------
    fes : numpy.ndarray
        Surface as (CV1 grid, CV2 grid, free energy).
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
        Stem for the saved files. Default is "fes_contourf".
    x_lab : str, optional
        Label for the x-axis. Default is "CV1".
    y_lab : str, optional
        Label for the y-axis. Default is "CV2".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.
    """
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
    """Overlay two free energy surfaces as contour lines on shared axes.

    Unfilled contours at matched levels let the two surfaces be compared
    in place, rather than side by side as in
    :func:`plot_fes_contourf_compare`.

    Parameters
    ----------
    fes_a : numpy.ndarray
        First surface as (CV1 grid, CV2 grid, free energy), drawn in blue.
    fes_b : numpy.ndarray
        Second surface, in the same layout, drawn in red.
    fig : matplotlib.figure.Figure, optional
        Figure to draw into. A new one is created if either this or `ax`
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new one is created if either this or `fig`
        is None.
    labels : list of str, optional
        Legend entries for the two surfaces. Default is ["MD", "PIMD"].
    save : bool, optional
        If True, write PNG and PDF copies. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    filename : str, optional
        Stem for the saved files. Default is "fes_contour_compare".
    x_lab : str, optional
        Label for the x-axis. Default is "CV1".
    y_lab : str, optional
        Label for the y-axis. Default is "CV2".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    if labels is None:
        labels = ["MD", "PIMD"]

    levels = np.linspace(0, 0.5, 6)
    ax.contour(*fes_a, colors="b", levels=levels)
    ax.contour(*fes_b, colors="r", levels=levels)

    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)

    # contour() registers no legend handles, so build proxy artists
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
    """Plot proton-transfer free energy profiles at two donor separations.

    Cuts through a two-dimensional surface at fixed heavy-atom separation,
    comparing classical and path-integral results at 2.6 and 2.7 A.

    Parameters
    ----------
    fes_a : numpy.ndarray
        Classical surface of shape (3, n_cv1, n_cv2), indexed as
        (grid axis, CV1, donor separation).
    fes_b : numpy.ndarray
        Path-integral surface, in the same layout.
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
        Stem for the saved files. Default is "energy_sep".
    fig_size : tuple of float, optional
        Figure size in inches. Default is (8, 3).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes that were drawn into.

    Notes
    -----
    The separations are hard-coded as grid columns 50 and 60, so this
    assumes the surfaces share the grid used to generate them.
    """
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


def plot_arrhenius(temperatures: list[float],
                   rates: list[float],
                   fig=None,
                   ax=None,
                   save=True,
                   show=True,
                   filename="arrhenius",
                   fig_size=(8, 3)) -> None:
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
                     fig_size=(8, 3)) -> None:
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
                           fig_size=(8, 3)) -> None:
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
                               fig_size=(8, 3)) -> None:
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
                         fig_size=(8, 3)) -> None:
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
                          ) -> None:
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
                      fig_size=(8, 3)):
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
