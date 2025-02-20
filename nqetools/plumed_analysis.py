import matplotlib.pyplot as plt

from .calcs import moving_average


def plot_time_potential_bias(data):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    ax.plot(
        data["time"],
        data["potential"],
        "r",
        label="potential",
    )
    ax.plot(
        data["time"],
        data["ensemble_bias"],
        "b",
        label="bias",
    )

    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"energy / eV")
    ax.legend(loc="upper left", ncols=1)
    plt.show()

    return None


def plot_time_temperature(data, window_size=100):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    min_val = int(window_size / 2)
    max_val = -int(window_size / 2 - 1)
    ax.plot(
        data["time"][min_val:max_val],
        moving_average(data["temperature"], window_size),
        "r",
        label=r"$T$",
    )

    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"temperature / K")
    ax.legend(loc="upper left", ncols=2)
    plt.show()
    return None
