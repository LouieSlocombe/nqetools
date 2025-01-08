import subprocess

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps


def plot_scm_tcm(file='COLVAR'):
    """
    Plot the SCM vs TCM from a given file.

    Parameters:
    file (str): The name of the file containing the SCM and TCM data. Default is 'COLVAR'.

    Returns:
    None
    """
    # Import free energy and reshape with the number of bins defined in the
    # reconstruction process.
    scm = np.loadtxt(file, usecols=1)
    tcm = np.loadtxt(file, usecols=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 9))

    # Plot free energy surface
    plt.plot(scm, tcm, 'o')
    # Plot parameters
    ax.set_xlabel('SCM', fontsize=40)
    ax.set_ylabel('TCM', fontsize=40)
    ax.tick_params(axis='y', labelsize=25)
    ax.tick_params(axis='x', labelsize=25)

    plt.tight_layout()
    plt.show()


def plot_fes(file='fes.dat'):
    """
    Plot the free energy surface from a given file.

    Parameters:
    file (str): The name of the file containing the free energy data. Default is 'fes.dat'.

    Returns:
    None
    """
    # Import free energy and reshape with the number of bins defined in the
    # reconstruction process.
    scm = np.loadtxt(file, usecols=0).reshape(301, 301)
    tcm = np.loadtxt(file, usecols=1).reshape(301, 301)
    fes = np.loadtxt(file, usecols=2).reshape(301, 301)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 9))

    # Plot free energy surface
    im = ax.contourf(scm, tcm, fes, 10, cmap=colormaps['Blues_r'])
    cp = ax.contour(scm, tcm, fes, 10,
                    linestyles='-', colors='darkgray', linewidths=1.2)

    # Plot parameters
    ax.set_xlabel('SCM', fontsize=40)
    ax.set_ylabel('TCM', fontsize=40)
    ax.tick_params(axis='y', labelsize=25)
    ax.tick_params(axis='x', labelsize=25)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label=r'FES[$\epsilon$]', fontsize=40)
    cbar.ax.tick_params(labelsize=32)

    plt.tight_layout()
    plt.show()


plot_scm_tcm('COLVAR')

# Reconstruct the free energy from HILLS using the plumed tool "sum_hills" :
p = subprocess.Popen("plumed sum_hills --hills HILLS --outfile fes.dat" +
                     " --bin 300,300 --min 0.3,-0.35 --max 1.2,1.56",
                     shell=True, stdout=subprocess.PIPE)
p.wait()

# Import free energy and reshape with the number of bins defined in the
# reconstruction process.
plot_fes('fes.dat')
