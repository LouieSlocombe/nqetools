import os
import sys
from scipy.constants import physical_constants

def add_ipi_paths(base=os.path.expanduser("~") + "/i-pi/"):
    """ Adds to system paths so that one can run from the jupyter
    notebooks without any settings after having just cloned the i-PI repo
    and compiled the FORTRAN driver. """
    sys.path.append(base)
    os.environ['PATH'] += (":" + base + "/bin/")


def rm_ipi_tmp(tmp=r"/tmp/ipi_localhost"):
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
    return None




# Conversion factor from Bohr to Angstrom
bohr_to_angstrom = physical_constants["Bohr radius"][0] * 1e10
