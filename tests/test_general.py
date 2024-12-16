import nqetools as nqe

"""
Test the exe functions for simple cases
optimise water
md of water and confirm and output file
vibrate water
TS of water

Check that the other calculators work

"""


def test_calculate_nbeads():
    # For water omega_max ~ 3800 invcm this number is ~ 18 at 300 K. So a safe choice is 32 replicas
    n_beads = nqe.calculate_nbeads(3800.0, 300.0)
    assert n_beads == 18
