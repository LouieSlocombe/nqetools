import ase.build
from ase.io import read
import nqetools as nqe
from ase.visualize import view


def test_zundel_driver():
    print(flush=True)
    print("Testing MACE calculator", flush=True)

    # Build the molecule
    atoms = read("../examples/zundel/h5o2+.xyz")

    # Make a directory to store everything
    directory = "zundel_opti"
    nqe.run_optimise(directory, atoms, driver='zundel', total_steps=2)
    pass


def test_ase_mace_driver():
    """
    Tests the MACE calculator using the ASE driver.

    This function builds a water molecule, centers it with a vacuum of 5.0 Å,
    sets up the MACE calculator, runs the optimization, and stores the results
    in a specified directory.

    Asserts:
        None
    """
    print(flush=True)
    print("Testing MACE calculator", flush=True)

    # Build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_opti"
    nqe.run_optimise(directory, atoms, driver='ase-mace', total_steps=2)
    pass


def test_ase_orca_driver():
    """
    Tests the ORCA driver using the ASE driver.

    This function builds a water molecule, centers it with a vacuum of 5.0 Å,
    sets up the ORCA driver, runs the optimization, and stores the results
    in a specified directory.

    Asserts:
        None
    """
    print(flush=True)
    print("Testing ORCA driver", flush=True)

    # Build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "orca_opti"
    nqe.run_optimise(directory, atoms, driver='ase-orca', total_steps=2)
    pass


def test_ase_nwchem_driver():
    """
    Tests the NWChem driver using the ASE driver.

    This function builds a water molecule, centers it with a vacuum of 5.0 Å,
    sets up the NWChem driver, runs the optimization, and stores the results
    in a specified directory.

    Asserts:
        None
    """
    print(flush=True)
    print("Testing NWChem driver", flush=True)

    # Build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "nwchem_opti"
    nqe.remove_directory(directory)

    # Run the optimization
    nqe.run_optimise(directory, atoms, driver='ase-nwchem', total_steps=2)
    pass


def test_nwchem_driver():
    """
    Tests the NWChem driver.

    This function builds a water molecule, centers it with a vacuum of 5.0 Å,
    sets up the NWChem driver, runs the optimization, and stores the results
    in a specified directory.

    Asserts:
        None
    """
    print(flush=True)
    print("Testing NWChem driver", flush=True)

    # Build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "nwchem_opti"
    nqe.remove_directory(directory)

    # Run the optimization
    nqe.run_optimise(directory, atoms, driver='nwchem', total_steps=2)
    pass
