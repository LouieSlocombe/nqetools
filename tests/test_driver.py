import getpass

import ase.build
import pytest

import nqetools as nqe


def test_zundel_driver():
    print(flush=True)
    print("Testing zundel driver", flush=True)

    atoms = nqe.read_ipi_xyz("data/h5o2+.xyz")[-1]

    directory = "zundel_opti"
    nqe.run_optimise(directory, atoms, driver='zundel', total_steps=2)
    nqe.remove_directory(directory)
    pass


def test_cbe_driver():
    print(flush=True)
    print("Testing cbe driver", flush=True)

    atoms = nqe.read_ipi_xyz("data/ch4hcbe.xyz")[-1]

    directory = "cbe_opti"
    nqe.run_optimise(directory, atoms, driver='cbe', total_steps=2)
    nqe.remove_directory(directory)
    pass


def test_ase_mace_driver():
    print(flush=True)
    print("Testing MACE driver", flush=True)

    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    directory = "mace_opti"
    nqe.run_optimise(directory, atoms, driver='ase-mace', total_steps=2)
    nqe.remove_directory(directory)
    pass


def test_mace_driver():
    print(flush=True)
    print("Testing MACE driver", flush=True)

    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    directory = "mace_opti"
    driver = 'mace'
    driver_args = {'model': f'/home/{getpass.getuser()}/.cache/mace/MACE-OFF23_small.model'}
    nqe.run_optimise(directory, atoms, driver=driver, driver_args=driver_args, total_steps=2)
    nqe.remove_directory(directory)
    pass


def test_ase_mace_driver_omol():
    print(flush=True)
    print("Testing MACE driver", flush=True)

    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    directory = "mace_opti"
    driver_args = {'model_type': 'omol',
                   'model': 'extra_large',
                   'device': 'cuda',
                   'default_dtype': 'float32'}
    nqe.run_optimise(directory, atoms, driver='ase-mace', driver_args=driver_args, total_steps=2)
    nqe.remove_directory(directory)
    pass


def test_ase_mace_driver_omol_eq():
    print(flush=True)
    print("Testing MACE driver", flush=True)

    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)
    from mace.calculators import mace_omol
    atoms.calc = mace_omol(device="cuda",
                           default_dtype="float64",
                           enable_cueq=False)

    directory = "mace_opti"
    nqe.remove_directory(directory)
    pass


def test_ase_mace_driver_qmmm():
    print(flush=True)
    print("Testing MACE driver", flush=True)

    atoms1 = ase.build.molecule('H2O')
    atoms2 = ase.build.molecule('H2O')
    atoms2.translate([0.0, 0.0, 3.0])  # Move the second water molecule
    atoms = atoms1 + atoms2

    atoms.center(vacuum=10.0)

    directory = "mace_opti"
    driver_args = {'qm_indices': [0],
                   'qm_model_type': 'omol',
                   'qm_model': 'extra_large',
                   'mm_model_type': 'off',
                   'mm_model': 'small',
                   'device': 'cuda',
                   'default_dtype': 'float32'}
    nqe.run_optimise(directory,
                     atoms,
                     driver='ase-qmmm-mace',
                     total_steps=2,
                     driver_args=driver_args)
    nqe.remove_directory(directory)
    pass


def test_ase_orca_driver():
    """Tests the ORCA driver using the ASE driver.

    This function builds a water molecule, centers it with a vacuum of 5.0 Å,
    sets up the ORCA driver, runs the optimisation, and stores the results
    in a specified directory.
    """
    print(flush=True)
    print("Testing ORCA driver", flush=True)

    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    directory = "orca_opti"
    nqe.run_optimise(directory, atoms, driver='ase-orca', driver_args={'n_procs': 1}, total_steps=2)
    nqe.remove_directory(directory)
    pass


def test_ase_orca_driver_qmmm():
    print(flush=True)
    print("Testing ORCA driver", flush=True)

    atoms1 = ase.build.molecule('H2O')
    atoms2 = ase.build.molecule('H2O')
    atoms2.translate([0.0, 0.0, 3.0])  # Move the second water molecule
    atoms = atoms1 + atoms2

    atoms.center(vacuum=10.0)

    directory = "orca_opti"
    nqe.run_optimise(directory,
                     atoms,
                     driver='ase-orca',
                     total_steps=2,
                     driver_args={'calc_type': 'QM/XTB2',
                                  'atom_list': '0:2',
                                  'n_procs': 1})
    nqe.remove_directory(directory)
    pass


@pytest.mark.fail
def test_ase_nwchem_driver():
    """Tests the NWChem driver using the ASE driver.

    This function builds a water molecule, centers it with a vacuum of 5.0 Å,
    sets up the NWChem driver, runs the optimisation, and stores the results
    in a specified directory.
    """
    print(flush=True)
    print("Testing NWChem driver", flush=True)

    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    directory = "nwchem_opti"

    nqe.run_optimise(directory, atoms, driver='ase-nwchem', total_steps=2)
    nqe.remove_directory(directory)
    pass


@pytest.mark.fail
def test_nwchem_driver():
    """Tests the NWChem driver.

    This function builds a water molecule, centers it with a vacuum of 5.0 Å,
    sets up the NWChem driver, runs the optimisation, and stores the results
    in a specified directory.
    """
    print(flush=True)
    print("Testing NWChem driver", flush=True)

    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    directory = "nwchem_opti"

    nqe.run_optimise(directory, atoms, driver='nwchem', total_steps=2)
    nqe.remove_directory(directory)
    pass
