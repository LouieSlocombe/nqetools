"""Tests for the molecular dynamics ensembles.

One short run per XML template in ``templates/``, confirming that each
ensemble, thermostat and barostat combination is set up correctly.
"""

import ase.build

import nqetools as nqe


def test_npt_bzp():
    """Run NPT dynamics with the Bussi-Zykova-Parrinello barostat."""
    print(flush=True)
    print("Testing NPT-BZP md", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "md"
    nqe.run_md(directory, atoms, driver="ase-mace", total_steps=10, md_type="NPT-BZP")
    nqe.remove_directory(directory)
    pass


def test_npt_mttk():
    """Run NPT dynamics with the Martyna-Tobias-Tuckerman-Klein barostat."""
    print(flush=True)
    print("Testing NPT-MTTK md", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "md"
    nqe.run_md(directory, atoms, driver="ase-mace", total_steps=10, md_type="NPT-MTTK")
    nqe.remove_directory(directory)
    pass


def test_npt_re():
    """Run NPT dynamics with replica exchange."""
    print(flush=True)
    print("Testing NPT-RE md", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "md"
    nqe.run_md(directory, atoms, driver="ase-mace", total_steps=10, md_type="NPT-RE")
    nqe.remove_directory(directory)
    pass


def test_nve():
    """Run microcanonical dynamics."""
    print(flush=True)
    print("Testing NVE md", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "md"
    nqe.run_md(directory, atoms, driver="ase-mace", total_steps=10, md_type="NVE")
    nqe.remove_directory(directory)
    pass


def test_nve_pimd():
    """Run microcanonical path-integral dynamics."""
    print(flush=True)
    print("Testing NVE md", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "pimd"
    nqe.run_md(
        directory, atoms, driver="ase-mace", total_steps=10, md_type="NVE", n_beads=2
    )
    nqe.remove_directory(directory)
    pass


def test_nvt():
    """Run canonical dynamics with the default thermostat."""
    print(flush=True)
    print("Testing NVT md", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "md"
    nqe.run_md(directory, atoms, driver="ase-mace", total_steps=10, md_type="NVT")
    nqe.remove_directory(directory)
    pass


def test_nvt_gle():
    """Run canonical dynamics with a generalised Langevin thermostat."""
    print(flush=True)
    print("Testing NVT-GLE md", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "md"
    nqe.run_md(directory, atoms, driver="ase-mace", total_steps=10, md_type="NVT-GLE")
    nqe.remove_directory(directory)
    pass


def test_nvt_langevin():
    """Run canonical dynamics with a Langevin thermostat."""
    print(flush=True)
    print("Testing NVT-LANGEVIN md", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "md"
    nqe.run_md(
        directory, atoms, driver="ase-mace", total_steps=10, md_type="NVT-LANGEVIN"
    )
    nqe.remove_directory(directory)
    pass


def test_nvt_pimd():
    """Run canonical path-integral dynamics."""
    print(flush=True)
    print("Testing NVT-PIMD md calculator", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "md"
    nqe.run_md(
        directory,
        atoms,
        driver="ase-mace",
        total_steps=10,
        md_type="NVT-PIMD",
        n_beads=2,
    )
    nqe.remove_directory(directory)
    pass


def test_nvt_pimd_sc():
    """Run canonical path-integral dynamics with Suzuki-Chin splitting."""
    print(flush=True)
    print("Testing NVT-PIMD-SC md", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "md"
    nqe.run_md(
        directory,
        atoms,
        driver="ase-mace",
        total_steps=10,
        md_type="NVT-PIMD-SC",
        n_beads=2,
    )
    nqe.remove_directory(directory)
    pass


def test_nvt_re():
    """Run canonical dynamics with replica exchange."""
    print(flush=True)
    print("Testing NVT-RE md", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "md"
    nqe.run_md(directory, atoms, driver="ase-mace", total_steps=10, md_type="NVT-RE")
    nqe.remove_directory(directory)
    pass


def test_nvt_svr():
    """Run canonical dynamics with a stochastic velocity rescaling thermostat."""
    print(flush=True)
    print("Testing NVT-SVR md", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "md"
    nqe.run_md(directory, atoms, driver="ase-mace", total_steps=10, md_type="NVT-SVR")
    nqe.remove_directory(directory)
    pass
