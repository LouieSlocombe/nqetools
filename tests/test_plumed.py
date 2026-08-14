"""Tests for PLUMED-biased molecular dynamics.

One short biased run per collective variable scheme, confirming that the
generated plumed.dat is accepted and that i-PI reads back the COLVAR
columns each writer declares.
"""

import ase.build

import nqetools as nqe


def test_plumed_zundel_mtd_pos():
    """Run metadynamics on a Zundel cation position with the analytic PES."""
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    atoms = nqe.read_ipi_xyz("data/h5o2+.xyz")[-1]

    directory = "mace_md"
    nqe.run_plumed_md(
        directory,
        atoms,
        driver="zundel",
        total_steps=10,
        md_type="NVT",
        plumed_type="mtd-pos",
    )
    nqe.remove_directory(directory)
    pass


def test_plumed_mtd_pos():
    """Run metadynamics on an atom position."""
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "mace_md"
    nqe.run_plumed_md(
        directory,
        atoms,
        driver="ase-mace",
        total_steps=10,
        md_type="NVT",
        plumed_type="mtd-pos",
    )
    nqe.remove_directory(directory)
    pass


def test_plumed_mtd_pos_pimd():
    """Run metadynamics on an atom position with path-integral dynamics."""
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)
    directory = "mace_md"
    nqe.run_plumed_md(
        directory,
        atoms,
        driver="ase-mace",
        total_steps=10,
        md_type="NVT-PIMD",
        n_beads=8,
        plumed_type="mtd-pos",
    )
    nqe.remove_directory(directory)
    pass


def test_plumed_opes_pos():
    """Run OPES on an atom position."""
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "mace_md"
    nqe.run_plumed_md(
        directory, atoms, driver="ase-mace", md_type="NVT", plumed_type="opes-pos"
    )
    nqe.remove_directory(directory)
    pass


def test_plumed_mtd_dist():
    """Run metadynamics on an interatomic distance."""
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "mace_md"
    nqe.run_plumed_md(
        directory, atoms, driver="ase-mace", md_type="NVT", plumed_type="mtd-dist"
    )
    nqe.remove_directory(directory)
    pass


def test_plumed_opes_dist():
    """Run OPES on an interatomic distance."""
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    atoms = ase.build.molecule("H2O")
    atoms.center(vacuum=5.0)

    directory = "mace_md"
    nqe.run_plumed_md(
        directory, atoms, driver="ase-mace", md_type="NVT", plumed_type="opes-dist"
    )
    nqe.remove_directory(directory)
    pass
