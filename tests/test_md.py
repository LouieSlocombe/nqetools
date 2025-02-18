import ase.build

import nqetools as nqe


def test_npt_bzp():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NPT-BZP")
    pass


def test_npt_mttk():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NPT-MTTK")
    pass


def test_npt_re():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NPT-RE")


def test_nve():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVE")
    pass


def test_nvt():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT")
    pass


def test_nvt_gle():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT-GLE")
    pass


def test_nvt_langevin():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT-LANGEVIN")
    pass


def test_nvt_pimd():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT-PIMD", n_beads=2)
    pass


def test_nvt_pimd_sc():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT-PIMD-SC", n_beads=2)
    pass


def test_nvt_re():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT-RE")
    pass


def test_nvt_svr():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "mace_md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT-SVR")
    pass
