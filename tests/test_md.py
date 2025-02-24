import ase.build

import nqetools as nqe


def test_npt_bzp():
    print(flush=True)
    print("Testing NPT-BZP md", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NPT-BZP", total_steps=10)
    nqe.remove_directory(directory)
    pass


def test_npt_mttk():
    print(flush=True)
    print("Testing NPT-MTTK md", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NPT-MTTK", total_steps=10)
    nqe.remove_directory(directory)
    pass


def test_npt_re():
    print(flush=True)
    print("Testing NPT-RE md", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NPT-RE", total_steps=10)
    nqe.remove_directory(directory)
    pass


def test_nve():
    print(flush=True)
    print("Testing NVE md", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVE", total_steps=10)
    nqe.remove_directory(directory)
    pass


def test_nve_pimd():
    print(flush=True)
    print("Testing NVE md", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "pimd"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVE", total_steps=10, n_beads=2)
    nqe.remove_directory(directory)
    pass


def test_nvt():
    print(flush=True)
    print("Testing NVT md", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT", total_steps=10)
    nqe.remove_directory(directory)
    pass


def test_nvt_gle():
    print(flush=True)
    print("Testing NVT-GLE md", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT-GLE", total_steps=10)
    nqe.remove_directory(directory)
    pass


def test_nvt_langevin():
    print(flush=True)
    print("Testing NVT-LANGEVIN md", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT-LANGEVIN", total_steps=10)
    nqe.remove_directory(directory)
    pass


def test_nvt_pimd():
    print(flush=True)
    print("Testing NVT-PIMD md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT-PIMD", total_steps=10, n_beads=2)
    nqe.remove_directory(directory)
    pass


def test_nvt_pimd_sc():
    print(flush=True)
    print("Testing NVT-PIMD-SC md", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT-PIMD-SC", total_steps=10, n_beads=2)
    nqe.remove_directory(directory)
    pass


def test_nvt_re():
    print(flush=True)
    print("Testing NVT-RE md", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT-RE", total_steps=10)
    nqe.remove_directory(directory)
    pass


def test_nvt_svr():
    print(flush=True)
    print("Testing NVT-SVR md", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    # Make a directory to store everything
    directory = "md"
    nqe.run_md(directory, atoms, driver='ase-mace', md_type="NVT-SVR", total_steps=10)
    nqe.remove_directory(directory)
    pass
