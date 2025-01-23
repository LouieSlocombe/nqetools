import ase.build

import nqetools as nqe


def test_plumed_mtd_pos():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    directory = "mace_md"
    # If the directory exists, remove it
    nqe.remove_directory(directory)
    atoms_out = nqe.run_plumed_md(directory,
                                  atoms,
                                  driver='ase-mace',
                                  md_type="NVT",
                                  plumed_type="mtd-pos")
    pass


def test_plumed_mtd_pos_pimd():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)
    directory = "mace_md"
    # If the directory exists, remove it
    nqe.remove_directory(directory)
    atoms_out = nqe.run_plumed_md(directory,
                                  atoms,
                                  driver='ase-mace',
                                  md_type="NVT-PIMD",
                                  n_beads=8,
                                  plumed_type="mtd-pos", )
    pass


def test_plumed_opes_pos():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    directory = "mace_md"
    # If the directory exists, remove it
    nqe.remove_directory(directory)
    atoms_out = nqe.run_plumed_md(directory,
                                  atoms,
                                  driver='ase-mace',
                                  md_type="NVT",
                                  plumed_type="opes-pos")
    pass


def test_plumed_mtd_dist():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    directory = "mace_md"
    # If the directory exists, remove it
    nqe.remove_directory(directory)
    atoms_out = nqe.run_plumed_md(directory,
                                  atoms,
                                  driver='ase-mace',
                                  md_type="NVT",
                                  plumed_type="mtd-dist")
    pass


def test_plumed_opes_dist():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    directory = "mace_md"
    # If the directory exists, remove it
    nqe.remove_directory(directory)
    atoms_out = nqe.run_plumed_md(directory,
                                  atoms,
                                  driver='ase-mace',
                                  md_type="NVT",
                                  plumed_type="opes-dist")
    pass
