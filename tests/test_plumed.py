import ase.build

import nqetools as nqe


def test_plumed_pos():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    directory = "mace_md"
    atoms_out = nqe.run_plumed_md(directory,
                                  atoms,
                                  driver='ase-mace',
                                  md_type="NVT",
                                  plumed_type="opes")
    pass


def test_plumed_pos_pimd():
    print(flush=True)
    print("Testing MACE md calculator", flush=True)
    # build the molecule
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=5.0)

    directory = "mace_md"
    atoms_out = nqe.run_plumed_md(directory,
                                  atoms,
                                  driver='ase-mace',
                                  md_type="NVT-PIMD",
                                  n_beads=8,
                                  plumed_type="pos-mtd", )
    pass
