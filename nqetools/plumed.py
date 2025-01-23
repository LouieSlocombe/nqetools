import os

from .conversions import (A_to_nm,
                          eV_to_kJpermol,
                          eVperA2_to_kJpermolpernm2)
from .tools import round_sf


def prep_plumed(plumed_type, atoms, args):
    if plumed_type == 'pos-mtd':
        return write_plumed_mtd_pos(**args)
    elif plumed_type == 'pos-opes':
        return write_plumed_opes_pos(**args)
    elif plumed_type == 'coord-mtd':
        return write_plumed_mtd_coord(atoms, **args)
    elif plumed_type == 'coord-opes':
        return write_plumed_opes_coord(atoms, **args)
    elif plumed_type == 'dist-mtd':
        return write_plumed_mtd_dist(**args)
    elif plumed_type == 'dist-opes':
        return write_plumed_opes_dist(**args)
    else:
        raise ValueError(f'Unknown plumed type: {plumed_type}')


def write_plumed_mtd_pos(directory=None,
                         idx_atom=0,
                         pace=20,
                         sigma=0.01,
                         height=1.0,
                         bias=2.5,
                         temperature=300,
                         stride=10):
    if directory is None:
        directory = os.getcwd()

    # Convert the height from eV to kJ/mol
    height = round_sf(height * eV_to_kJpermol)

    # Update the index as it starts from 1
    idx_atom += 1
    impt = f"""
q: POSITION ATOM={idx_atom}
mtd: METAD ARG=q.x,q.y,q.z PACE={pace} SIGMA={sigma},{sigma},{sigma} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=q.*,mtd.* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    # Write the input file
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['q', 'mtd.bias']


def write_plumed_opes_pos(directory=None,
                          idx_atom=0,
                          pace=20,
                          barrier=1.0,
                          temperature=300,
                          stride=10):
    if directory is None:
        directory = os.getcwd()

    # Update the index as it starts from 1
    idx_atom += 1

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)

    impt = f"""
q: POSITION ATOM={idx_atom}
opes: OPES_METAD ARG=d,dc PACE={pace} BARRIER={barrier} TEMP={temperature}

PRINT ARG=q.*,mtd.* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    # Write the input file
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['q', 'mtd.bias']


def write_plumed_mtd_coord(atoms,
                           directory=None,
                           idx_atom1=0,
                           idx_atom2=1,
                           temperature=300,
                           sigma=None,
                           d_low=1.4,
                           d_upper=4.0,
                           kappa=0.026,
                           pace=10,
                           stride=10,
                           height=0.041,
                           bias=10,
                           ):
    if directory is None:
        directory = os.getcwd()

    if sigma is None:
        sigma = [0.05, 0.05]

    # Convert d_low and d_upper from A to nm
    d_low = d_low * A_to_nm
    d_upper = d_upper * A_to_nm

    # eV/Å² to 1 kJ/(mol·nm²)
    kappa = round_sf(kappa * eVperA2_to_kJpermolpernm2)

    # convert the height from eV to kJ/mol
    height = round_sf(height * eV_to_kJpermol)

    # Fix the indexing as it starts from 1
    idx_atom1 += 1
    idx_atom2 += 1

    # Indexing starts from 1
    idx_group = f"{max([idx_atom1, idx_atom2]) + 1}-{len(atoms) + 1}"

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    impt = f"""
d: DISTANCE ATOMS={idx_atom1},{idx_atom2} 
c1: DISTANCES GROUPA={idx_atom1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx_atom2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
mtd:   METAD ARG=d,dc PACE={pace} SIGMA={sigma[0]},{sigma[1]} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}
uwall: UPPER_WALLS ARG=d AT={d_upper} KAPPA={kappa}

PRINT ARG=d,c1.*,c2.*,dc,mtd.*,uwall.* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    # Write the input file
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d', 'c1.lessthan', 'c2.lessthan', 'dc', 'mtd.bias']


def write_plumed_opes_coord(atoms,
                            directory=None,
                            idx_atom1=0,
                            idx_atom2=1,
                            temperature=300,
                            d_low=1.4,
                            d_upper=4.0,
                            kappa=0.026,
                            pace=10,
                            stride=10,
                            barrier=0.041,
                            ):
    if directory is None:
        directory = os.getcwd()

    # Convert d_low and d_upper from A to nm
    d_low = d_low * A_to_nm
    d_upper = d_upper * A_to_nm

    # eV/Å² to 1 kJ/(mol·nm²)
    kappa = round_sf(kappa * eVperA2_to_kJpermolpernm2)

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)

    # Fix the indexing as it starts from 1
    idx_atom1 += 1
    idx_atom2 += 1
    # Indexing starts from 1
    idx_group = f"{max([idx_atom1, idx_atom2]) + 1}-{len(atoms) + 1}"

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    impt = f"""
d: DISTANCE ATOMS={idx_atom1},{idx_atom2} 
c1: DISTANCES GROUPA={idx_atom1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx_atom2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
opes: OPES_METAD ARG=d,dc PACE={pace} BARRIER={barrier} TEMP={temperature}
uwall: UPPER_WALLS ARG=d AT={d_upper} KAPPA={kappa}

PRINT ARG=d,c1.*,c2.*,dc,mtd.*,uwall.* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    # Write the input file
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d', 'c1.lessthan', 'c2.lessthan', 'dc', 'opes.bias']


def write_plumed_mtd_dist(directory=None,
                          idx1=0,
                          idx2=1,
                          idx3=2,
                          temperature=300,
                          sigma=None,
                          pace=10,
                          stride=10,
                          height=0.041,
                          bias=10,
                          ):
    if directory is None:
        directory = os.getcwd()

    if sigma is None:
        sigma = [0.05, 0.05]

    # Convert the height from eV to kJ/mol
    height = round_sf(height * eV_to_kJpermol)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1
    idx3 += 1

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx2},{idx3}
mtd: METAD ARG=d1,d2 PACE={pace} SIGMA={sigma[0]},{sigma[1]} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=d1,d2,mtd.* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    # Write the input file
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'd2', 'mtd.bias']


def write_plumed_opes_dist(directory=None,
                           idx1=0,
                           idx2=1,
                           idx3=2,
                           temperature=300,
                           pace=10,
                           stride=10,
                           barrier=0.041,
                           ):
    if directory is None:
        directory = os.getcwd()

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1
    idx3 += 1

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx2},{idx3}
opes: OPES_METAD ARG=d1,d2 PACE={pace} BARRIER={barrier} TEMP={temperature}

PRINT ARG=d1,d2,opes.* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    # Write the input file
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'd2', 'opes.bias']
