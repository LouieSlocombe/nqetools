import os

from .conversions import (A_to_nm,
                          eV_to_kJpermol,
                          eVperA2_to_kJpermolpernm2)
from .tools import round_sf, get_distance


def prep_plumed(atoms, plumed_type, plumed_args):
    if plumed_type == 'mtd-pos':
        return write_plumed_mtd_pos(**plumed_args)
    elif plumed_type == 'opes-pos':
        return write_plumed_opes_pos(**plumed_args)
    elif plumed_type == 'mtd-coord':
        return write_plumed_mtd_coord(atoms, **plumed_args)
    elif plumed_type == 'opes-coord':
        return write_plumed_opes_coord(atoms, **plumed_args)
    elif plumed_type == 'mtd-dists':
        return write_plumed_mtd_dists(**plumed_args)
    elif plumed_type == 'opes-dists':
        return write_plumed_opes_dists(**plumed_args)
    elif plumed_type == 'mtd-dist':
        return write_plumed_mtd_dist(**plumed_args)
    elif plumed_type == 'opes-dist':
        return write_plumed_opes_dist(**plumed_args)
    elif plumed_type == 'mtd-diff1':
        return write_plumed_mtd_diff1(**plumed_args)
    elif plumed_type == 'opes-diff1':
        return write_plumed_opes_diff1(**plumed_args)
    elif plumed_type == 'mtd-diff2':
        return write_plumed_mtd_diff2(**plumed_args)
    elif plumed_type == 'opes-diff2':
        return write_plumed_opes_diff2(**plumed_args)
    elif plumed_type == 'mtd-pt1':
        return write_plumed_mtd_pt1(atoms, **plumed_args)
    elif plumed_type == 'opes-pt1':
        return write_plumed_opes_pt1(atoms, **plumed_args)
    elif plumed_type == 'mtd-pt2_a':
        return write_plumed_mtd_pt2_a(atoms, **plumed_args)
    elif plumed_type == 'opes-pt2_a':
        return write_plumed_opes_pt2_a(atoms, **plumed_args)

    elif plumed_type == 'mtd-pt-wob':
        return write_plumed_mtd_pt_wob(atoms, **plumed_args)
    elif plumed_type == 'opes-pt-wob':
        return write_plumed_opes_pt_wob(atoms, **plumed_args)
    elif plumed_type == 'mtd-pt-wob-sep':
        return write_plumed_mtd_pt_wob_sep(atoms, **plumed_args)
    elif plumed_type == 'opes-pt-wob-sep':
        return write_plumed_opes_pt_wob_sep(atoms, **plumed_args)
    elif plumed_type == 'opes-pt-wob-dist':
        return write_plumed_opes_pt_wob_dist(atoms, **plumed_args)

    elif plumed_type == 'opes_com':
        return write_plumed_opes_com(**plumed_args)
    elif plumed_type == 'opes_1pt':
        return write_plumed_opes_1pt(**plumed_args)
    elif plumed_type == 'opes_1pt_coord':
        return write_plumed_opes_1pt_coord(**plumed_args)
    elif plumed_type == 'opes_1pt_3donor_coord':
        return write_plumed_opes_1pt_3donor_coord(**plumed_args)


    elif plumed_type == 'opes_2pt_2d':
        return write_plumed_opes_2pt_2d(**plumed_args)
    elif plumed_type == 'opes_2pt_2d_coord':
        return write_plumed_opes_2pt_2d_coord(**plumed_args)
    elif plumed_type == 'opes_2pt_1d':
        return write_plumed_opes_2pt_1d(**plumed_args)
    elif plumed_type == 'opes_2pt_1d_coord':
        return write_plumed_opes_2pt_1d_coord(**plumed_args)
    elif plumed_type == 'opes_2pt_1d_coord_com':
        return write_plumed_opes_2pt_1d_coord_com(**plumed_args)

    elif plumed_type == 'custom':
        with open(os.path.join(plumed_args['directory'], "plumed.dat"), "w") as f:
            f.write(plumed_args['input'])
        return plumed_args['output']

    else:
        raise ValueError(f'Unknown plumed type: {plumed_type}')


def write_plumed_mtd_pos(directory=None,
                         idx_atom=0,
                         pace=20,
                         sigma=0.01,
                         height=1.0,
                         bias=2.5,
                         temperature=300,
                         stride=10,
                         ):
    if directory is None:
        directory = os.getcwd()

    # Convert the height from eV to kJ/mol
    height = round_sf(height * eV_to_kJpermol)

    # Update the index as it starts from 1
    idx_atom += 1
    impt = f"""
q: POSITION ATOM={idx_atom}
mtd: METAD ARG=q.x PACE={pace} SIGMA={sigma} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['q.x', 'mtd.bias']


def write_plumed_opes_pos(directory=None,
                          idx_atom=0,
                          pace=20,
                          barrier=1.0,
                          temperature=300,
                          stride=10,
                          stride_hills=100,
                          explore=False
                          ):
    if directory is None:
        directory = os.getcwd()

    # Update the index as it starts from 1
    idx_atom += 1

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
q: POSITION ATOM={idx_atom}
opes: {opes_command} ARG=q.x PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['q.x', 'opes.bias']


def write_plumed_mtd_coord(atoms,
                           directory=None,
                           idx1=0,
                           idx2=1,
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
        sigma = [0.005, 0.05]

    # Convert d_low and d_upper from A to nm
    d_low = d_low * A_to_nm
    d_upper = d_upper * A_to_nm

    # eV/Å² to 1 kJ/(mol·nm²)
    kappa = round_sf(kappa * eVperA2_to_kJpermolpernm2)

    # convert the height from eV to kJ/mol
    height = round_sf(height * eV_to_kJpermol)

    # Get a list of all the atom indexes
    group_idx = list(range(len(atoms)))

    # Remove the indexes of the atoms which are acceptors or donors
    group_idx.remove(idx1)
    group_idx.remove(idx2)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]

    # Indexing starts from 1
    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    impt = f"""
d: DISTANCE ATOMS={idx1},{idx2} 
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
mtd:   METAD ARG=d,dc PACE={pace} SIGMA={sigma[0]},{sigma[1]} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}
uwall: UPPER_WALLS ARG=d AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d', 'c1.lessthan', 'c2.lessthan', 'dc', 'mtd.bias']


def write_plumed_opes_coord(atoms,
                            directory=None,
                            idx1=0,
                            idx2=1,
                            temperature=300,
                            d_low=1.4,
                            d_upper=4.0,
                            kappa=0.026,
                            pace=10,
                            stride=10,
                            barrier=0.041,
                            stride_hills=100,
                            explore=False,
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

    # Get a list of all the atom indexes
    group_idx = list(range(len(atoms)))

    # Remove the indexes of the atoms which are acceptors or donors
    group_idx.remove(idx1)
    group_idx.remove(idx2)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]

    # Indexing starts from 1
    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d: DISTANCE ATOMS={idx1},{idx2} 
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=d,dc PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES 
uwall: UPPER_WALLS ARG=d AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d', 'c1.lessthan', 'c2.lessthan', 'dc', 'opes.bias']


def write_plumed_mtd_dists(directory=None,
                           idx1=0,
                           idx2=1,
                           idx3=2,
                           idx4=3,
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
    idx4 += 1

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx3},{idx4}
mtd: METAD ARG=d1,d2 PACE={pace} SIGMA={sigma[0]},{sigma[1]} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'd2', 'mtd.bias']


def write_plumed_opes_dists(directory=None,
                            idx1=0,
                            idx2=1,
                            idx3=2,
                            idx4=3,
                            temperature=300,
                            pace=10,
                            stride=10,
                            barrier=0.041,
                            stride_hills=100,
                            explore=False,
                            ):
    if directory is None:
        directory = os.getcwd()

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx3},{idx4}
opes: {opes_command} ARG=d1,d2 PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'd2', 'opes.bias']


def write_plumed_mtd_dist(directory=None,
                          idx1=0,
                          idx2=1,
                          temperature=300,
                          sigma=0.05,
                          pace=10,
                          stride=10,
                          height=0.041,
                          bias=10,
                          ):
    if directory is None:
        directory = os.getcwd()

    # Convert the height from eV to kJ/mol
    height = round_sf(height * eV_to_kJpermol)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
mtd: METAD ARG=d1 PACE={pace} SIGMA={sigma} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'mtd.bias']


def write_plumed_opes_dist(directory=None,
                           idx1=0,
                           idx2=1,
                           temperature=300,
                           pace=10,
                           stride=10,
                           barrier=0.041,
                           stride_hills=100,
                           explore=False,
                           ):
    if directory is None:
        directory = os.getcwd()

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
opes: {opes_command} ARG=d1 PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'opes.bias']


def write_plumed_mtd_diff1(directory=None,
                           idx1=0,
                           idx2=1,
                           idx3=2,
                           temperature=300,
                           sigma=0.05,
                           pace=10,
                           stride=10,
                           height=0.041,
                           bias=10,
                           ):
    if directory is None:
        directory = os.getcwd()

    # Convert the height from eV to kJ/mol
    height = round_sf(height * eV_to_kJpermol)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1  # Assumed transferring atom
    idx3 += 1

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx2},{idx3}
diff: COMBINE ARG=d1,d2 COEFFICIENTS=1,-1 PERIODIC=NO
mtd: METAD ARG=diff PACE={pace} SIGMA={sigma} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'd2', 'diff', 'mtd.bias']


def write_plumed_opes_diff1(directory=None,
                            idx1=0,
                            idx2=1,
                            idx3=2,
                            temperature=300,
                            pace=10,
                            stride=10,
                            barrier=0.041,
                            stride_hills=100,
                            explore=False,
                            ):
    if directory is None:
        directory = os.getcwd()

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1  # Assumed transferring atom
    idx3 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx2},{idx3}
diff: COMBINE ARG=d1,d2 COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=diff PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'd2', 'diff', 'opes.bias']


def write_plumed_mtd_diff2(directory=None,
                           idx1=0,
                           idx2=1,
                           idx3=2,
                           idx4=3,
                           idx5=4,
                           idx6=5,
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
    idx2 += 1  # Assumed transferring atom
    idx3 += 1
    idx4 += 1
    idx5 += 1  # Assumed transferring atom
    idx6 += 1

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx2},{idx3}

d3: DISTANCE ATOMS={idx4},{idx5}
d4: DISTANCE ATOMS={idx5},{idx6}

diff1: COMBINE ARG=d1,d2 COEFFICIENTS=1,-1 PERIODIC=NO
diff2: COMBINE ARG=d3,d4 COEFFICIENTS=1,-1 PERIODIC=NO

mtd: METAD ARG=diff1,diff2 PACE={pace} SIGMA={sigma[0]},{sigma[1]} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'd2', 'd3', 'd4', 'diff1', 'diff2', 'mtd.bias']


def write_plumed_opes_diff2(directory=None,
                            idx1=0,
                            idx2=1,
                            idx3=2,
                            idx4=3,
                            idx5=4,
                            idx6=5,
                            temperature=300,
                            pace=10,
                            stride=10,
                            barrier=0.041,
                            stride_hills=100,
                            explore=False,
                            ):
    if directory is None:
        directory = os.getcwd()

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1  # Assumed transferring atom
    idx3 += 1
    idx4 += 1
    idx5 += 1  # Assumed transferring atom
    idx6 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx2},{idx3}

d3: DISTANCE ATOMS={idx4},{idx5}
d4: DISTANCE ATOMS={idx5},{idx6}

diff1: COMBINE ARG=d1,d2 COEFFICIENTS=1,-1 PERIODIC=NO
diff2: COMBINE ARG=d3,d4 COEFFICIENTS=1,-1 PERIODIC=NO

opes: {opes_command} ARG=diff1,diff2 PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'd2', 'd3', 'd4', 'diff1', 'diff2', 'opes.bias']


def write_plumed_mtd_pt1(atoms,
                         directory=None,
                         idx1=0,
                         idx2=1,
                         temperature=300,
                         sigma=0.005,
                         d_low=1.4,
                         pace=10,
                         stride=10,
                         height=0.041,
                         bias=10,
                         ):
    if directory is None:
        directory = os.getcwd()

    # Convert d_low and d_upper from A to nm
    d_low = d_low * A_to_nm

    # convert the height from eV to kJ/mol
    height = round_sf(height * eV_to_kJpermol)

    # Get a list of all the atom indexes
    group_idx = list(range(len(atoms)))

    # Remove the indexes of the atoms which are acceptors or donors
    group_idx.remove(idx1)
    group_idx.remove(idx2)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]

    # Indexing starts from 1
    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    impt = f"""
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
mtd:   METAD ARG=dc PACE={pace} SIGMA={sigma} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['c1.lessthan', 'c2.lessthan', 'dc', 'mtd.bias']


def write_plumed_opes_pt1(atoms,
                          directory=None,
                          idx1=0,
                          idx2=1,
                          temperature=300,
                          d_low=1.4,
                          pace=10,
                          stride=10,
                          barrier=0.041,
                          stride_hills=100,
                          explore=False,
                          ):
    if directory is None:
        directory = os.getcwd()

    # Convert d_low and d_upper from A to nm
    d_low = d_low * A_to_nm

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)

    # Get a list of all the atom indexes
    group_idx = list(range(len(atoms)))

    # Remove the indexes of the atoms which are acceptors or donors
    group_idx.remove(idx1)
    group_idx.remove(idx2)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]

    # Indexing starts from 1
    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=dc PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES 

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['c1.lessthan', 'c2.lessthan', 'dc', 'opes.bias']


def write_plumed_mtd_pt2_a(atoms,
                           directory=None,
                           idx1=0,
                           idx2=1,
                           idx3=2,
                           idx4=3,
                           temperature=300,
                           sigma=0.005,
                           d_low=1.4,
                           pace=10,
                           stride=10,
                           height=0.041,
                           bias=10,
                           ):
    if directory is None:
        directory = os.getcwd()

    # Convert d_low and d_upper from A to nm
    d_low = d_low * A_to_nm

    # convert the height from eV to kJ/mol
    height = round_sf(height * eV_to_kJpermol)

    # Get a list of all the atom indexes
    group_idx = list(range(len(atoms)))

    # Remove the indexes of the atoms which are acceptors or donors
    group_idx.remove(idx1)
    group_idx.remove(idx2)
    group_idx.remove(idx3)
    group_idx.remove(idx4)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1
    group_idx = [x + 1 for x in group_idx]

    # Indexing starts from 1
    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    impt = f"""
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c3: DISTANCES GROUPA={idx3} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c4: DISTANCES GROUPA={idx4} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
dc2: COMBINE ARG=c3.lessthan,c4.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
mtd:   METAD ARG=dc1,dc2 PACE={pace} SIGMA={sigma} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['c1.lessthan', 'c2.lessthan', 'c3.lessthan', 'c4.lessthan', 'dc1', 'dc2', 'mtd.bias']


def write_plumed_opes_pt2_a(atoms,
                            directory=None,
                            idx1=0,
                            idx2=1,
                            idx3=2,
                            idx4=3,
                            temperature=300,
                            d_low=1.4,
                            pace=10,
                            stride=10,
                            barrier=0.041,
                            stride_hills=100,
                            explore=False,
                            ):
    if directory is None:
        directory = os.getcwd()

    # Convert d_low and d_upper from A to nm
    d_low = d_low * A_to_nm

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)

    # Get a list of all the atom indexes
    group_idx = list(range(len(atoms)))

    # Remove the indexes of the atoms which are acceptors or donors
    group_idx.remove(idx1)
    group_idx.remove(idx2)
    group_idx.remove(idx3)
    group_idx.remove(idx4)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1
    group_idx = [x + 1 for x in group_idx]

    # Indexing starts from 1
    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c3: DISTANCES GROUPA={idx3} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c4: DISTANCES GROUPA={idx4} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
dc2: COMBINE ARG=c3.lessthan,c4.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=dc1,dc2  PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES 

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['c1.lessthan', 'c2.lessthan', 'c3.lessthan', 'c4.lessthan', 'dc1', 'dc2', 'opes.bias']


def write_plumed_mtd_pt_wob(atoms,
                            directory=None,
                            idx1=0,
                            idx2=1,
                            temperature=300,
                            sigma=0.005,
                            d_low=1.4,
                            pace=10,
                            stride=10,
                            height=0.041,
                            bias=10,
                            ):
    if directory is None:
        directory = os.getcwd()

    # Convert d_low and d_upper from A to nm
    d_low = d_low * A_to_nm

    # convert the height from eV to kJ/mol
    height = round_sf(height * eV_to_kJpermol)

    # Get a list of all the atom indexes
    group_idx = list(range(len(atoms)))

    # Remove the indexes of the atoms which are acceptors or donors
    group_idx.remove(idx1)
    group_idx.remove(idx2)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]

    # Indexing starts from 1
    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    impt = f"""
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
mtd: METAD ARG=dc1 PACE={pace} SIGMA={sigma} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['c1.lessthan', 'c2.lessthan', 'dc1', 'mtd.bias']


def write_plumed_opes_pt_wob(atoms,
                             directory=None,
                             idx1=0,
                             idx2=1,
                             temperature=300,
                             d_low=1.4,
                             pace=10,
                             stride=10,
                             barrier=0.041,
                             stride_hills=100,
                             explore=False,
                             ):
    if directory is None:
        directory = os.getcwd()

    # Convert d_low and d_upper from A to nm
    d_low = d_low * A_to_nm

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)

    # Get a list of all the atom indexes
    group_idx = list(range(len(atoms)))

    # Remove the indexes of the atoms which are acceptors or donors
    group_idx.remove(idx1)
    group_idx.remove(idx2)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]

    # Indexing starts from 1
    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=dc1  PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES 

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['c1.lessthan', 'c2.lessthan', 'dc1', 'opes.bias']


def write_plumed_mtd_pt_wob_sep(atoms,
                                directory=None,
                                idx1=0,
                                idx2=1,
                                list_1=None,
                                list_2=None,
                                temperature=300,
                                sigma=None,
                                d_low=1.4,
                                d_upper=4.0,
                                kappa=0.1,
                                pace=10,
                                stride=10,
                                height=0.041,
                                bias=10,
                                ):
    if directory is None:
        directory = os.getcwd()

    if sigma is None:
        sigma = [0.005, 0.05]

    # Convert d_low and d_upper from A to nm
    d_low = d_low * A_to_nm
    d_upper = d_upper * A_to_nm

    # convert the height from eV to kJ/mol
    height = round_sf(height * eV_to_kJpermol)

    # Get a list of all the atom indexes
    group_idx = list(range(len(atoms)))

    # Remove the indexes of the atoms which are acceptors or donors
    group_idx.remove(idx1)
    group_idx.remove(idx2)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]
    list_1 = [x + 1 for x in list_1]
    list_2 = [x + 1 for x in list_2]

    # Indexing starts from 1
    idx_group = ",".join([str(x) for x in group_idx])
    idx_list_1 = ",".join([str(x) for x in list_1])
    idx_list_2 = ",".join([str(x) for x in list_2])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    impt = f"""
com1: COM ATOMS={idx_list_1}
com2: COM ATOMS={idx_list_2}

c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}

d: DISTANCE ATOMS=com1,com2

dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
mtd: METAD ARG=d,dc1 PACE={pace} SIGMA={sigma[0]},{sigma[1]} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}
uwall: UPPER_WALLS ARG=d AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d', 'c1.lessthan', 'c2.lessthan', 'dc1', 'mtd.bias']


def write_plumed_opes_pt_wob_sep(atoms,
                                 directory=None,
                                 idx1=0,
                                 idx2=1,
                                 list_1=None,
                                 list_2=None,
                                 temperature=300,
                                 d_low=1.4,
                                 d_upper=4.0,
                                 kappa=0.1,
                                 pace=10,
                                 stride=10,
                                 barrier=0.041,
                                 stride_hills=100,
                                 explore=False,
                                 ):
    if directory is None:
        directory = os.getcwd()

    # Convert d_low and d_upper from A to nm
    d_low = d_low * A_to_nm
    d_upper = d_upper * A_to_nm

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)

    # Get a list of all the atom indexes
    group_idx = list(range(len(atoms)))

    # Remove the indexes of the atoms which are acceptors or donors
    group_idx.remove(idx1)
    group_idx.remove(idx2)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]
    list_1 = [x + 1 for x in list_1]
    list_2 = [x + 1 for x in list_2]

    # Indexing starts from 1
    idx_group = ",".join([str(x) for x in group_idx])
    idx_list_1 = ",".join([str(x) for x in list_1])
    idx_list_2 = ",".join([str(x) for x in list_2])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
com1: COM ATOMS={idx_list_1}
com2: COM ATOMS={idx_list_2}

c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}

d: DISTANCE ATOMS=com1,com2

dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=d,dc1  PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES 
uwall: UPPER_WALLS ARG=d AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d', 'c1.lessthan', 'c2.lessthan', 'dc1', 'opes.bias']


def write_plumed_opes_pt_wob_dist(atoms,
                                  directory=None,
                                  idx1=0,
                                  idx2=1,
                                  idx3=2,
                                  idx4=3,
                                  idx5=4,
                                  idx6=5,
                                  idx7=6,
                                  idx8=7,
                                  temperature=300,
                                  d_low=1.4,
                                  pace=10,
                                  stride=10,
                                  barrier=0.041,
                                  stride_hills=100,
                                  explore=False,
                                  ):
    if directory is None:
        directory = os.getcwd()

    # Convert d_low and d_upper from A to nm
    d_low = d_low * A_to_nm

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)

    # Get a list of all the atom indexes
    group_idx = list(range(len(atoms)))

    # Remove the indexes of the atoms which are acceptors or donors
    group_idx.remove(idx1)
    group_idx.remove(idx2)

    # Fix the indexing as it starts from 1
    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]

    # Indexing starts from 1
    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=dc1  PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES 

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """

    impt = f"""
# Compute distances between specified atoms
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx3},{idx4}
d3: DISTANCE ATOMS={idx5},{idx6}
d4: DISTANCE ATOMS={idx7},{idx8}

# Compute sums of distances
sum1: COMBINE ARG=d1,d3 COEFFICIENTS=1,1 PERIODIC=NO
sum2: COMBINE ARG=d2,d4 COEFFICIENTS=1,1 PERIODIC=NO

# Compute the final difference
dc1: COMBINE ARG=sum1,sum2 COEFFICIENTS=1,-1 PERIODIC=NO

opes: {opes_command} ARG=dc1  PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES 

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1

    """

    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['dc1', 'opes.bias']


def plumed_input_dpt(
        output_file: str,
        h1_index: int,
        a1_index: int,
        a2_index: int,
        h2_index: int,
        b1_index: int,
        b2_index: int,
        r0: float = 1.5,
        d_max: float = 2.5,
        height: float = 0.2,
        sigma: float = 0.1,
        pace: int = 500,
        bias_factor: float = 10.0,
        temp: float = 300.0,
        stride: int = 100,
):
    """Writes a PLUMED input file for metadynamics to study double proton transfer.

    Parameters
    ----------
    output_file : str
        Name of the output PLUMED input file.
    h1_index : int
        Index of the first proton (H1).
    a1_index : int
        Index of the first donor/acceptor atom for H1.
    a2_index : int
        Index of the second donor/acceptor atom for H1.
    h2_index : int
        Index of the second proton (H2).
    b1_index : int
        Index of the first donor/acceptor atom for H2.
    b2_index : int
        Index of the second donor/acceptor atom for H2.
    r0 : float, optional
        R_0 parameter for the switching function (default: 1.5).
    d_max : float, optional
        D_MAX parameter for the switching function (default: 2.5).
    height : float, optional
        Height of the Gaussian hills in metadynamics (default: 0.2).
    sigma : float, optional
        Width of the Gaussian hills in metadynamics (default: 0.1).
    pace : int, optional
        Frequency of Gaussian hill deposition (default: 500).
    bias_factor : float, optional
        Bias factor for well-tempered metadynamics (default: 10.0).
    temp : float, optional
        Temperature of the system in Kelvin (default: 300.0).
    stride : int, optional
        Frequency of writing output to the COLVAR file (default: 100).
    """
    plumed_input = f"""
# Define the atoms involved in the proton transfer
# H1: Proton 1, A1/A2: Donor/acceptor for H1
# H2: Proton 2, B1/B2: Donor/acceptor for H2

# Define coordination numbers for the two protons
cn1: COORDINATIONNUMBER SPECIES={h1_index},{a1_index},{a2_index} SWITCH={{RATIONAL R_0={r0} D_MAX={d_max}}}
cn2: COORDINATIONNUMBER SPECIES={h2_index},{b1_index},{b2_index} SWITCH={{RATIONAL R_0={r0} D_MAX={d_max}}}

# Calculate the difference in coordination numbers
diff_cn: COMBINE ARG=cn1,cn2 COEFFICIENTS=1,-1 PERIODIC=NO

# Metadynamics setup
metad: METAD ARG=diff_cn HEIGHT={height} SIGMA={sigma} PACE={pace} FILE=HILLS BIASFACTOR={bias_factor} TEMP={temp}

# Print the collective variable and bias potential to a file
PRINT ARG=diff_cn,metad.bias STRIDE={stride} FILE=COLVAR
"""

    with open(output_file, "w") as f:
        f.write(plumed_input)

    print(f"PLUMED input file written to {output_file}")


def write_plumed_opes_com(directory=None,
                          group_1=[0],
                          group_2=[1],
                          temperature=300.0,
                          pace=10,
                          stride=10,
                          barrier=0.5,
                          d_upper=5.0,
                          kappa=500.0,
                          stride_hills=100,
                          explore=False,
                          ):
    if directory is None:
        directory = os.getcwd()

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)
    # Convert d_upper from A to nm
    d_upper = round_sf(d_upper * A_to_nm)

    # Fix the indexing as it starts from 1
    group_1 = [x + 1 for x in group_1]
    group_2 = [x + 1 for x in group_2]

    group_1 = ",".join([str(x) for x in group_1])
    group_2 = ",".join([str(x) for x in group_2])

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'
    #
    impt = f"""
com1: COM ATOMS={group_1}
com2: COM ATOMS={group_2}
d12: DISTANCE ATOMS=com1,com2
opes: {opes_command} ARG=d12 PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES
upperwall: UPPER_WALLS ARG=d12 AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d12', 'opes.bias', 'upperwall.bias']


def write_plumed_opes_1pt(directory=None,
                          idx_d=0,
                          idx_h=1,
                          idx_a=2,
                          temperature=300.0,
                          pace=10,
                          stride=10,
                          barrier=0.5,
                          d_upper=3.5,
                          kappa=500.0,
                          stride_hills=100,
                          explore=False,
                          ):
    if directory is None:
        directory = os.getcwd()

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)
    # Convert d_upper from A to nm
    d_upper = round_sf(d_upper * A_to_nm)

    # Fix the indexing as it starts from 1
    idx_d += 1
    idx_h += 1  # Assumed transferring atom
    idx_a += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d_dh: DISTANCE ATOMS={idx_d},{idx_h}
d_ah: DISTANCE ATOMS={idx_a},{idx_h}
d_da: DISTANCE ATOMS={idx_d},{idx_a}

diff: COMBINE ARG=d_dh,d_ah COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=diff PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

upperwall: UPPER_WALLS ARG=d_da AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d_dh', 'd_ah', 'diff', 'opes.bias', 'upperwall.bias']


def write_plumed_opes_1pt_coord(directory=None,
                                idx_d=0,
                                idx_h=1,
                                idx_a=2,
                                temperature=300.0,
                                pace=10,
                                stride=10,
                                barrier=0.5,
                                r0=1.5,
                                stride_hills=100,
                                explore=False):
    if directory is None:
        directory = os.getcwd()

    # Convert barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)
    # Convert r0 from A to nm
    r0 = round_sf(r0 * A_to_nm)

    # Fix indexing to start from 1 (PLUMED convention)
    idx_d += 1
    idx_h += 1
    idx_a += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
c_dh: COORDINATION GROUPA={idx_d} GROUPB={idx_h} R_0={r0}
c_ah: COORDINATION GROUPA={idx_a} GROUPB={idx_h} R_0={r0}
c_da: COORDINATION GROUPA={idx_d} GROUPB={idx_a} R_0={r0}

diff: COMBINE ARG=c_dh,c_ah COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=diff PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['c_dh', 'c_ah', 'diff', 'opes.bias']


def write_plumed_opes_1pt_3donor_coord(atoms,
                                       directory=None,
                                       idx_n3=0,
                                       idx_h3=1,
                                       idx_o6=2,
                                       idx_o4=2,
                                       idx_n1=3,
                                       idx_h1=4,
                                       idx_o2=5,
                                       idx_n2=6,
                                       temperature=300.0,
                                       pace=10,
                                       stride=10,
                                       barrier=0.5,
                                       r0=1.1,
                                       stride_hills=100,
                                       explore=False,
                                       d_upper=4.0,
                                       kappa=500.0):
    if directory is None:
        directory = os.getcwd()

    # Convert barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)
    # Convert r0 from A to nm
    # r0 = round_sf(r0 * A_to_nm)

    # Convert d_upper from A to nm
    d_upper = round_sf(d_upper * A_to_nm)

    r_1 = round_sf(get_distance(atoms, idx_n3, idx_h3) * r0 * A_to_nm)
    r_2 = round_sf(get_distance(atoms, idx_o6, idx_h3) * r0 * A_to_nm)
    r_3 = round_sf(get_distance(atoms, idx_o6, idx_h3) * r0 * A_to_nm)
    r_4 = round_sf(get_distance(atoms, idx_o4, idx_h3) * r0 * A_to_nm)
    r_5 = round_sf(get_distance(atoms, idx_n1, idx_h1) * r0 * A_to_nm)
    r_6 = round_sf(get_distance(atoms, idx_n3, idx_h1) * r0 * A_to_nm)
    r_7 = round_sf(get_distance(atoms, idx_n1, idx_o2) * r0 * A_to_nm)
    r_8 = round_sf(get_distance(atoms, idx_n2, idx_o2) * r0 * A_to_nm)
    r_9 = round_sf(get_distance(atoms, idx_n2, idx_o2) * r0 * A_to_nm)
    r_10 = round_sf(get_distance(atoms, idx_n1, idx_n3) * r0 * A_to_nm)

    # Fix indexing to start from 1 (PLUMED convention)
    idx_n3 += 1
    idx_h3 += 1
    idx_o6 += 1
    idx_o4 += 1
    idx_n1 += 1
    idx_h1 += 1
    idx_o2 += 1
    idx_h1 += 1
    idx_n2 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
# z1: top PT reaction coordinate
c_1: COORDINATION GROUPA={idx_n3} GROUPB={idx_h3} R_0={r_1}
c_2: COORDINATION GROUPA={idx_o6} GROUPB={idx_h3} R_0={r_2}

# c_1: DISTANCE ATOMS={idx_n3},{idx_h3}
# c_2: DISTANCE ATOMS={idx_o6},{idx_h3}

z1: COMBINE ARG=c_1,c_2 COEFFICIENTS=1,-1 PERIODIC=NO

# z2: top PT reaction coordinate
c_3: COORDINATION GROUPA={idx_o6} GROUPB={idx_h3} R_0={r_3}
c_4: COORDINATION GROUPA={idx_o4} GROUPB={idx_h3} R_0={r_4}

# c_3: DISTANCE ATOMS={idx_o6},{idx_h3}
# c_4: DISTANCE ATOMS={idx_o4},{idx_h3}

z2: COMBINE ARG=c_3,c_4 COEFFICIENTS=1,-1 PERIODIC=NO

# z3: second PT reaction coordinate
c_5: COORDINATION GROUPA={idx_n1} GROUPB={idx_h1} R_0={r_5}
c_6: COORDINATION GROUPA={idx_n3} GROUPB={idx_h1} R_0={r_6}

# c_5: DISTANCE ATOMS={idx_n1},{idx_h1}
# c_6: DISTANCE ATOMS={idx_n3},{idx_h1}

z3: COMBINE ARG=c_5,c_6 COEFFICIENTS=1,-1 PERIODIC=NO
    
# z4
c_7: COORDINATION GROUPA={idx_n1} GROUPB={idx_o2} R_0={r_7}
c_8: COORDINATION GROUPA={idx_n2} GROUPB={idx_o2} R_0={r_8}

# c_7: DISTANCE ATOMS={idx_n1},{idx_o2}
# c_8: DISTANCE ATOMS={idx_n2},{idx_o2}

z4: COMBINE ARG=c_7,c_8 COEFFICIENTS=1,-1 PERIODIC=NO

# z5
c_9: COORDINATION GROUPA={idx_n2} GROUPB={idx_o2} R_0={r_9}
c_10: COORDINATION GROUPA={idx_n1} GROUPB={idx_n3} R_0={r_10}

# c_9: DISTANCE ATOMS={idx_n2},{idx_o2} 
# c_10: DISTANCE ATOMS={idx_n1},{idx_n3}

z5: COMBINE ARG=c_9,c_10 COEFFICIENTS=1,-1 PERIODIC=NO

z:   COMBINE ARG=z1,z2,z3,z4,z5 COEFFICIENTS=1,1,1,1,1 PERIODIC=NO

opes: {opes_command} ARG=z PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

d1: DISTANCE ATOMS={idx_o6},{idx_o4} 
d2: DISTANCE ATOMS={idx_n1},{idx_n3}
d3: DISTANCE ATOMS={idx_n2},{idx_o2}


uw1: UPPER_WALLS ARG=d1 AT={d_upper} KAPPA={kappa}
uw2: UPPER_WALLS ARG=d2 AT={d_upper} KAPPA={kappa}
uw3: UPPER_WALLS ARG=d3 AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['z', 'opes.bias']


def write_plumed_opes_2pt_2d(directory=None,
                             idx_d1=0,
                             idx_h1=1,
                             idx_a1=2,
                             idx_d2=3,
                             idx_h2=4,
                             idx_a2=5,
                             temperature=300.0,
                             pace=10,
                             stride=10,
                             barrier=0.5,
                             d_upper=3.5,
                             kappa=500.0,
                             stride_hills=100,
                             explore=False,
                             ):
    if directory is None:
        directory = os.getcwd()

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)
    # Convert d_upper from A to nm
    d_upper = round_sf(d_upper * A_to_nm)

    # Fix the indexing as it starts from 1
    idx_d1 += 1
    idx_h1 += 1  # Assumed transferring atom
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1  # Assumed transferring atom
    idx_a2 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d_dh1: DISTANCE ATOMS={idx_d1},{idx_h1}
d_ah1: DISTANCE ATOMS={idx_a1},{idx_h1}
d_da1: DISTANCE ATOMS={idx_d1},{idx_a1}

d_dh2: DISTANCE ATOMS={idx_d2},{idx_h2}
d_ah2: DISTANCE ATOMS={idx_a2},{idx_h2}
d_da2: DISTANCE ATOMS={idx_d2},{idx_a2}

diff1: COMBINE ARG=d_dh1,d_ah1 COEFFICIENTS=1,-1 PERIODIC=NO
diff2: COMBINE ARG=d_dh2,d_ah2 COEFFICIENTS=1,-1 PERIODIC=NO

opes: {opes_command} ARG=diff1,diff2 PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

upperwall1: UPPER_WALLS ARG=d_da1 AT={d_upper} KAPPA={kappa}
upperwall2: UPPER_WALLS ARG=d_da2 AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d_dh1', 'd_ah1', 'diff1', 'd_dh2', 'd_ah2', 'diff2', 'opes.bias', 'upperwall1.bias', 'upperwall2.bias']


def write_plumed_opes_2pt_2d_coord(directory=None,
                                   idx_d1=0,
                                   idx_h1=1,
                                   idx_a1=2,
                                   idx_d2=3,
                                   idx_h2=4,
                                   idx_a2=5,
                                   temperature=300.0,
                                   pace=10,
                                   stride=10,
                                   barrier=0.5,
                                   stride_hills=100,
                                   explore=False,
                                   r0=1.5,
                                   ):
    if directory is None:
        directory = os.getcwd()

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)
    # Convert r0 from A to nm
    r0 = round_sf(r0 * A_to_nm)

    # PLUMED uses 1-based indexing
    idx_d1 += 1
    idx_h1 += 1
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1
    idx_a2 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
# Coordination numbers for each donor–H–acceptor triplet (1-based indices)
c_dh1: COORDINATION GROUPA={idx_d1} GROUPB={idx_h1} R_0={r0}
c_ah1: COORDINATION GROUPA={idx_a1} GROUPB={idx_h1} R_0={r0}
c_da1: COORDINATION GROUPA={idx_d1} GROUPB={idx_a1} R_0={r0} 

c_dh2: COORDINATION GROUPA={idx_d2} GROUPB={idx_h2} R_0={r0} 
c_ah2: COORDINATION GROUPA={idx_a2} GROUPB={idx_h2} R_0={r0} 
c_da2: COORDINATION GROUPA={idx_d2} GROUPB={idx_a2} R_0={r0} 

# Proton-transfer-like coordinates (donor–H minus acceptor–H)
diff1: COMBINE ARG=c_dh1,c_ah1 COEFFICIENTS=1,-1 PERIODIC=NO
diff2: COMBINE ARG=c_dh2,c_ah2 COEFFICIENTS=1,-1 PERIODIC=NO

# 2D OPES bias on (diff1, diff2)
opes: {opes_command} ARG=diff1,diff2 PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
"""
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)

    return ['c_dh1', 'c_ah1', 'diff1', 'c_dh2', 'c_ah2', 'diff2', 'opes.bias']


def write_plumed_opes_2pt_1d(directory=None,
                             idx_d1=0,
                             idx_h1=1,
                             idx_a1=2,
                             idx_d2=3,
                             idx_h2=4,
                             idx_a2=5,
                             temperature=300.0,
                             pace=10,
                             stride=10,
                             barrier=0.5,
                             d_upper=3.5,
                             kappa=500.0,
                             stride_hills=100,
                             explore=False,
                             ):
    if directory is None:
        directory = os.getcwd()

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)
    # Convert d_upper from A to nm
    d_upper = round_sf(d_upper * A_to_nm)

    # Fix the indexing as it starts from 1
    idx_d1 += 1
    idx_h1 += 1  # Assumed transferring atom
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1  # Assumed transferring atom
    idx_a2 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d_dh1: DISTANCE ATOMS={idx_d1},{idx_h1}
d_ah1: DISTANCE ATOMS={idx_a1},{idx_h1}
d_da1: DISTANCE ATOMS={idx_d1},{idx_a1}

d_dh2: DISTANCE ATOMS={idx_d2},{idx_h2}
d_ah2: DISTANCE ATOMS={idx_a2},{idx_h2}
d_da2: DISTANCE ATOMS={idx_d2},{idx_a2}

diff1: COMBINE ARG=d_dh1,d_ah1 COEFFICIENTS=1,-1 PERIODIC=NO
diff2: COMBINE ARG=d_dh2,d_ah2 COEFFICIENTS=1,-1 PERIODIC=NO
pt_cv: COMBINE ARG=diff1,diff2 COEFFICIENTS=0.5,0.5 PERIODIC=NO

opes: {opes_command} ARG=pt_cv PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

upperwall1: UPPER_WALLS ARG=d_da1 AT={d_upper} KAPPA={kappa}
upperwall2: UPPER_WALLS ARG=d_da2 AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d_dh1', 'd_ah1', 'diff1', 'd_dh2', 'd_ah2', 'diff2', 'pt_cv', 'opes.bias', 'upperwall1.bias',
            'upperwall2.bias']


def write_plumed_opes_2pt_1d_coord(directory=None,
                                   idx_d1=0,
                                   idx_h1=1,
                                   idx_a1=2,
                                   idx_d2=3,
                                   idx_h2=4,
                                   idx_a2=5,
                                   temperature=300.0,
                                   pace=10,
                                   stride=10,
                                   barrier=0.5,
                                   stride_hills=100,
                                   explore=False,
                                   r0=1.5,
                                   ):
    if directory is None:
        directory = os.getcwd()

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)
    # Convert r0 from A to nm
    r0 = round_sf(r0 * A_to_nm)

    # 1-based indexing for PLUMED
    idx_d1 += 1
    idx_h1 += 1
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1
    idx_a2 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
# Coordination numbers (donor–H, acceptor–H, donor–acceptor) for both paths
c_dh1: COORDINATION GROUPA={idx_d1} GROUPB={idx_h1} R_0={r0}
c_ah1: COORDINATION GROUPA={idx_a1} GROUPB={idx_h1} R_0={r0} 
c_da1: COORDINATION GROUPA={idx_d1} GROUPB={idx_a1} R_0={r0} 

c_dh2: COORDINATION GROUPA={idx_d2} GROUPB={idx_h2} R_0={r0} 
c_ah2: COORDINATION GROUPA={idx_a2} GROUPB={idx_h2} R_0={r0}
c_da2: COORDINATION GROUPA={idx_d2} GROUPB={idx_a2} R_0={r0} 

# Two proton-transfer-like coordinates
diff1: COMBINE ARG=c_dh1,c_ah1 COEFFICIENTS=1,-1 PERIODIC=NO
diff2: COMBINE ARG=c_dh2,c_ah2 COEFFICIENTS=1,-1 PERIODIC=NO

# 1D collective variable (average of the two)
pt_cv: COMBINE ARG=diff1,diff2 COEFFICIENTS=0.5,0.5 PERIODIC=NO

# OPES on the 1D CV
opes: {opes_command} ARG=pt_cv PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
"""
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)

    return [
        'c_dh1', 'c_ah1', 'diff1',
        'c_dh2', 'c_ah2', 'diff2',
        'pt_cv', 'opes.bias',
    ]


def write_plumed_opes_2pt_1d_coord_com(directory=None,
                                       idx_d1=0,
                                       idx_h1=1,
                                       idx_a1=2,
                                       idx_d2=3,
                                       idx_h2=4,
                                       idx_a2=5,
                                       group_1=[0],
                                       group_2=[1],
                                       d_upper=5.0,
                                       kappa=500.0,
                                       temperature=300.0,
                                       pace=10,
                                       stride=10,
                                       barrier=0.5,
                                       stride_hills=100,
                                       explore=False,
                                       r0=1.5,
                                       ):
    if directory is None:
        directory = os.getcwd()

    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)
    # Convert r0 from A to nm
    r0 = round_sf(r0 * A_to_nm)
    # Convert d_upper from A to nm
    d_upper = round_sf(d_upper * A_to_nm)

    # 1-based indexing for PLUMED
    idx_d1 += 1
    idx_h1 += 1
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1
    idx_a2 += 1

    # Fix the indexing as it starts from 1
    group_1 = [x + 1 for x in group_1]
    group_2 = [x + 1 for x in group_2]

    group_1 = ",".join([str(x) for x in group_1])
    group_2 = ",".join([str(x) for x in group_2])

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
# Coordination numbers (donor–H, acceptor–H, donor–acceptor) for both paths
c_dh1: COORDINATION GROUPA={idx_d1} GROUPB={idx_h1} R_0={r0}
c_ah1: COORDINATION GROUPA={idx_a1} GROUPB={idx_h1} R_0={r0} 
c_da1: COORDINATION GROUPA={idx_d1} GROUPB={idx_a1} R_0={r0} 

c_dh2: COORDINATION GROUPA={idx_d2} GROUPB={idx_h2} R_0={r0} 
c_ah2: COORDINATION GROUPA={idx_a2} GROUPB={idx_h2} R_0={r0}
c_da2: COORDINATION GROUPA={idx_d2} GROUPB={idx_a2} R_0={r0} 

# Two proton-transfer-like coordinates
diff1: COMBINE ARG=c_dh1,c_ah1 COEFFICIENTS=1,-1 PERIODIC=NO
diff2: COMBINE ARG=c_dh2,c_ah2 COEFFICIENTS=1,-1 PERIODIC=NO

# 1D collective variable (average of the two)
pt_cv: COMBINE ARG=diff1,diff2 COEFFICIENTS=0.5,0.5 PERIODIC=NO

# Center of mass for the two groups
com1: COM ATOMS={group_1}
com2: COM ATOMS={group_2}
d12: DISTANCE ATOMS=com1,com2

# OPES on the 1D CV
opes: {opes_command} ARG=pt_cv,d12 PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES
upperwall: UPPER_WALLS ARG=d12 AT={d_upper} KAPPA={kappa}
PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
"""
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)

    return [
        'c_dh1', 'c_ah1', 'diff1',
        'c_dh2', 'c_ah2', 'diff2',
        'pt_cv', 'opes.bias',
        'd12', 'upperwall.bias',
    ]
