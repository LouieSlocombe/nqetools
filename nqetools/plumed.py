import os

from .conversions import (A_to_nm,
                          eV_to_kJpermol,
                          eVperA2_to_kJpermolpernm2)
from .tools import round_sf

def prep_plumed(plumed_type, args):
    if plumed_type == 'pos':
        return write_plumed_pos(**args)
    elif plumed_type == 'coord':
        return write_plumed_coord(**args)
    elif plumed_type == 'dist':
        return write_plumed_dist(**args)
    else:
        raise ValueError(f'Unknown plumed type: {plumed_type}')



def write_plumed_pos(directory=None,
                     idx_atom=0,
                     pace=20,
                     sigma=0.01,
                     height=1.0,
                     bias=2.5,
                     temperature=300,
                     stride=10):
    if directory is None:
        directory = os.getcwd()

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


def write_plumed_coord(atoms,
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

    # https://www.plumed.org/doc-master/user-doc/html/_d_i_s_t_a_n_c_e_s.html
    # https://www.plumed.org/doc-v2.9/user-doc/html/_u_p_p_e_r__w_a_l_l_s.html

    if sigma is None:
        sigma = [0.005, 0.05]

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

    # default units are LENGTH=nm ENERGY=kJ/mol TIME=ps
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


# Just distance
def write_plumed_dist(directory=None,
                      idx_atom1=0,
                      idx_atom2=1,
                      temperature=300,
                      barrier=0.5,
                      d_upper=4.0,
                      kappa=0.026,
                      pace=10,
                      stride=10,
                      ):
    if directory is None:
        directory = os.getcwd()

    # https://www.plumed-nest.org/eggs/24/021/
    # https://www.plumed-nest.org/eggs/24/021/data/molecular-dynamics/ion-pairing/caco3-batches/1/plumed.dat.html
    # https://www.plumed-nest.org/eggs/24/035/data/notebooks/1_exploration/N2_flooding_inputs/plumed-fresh.dat.html

    # Fix the indexing as it starts from 1
    idx_atom1 += 1
    idx_atom2 += 1
    # Convert the barrier from eV to kJ/mol
    barrier = round_sf(barrier * eV_to_kJpermol)

    impt = f"""
d: DISTANCE ATOMS={idx_atom1},{idx_atom2} 
opes: OPES_METAD ARG=d PACE={pace} BARRIER={barrier} TEMP={temperature}
uwall: UPPER_WALLS ARG=d AT={d_upper} KAPPA={kappa}

PRINT ARG STRIDE={stride} FILE=COLVAR
    """
    # Write the input file
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d', 'opes.bias']
