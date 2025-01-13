from scipy.constants import physical_constants

from .tools import round_sf

# General
def write_plumed_input(temperature=300, sigma=[0.005, 0.05], ):
    impt = """# default units are LENGTH=nm ENERGY=kJ/mol TIME=ps\n"""

    restraints = {}

    restraints["doo"] = "DISTANCE ATOMS=1,2"
    restraints["co1"] = "DISTANCES GROUPA=1 GROUPB=3-7 LESS_THAN={RATIONAL R_0=0.14}"
    restraints["co2"] = "DISTANCES GROUPA=2 GROUPB=3-7 LESS_THAN={RATIONAL R_0=0.14}"
    restraints["dc"] = "COMBINE ARG=co1.lessthan,co2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO"

    # Iterate over the restraints and add them to the input file
    for key, value in restraints.items():
        impt += f"{key}: {value}\n"

    restraints_keys = list(restraints.keys())
    # convert the keys to a string
    restraints_str = ",".join(restraints_keys)

    # mtd line
    mtd_line = f"mtd: METAD ARG={restraints_str} PACE=10 \n"
    # add the mtd line to the input file
    impt += mtd_line
    #

    return impt


def write_plumed_input_coordination(atoms,
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
                                    biasfactor=10,
                                    ):
    # https://www.plumed.org/doc-master/user-doc/html/_d_i_s_t_a_n_c_e_s.html
    # https://www.plumed.org/doc-v2.9/user-doc/html/_u_p_p_e_r__w_a_l_l_s.html

    # Direct calculation of conversion constants
    avogadro_number = physical_constants["Avogadro constant"][0]  # mol^-1
    eV_to_J = physical_constants["electron volt-joule relationship"][0]  # 1 eV = this many joules
    J_to_kJ = 1e-3  # 1 J = 0.001 kJ

    A_to_nm = 10  # Å to nm

    if sigma is None:
        sigma = [0.005, 0.05]

    # Convert d_low and d_upper from A to nm
    d_low = d_low / A_to_nm
    d_upper = d_upper / A_to_nm

    # eV/Å² to 1 kJ/(mol·nm²)
    kappa = round_sf(kappa * eV_to_J * J_to_kJ * avogadro_number * A_to_nm ** 2)
    print(f"kappa: {kappa}")

    print(f"inverse kappa {250 / (eV_to_J * J_to_kJ * avogadro_number * A_to_nm ** 2)}")

    # convert the height from eV to kJ/mol
    height = round_sf(height * eV_to_J * J_to_kJ * avogadro_number)

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
mtd:   METAD ARG=d,dc PACE={pace} SIGMA={sigma[0]},{sigma[1]} HEIGHT={height} FILE=HILLS BIASFACTOR={biasfactor} TEMP={temperature}
uwall: UPPER_WALLS ARG=d AT={d_upper} KAPPA={kappa}

PRINT ARG=d,c1.*,c2.*,dc,mtd.*,uwall.* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """

    return impt


# Just distance
def write_plumed_input_distance():
    # https://www.plumed-nest.org/eggs/24/021/
    # https://www.plumed-nest.org/eggs/24/021/data/molecular-dynamics/ion-pairing/caco3-batches/1/plumed.dat.html
    tmp = """
    d: DISTANCE ATOMS=1,2 

    opes: OPES_METAD ARG=d PACE=500 BARRIER=50 TEMP=330

    uwall: UPPER_WALLS ARG=d AT=0.9 KAPPA=1000.0

    PRINT ARG STRIDE=100 FILE=COLVAR
    """

    return None
