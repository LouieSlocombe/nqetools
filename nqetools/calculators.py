import os

from ase.calculators.nwchem import NWChem
from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile

from .qchem_mod import QChem


def nwchem_calc_preset(task=None,
                       charge=0,
                       xc="B3LYP",
                       multiplicity=1,
                       basis_set="6-311++G**",
                       disp=None,
                       solv=None):
    """
    Set up and return an NWChem calculator object with the specified settings.

    Parameters:
    task (str, optional): Specific task for the calculation (e.g., "freq" for frequency calculations). Default is None.
    charge (int, optional): Charge of the molecule. Default is 0.
    xc (str, optional): Exchange-correlation functional to use. Default is "B3LYP".
    multiplicity (int, optional): Multiplicity of the molecule. Default is 1.
    basis_set (str, optional): Basis set to use. Default is "6-311++G**".
    disp (str, optional): Dispersion correction method to use ("XDM" or "D3"). Default is None.
    solv (str, optional): Solvent model to use ("WATER" or "PROTEIN"). Default is None.

    Returns:
    NWChem: An NWChem calculator object with the specified settings.
    """
    # Init the nwchem dictionary
    tmp = dict(charge=charge, basis=basis_set)
    # Make the standard dft block
    tmp["dft"] = dict(maxiter=2000,
                      iterations=1000,
                      grid="fine nodisk",
                      print="medium",
                      direct=" ",
                      noio=" ",
                      xc=str(xc).upper(),
                      mult=multiplicity)
    if disp is not None:
        if disp.upper() == "XDM":
            # https://nwchemgit.github.io/Density-Functional-Theory-for-Molecules.html#xdm-exchange-hole-dipole-moment-dispersion-model
            val = tmp.get("dft")  # Key the key value
            val["xdm "] = "a1 0.6224 a2 1.7068"  # modify the value
            tmp["dft"] = val  # Put it back

        elif disp.upper() == "D3":
            # https://nwchemgit.github.io/Density-Functional-Theory-for-Molecules.html#disp-empirical-long-range-contribution-vdw
            val = tmp.get("dft")  # Key the key value
            val["disp"] = "vdw 3"  # modify the value
            tmp["dft"] = val  # Put it back

    if solv is not None:
        if solv.upper() == "WATER":
            # https://nwchemgit.github.io/COSMO-Solvation-Model.html
            tmp["cosmo"] = dict(do_cosmo_smd=True, solvent='water')

        if solv.upper() == "PROTEIN":
            # https://nwchemgit.github.io/COSMO-Solvation-Model.html
            tmp["cosmo"] = dict(do_cosmo_smd=True, dielec=8.0)

    # Add the task if specified, this is for frequency calculations
    # https://nwchemgit.github.io/Vibration.html#an-example-input-deck
    if task is not None:
        tmp["task"] = task

    return NWChem(**tmp)


def orca_calc_preset(orca_path=None,
                     calc_type="DFT",
                     xc="B3LYP",
                     charge=0,
                     multiplicity=1,
                     basis_set="6-31+G(d,p)",
                     nprocs=1,
                     f_solv=True,
                     f_disp=True,
                     atom_list=None,
                     calc_extra=None,
                     scf_option=None):
    """
    Set up and return an ORCA calculator object with the specified settings.

    Parameters:
    orca_path (str, optional): Path to the ORCA executable. Default is None.
    calc_type (str, optional): The type of calculation to perform. Default is "DFT".
    xc (str, optional): The exchange-correlation functional to use. Default is "B3LYP".
    charge (int, optional): The charge of the molecule. Default is 0.
    multiplicity (int, optional): The multiplicity of the molecule. Default is 1.
    basis_set (str, optional): The basis set to use. Default is "6-31+G(d,p)".
    nprocs (int, optional): The number of processors to use. Default is 1.
    f_solv (bool or str, optional): Whether to use a solvent model. If True, uses "WATER". Default is True.
    f_disp (bool or str, optional): Whether to include dispersion corrections. If True, uses "D4". Default is True.
    atom_list (list, optional): List of atoms for QM/MM calculations. Default is None.
    calc_extra (str, optional): Additional calculation options. Default is None.
    scf_option (str, optional): Additional SCF options. Default is None.

    Returns:
    ORCA: An ORCA calculator object with the specified settings.
    """
    if orca_path is None:
        # try and read the path from the environment
        orca_path = os.environ.get("ORCA_PATH")

    profile = OrcaProfile(command=orca_path)

    if nprocs > 1:
        inpt_procs = "%pal nprocs {} end".format(nprocs)
    else:
        inpt_procs = ""

    if f_solv is not None and f_solv is not False:
        if f_solv:
            f_solv = "WATER"
        inpt_solv = """
        %CPCM SMD TRUE
            SMDSOLVENT "{}"
        END""".format(f_solv)
    else:
        inpt_solv = ""

    if f_disp is None or f_disp is False:
        inpt_disp = ""
    else:
        if f_disp:
            f_disp = "D4"
        inpt_disp = f_disp

    if atom_list is not None:
        inpt_xtb = """
        %QMMM QMATOMS {{}} END END
        """.format(str(atom_list).strip("[").strip("]"))

    inpt_blocks = inpt_procs + inpt_solv

    if calc_type == "DFT":
        inpt_simple = "{} {} {}".format(xc, inpt_disp, basis_set)
    elif calc_type == "MP2":
        inpt_simple = "DLPNO-{} {} {}/C".format(calc_type, basis_set, basis_set)
    elif calc_type == "CCSD":
        inpt_simple = "DLPNO-{}(T) {} {}/C".format(calc_type, basis_set, basis_set)
    elif calc_type == "QM/XTB2":
        inpt_simple = "{} {} {} {}".format(calc_type, xc, inpt_disp, basis_set)
        inpt_blocks = inpt_procs + inpt_solv + inpt_xtb
    else:
        inpt_simple = "{} {}".format(calc_type, basis_set)

    # Add the scf option
    if scf_option is not None:
        inpt_simple += " " + scf_option

    # Add the extra options
    if calc_extra is not None:
        inpt_simple += " " + calc_extra

    calc = ORCA(
        profile=profile,
        charge=charge,
        mult=multiplicity,
        directory='data',
        orcasimpleinput=inpt_simple,
        orcablocks=inpt_blocks
    )
    calc._label = "check"
    return calc


def qchem_calc_preset(charge=0,
                      multiplicity=1,
                      xc="BLYP",  # wB97X-V B3LYP
                      basis="6-311G**",  # 6-31G* 6-311G** 6-31G(d,p) 6-311++G**
                      f_fast=False,
                      f_solv=False,
                      f_disp=False,
                      f_neo=False,
                      neo_idx=None,
                      neo_epc='epc19',  # LDA epc172, GGA epc19
                      neo_preset="PB4-D",
                      neo_isotope="1",
                      scf_algorithm="DIIS",  # DIIS GDM DIIS_GDM
                      solv_extra=None
                      ):
    if neo_idx is None:
        neo_idx = [0]
    inpt_dict = {
        'label': 'calc/data',
        'charge': charge,
        'multiplicity': multiplicity,
        'method': xc,
        'basis': basis,
        'n_t': os.environ.get('OMP_NUM_THREADS'),
        'scf_convergence': "9",
        'thresh': '14',
        'max_scf_cycles': "100",
        'scf_algorithm': scf_algorithm,
    }

    if f_solv:
        inpt_dict.update({'solvent_method': 'PCM'})  # kirkwood, COSMO, PCM, SMD

    if f_disp:
        inpt_dict.update({'dft_d': 'D4'})
        # inpt_dict.update({'dft_d': 'D3_BJ'})

    if f_fast:
        inpt_dict.update({'fast_xc': 'True'})
        inpt_dict.update({'xc_smart_grid': 'True'})

    if f_neo:
        inpt_dict.update({'neo': 'True'})
        inpt_dict.update({'point_group_symmetry': 'False'})
        inpt_dict.update({'neo_epc': neo_epc})
        inpt_dict.update({'neo_preset': neo_preset})
        inpt_dict.update({'neo_idx': neo_idx})
        inpt_dict.update({'neo_isotope': neo_isotope})
    # Add solvent extra
    if solv_extra is not None and f_solv is True:
        return QChem(solv_extra=solv_extra, **inpt_dict)
    else:
        return QChem(**inpt_dict)
