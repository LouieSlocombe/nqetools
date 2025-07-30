import os
import tempfile

import numpy as np
from ase import Atoms
from ase.calculators.nwchem import NWChem
from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile

from .qchem_mod import QChem


def nwchem_calc_preset(directory=None,
                       task=None,
                       charge=0,
                       xc='B3LYP',
                       multiplicity=1,
                       basis_set='6-311++G**',
                       disp=None,
                       solv=None,
                       host=None):
    if directory is None:
        directory = os.path.join(tempfile.mkdtemp(), 'nwchem')

    tmp = dict(
        label=directory,
        charge=charge,
        basis=basis_set,
        dft=dict(
            maxiter=2000,
            iterations=1000,
            grid='fine nodisk',
            print='medium',
            direct=' ',
            noio=' ',
            xc=xc.upper(),
            mult=multiplicity
        )
    )
    if host is not None:
        tmp['driver'] = dict(socket=dict(unix=host))

    if disp:
        if disp.upper() == 'XDM':
            tmp['dft']['xdm '] = 'a1 0.6224 a2 1.7068'
        elif disp.upper() == 'D3':
            tmp['dft']['disp'] = 'vdw 3'

    if solv:
        if solv.upper() == 'WATER':
            tmp['cosmo'] = dict(do_cosmo_smd=True, solvent='water')
        elif solv.upper() == 'PROTEIN':
            tmp['cosmo'] = dict(do_cosmo_smd=True, dielec=8.0)

    if task:
        tmp['task'] = task

    return NWChem(**tmp)


def get_total_electrons(atoms: Atoms) -> int:
    """
    Calculate the total number of electrons in a molecule.

    This function computes the total number of electrons in a molecule
    represented by an ASE `Atoms` object. It sums the atomic numbers (Z)
    of all atoms in the molecule and adjusts for the explicit charge
    provided in the `Atoms.info` dictionary.

    Parameters:
    -----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecule.

    Returns:
    --------
    int
        The total number of electrons in the molecule, corrected for its charge.
    """
    # Sum atomic numbers (Z) for every atom in the molecule
    n_electrons = int(np.sum(atoms.get_atomic_numbers()))

    # Correct for explicit total charge, if provided in the `Atoms.info` dictionary
    charge = atoms.info.get('charge', 0.0)
    n_electrons -= int(round(charge))

    return n_electrons


def round_to_nearest_two(number):
    """
    Round a number to the nearest multiple of 2.
    If the result would be 0, return 1 instead.

    Parameters:
    -----------
    number : float or int
        The number to be rounded

    Returns:
    --------
    int
        The nearest multiple of 2, or 1 if result would be 0
    """
    # Round to nearest multiple of 2
    result = round(number / 2) * 2

    # If result is 0, set it to 1
    if result == 0:
        result = 1

    return result


def calculate_ccsd_energy(atoms,
                          charge=0,
                          multiplicity=1,
                          orca_path=None,
                          basis_set='def2-TZVPP',
                          n_procs=10):
    """
    Perform a CCSD (Coupled Cluster Single and Double) energy calculation using the ORCA quantum chemistry package.

    This function sets up and executes a CCSD energy calculation for a molecule represented by an ASE `Atoms` object.
    It ensures that the number of processors used does not exceed the total number of electrons in the system.

    Parameters:
    -----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecule.
    charge : int, optional
        Total charge of the molecule. Default is 0.
    multiplicity : int, optional
        Spin multiplicity of the molecule. Default is 1.
    orca_path : str, optional
        Path to the ORCA executable. If None, it will attempt to read from the environment variable 'ORCA_PATH'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-TZVPP'.
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 10.

    Returns:
    --------
    float
        The CCSD energy of the molecule in eV.

    Raises:
    -------
    ValueError
        If the number of processors exceeds the adjusted limit based on the total number of electrons.
    """
    # If no ORCA path is provided, try to read it from the environment variable
    orca_path = os.path.abspath(orca_path or os.getenv('ORCA_PATH', 'orca'))

    # Get the total number of electrons in the system
    total_electrons = get_total_electrons(atoms)
    # Prevent too many processors being used
    if n_procs > total_electrons:
        n_procs = round_to_nearest_two(total_electrons - 2)

    # Create a temporary directory for the ORCA calculation
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up the ORCA calculator with the specified parameters
        calc = orca_calc_preset(orca_path=orca_path,
                                directory=temp_dir,
                                calc_type='CCSD',
                                charge=charge,
                                multiplicity=multiplicity,
                                basis_set=basis_set,
                                n_procs=n_procs)
        # Attach the ORCA calculator to the ASE Atoms object
        atoms.calc = calc

        # Perform the energy calculation
        return atoms.get_potential_energy()


def orca_calc_preset(orca_path=None,
                     directory=None,
                     calc_type='DFT',
                     xc='B3LYP',
                     charge=0,
                     multiplicity=1,
                     basis_set='6-311G',  # 'cc-pVDZ', '6-31+G(d,p)',
                     nprocs=1,
                     f_solv=True,
                     f_disp=True,
                     atom_list=None,
                     calc_extra=None,
                     blocks_extra=None,
                     scf_option=None):
    if orca_path is None:
        # try and read the path from the environment
        orca_path = os.environ.get('ORCA_PATH')
    if directory is None:
        directory = os.path.join(tempfile.mkdtemp(), 'orca')

    profile = OrcaProfile(command=orca_path)

    if nprocs > 1:
        inpt_procs = '%pal nprocs {} end'.format(nprocs)
    else:
        inpt_procs = ''

    if f_solv is not None and f_solv is not False:
        if f_solv:
            f_solv = 'WATER'
        inpt_solv = '''
        %CPCM SMD TRUE
            SMDSOLVENT "{}"
        END'''.format(f_solv)
    else:
        inpt_solv = ''

    if f_disp is None or f_disp is False:
        inpt_disp = ''
    else:
        if f_disp:
            f_disp = 'D4'
        inpt_disp = f_disp

    if atom_list is not None and calc_type == 'QM/XTB2':
        inpt_xtb = '''
        %QMMM QMATOMS {{}} END END
        '''.format(str(atom_list).strip('[').strip(']'))
    else:
        inpt_xtb = ''

    if blocks_extra is None:
        blocks_extra = ''

    inpt_blocks = inpt_procs + inpt_solv + blocks_extra

    if calc_type == 'DFT':
        inpt_simple = '{} {} {}'.format(xc, inpt_disp, basis_set)
    elif calc_type == 'MP2':
        inpt_simple = 'DLPNO-{} {} {}/C'.format(calc_type, basis_set, basis_set)
    elif calc_type == 'CCSD':
        inpt_simple = 'DLPNO-{}(T) {} {}/C'.format(calc_type, basis_set, basis_set)
    elif calc_type == 'QM/XTB2':
        inpt_simple = '{} {} {} {}'.format(calc_type, xc, inpt_disp, basis_set)
        inpt_blocks = inpt_procs + inpt_solv + inpt_xtb
    else:
        inpt_simple = '{} {}'.format(calc_type, basis_set)

    # Add the scf option
    if scf_option is not None:
        inpt_simple += ' ' + scf_option

    # Add the extra options
    if calc_extra is not None:
        inpt_simple += ' ' + calc_extra

    calc = ORCA(
        profile=profile,
        charge=charge,
        mult=multiplicity,
        directory=directory,
        orcasimpleinput=inpt_simple + ' EnGrad',
        orcablocks=inpt_blocks
    )
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


orca_preset_dft_cheap = {
    'calc_type': 'DFT',
    'xc': 'BLYP',
    'basis_set': '6-31+G(d,p)',
    'f_disp': False,
    'f_solv': False,
    'atom_list': None,
    'calc_extra': None,
    'scf_option': None
}

orca_preset_dft_gold = {
    'calc_type': 'DFT',
    'xc': 'B3LYP',
    'basis_set': 'DEF2-SVP',
    'f_disp': True,
    'f_solv': True,
    'atom_list': None,
    'calc_extra': None,
    'scf_option': None
}

orca_preset_xtb = {
    'calc_type': 'XTB2',
    'xc': '',
    'basis_set': '',
    'f_disp': False,
    'f_solv': False,
    'atom_list': None,
    'calc_extra': None,
    'scf_option': None
}

orca_preset_mp2_gold = {
    'calc_type': 'MP2',
    'xc': '',
    'basis_set': 'DEF2-TZVPP',
    'f_disp': False,
    'f_solv': True,
    'atom_list': None,
    'calc_extra': None,
    'scf_option': None
}

orca_preset_ccsd_gold = {
    'calc_type': 'CCSD(T)',
    'xc': '',
    'basis_set': 'DEF2-TZVPP',
    'f_disp': False,
    'f_solv': True,
    'atom_list': None,
    'calc_extra': None,
    'scf_option': None
}
