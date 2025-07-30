import os
import re
import tempfile

import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.nwchem import NWChem
from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile
from ase.io import read
from ase.units import Hartree

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


def orca_calc_preset(orca_path=None,
                     directory=None,
                     calc_type='DFT',
                     xc='wB97X',
                     charge=0,
                     multiplicity=1,
                     basis_set='def2-SVP',
                     n_procs=10,
                     f_solv=False,
                     f_disp=False,
                     atom_list=None,
                     calc_extra=None,
                     blocks_extra=None,
                     scf_option=None):
    """
    Create and configure an ORCA calculator preset for quantum chemistry calculations.

    Parameters:
    -----------
    orca_path : str, optional
        Path to the ORCA executable. If None, it will attempt to read from the environment variable 'ORCA_PATH'.
    directory : str, optional
        Directory where the calculation will be performed. Defaults to a temporary directory.
    calc_type : str, optional
        Type of calculation to perform (e.g., 'DFT', 'MP2', 'CCSD', 'QM/XTB2'). Default is 'DFT'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'wB97X'.
    charge : int, optional
        Total charge of the system. Default is 0.
    multiplicity : int, optional
        Spin multiplicity of the system. Default is 1.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-SVP'.
    n_procs : int, optional
        Number of processors to use. Default is 10.
    f_solv : bool or str, optional
        Solvent model to use. If True, defaults to 'WATER'. Default is False (no solvent).
    f_disp : bool or str, optional
        Dispersion correction to use. If True, defaults to 'D4'. Default is False (no dispersion correction).
    atom_list : list, optional
        List of atoms for QM/MM calculations. Only used if `calc_type` is 'QM/XTB2'. Default is None.
    calc_extra : str, optional
        Additional calculation options to include in the ORCA input. Default is None.
    blocks_extra : str, optional
        Additional ORCA input blocks to include. Default is None.
    scf_option : str, optional
        Additional SCF options to include in the ORCA input. Default is None.

    Returns:
    --------
    ORCA
        Configured ORCA calculator object.
    """
    if orca_path is None:
        # Try and read the path from the environment
        orca_path = os.environ.get('ORCA_PATH')
    if directory is None:
        # Create a temporary directory for the calculation
        directory = os.path.join(tempfile.mkdtemp(), 'orca')

    # Create an ORCA profile with the specified command
    profile = OrcaProfile(command=orca_path)

    # Configure the number of processors
    if n_procs > 1:
        inpt_procs = '%pal nprocs {} end'.format(n_procs)
    else:
        inpt_procs = ''

    # Configure the solvent model
    if f_solv is not None and f_solv is not False:
        if f_solv:
            f_solv = 'WATER'
        inpt_solv = '''
                                              %CPCM SMD TRUE
                                                  SMDSOLVENT "{}"
                                              END'''.format(f_solv)
    else:
        inpt_solv = ''

    # Configure the dispersion correction
    if f_disp is None or f_disp is False:
        inpt_disp = ''
    else:
        if f_disp:
            f_disp = 'D4'
        inpt_disp = f_disp

    # Configure QM/MM atom list for QM/XTB2 calculations
    if atom_list is not None and calc_type == 'QM/XTB2':
        inpt_xtb = '''
                                              %QMMM QMATOMS {{}} END END
                                              '''.format(str(atom_list).strip('[').strip(']'))
    else:
        inpt_xtb = ''

    # Add any additional input blocks
    if blocks_extra is None:
        blocks_extra = ''

    # Combine all input blocks
    inpt_blocks = inpt_procs + inpt_solv + blocks_extra

    # Configure the main calculation input based on the calculation type
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

    if multiplicity > 1:
        if calc_type == 'DFT' or calc_type == 'QM/XTB2':
            inpt_simple = 'UKS  ' + inpt_simple
        elif calc_type == 'MP2' or calc_type == 'CCSD':
            inpt_simple = 'UKS ' + inpt_simple

    # Add the SCF option if provided
    if scf_option is not None:
        inpt_simple += ' ' + scf_option

    # Add any extra calculation options
    if calc_extra is not None:
        inpt_simple += ' ' + calc_extra

    # Create and return the ORCA calculator object
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


def optimise_atoms(atoms,
                   charge=0,
                   multiplicity=1,
                   orca_path=None,
                   xc='r2SCAN-3c',
                   basis_set='def2-QZVP',
                   tight_opt=False,
                   tight_scf=False,
                   f_solv=False,
                   f_disp=False,
                   n_procs=10):
    """
    Optimise the geometry of a molecule using the ORCA quantum chemistry package.

    This function sets up an ORCA calculation to optimise the geometry of a molecule
    represented by an ASE `Atoms` object. It supports various calculation options,
    including tight optimisation, solvent effects, and dispersion corrections.

    Parameters:
    -----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecule to be optimised.
    charge : int, optional
        Total charge of the molecule. Default is 0.
    multiplicity : int, optional
        Spin multiplicity of the molecule. Default is 1.
    orca_path : str, optional
        Path to the ORCA executable. If None, it will attempt to read from the environment variable 'ORCA_PATH'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'r2SCAN-3c'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-QZVP'.
    tight_opt : bool, optional
        Whether to use tight geometry optimisation. Default is False.
    tight_scf : bool, optional
        Whether to use tight SCF convergence criteria. Default is False.
    f_solv : bool, optional
        Whether to include solvent effects in the calculation. Default is False.
    f_disp : bool, optional
        Whether to include dispersion corrections in the calculation. Default is False.
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 10.

    Returns:
    --------
    ase.Atoms
        An ASE `Atoms` object representing the optimised geometry of the molecule.

    Raises:
    -------
    ValueError
        If the ORCA path cannot be determined or the calculation fails.
    """
    # Determine the ORCA path
    if orca_path is None:
        # Try to read the path from the environment variable
        orca_path = os.environ.get('ORCA_PATH')
    else:
        # Convert the provided path to an absolute path
        orca_path = os.path.abspath(orca_path)

    if tight_opt:
        # Set up geometry optimization and frequency calculation parameters
        opt_option = 'TIGHTOPT'
    else:
        # Set up frequency calculation parameters only
        opt_option = 'OPT'

    if tight_scf:
        # Set up tight SCF convergence parameters
        calc_extra = f'{opt_option} TIGHTSCF'
    else:
        # Use default SCF convergence parameters
        calc_extra = f'{opt_option}'

    # Create a temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        orca_file = os.path.join(temp_dir, "orca.xyz")

        # Set up the ORCA calculator with the specified parameters
        calc = orca_calc_preset(orca_path=orca_path,
                                directory=temp_dir,
                                charge=charge,
                                multiplicity=multiplicity,
                                xc=xc,
                                basis_set=basis_set,
                                n_procs=n_procs,
                                f_solv=f_solv,
                                f_disp=f_disp,
                                calc_extra=calc_extra)
        # Assign the calculator to the molecule
        atoms.calc = calc

        # Trigger the calculation to optimise the geometry
        _ = atoms.get_potential_energy()

        # Load the optimised geometry from the ORCA output file
        return read(orca_file, format="xyz")


def load_ir_data(filename):
    """
    Load IR spectrum data from ORCA output file into a pandas DataFrame.

    This function reads an ORCA output file containing IR frequency data,
    extracts the relevant information, and returns it as a pandas DataFrame.

    Parameters:
    -----------
    filename : str
        Path to the IR frequency output file.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the following columns:
        - 'Mode': Mode number (int).
        - 'Frequency (cm^-1)': Frequency in inverse centimeters (float).
        - 'Epsilon': Epsilon value (float).
        - 'Intensity (km/mol)': Intensity in km/mol (float).

    Raises:
    -------
    ValueError
        If the IR spectrum data cannot be found in the file.
    """
    # Read the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the start of the IR spectrum data
    start_idx = None
    for i, line in enumerate(lines):
        if 'Mode   freq       eps      Int' in line:
            start_idx = i + 2  # Skip header and separator line
            break

    if start_idx is None:
        raise ValueError("Could not find IR spectrum data in the file")

    # Extract data
    data = []
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('*') or line.startswith('The first'):
            break

        # Parse the line using regex to handle varying whitespace
        match = re.match(r'\s*(\d+):\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
        if match:
            mode = int(match.group(1))  # Mode number
            freq = float(match.group(2))  # Frequency in cm^-1
            eps = float(match.group(3))  # Epsilon value
            intensity = float(match.group(4))  # Intensity in km/mol
            data.append([mode, freq, eps, intensity])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Mode', 'Frequency (cm^-1)', 'Epsilon', 'Intensity (km/mol)'])

    return df


def load_raman_data(filename):
    """
    Load Raman spectrum data from ORCA output file into a pandas DataFrame.

    This function reads an ORCA output file containing Raman frequency data,
    extracts the relevant information, and returns it as a pandas DataFrame.

    Parameters:
    -----------
    filename : str
        Path to the Raman frequency output file.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the following columns:
        - 'Mode': Mode number (int).
        - 'Frequency (cm^-1)': Frequency in inverse centimeters (float).
        - 'Intensity (km/mol)': Intensity in km/mol (float).
        - 'Depolarization': Depolarization value (float).

    Raises:
    -------
    ValueError
        If the Raman spectrum data cannot be found in the file.
    """
    # Read the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the start of the Raman spectrum data
    start_idx = None
    for i, line in enumerate(lines):
        if 'Mode    freq (cm**-1)   Activity   Depolarization' in line:
            start_idx = i + 2  # Skip header and separator line
            break

    if start_idx is None:
        raise ValueError("Could not find Raman spectrum data in the file")

    # Extract data
    data = []
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('The first'):
            break

        # Parse the line using regex to handle varying whitespace
        match = re.match(r'\s*(\d+):\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
        if match:
            mode = int(match.group(1))  # Mode number
            freq = float(match.group(2))  # Frequency in cm^-1
            activity = float(match.group(3))  # Intensity in km/mol
            depolarization = float(match.group(4))  # Depolarization value
            data.append([mode, freq, activity, depolarization])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Mode', 'Frequency (cm^-1)', 'Intensity (km/mol)', 'Depolarization'])

    return df


def load_vib_data(filename):
    """
    Load vibrational spectrum data from a file into a pandas DataFrame.

    Parameters:
    -----------
    filename : str
        Path to the file containing vibrational spectrum data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the following columns:
        - 'Mode': Mode number (int).
        - 'Frequency (cm^-1)': Frequency in inverse centimeters (float).
        - 'Epsilon': Default value set to 1.0 (float).
        - 'Intensity (km/mol)': Default value set to 1.0 (float).

    Raises:
    -------
    ValueError
        If the vibrational spectrum data cannot be found in the file.
    """
    # Read the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the start of the IR spectrum data
    start_idx = None
    for i, line in enumerate(lines):
        if 'Mode   freq       eps      Int' in line:
            start_idx = i + 2  # Skip header and separator line
            break

    if start_idx is None:
        raise ValueError("Could not find IR spectrum data in the file")

    # Extract data
    data = []
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('*') or line.startswith('The first'):
            break

        # Parse the line using regex to handle varying whitespace
        match = re.match(r'\s*(\d+):\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
        if match:
            mode = int(match.group(1))  # Mode number
            freq = float(match.group(2))  # Frequency in cm^-1
            eps = 1.0  # Default epsilon value
            intensity = 1.0  # Default intensity value
            data.append([mode, freq, eps, intensity])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Mode', 'Frequency (cm^-1)', 'Epsilon', 'Intensity (km/mol)'])

    return df


def calculate_vib_spectrum(atoms,
                           charge=0,
                           multiplicity=1,
                           orca_path=None,
                           xc='r2SCAN-3c',
                           basis_set='def2-QZVP',
                           tight_opt=False,
                           tight_scf=False,
                           f_solv=False,
                           f_disp=False,
                           n_procs=10):
    """
    Calculate vibrational spectrum data using the ORCA quantum chemistry package.

    This function sets up and performs a vibrational spectrum calculation for a molecule
    represented by an ASE `Atoms` object. It computes IR, Raman, and vibrational spectrum data.

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
    xc : str, optional
        Exchange-correlation functional to use. Default is 'r2SCAN-3c'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-QZVP'.
    tight_opt : bool, optional
        Whether to use tight geometry optimisation. Default is False.
    tight_scf : bool, optional
        Whether to use tight SCF convergence criteria. Default is False.
    f_solv : bool, optional
        Whether to include solvent effects in the calculation. Default is False.
    f_disp : bool, optional
        Whether to include dispersion corrections in the calculation. Default is False.
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 10.

    Returns:
    --------
    tuple
        A tuple containing three pandas DataFrames:
        - data_ir : pd.DataFrame
            IR spectrum data.
        - data_raman : pd.DataFrame
            Raman spectrum data.
        - data_vib : pd.DataFrame
            Vibrational spectrum data.

    Raises:
    -------
    ValueError
        If the ORCA path cannot be determined or the calculation fails.
    """
    # Determine the ORCA path
    if orca_path is None:
        # Try to read the path from the environment variable
        orca_path = os.environ.get('ORCA_PATH')
    else:
        # Convert the provided path to an absolute path
        orca_path = os.path.abspath(orca_path)

    if tight_opt:
        # Set up geometry optimization and frequency calculation parameters
        opt_option = 'TIGHTOPT'
    else:
        # Set up frequency calculation parameters only
        opt_option = 'OPT'

    if tight_scf:
        # Set up tight SCF convergence parameters
        calc_extra = f'{opt_option} TIGHTSCF FREQ'
    else:
        # Use default SCF convergence parameters
        calc_extra = f'{opt_option} FREQ'

    blocks_extra = '''
                          %ELPROP
                              POLAR 1
                          END'''

    # Create a temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        orca_file = os.path.join(temp_dir, 'orca.out')

        # Set up the ORCA calculator with the specified parameters
        calc = orca_calc_preset(orca_path=orca_path,
                                directory=temp_dir,
                                charge=charge,
                                multiplicity=multiplicity,
                                xc=xc,
                                basis_set=basis_set,
                                n_procs=n_procs,
                                f_solv=f_solv,
                                f_disp=f_disp,
                                calc_extra=calc_extra,
                                blocks_extra=blocks_extra)

        # Attach the calculator to the ASE Atoms object
        atoms.calc = calc

        # Perform the calculation (this will write the output to the ORCA file)
        _ = atoms.get_potential_energy()

        # Load IR spectrum data
        data_ir = load_ir_data(orca_file)

        # Load Raman spectrum data
        data_raman = load_raman_data(orca_file)

        # Load vibrational spectrum data
        data_vib = load_vib_data(orca_file)

        return data_ir, data_raman, data_vib


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


def grab_value(orca_file, term, splitter):
    """
    Extract a specific numerical value from an ORCA output file.

    This function reads an ORCA output file in reverse order, searches for a specific term,
    and extracts the numerical value associated with it. The value is converted from Hartree
    units to eV using the ASE `Hartree` constant.

    Parameters:
    -----------
    orca_file : str
        Path to the ORCA output file.
    term : str
        The term to search for in the file.
    splitter : str
        The delimiter used to split the line containing the term.

    Returns:
    --------
    float or None
        The extracted value in eV, or None if the term is not found.
    """
    with open(orca_file, 'r') as f:
        for line in reversed(f.readlines()):
            if term in line:
                return float(line.split(splitter)[-1].split('Eh')[0]) * Hartree
        return None


def calculate_free_energy(atoms,
                          charge=0,
                          multiplicity=1,
                          temperature=None,
                          pressure=None,
                          orca_path=None,
                          xc='r2SCAN-3c',
                          basis_set='def2-QZVP',
                          tight_opt=False,
                          tight_scf=False,
                          f_solv=False,
                          f_disp=False,
                          n_procs=10,
                          use_ccsd=False,
                          ccsd_energy=None):
    """
    Calculate the Gibbs free energy, enthalpy, and entropy of a molecule.

    This function performs a quantum chemistry calculation using the ORCA package to compute
    the Gibbs free energy, enthalpy, and entropy of a molecule represented by an ASE `Atoms` object.
    It supports temperature and pressure adjustments, CCSD energy calculations, and various ORCA options.

    Parameters:
    -----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecule.
    charge : int, optional
        Total charge of the molecule. Default is 0.
    multiplicity : int, optional
        Spin multiplicity of the molecule. Default is 1.
    temperature : float, optional
        Temperature in Kelvin for the calculation. Default is None.
    pressure : float, optional
        Pressure in atm for the calculation. Default is None.
    orca_path : str, optional
        Path to the ORCA executable. If None, it will attempt to read from the environment variable 'ORCA_PATH'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'r2SCAN-3c'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-QZVP'.
    tight_opt : bool, optional
        Whether to use tight geometry optimization. Default is False.
    tight_scf : bool, optional
        Whether to use tight SCF convergence criteria. Default is False.
    f_solv : bool, optional
        Whether to include solvent effects in the calculation. Default is False.
    f_disp : bool, optional
        Whether to include dispersion corrections in the calculation. Default is False.
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 10.
    use_ccsd : bool, optional
        Whether to use CCSD energy calculations. Default is False.
    ccsd_energy : float, optional
        Precomputed CCSD energy in eV. If None, CCSD energy will be calculated if `use_ccsd` is True.

    Returns:
    --------
    tuple
        A tuple containing:
        - energy : float
            The Gibbs free energy in eV.
        - enthalpy : float
            The enthalpy in eV.
        - entropy : float
            The entropy correction in eV.

    Raises:
    -------
    ValueError
        If the CCSD energy calculation fails or the ORCA setup is invalid.
    """
    # Determine the ORCA path
    orca_path = os.path.abspath(orca_path or os.getenv('ORCA_PATH', 'orca'))

    # Set optimization flags
    opt_flag = 'TIGHTOPT' if tight_opt else 'OPT'
    if len(atoms) == 1:  # Skip optimization for single atoms
        opt_flag = ''

    # Set SCF flags
    scf_flag = 'TIGHTSCF' if tight_scf else ''
    calc_extra = f'{opt_flag} {scf_flag} FREQ'.strip()

    # Set up the %thermo block for this temperature and pressure
    if temperature is not None and pressure is None:
        blocks_extra = f'''
                                  %freq
                                      Temp {temperature}
                                  end
                                  '''
    elif pressure is not None and temperature is None:
        blocks_extra = f'''
                                          %freq
                                              Pressure {pressure}
                                          end
                                          '''
    elif pressure is None and temperature is not None:
        blocks_extra = f'''
                                          %freq
                                              Temp {temperature}
                                              Pressure {pressure}
                                          end
                                          '''
    else:
        blocks_extra = None

    # Perform CCSD energy calculation if required and not provided
    if use_ccsd and ccsd_energy is None:
        ccsd_energy = calculate_ccsd_energy(atoms,
                                            orca_path=orca_path,
                                            charge=charge,
                                            multiplicity=multiplicity,
                                            n_procs=n_procs)
        if ccsd_energy is None:
            raise ValueError("CCSD energy calculation failed. Please check the ORCA setup.")

    # Create a temporary directory for the calculation
    with tempfile.TemporaryDirectory() as temp_dir:
        orca_file = os.path.join(temp_dir, 'orca.out')

        # Set up the ORCA calculator
        calc = orca_calc_preset(orca_path=orca_path,
                                directory=temp_dir,
                                charge=charge,
                                multiplicity=multiplicity,
                                xc=xc,
                                basis_set=basis_set,
                                n_procs=n_procs,
                                f_solv=f_solv,
                                f_disp=f_disp,
                                calc_extra=calc_extra,
                                blocks_extra=blocks_extra)
        atoms.calc = calc

        # Trigger the calculation
        _ = atoms.get_potential_energy()

        # Extract entropy correction
        entropy = grab_value(orca_file, 'Total entropy correction', '...')

        # Calculate Gibbs free energy based on CCSD or DFT results
        if use_ccsd:
            g_e_ele = grab_value(orca_file, 'G-E(el)', '...')
            g_e_solv = grab_value(orca_file, 'Free-energy (cav+disp)', ':') if f_solv else 0.0
            energy = ccsd_energy + g_e_ele + g_e_solv
        else:
            energy = grab_value(orca_file, 'Final Gibbs free energy', '...')

        # Return energy, enthalpy, and entropy
        return energy, energy - entropy, entropy


def calculate_hessian(atoms,
                      charge=0,
                      multiplicity=1,
                      orca_path=None,
                      xc='r2SCAN-3c',
                      basis_set='def2-QZVP',
                      tight_opt=False,
                      tight_scf=False,
                      f_solv=False,
                      f_disp=False,
                      n_procs=10):
    """
    Perform a Hessian matrix calculation using the ORCA quantum chemistry package.

    This function sets up and executes a Hessian matrix calculation for a molecule
    represented by an ASE `Atoms` object. It optimizes the geometry and computes
    the Hessian matrix, which is used for vibrational analysis.

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
    xc : str, optional
        Exchange-correlation functional to use. Default is 'r2SCAN-3c'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-QZVP'.
    tight_opt : bool, optional
        Whether to use tight geometry optimization. Default is False.
    tight_scf : bool, optional
        Whether to use tight SCF convergence criteria. Default is False.
    f_solv : bool, optional
        Whether to include solvent effects in the calculation. Default is False.
    f_disp : bool, optional
        Whether to include dispersion corrections in the calculation. Default is False.
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 10.

    Returns:
    --------
    tuple
        A tuple containing:
        - atoms : ase.Atoms
            The optimized geometry of the molecule.
        - hessian_file : str
            Path to the file containing the Hessian matrix.

    Raises:
    -------
    ValueError
        If the ORCA path cannot be determined or the calculation fails.
    """
    # Determine the ORCA path
    if orca_path is None:
        # Try to read the path from the environment variable
        orca_path = os.environ.get('ORCA_PATH')
    else:
        # Convert the provided path to an absolute path
        orca_path = os.path.abspath(orca_path)

    if tight_opt:
        # Set up geometry optimization and frequency calculation parameters
        opt_option = 'TIGHTOPT'
    else:
        # Set up frequency calculation parameters only
        opt_option = 'OPT'

    if tight_scf:
        # Set up tight SCF convergence parameters
        calc_extra = f'{opt_option} TIGHTSCF FREQ'
    else:
        # Use default SCF convergence parameters
        calc_extra = f'{opt_option} FREQ'

    # Create a temporary directory for the ORCA calculation
    with tempfile.TemporaryDirectory() as temp_dir:

        # Set up the ORCA calculator with the specified parameters
        calc = orca_calc_preset(orca_path=orca_path,
                                directory=temp_dir,
                                charge=charge,
                                multiplicity=multiplicity,
                                xc=xc,
                                basis_set=basis_set,
                                n_procs=n_procs,
                                f_solv=f_solv,
                                f_disp=f_disp,
                                calc_extra=calc_extra)

        # Attach the ORCA calculator to the ASE Atoms object
        atoms.calc = calc

        # Perform the energy calculation
        _ = atoms.get_potential_energy()

        # Load the optimized geometry from the ORCA output file
        atoms_file = os.path.join(temp_dir, "orca.xyz")
        hessian_file = os.path.join(temp_dir, "orca.hess")
        return read(atoms_file, format="xyz"), hessian_file
