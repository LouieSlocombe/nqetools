"""ASE calculator presets and quantum chemistry driver routines.

Wraps NWChem and Q-Chem behind preset factory functions, so a level of
theory can be selected by name rather than assembled by hand each time.

Also provides the calculations that run on top of those calculators:
vibrational spectra, Hessians, free energies and coupled-cluster energies.

The ORCA calculator itself is not built here. It lives in
:mod:`reactiontools.tools_orca`, together with the graded ``orca_preset_*``
levels of theory, and is imported below - one copy of that code rather than
two drifting apart.
"""

import os
import re
import tempfile

import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.nwchem import NWChem
from ase.io import read
from ase.units import Hartree
from reactiontools import orca_calc_preset

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
    """Create and configure an NWChem calculator preset for quantum chemistry calculations.

    This function sets up the input parameters for an NWChem calculation, including
    the directory, charge, exchange-correlation functional, basis set, dispersion
    corrections, solvent effects, and other options.

    Parameters
    ----------
    directory : str, optional
        Directory where the calculation will be performed. Defaults to a temporary directory.
    task : str, optional
        The specific task to perform (e.g., 'energy', 'optimize'). Default is None.
    charge : int, optional
        Total charge of the system. Default is 0.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'B3LYP'.
    multiplicity : int, optional
        Spin multiplicity of the system. Default is 1.
    basis_set : str, optional
        Basis set to use for the calculation. Default is '6-311++G**'.
    disp : str, optional
        Dispersion correction method to use ('XDM' or 'D3'). Default is None.
    solv : str, optional
        Solvent model to use ('WATER' or 'PROTEIN'). Default is None.
    host : str, optional
        Path to a Unix socket for distributed calculations. Default is None.

    Returns
    -------
    NWChem
        Configured NWChem calculator object.
    """
    if directory is None:
        directory = os.path.join(tempfile.mkdtemp(), 'nwchem')

    tmp = {
        'label': directory,
        'charge': charge,
        'basis': basis_set,
        'dft': {
            'maxiter': 2000,
            'iterations': 1000,
            'grid': 'fine nodisk',
            'print': 'medium',
            'direct': ' ',
            'noio': ' ',
            'xc': xc.upper(),
            'mult': multiplicity
        }
    }
    if host is not None:
        tmp['driver'] = {'socket': {'unix': host}}

    if disp:
        if disp.upper() == 'XDM':
            tmp['dft']['xdm '] = 'a1 0.6224 a2 1.7068'
        elif disp.upper() == 'D3':
            tmp['dft']['disp'] = 'vdw 3'

    if solv:
        if solv.upper() == 'WATER':
            tmp['cosmo'] = {'do_cosmo_smd': True, 'solvent': 'water'}
        elif solv.upper() == 'PROTEIN':
            tmp['cosmo'] = {'do_cosmo_smd': True, 'dielec': 8.0}

    if task:
        tmp['task'] = task

    return NWChem(**tmp)


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
                      solv_extra=None):
    """Create and configure a Q-Chem calculator preset for quantum chemistry calculations.

    This function sets up the input parameters for a Q-Chem calculation, including
    charge, multiplicity, exchange-correlation functional, basis set, solvent effects,
    dispersion corrections, and other advanced options.

    Parameters
    ----------
    charge : int, optional
        Total charge of the system. Default is 0.
    multiplicity : int, optional
        Spin multiplicity of the system. Default is 1.
    xc : str, optional
        Exchange-correlation functional to use. Default is "BLYP".
    basis : str, optional
        Basis set to use for the calculation. Default is "6-311G**".
    f_fast : bool, optional
        Whether to enable fast exchange-correlation calculations. Default is False.
    f_solv : bool, optional
        Whether to include solvent effects in the calculation. Default is False.
    f_disp : bool, optional
        Whether to include dispersion corrections in the calculation. Default is False.
    f_neo : bool, optional
        Whether to enable NEO (Nuclear-Electronic Orbital) calculations. Default is False.
    neo_idx : list, optional
        List of indices for NEO calculations. Default is None.
    neo_epc : str, optional
        EPC (Electron-Proton Correlation) method for NEO calculations. Default is "epc19".
    neo_preset : str, optional
        Preset for NEO calculations. Default is "PB4-D".
    neo_isotope : str, optional
        Isotope for NEO calculations. Default is "1".
    scf_algorithm : str, optional
        SCF algorithm to use. Default is "DIIS".
    solv_extra : str, optional
        Additional solvent options for the calculation. Default is None.

    Returns
    -------
    QChem
        Configured Q-Chem calculator object.
    """
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
    if solv_extra is not None and f_solv is True:
        return QChem(solv_extra=solv_extra, **inpt_dict)
    else:
        return QChem(**inpt_dict)


def load_ir_data(filename):
    """Load IR spectrum data from ORCA output file into a pandas DataFrame.

    This function reads an ORCA output file containing IR frequency data,
    extracts the relevant information, and returns it as a pandas DataFrame.

    Parameters
    ----------
    filename : str
        Path to the IR frequency output file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the following columns:
        - 'Mode': Mode number (int).
        - 'Frequency (cm^-1)': Frequency in inverse centimeters (float).
        - 'Epsilon': Epsilon value (float).
        - 'Intensity (km/mol)': Intensity in km/mol (float).

    Raises
    ------
    ValueError
        If the IR spectrum data cannot be found in the file.
    """
    with open(filename) as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if 'Mode   freq       eps      Int' in line:
            start_idx = i + 2  # Skip header and separator line
            break

    if start_idx is None:
        raise ValueError("Could not find IR spectrum data in the file")

    data = []
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('*') or line.startswith('The first'):
            break

        match = re.match(r'\s*(\d+):\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
        if match:
            mode = int(match.group(1))  # Mode number
            freq = float(match.group(2))  # Frequency in cm^-1
            eps = float(match.group(3))  # Epsilon value
            intensity = float(match.group(4))  # Intensity in km/mol
            data.append([mode, freq, eps, intensity])

    df = pd.DataFrame(data, columns=['Mode', 'Frequency (cm^-1)', 'Epsilon', 'Intensity (km/mol)'])

    return df


def load_raman_data(filename):
    """Load Raman spectrum data from ORCA output file into a pandas DataFrame.

    This function reads an ORCA output file containing Raman frequency data,
    extracts the relevant information, and returns it as a pandas DataFrame.

    Parameters
    ----------
    filename : str
        Path to the Raman frequency output file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the following columns:
        - 'Mode': Mode number (int).
        - 'Frequency (cm^-1)': Frequency in inverse centimeters (float).
        - 'Intensity (km/mol)': Intensity in km/mol (float).
        - 'Depolarization': Depolarization value (float).

    Raises
    ------
    ValueError
        If the Raman spectrum data cannot be found in the file.
    """
    with open(filename) as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if 'Mode    freq (cm**-1)   Activity   Depolarization' in line:
            start_idx = i + 2  # Skip header and separator line
            break

    if start_idx is None:
        raise ValueError("Could not find Raman spectrum data in the file")

    data = []
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('The first'):
            break

        match = re.match(r'\s*(\d+):\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
        if match:
            mode = int(match.group(1))  # Mode number
            freq = float(match.group(2))  # Frequency in cm^-1
            activity = float(match.group(3))  # Intensity in km/mol
            depolarization = float(match.group(4))  # Depolarization value
            data.append([mode, freq, activity, depolarization])

    df = pd.DataFrame(data, columns=['Mode', 'Frequency (cm^-1)', 'Intensity (km/mol)', 'Depolarization'])

    return df


def load_vib_data(filename):
    """Load vibrational spectrum data from a file into a pandas DataFrame.

    Parameters
    ----------
    filename : str
        Path to the file containing vibrational spectrum data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the following columns:
        - 'Mode': Mode number (int).
        - 'Frequency (cm^-1)': Frequency in inverse centimeters (float).
        - 'Epsilon': Default value set to 1.0 (float).
        - 'Intensity (km/mol)': Default value set to 1.0 (float).

    Raises
    ------
    ValueError
        If the vibrational spectrum data cannot be found in the file.
    """
    with open(filename) as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if 'Mode   freq       eps      Int' in line:
            start_idx = i + 2  # Skip header and separator line
            break

    if start_idx is None:
        raise ValueError("Could not find IR spectrum data in the file")

    data = []
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('*') or line.startswith('The first'):
            break

        match = re.match(r'\s*(\d+):\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
        if match:
            mode = int(match.group(1))  # Mode number
            freq = float(match.group(2))  # Frequency in cm^-1
            eps = 1.0  # Default epsilon value
            intensity = 1.0  # Default intensity value
            data.append([mode, freq, eps, intensity])

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
    """Calculate vibrational spectrum data using the ORCA quantum chemistry package.

    This function sets up and performs a vibrational spectrum calculation for a molecule
    represented by an ASE `Atoms` object. It computes IR, Raman, and vibrational spectrum data.

    Parameters
    ----------
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

    Returns
    -------
    tuple
        A tuple containing three pandas DataFrames:
        - data_ir : pd.DataFrame
            IR spectrum data.
        - data_raman : pd.DataFrame
            Raman spectrum data.
        - data_vib : pd.DataFrame
            Vibrational spectrum data.

    Raises
    ------
    ValueError
        If the ORCA path cannot be determined or the calculation fails.
    """
    if orca_path is None:
        orca_path = os.environ.get('ORCA_PATH')
    else:
        orca_path = os.path.abspath(orca_path)

    if tight_opt:
        opt_option = 'TIGHTOPT'
    else:
        opt_option = 'OPT'

    if tight_scf:
        calc_extra = f'{opt_option} TIGHTSCF FREQ'
    else:
        calc_extra = f'{opt_option} FREQ'

    blocks_extra = '''
                          %ELPROP
                              POLAR 1
                          END'''

    with tempfile.TemporaryDirectory() as temp_dir:
        orca_file = os.path.join(temp_dir, 'orca.out')

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

        _ = atoms.get_potential_energy()

        data_ir = load_ir_data(orca_file)

        data_raman = load_raman_data(orca_file)

        data_vib = load_vib_data(orca_file)

        return data_ir, data_raman, data_vib


def get_total_electrons(atoms: Atoms) -> int:
    """Calculate the total number of electrons in a molecule.

    This function computes the total number of electrons in a molecule
    represented by an ASE `Atoms` object. It sums the atomic numbers (Z)
    of all atoms in the molecule and adjusts for the explicit charge
    provided in the `Atoms.info` dictionary.

    Parameters
    ----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecule.

    Returns
    -------
    int
        The total number of electrons in the molecule, corrected for its charge.
    """
    n_electrons = int(np.sum(atoms.get_atomic_numbers()))

    charge = atoms.info.get('charge', 0.0)
    n_electrons -= round(charge)

    return n_electrons


def round_to_nearest_two(number):
    """Round a number to the nearest multiple of 2.
    If the result would be 0, return 1 instead.

    Parameters
    ----------
    number : float or int
        The number to be rounded

    Returns
    -------
    int
        The nearest multiple of 2, or 1 if result would be 0
    """
    result = round(number / 2) * 2

    if result == 0:
        result = 1

    return result


def calculate_ccsd_energy(atoms,
                          charge=0,
                          multiplicity=1,
                          orca_path=None,
                          basis_set='def2-TZVPP',
                          n_procs=10):
    """Perform a CCSD (Coupled Cluster Single and Double) energy calculation using the ORCA quantum chemistry package.

    This function sets up and executes a CCSD energy calculation for a molecule represented by an ASE `Atoms` object.
    It ensures that the number of processors used does not exceed the total number of electrons in the system.

    Parameters
    ----------
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

    Returns
    -------
    float
        The CCSD energy of the molecule in eV.

    Raises
    ------
    ValueError
        If the number of processors exceeds the adjusted limit based on the total number of electrons.
    """
    orca_path = os.path.abspath(orca_path or os.getenv('ORCA_PATH', 'orca'))

    total_electrons = get_total_electrons(atoms)
    # Prevent too many processors being used
    if n_procs > total_electrons:
        n_procs = round_to_nearest_two(total_electrons - 2)

    with tempfile.TemporaryDirectory() as temp_dir:
        calc = orca_calc_preset(orca_path=orca_path,
                                directory=temp_dir,
                                calc_type='CCSD',
                                charge=charge,
                                multiplicity=multiplicity,
                                basis_set=basis_set,
                                n_procs=n_procs)
        atoms.calc = calc

        return atoms.get_potential_energy()


def grab_value(orca_file, term, splitter):
    """Extract a specific numerical value from an ORCA output file.

    This function reads an ORCA output file in reverse order, searches for a specific term,
    and extracts the numerical value associated with it. The value is converted from Hartree
    units to eV using the ASE `Hartree` constant.

    Parameters
    ----------
    orca_file : str
        Path to the ORCA output file.
    term : str
        The term to search for in the file.
    splitter : str
        The delimiter used to split the line containing the term.

    Returns
    -------
    float or None
        The extracted value in eV, or None if the term is not found.
    """
    with open(orca_file) as f:
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
                          opt=False,
                          tight_opt=False,
                          tight_scf=False,
                          f_solv=False,
                          f_disp=False,
                          n_procs=10,
                          use_ccsd=False,
                          ccsd_energy=None):
    """Calculate the Gibbs free energy of a molecule using the ORCA quantum chemistry package.

    This function sets up and performs a vibrational frequency calculation for a molecule
    represented by an ASE `Atoms` object. It supports various options, including geometry
    optimization, solvent effects, dispersion corrections, and CCSD energy calculations.

    Parameters
    ----------
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
    opt : bool, optional
        Whether to perform geometry optimization. Default is False.
    tight_opt : bool, optional
        Whether to use tight geometry optimization criteria. Default is False.
    tight_scf : bool, optional
        Whether to use tight SCF convergence criteria. Default is False.
    f_solv : bool, optional
        Whether to include solvent effects in the calculation. Default is False.
    f_disp : bool, optional
        Whether to include dispersion corrections in the calculation. Default is False.
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 10.
    use_ccsd : bool, optional
        Whether to use CCSD energy for the calculation. Default is False.
    ccsd_energy : float, optional
        Precomputed CCSD energy in eV. If None, it will be calculated if `use_ccsd` is True. Default is None.

    Returns
    -------
    tuple
        A tuple containing:
        - energy : float
            The Gibbs free energy in eV.
        - enthalpy : float
            The enthalpy in eV.
        - entropy : float
            The entropy in eV.

    Raises
    ------
    ValueError
        If the CCSD energy calculation fails or the ORCA setup is incorrect.
    """
    orca_path = os.path.abspath(orca_path or os.getenv('ORCA_PATH', 'orca'))

    if opt:
        opt_flag = 'TIGHTOPT' if tight_opt else 'OPT'
        if len(atoms) == 1:  # Skip optimization for single atoms
            opt_flag = ''
    else:
        opt_flag = ''

    scf_flag = 'TIGHTSCF' if tight_scf else ''
    calc_extra = f'{opt_flag} {scf_flag} FREQ'.strip()

    # Build the %freq block from whichever of temperature/pressure were supplied,
    # so that setting both is honoured rather than silently discarded
    freq_settings = []
    if temperature is not None:
        freq_settings.append(f'Temp {temperature}')
    if pressure is not None:
        freq_settings.append(f'Pressure {pressure}')

    if freq_settings:
        freq_lines = '\n'.join(f'    {setting}' for setting in freq_settings)
        blocks_extra = f'\n%freq\n{freq_lines}\nend\n'
    else:
        blocks_extra = None

    if use_ccsd and ccsd_energy is None:
        ccsd_energy = calculate_ccsd_energy(atoms,
                                            orca_path=orca_path,
                                            charge=charge,
                                            multiplicity=multiplicity,
                                            n_procs=n_procs)
        if ccsd_energy is None:
            raise ValueError("CCSD energy calculation failed. Please check the ORCA setup.")

    with tempfile.TemporaryDirectory() as temp_dir:
        orca_file = os.path.join(temp_dir, 'orca.out')

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

        _ = atoms.get_potential_energy()

        entropy = grab_value(orca_file, 'Total entropy correction', '...')

        if use_ccsd:
            g_e_ele = grab_value(orca_file, 'G-E(el)', '...')
            g_e_solv = grab_value(orca_file, 'Free-energy (cav+disp)', ':') if f_solv else 0.0
            energy = ccsd_energy + g_e_ele + g_e_solv
        else:
            energy = grab_value(orca_file, 'Final Gibbs free energy', '...')

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
    """Perform a Hessian matrix calculation using the ORCA quantum chemistry package.

    This function sets up and executes a Hessian matrix calculation for a molecule
    represented by an ASE `Atoms` object. It optimizes the geometry and computes
    the Hessian matrix, which is used for vibrational analysis.

    Parameters
    ----------
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

    Returns
    -------
    tuple
        A tuple containing:
        - atoms : ase.Atoms
            The optimized geometry of the molecule.
        - hessian_file : str
            Path to the file containing the Hessian matrix.

    Raises
    ------
    ValueError
        If the ORCA path cannot be determined or the calculation fails.
    """
    if orca_path is None:
        orca_path = os.environ.get('ORCA_PATH')
    else:
        orca_path = os.path.abspath(orca_path)

    if tight_opt:
        opt_option = 'TIGHTOPT'
    else:
        opt_option = 'OPT'

    if tight_scf:
        calc_extra = f'{opt_option} TIGHTSCF FREQ'
    else:
        calc_extra = f'{opt_option} FREQ'

    with tempfile.TemporaryDirectory() as temp_dir:

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

        atoms.calc = calc

        _ = atoms.get_potential_energy()

        atoms_file = os.path.join(temp_dir, "orca.xyz")
        hessian_file = os.path.join(temp_dir, "orca.hess")
        return read(atoms_file, format="xyz"), hessian_file
