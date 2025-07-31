import os

import ipi

from .calculators import (nwchem_calc_preset)
from .tools import get_ipi_driver


def write_ase_mace_driver(
        directory,
        out_file="run-ase-mace.py",
        in_file="init.xyz",
        host="driver",
        model="small",
        model_type="off",
        device="cpu",
        default_dtype="float64"):
    assert model_type in ["off", "mp", "anicc"]
    in_str = f"""
from ase.io import read
from mace.calculators import mace_{model_type}
from ase.calculators.socketio import SocketClient
atoms = read('{in_file}', 0)
atoms.calc = mace_{model_type}(model='{model}', device='{device}', default_dtype='{default_dtype}')
client = SocketClient(unixsocket='{host}')
client.run(atoms, use_stress=True)
    """
    # Write the file
    with open(os.path.join(directory, out_file), "w") as f:
        f.write(in_str)
    return None


def write_cp2k_driver():
    # which cp2k.ssmp
    # /home/louie/anaconda3/envs/ipi_env/bin/cp2k.ssmp
    # https://github.com/i-pi/i-pi/tree/main/examples/clients/cp2k/npt_classical

    raise ValueError(f"Driver cp2k is not recognized.")


def write_ase_nwchem_driver(
        directory,
        in_file="init.xyz",
        out_file="run-ase-nwchem.py",
        charge=0,
        xc="B3LYP",
        multi=1,
        basis_set="6-31G**",
        disp=None,
        solv=None,
        host="driver"):
    in_str = f"""
import os
import tempfile

from ase.calculators.nwchem import NWChem
from ase.io import read
from ase.calculators.socketio import SocketClient

def nwchem_calc_preset(directory=None,
                       task=None,
                       charge=0,
                       xc='B3LYP',
                       multiplicity=1,
                       basis_set='6-311++G**',
                       disp=None,
                       solv=None):
    if directory is None:
        #directory = os.path.join(tempfile.mkdtemp(), 'nwchem')
        directory = os.path.join(os.getcwd(), 'nwchem')

    tmp = dict(
        label=directory,
        charge=charge,
        basis=basis_set,
        driver=dict(socket=dict(unix='{host}')),
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
    
atoms = read('{in_file}', 0)
atoms.calc = nwchem_calc_preset(charge={charge}, 
                         xc='{xc}',
                         task='optimize', 
                         multiplicity={multi},
                         basis_set='{basis_set}',
                         disp={disp},
                         solv={solv})
client = SocketClient(unixsocket='{host}')
client.run(atoms, use_stress=False)

    """
    # Write the file
    with open(os.path.join(directory, out_file), "w") as f:
        f.write(in_str)
    return None


def write_nwchem_driver(atoms,
                        directory,
                        task='optimize',
                        charge=0,
                        xc='B3LYP',
                        multiplicity=1,
                        basis_set='6-311++G**',
                        disp=None,
                        solv=None,
                        host='driver'):
    """
    Prepares and writes the input file for an NWChem calculation.

    Parameters:
    atoms (object): An ASE Atoms object representing the atomic structure.
    directory (str): The directory to write the NWChem input file to.
    task (str, optional): The NWChem task to perform
    charge (int, optional): The charge of the system. Default is 0.
    xc (str, optional): The exchange-correlation functional to use. Default is 'B3LYP'.
    multiplicity (int, optional): The spin multiplicity of the system. Default is 1.
    basis_set (str, optional): The basis set to use. Default is '6-311++G**'.
    disp (str, optional): The dispersion correction to use. Default is None.
    solv (str, optional): The solvation model to use. Default is None.
    host (str, optional): The host for the SocketClient. Default is 'driver'.

    Returns:
    None
    """
    # Prepare the NWChem calculator
    calc = nwchem_calc_preset(directory=directory,
                              task=task,
                              charge=charge,
                              xc=xc,
                              multiplicity=multiplicity,
                              basis_set=basis_set,
                              disp=disp,
                              solv=solv,
                              host=host)
    # Set the correct pathing
    calc.prefix = os.path.join(directory, 'nwchem')
    # Write the input file
    calc.write_input(atoms)
    return None


def write_ase_orca_driver(
        directory,
        in_file="init.xyz",
        out_file="run-ase-orca.py",
        charge=0,
        xc="BLYP",
        multi=1,
        basis_set="6-311G",
        disp=False,
        solv=False,
        calc_type='DFT',
        atom_list=None,
        calc_extra=None,
        scf_option=None):
    in_str_1 = """
import os
import tempfile

from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile
from ase.io import read
from ase.calculators.socketio import SocketClient

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
        atom_list = '{'+atom_list+'}'
        inpt_xtb = f'''
        %QMMM QMATOMS {atom_list} END END
                   '''
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

    """
    in_str_2 = f"""
    
atoms = read('{in_file}', 0)
atoms.set_pbc(False)
atoms.calc = orca_calc_preset(calc_type='{calc_type}',
                       xc='{xc}',
                       charge={charge},
                       multiplicity={multi},
                       basis_set='{basis_set}',
                       f_disp={disp},
                       f_solv={solv},
                       atom_list={atom_list},
                       calc_extra={calc_extra},
                       scf_option={scf_option})
# Create Client
port = 10200
host = "localhost"
client = SocketClient(host=host, port=port)
client.run(atoms)

"""
    # join the strings
    in_str = in_str_1 + in_str_2

    # Write the file
    with open(os.path.join(directory, out_file), "w") as f:
        f.write(in_str)


def move_zundel_driver_pes_files(directory):
    """
    Copies the Zundel driver PES (Potential Energy Surface) files to the specified directory.

    Parameters:
    directory (str): The target directory where the PES files will be copied.

    Returns:
    None
    """
    # Determine the base directory where the PES files are located
    base = os.path.join(ipi.__file__.split('__init__.py')[0], 'drivers', 'f90', 'pes')
    # List of PES files to be copied
    files = ['h5o2.dms4B.coeff.com.dat', 'h5o2.pes4B.coeff.dat']
    # Copy each file from the base directory to the target directory
    for file in files:
        os.system(f"cp {os.path.join(base, file)} {directory}")


def prep_driver(atoms, directory, f_driver, driver_args):
    """
    Prepares the driver command based on the specified driver type and parameters.

    Parameters:
    atoms (object): An ASE Atoms object representing the atomic structure.
    directory (str): The directory where driver files will be written.
    f_driver (str): The type of driver to prepare. Must be one of ["cbe", "zundel", "ase-mace", "ase-nwchem", "ase-orca", "nwchem"].
    driver_dict (dict): A dictionary of additional parameters specific to the driver type.

    Returns:
    str: The command to run the prepared driver.
    """
    driver_path = get_ipi_driver()
    if f_driver == "cbe":
        return f"{driver_path} -m ch4hcbe -u"
    elif f_driver == "zundel":
        move_zundel_driver_pes_files(directory)
        return f"{driver_path} -u -a zundel -m zundel"
    elif f_driver == "ase-mace":
        # If the driver is an ASE-MACE driver, write the driver file
        write_ase_mace_driver(directory, **driver_args)
        return "python3 run-ase-mace.py"
    elif f_driver == "ase-nwchem":
        # If the driver is an ASE-NWChem driver, write the driver file
        write_ase_nwchem_driver(directory, **driver_args)
        return "python3 run-ase-nwchem.py"
    elif f_driver == "ase-orca":
        # If the driver is an ASE-ORCA driver, write the driver file
        write_ase_orca_driver(directory, **driver_args)
        return "python3 run-ase-orca.py"
    elif f_driver == "nwchem":
        write_nwchem_driver(atoms, directory, **driver_args)
        return "nwchem nwchem.nwi > nwchem.out"
    else:
        # If not a recognised driver, raise an error
        raise ValueError(f"Driver {f_driver} is not recognized.")
