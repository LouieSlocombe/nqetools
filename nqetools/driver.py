def write_mace_driver(
        out_file="run-ase-mace.py",
        in_file="init.xyz",
        host="driver",
        model="small",
        model_type="off",
        device="cpu",
        default_dtype="float32"):
    """
    Writes a Python script to run a MACE driver with specified parameters.

    Parameters:
    model_type (str): The type of MACE model to use. Must be one of ["off", "mp", "anicc"].
    out_file (str): The name of the output file to write the script to.
    in_file (str): The input file containing atomic structure data.
    host (str): The host for the SocketClient.
    model (str): The model name to use in the MACE calculator.
    device (str): The device to run the MACE model on (e.g., "cpu").
    default_dtype (str): The default data type for the MACE model.

    Returns:
    None
    """
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
    with open(out_file, "w") as f:
        f.write(in_str)
    return None


def write_cp2k_driver():
    raise ValueError(f"Driver cp2k is not recognized.")


def write_nwchem_driver(
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

def nwchem_calc(charge=0,
                xc='B3LYP',
                multi=1,
                basis_set='6-31G**',
                disp=None,
                solv=None):
    tmp = dict(label=os.path.join(tempfile.mkdtemp(), 'nwchem'), charge=charge, basis=basis_set)

    if xc.upper() == 'CAM-B3LYP':
        tmp['dft'] = dict(
            maxiter=2000,
            iterations=1000,
            grid='fine nodisk',
            direct=' ',
            noio=' ',
            xc='xcamb88 1.00 lyp 0.81 vwn_5 0.19 hfexch 1.00',
            cam='0.33 cam_alpha 0.19 cam_beta 0.46',
            mult=multi
        )
    else:
        tmp['dft'] = dict(
            maxiter=2000,
            iterations=1000,
            grid='fine nodisk',
            direct=' ',
            noio=' ',
            xc=xc.upper(),
            mult=multi
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

    return NWChem(**tmp)
    
atoms = read('{in_file}', 0)
atoms.calc = nwchem_calc(charge={charge}, 
                         xc={xc}, 
                         multi={multi},
                         basis_set={basis_set},
                         disp={disp},
                         solv={solv})
client = SocketClient(unixsocket='{host}')
client.run(atoms, use_stress=True)

    """
    # Write the file
    with open(out_file, "w") as f:
        f.write(in_str)
    return None


def write_qchem_driver():
    raise ValueError(f"Driver qchem is not recognized.")


def write_orca_driver(
        in_file="init.xyz",
        out_file="run-ase-orca.py",
        charge=0,
        xc="B3LYP",
        multi=1,
        basis_set="6-31G**",
        disp=None,
        solv=None,
        calc_type='DFT',
        atom_list=None,
        calc_extra=None,
        scf_option=None,
        host="driver"):
    in_str_1 = """
import os
import tempfile

from ase.calculators.orca import ORCA
from ase.io import read
from ase.calculators.socketio import SocketClient

def orca_calc(calc_type='DFT',
              xc='B3LYP',
              charge=0,
              mult=1,
              basis_set='6-31+G(d,p)',
              nprocs=1,
              solv=True,
              disp=True,
              atom_list=None,
              calc_extra=None,
              scf_option=None):
    inpt_procs = f'%pal nprocs {nprocs} end' if nprocs > 1 else ''
    inpt_solv = f'''
    %CPCM SMD TRUE
        SMDSOLVENT '{solv if solv else "WATER"}'
    END''' if solv else ''
    inpt_disp = disp if disp else ''
    inpt_xtb = f'''
    %QMMM QMATOMS {str(atom_list).strip('[]')} END END
    ''' if atom_list else ''
    inpt_blocks = inpt_procs + inpt_solv + (inpt_xtb if calc_type == 'QM/XTB2' else '')

    if calc_type == 'DFT':
        inpt_simple = f'{xc} {inpt_disp} {basis_set}'
    elif calc_type in ['MP2', 'CCSD']:
        inpt_simple = f'DLPNO-{calc_type} {basis_set} {basis_set}/C'
    elif calc_type == 'QM/XTB2':
        inpt_simple = f'{calc_type} {xc} {inpt_disp} {basis_set}'
    else:
        inpt_simple = f'{calc_type} {basis_set}'

    if scf_option:
        inpt_simple += f' {scf_option}'
    if calc_extra:
        inpt_simple += f' {calc_extra}'

    return ORCA(
        charge=charge,
        mult=mult,
        directory=os.path.join(tempfile.mkdtemp(), 'orca'),
        orcasimpleinput=inpt_simple,
        orcablocks=inpt_blocks
    )

    """
    in_str_2 = f"""
    
atoms = read('{in_file}', 0)
atoms.calc = orca_calc(calc_type='{calc_type}',
                       xc='{xc}',
                       charge={charge},
                       multi={multi},
                       basis_set='{basis_set}',
                       disp={disp},
                       solv={solv},
                       atom_list={atom_list},
                       calc_extra={calc_extra},
                       scf_option={scf_option})
client = SocketClient(unixsocket='{host}')
client.run(atoms, use_stress=True)

"""
    # join the strings
    in_str = in_str_1 + in_str_2

    # Write the file
    with open(out_file, "w") as f:
        f.write(in_str)


def prep_driver(f_driver, driver_dict):
    """
    Prepares the driver command based on the specified driver type.

    Parameters:
    f_driver (str): The type of driver to prepare. Can be "cbe" or contain "ase-mace".
    driver_dict (dict): A dictionary of parameters to pass to the `write_mace_driver` function if the driver type is "ase-mace".

    Returns:
    str: The command to run the specified driver.

    Raises:
    ValueError: If the driver type is not recognized.
    """
    if f_driver == "cbe":
        return "i-pi-driver -m ch4hcbe -u"
    elif "ase-mace" in f_driver:
        # If the driver is an ASE-MACE driver, write the driver file
        write_mace_driver(**driver_dict)
        return "python3 run-ase-mace.py"
    elif "ase-nwchem" in f_driver:
        # If the driver is an ASE-NWChem driver, write the driver file
        write_nwchem_driver(**driver_dict)
        return "python3 run-ase-nwchem.py"
    elif "ase-orca" in f_driver:
        # If the driver is an ASE-ORCA driver, write the driver file
        write_orca_driver(**driver_dict)
        return "python3 run-ase-orca.py"
    else:
        # If not a recognized driver, raise an error
        raise ValueError(f"Driver {f_driver} is not recognized.")
