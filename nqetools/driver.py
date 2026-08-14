"""Generation of i-PI client drivers.

i-PI splits a calculation between a server, which propagates the nuclei,
and a client, which returns forces over a socket. This module writes the
client side: either a standalone Python script that builds an ASE
calculator and connects back, or an input deck for a code that speaks the
i-PI protocol natively.

:func:`prep_driver` is the entry point, dispatching on a driver name and
returning the shell command that launches the client.
"""

import ipi
import os
import shutil
import torch

from .calculators import nwchem_calc_preset
from .tools import get_ipi_driver


def write_ase_mace_driver(
    directory,
    out_file="run-ase-mace.py",
    in_file="init.xyz",
    host="driver",
    model="small",
    model_type="off",
    device=None,
    default_dtype="float64",
    enable_cueq=False,
):
    """Write an i-PI client script driven by a MACE machine-learning potential.

    Parameters
    ----------
    directory : str
        The directory to write the client script to.
    out_file : str, optional
        Name of the generated script. Default is "run-ase-mace.py".
    in_file : str, optional
        Structure file the client reads to build its Atoms object. Default
        is "init.xyz".
    host : str, optional
        Unix socket name used to reach the i-PI server. Default is "driver".
    model : str, optional
        Model size or path passed to the MACE calculator. Default is "small".
    model_type : {"off", "mp", "anicc", "omol"}, optional
        Which pre-trained MACE family to load. Default is "off".
    device : torch.device or str, optional
        Device to run inference on. Defaults to CUDA when available,
        otherwise CPU.
    default_dtype : str, optional
        Floating-point precision for the model. Default is "float64".
    enable_cueq : bool, optional
        If True, enable cuEquivariance acceleration. Default is False.

    Raises
    ------
    AssertionError
        If `model_type` is not a recognised MACE family.

    Notes
    -----
    The 'anicc' family takes no model or dtype arguments, so it is emitted
    with a reduced call signature.
    """
    assert model_type in ["off", "mp", "anicc", "omol"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "off" or model_type == "mp" or model_type == "omol":
        in_str = f"""
from ase.io import read
from mace.calculators import mace_{model_type}
from ase.calculators.socketio import SocketClient
atoms = read('{in_file}', 0)
atoms.calc = mace_{model_type}(model='{model}', device='{device}', default_dtype='{default_dtype}', enable_cueq={enable_cueq})
client = SocketClient(unixsocket='{host}')
client.run(atoms, use_stress=True)
        """
    else:
        in_str = f"""
from ase.io import read
from mace.calculators import mace_{model_type}
from ase.calculators.socketio import SocketClient
atoms = read('{in_file}', 0)
atoms.calc = mace_{model_type}(device='{device}')
client = SocketClient(unixsocket='{host}')
client.run(atoms, use_stress=True)
            """
    with open(os.path.join(directory, out_file), "w") as f:
        f.write(in_str)


def write_ase_qmmm_mace_driver(
    directory,
    out_file="run-ase-qmmm-mace.py",
    in_file="init.xyz",
    qm_indices=None,
    qm_model_type="omol",
    qm_model="medium",
    mm_model_type="off",
    mm_model="small",
    device=None,
    default_dtype="float64",
    enable_cueq=False,
    host="driver",
):
    """Write an i-PI client script using a MACE-based QM/MM partition.

    Two MACE models are combined through ASE's ``SimpleQMMM``: an accurate
    model for the reacting subsystem and a cheaper one for the surroundings.

    Parameters
    ----------
    directory : str
        The directory to write the client script to.
    out_file : str, optional
        Name of the generated script. Default is "run-ase-qmmm-mace.py".
    in_file : str, optional
        Structure file the client reads to build its Atoms object. Default
        is "init.xyz".
    qm_indices : list of int, optional
        Indices of the atoms treated at the QM level. Default is [0].
    qm_model_type : {"off", "mp", "anicc", "omol"}, optional
        MACE family for the QM region. Default is "omol".
    qm_model : str, optional
        Model size or path for the QM region. Default is "medium".
    mm_model_type : {"off", "mp", "anicc", "omol"}, optional
        MACE family for the MM region. Default is "off".
    mm_model : str, optional
        Model size or path for the MM region. Default is "small".
    device : torch.device or str, optional
        Device to run inference on. Defaults to CUDA when available,
        otherwise CPU.
    default_dtype : str, optional
        Floating-point precision for the models. Default is "float64".
    enable_cueq : bool, optional
        If True, enable cuEquivariance acceleration. Default is False.
    host : str, optional
        Unix socket name used to reach the i-PI server. Default is "driver".

    Raises
    ------
    AssertionError
        If either model type is not a recognised MACE family.

    Notes
    -----
    ``SimpleQMMM`` is passed the MM calculator twice, so the MM model is
    evaluated on both the QM subsystem and the full system in order to
    subtract the double-counted contribution.
    """
    if qm_indices is None:
        qm_indices = [0]

    assert qm_model_type in ["off", "mp", "anicc", "omol"], (
        "QM model type must be one of: off, mp, anicc, omol"
    )
    assert mm_model_type in ["off", "mp", "anicc", "omol"], (
        "MM model type must be one of: off, mp, anicc, omol"
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_str = f"""
from ase.io import read
from mace.calculators import mace_{qm_model_type}, mace_{mm_model_type}
from ase.calculators.qmmm import SimpleQMMM
from ase.calculators.socketio import SocketClient
atoms = read('{in_file}', 0)
"""

    if qm_model_type in ["off", "mp", "omol"]:
        in_str += f"qm_calc = mace_{qm_model_type}(model='{qm_model}', device='{device}', default_dtype='{default_dtype}', enable_cueq={enable_cueq})\n"
    else:  # anicc takes no model/dtype kwargs
        in_str += f"qm_calc = mace_{qm_model_type}(device='{device}')\n"

    in_str += "\n# Set up MM calculator\n"
    if mm_model_type in ["off", "mp", "omol"]:
        in_str += f"mm_calc = mace_{mm_model_type}(model='{mm_model}', device='{device}', default_dtype='{default_dtype}', enable_cueq={enable_cueq})\n"
    else:  # anicc takes no model/dtype kwargs
        in_str += f"mm_calc = mace_{mm_model_type}(device='{device}')\n"

    in_str += f"""
qm_indices = {qm_indices}
qmmm_calc = SimpleQMMM(qm_indices, qm_calc, mm_calc, mm_calc)
atoms.calc = qmmm_calc
client = SocketClient(unixsocket='{host}')
client.run(atoms)
"""

    with open(os.path.join(directory, out_file), "w") as f:
        f.write(in_str)


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
    host="driver",
):
    """Write an i-PI client script driven by an NWChem DFT calculation.

    Parameters
    ----------
    directory : str
        The directory to write the client script to.
    in_file : str, optional
        Structure file the client reads to build its Atoms object. Default
        is "init.xyz".
    out_file : str, optional
        Name of the generated script. Default is "run-ase-nwchem.py".
    charge : int, optional
        Net charge of the system. Default is 0.
    xc : str, optional
        Exchange-correlation functional. Default is "B3LYP".
    multi : int, optional
        Spin multiplicity of the system. Default is 1.
    basis_set : str, optional
        Basis set to use. Default is "6-31G**".
    disp : str, optional
        Dispersion correction, either 'XDM' or 'D3'. Default is None.
    solv : str, optional
        Implicit solvent model, either 'WATER' or 'PROTEIN'. Default is None.
    host : str, optional
        Unix socket name used to reach the i-PI server. Default is "driver".

    Notes
    -----
    The calculator preset is inlined into the generated script rather than
    imported, so the client has no dependency on this package at run time.
    """
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
                         disp={disp!r},
                         solv={solv!r})
client = SocketClient(unixsocket='{host}')
client.run(atoms, use_stress=False)

    """
    with open(os.path.join(directory, out_file), "w") as f:
        f.write(in_str)


def write_nwchem_driver(
    atoms,
    directory,
    task="optimize",
    charge=0,
    xc="B3LYP",
    multiplicity=1,
    basis_set="6-311++G**",
    disp=None,
    solv=None,
    host="driver",
):
    """Prepares and writes the input file for an NWChem calculation.

    Parameters
    ----------
    atoms : object
        An ASE Atoms object representing the atomic structure.
    directory : str
        The directory to write the NWChem input file to.
    task : str, optional
        The NWChem task to perform
    charge : int, optional
        The charge of the system. Default is 0.
    xc : str, optional
        The exchange-correlation functional to use. Default is 'B3LYP'.
    multiplicity : int, optional
        The spin multiplicity of the system. Default is 1.
    basis_set : str, optional
        The basis set to use. Default is '6-311++G**'.
    disp : str, optional
        The dispersion correction to use. Default is None.
    solv : str, optional
        The solvation model to use. Default is None.
    host : str, optional
        The host for the SocketClient. Default is 'driver'.
    """
    calc = nwchem_calc_preset(
        directory=directory,
        task=task,
        charge=charge,
        xc=xc,
        multiplicity=multiplicity,
        basis_set=basis_set,
        disp=disp,
        solv=solv,
        host=host,
    )
    calc.prefix = os.path.join(directory, "nwchem")
    calc.write_input(atoms)


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
    calc_type="DFT",
    atom_list=None,
    calc_extra=None,
    scf_option=None,
    n_procs=10,
):
    """Write an i-PI client script driven by an ORCA calculation.

    Parameters
    ----------
    directory : str
        The directory to write the client script to.
    in_file : str, optional
        Structure file the client reads to build its Atoms object. Default
        is "init.xyz".
    out_file : str, optional
        Name of the generated script. Default is "run-ase-orca.py".
    charge : int, optional
        Net charge of the system. Default is 0.
    xc : str, optional
        Exchange-correlation functional, used by the DFT and QM/XTB2
        calculation types. Default is "BLYP".
    multi : int, optional
        Spin multiplicity. Values above 1 switch the reference to
        unrestricted. Default is 1.
    basis_set : str, optional
        Basis set to use. Default is "6-311G".
    disp : str or bool, optional
        Dispersion correction; True selects 'D4'. Default is False.
    solv : str or bool, optional
        SMD implicit solvent; True selects 'WATER'. Default is False.
    calc_type : {"DFT", "MP2", "CCSD", "QM/XTB2"}, optional
        Level of theory. Default is "DFT".
    atom_list : str, optional
        ORCA-format atom selection for the QM region, used only when
        `calc_type` is 'QM/XTB2'. Default is None.
    calc_extra : str, optional
        Extra keywords appended to the ORCA simple input line. Default is
        None.
    scf_option : str, optional
        Additional SCF convergence keyword. Default is None.
    n_procs : int, optional
        Number of MPI processes for ORCA. Default is 10.

    Notes
    -----
    Unlike the other ASE clients, this one connects over a TCP socket on
    port 10200 rather than a Unix socket.
    """
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
        # `is True`, not truthiness: a solvent name is truthy too, and testing
        # it that way overwrote every name with WATER.
        if f_solv is True:
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
        inpt_disp = 'D4' if f_disp is True else f_disp

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
                       f_disp={disp!r},
                       f_solv={solv!r},
                       atom_list={atom_list!r},
                       calc_extra={calc_extra!r},
                       scf_option={scf_option!r},
                       n_procs={n_procs})
# Create Client
port = 10200
host = "localhost"
client = SocketClient(host=host, port=port)
client.run(atoms)

"""
    in_str = in_str_1 + in_str_2

    with open(os.path.join(directory, out_file), "w") as f:
        f.write(in_str)


def move_zundel_driver_pes_files(directory):
    """Copies the Zundel driver PES (Potential Energy Surface) files to the specified directory.

    Parameters
    ----------
    directory : str
        The target directory where the PES files will be copied.
    """
    base = os.path.join(ipi.__file__.split("__init__.py")[0], "drivers", "f90", "pes")
    files = ["h5o2.dms4B.coeff.com.dat", "h5o2.pes4B.coeff.dat"]
    for file in files:
        shutil.copy(os.path.join(base, file), directory)


def prep_driver(atoms, directory, f_driver, driver_args):
    """Prepares the driver command based on the specified driver type and parameters.

    Parameters
    ----------
    atoms : object
        An ASE Atoms object representing the atomic structure.
    directory : str
        The directory where driver files will be written.
    f_driver : str
        The type of driver to prepare. Must be one of ["cbe", "zundel", "mace", "ase-mace",
        "ase-qmmm-mace", "ase-nwchem", "ase-orca", "nwchem"].
    driver_args : dict
        A dictionary of additional parameters specific to the driver type.

    Returns
    -------
    str
        The command to run the prepared driver.
    """
    driver_path = get_ipi_driver()
    if f_driver == "cbe":
        return f"{driver_path} -m ch4hcbe -u"

    elif f_driver == "zundel":
        move_zundel_driver_pes_files(directory)
        return f"{driver_path} -u -a zundel -m zundel"

    elif f_driver == "mace":
        write_ase_mace_driver(directory, **driver_args)
        f_model = driver_args.get("model", "small")
        return f"i-pi-py_driver -a driver -u -m mace -o init.xyz,{f_model}"

    elif f_driver == "ase-mace":
        write_ase_mace_driver(directory, **driver_args)
        return "python3 run-ase-mace.py"

    elif f_driver == "ase-qmmm-mace":
        write_ase_qmmm_mace_driver(directory, **driver_args)
        return "python3 run-ase-qmmm-mace.py"

    elif f_driver == "ase-nwchem":
        write_ase_nwchem_driver(directory, **driver_args)
        return "python3 run-ase-nwchem.py"

    elif f_driver == "ase-orca":
        write_ase_orca_driver(directory, **driver_args)
        return "python3 run-ase-orca.py"

    elif f_driver == "nwchem":
        write_nwchem_driver(atoms, directory, **driver_args)
        # Driver stdout is captured by run_ipi; a shell redirect here would be
        # passed to nwchem as literal arguments, since the driver is not run
        # through a shell
        return "nwchem nwchem.nwi"
    else:
        raise ValueError(f"Driver {f_driver} is not recognized.")
