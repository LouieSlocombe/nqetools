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
    """
    """
    return None


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
from ase.calculators.nwchem import NWChem
from ase.io import read
from ase.calculators.socketio import SocketClient

def get_nwchem_calculator(charge=0,
                          xc='B3LYP',
                          multi=1,
                          basis_set='6-31G**',
                          disp=None,
                          solv=None):
    tmp = dict(label='calc/nwchem', charge=charge, basis=basis_set)

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
atoms.calc = get_nwchem_calculator(charge={charge}, 
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
    return None


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
    else:
        # If not a recognized driver, raise an error
        raise ValueError(f"Driver {f_driver} is not recognized.")
