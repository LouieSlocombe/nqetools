def write_mace_driver(
        out_file="run-ase.py",
        in_file="init.xyz",
        host="driver",
        model="small",
        device="cpu",
        default_dtype="float32"):
    """
    Writes a Python script to run an ASE calculation with MACE and a socket client.

    Parameters:
    out_file (str, optional): The name of the output Python script file. Default is "run-ase.py".
    in_file (str, optional): The name of the input XYZ file. Default is "init.xyz".
    host (str, optional): The host for the socket client. Default is "driver".
    model (str, optional): The MACE model to use. Default is "small".
    device (str, optional): The device to run the calculation on. Default is "cpu".
    default_dtype (str, optional): The default data type for the calculation. Default is "float32".

    Returns:
    None
    """
    in_str = f"""
from ase.io import read
from mace.calculators import mace_off
from ase.calculators.socketio import SocketClient
atoms = read('{in_file}', 0)
atoms.calc = mace_off(model='{model}', device='{device}', default_dtype='{default_dtype}')
client = SocketClient(unixsocket='{host}')
client.run(atoms, use_stress=True)
    """
    # Write the file
    with open(out_file, "w") as f:
        f.write(in_str)
    return None


def prep_driver(f_driver):
    """
    Determines the appropriate driver command for the simulation based on the input.

    Parameters:
    f_driver (str): The input driver identifier.

    Returns:
    str: The command to run the driver.
    """
    # Sorting out the driver for the simulation
    if f_driver == "cbe":
        driver = "i-pi-driver -m ch4hcbe -u"
    elif "ase-mace" in f_driver:
        driver = "python3 run-ase.py"
    else:
        driver = f_driver
    return driver
