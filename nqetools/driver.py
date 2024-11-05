def write_mace_driver(
        model_type="off",
        out_file="run-ase.py",
        in_file="init.xyz",
        host="driver",
        model="small",
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
        write_mace_driver(**driver_dict)
        return "python3 run-ase.py"
    else:
        # If not a recognized driver, raise an error
        raise ValueError(f"Driver {f_driver} is not recognized.")
