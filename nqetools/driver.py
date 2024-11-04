import xml.etree.ElementTree as ET


def write_mace_driver(
        out_file="run-ase.py",
        in_file="init.xyz",
        host="driver",
        model="small",
        device="cpu",
        default_dtype="float32"):
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
    # Sorting out the driver for the simulation
    if f_driver == "cbe":
        driver = "i-pi-driver -m ch4hcbe -u"
    elif "ase-mace" in f_driver:
        driver = "python3 run-ase.py"
    else:
        driver = f_driver
    return driver


