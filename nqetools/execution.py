import os
import subprocess
import time
import xml.etree.ElementTree as ET

import ase.io

from .driver import prep_driver
from .io import write_xml, copy_xyz, get_final_xyz, copy_hess, get_final_hess, write_xyz
from .tools import rm_ipi_tmp
from .xml_parse import update_cell, update_mass, update_driver, update_title, \
    update_total_steps, update_optimizer, update_tol, update_nbeads, update_hessian, \
    update_open_paths, update_temperature


def run_ipi(server,
            driver,
            outfile,
            n=1,
            t_sleep=5.0,
            cwd=None):
    """
    Runs the i-PI server and driver processes for a given simulation.

    Parameters:
    server (str): Command to start the i-PI server.
    driver (str): Command to start the driver process.
    outfile (str): Output file name to check if the simulation has already been run.
    n (int, optional): Number of driver processes to start. Default is 1.
    t_sleep (float, optional): Time to wait for the i-PI server to start in seconds. Default is 5.0.
    cwd (str, optional): Directory to change to before running the simulation. Default is None.

    Returns:
    None
    """
    # Change to the working directory
    dir_base = os.getcwd()
    if cwd is not None:
        os.chdir(cwd)
    # Remove the tmp file if it exists
    rm_ipi_tmp()
    # Start the i-PI server and the driver processes
    ipi_proc = None
    driver_proc = None
    if not os.path.exists(outfile):
        # Don't rerun if the outputs already exist
        ipi_proc = subprocess.Popen(server.split())
        # Wait for i-PI to start
        time.sleep(t_sleep)
        # Start the driver processes
        driver_proc = [subprocess.Popen(driver.split()) for _ in range(n)]
    # Wait for all the simulations to finish
    if ipi_proc is not None:
        ipi_proc.wait()
        for process in driver_proc:
            process.wait()
    # Move back to the original directory
    os.chdir(dir_base)
    return None


def prep_optimise(directory,
                  atoms,
                  outfile="min",
                  driver="ase-mace",
                  total_steps=1000,
                  deut=False,
                  optimizer="cg",
                  tol_energy=5.0e-6,
                  tol_force=5.0e-6,
                  tol_position=1.0e-6):
    """
    Prepares the minimization XML file for an optimization run.

    Parameters:
    directory (str): Directory where the XML file will be saved.
    atoms (ase.Atoms): ASE Atoms object containing the atomic structure.
    outfile (str, optional): Output file name. Default is "min".
    driver (str, optional): Driver to be used for the optimization. Default is "ase-mace".
    total_steps (int, optional): Total number of optimization steps. Default is 1000.
    deut (bool, optional): Whether to use deuterium masses. Default is False.
    optimizer (str, optional): Optimizer to be used. Default is "cg".
    tol_energy (float, optional): Energy tolerance for the optimization. Default is 5.0e-6.
    tol_force (float, optional): Force tolerance for the optimization. Default is 5.0e-6.
    tol_position (float, optional): Position tolerance for the optimization. Default is 1.0e-6.

    Returns:
    None
    """
    # Prepare the minimization xml file
    tree = ET.parse(os.path.abspath("../templates/MIN.xml"))
    root = tree.getroot()

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, deut=deut)

    # Change the driver ff
    update_driver(root, atoms, driver)

    # Change the title
    update_title(root, outfile)

    # Update the total_steps
    update_total_steps(root, total_steps)

    # Update the optimizer
    update_optimizer(root, optimizer)

    # Update the tolerances
    update_tol(root, tol_energy, tol_force, tol_position)

    # Write the file
    write_xml(root, f'{directory}input.xml')
    return None


def run_optimise(directory,
                 atoms,
                 server="i-pi input.xml",
                 outfile="min",
                 driver="ase-mace",
                 driver_dict=None,
                 total_steps=1000,
                 deut=False,
                 optimizer="cg",
                 tol_energy=5.0e-6,
                 tol_force=5.0e-6,
                 tol_position=1.0e-6):
    """
    Runs the optimization process by preparing the necessary XML file and executing the i-PI server and driver.

    Parameters:
    directory (str): Directory where the XML file will be saved and the simulation will be run.
    atoms (ase.Atoms): ASE Atoms object containing the atomic structure.
    server (str, optional): Command to start the i-PI server. Default is "i-pi input.xml".
    outfile (str, optional): Output file name. Default is "min".
    driver (str, optional): Driver to be used for the optimization. Default is "ase-mace".
    driver_dict (dict, optional): Dictionary of driver parameters. Default is None.
    total_steps (int, optional): Total number of optimization steps. Default is 1000.
    deut (bool, optional): Whether to use deuterium masses. Default is False.
    optimizer (str, optional): Optimizer to be used. Default is "cg".
    tol_energy (float, optional): Energy tolerance for the optimization. Default is 5.0e-6.
    tol_force (float, optional): Force tolerance for the optimization. Default is 5.0e-6.
    tol_position (float, optional): Position tolerance for the optimization. Default is 1.0e-6.

    Returns:
    None
    """
    # Prepare the minimization xml file
    if driver_dict is None:
        driver_dict = {}
    prep_optimise(directory, atoms,
                  outfile=outfile,
                  driver=driver,
                  total_steps=total_steps,
                  deut=deut,
                  optimizer=optimizer,
                  tol_energy=tol_energy,
                  tol_force=tol_force,
                  tol_position=tol_position)
    # Prepare the driver
    driver = prep_driver(driver, driver_dict)

    # Run the minimization
    print(f"Running the minimization with the driver: {driver}", flush=True)
    run_ipi(server, driver, outfile + ".out", cwd=directory)

    return None


def prep_phonons(directory,
                 atoms,
                 outfile="phonon",
                 driver="ase-mace",
                 total_steps=1000,
                 deut=False):
    """
    Prepares the phonon XML file for a phonon calculation.

    Parameters:
    directory (str): Directory where the XML file will be saved.
    atoms (ase.Atoms): ASE Atoms object containing the atomic structure.
    outfile (str, optional): Output file name. Default is "phonon".
    driver (str, optional): Driver to be used for the phonon calculation. Default is "ase-mace".
    total_steps (int, optional): Total number of steps for the phonon calculation. Default is 1000.
    deut (bool, optional): Whether to use deuterium masses. Default is False.

    Returns:
    None
    """
    # Prepare the phonon xml file
    tree = ET.parse(os.path.abspath("../templates/PHO.xml"))
    root = tree.getroot()

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, deut=deut)

    # Change the driver ff
    update_driver(root, atoms, driver)

    # Change the title
    update_title(root, outfile)

    # Update the total_steps
    update_total_steps(root, total_steps)

    # Write the file
    write_xml(root, f'{directory}input.xml')
    return None


def ipi_run_phonons(directory,
                    atoms,
                    min_file_path,
                    server="i-pi input.xml",
                    outfile="phonon",
                    driver="ase-mace",
                    driver_dict=None,
                    total_steps=1000,
                    deut=False):
    """
    Runs the phonon calculation by preparing the necessary XML file and executing the i-PI server and driver.

    Parameters:
    directory (str): Directory where the XML file will be saved and the simulation will be run.
    atoms (ase.Atoms): ASE Atoms object containing the atomic structure.
    min_file_path (str): Path to the minimized structure file.
    server (str, optional): Command to start the i-PI server. Default is "i-pi input.xml".
    outfile (str, optional): Output file name. Default is "phonon".
    driver (str, optional): Driver to be used for the phonon calculation. Default is "ase-mace".
    driver_dict (dict, optional): Dictionary of driver parameters. Default is None.
    total_steps (int, optional): Total number of steps for the phonon calculation. Default is 1000.
    deut (bool, optional): Whether to use deuterium masses. Default is False.

    Returns:
    None
    """
    if driver_dict is None:
        driver_dict = {}
    # Get the directory and the file name
    dir_react_min, outfile_min = os.path.split(min_file_path)
    # Prepare the phonon xyz file
    copy_xyz(get_final_xyz(dir_react_min, sub=f"{outfile_min}*"), directory)
    # Prepare the phonon xml file
    prep_phonons(directory, atoms,
                 outfile=outfile,
                 driver=driver,
                 total_steps=total_steps,
                 deut=deut)
    # Prepare the driver
    driver = prep_driver(driver, driver_dict)
    # Run the phonons
    run_ipi(server, driver, outfile + ".out", cwd=directory)


def prep_ts(directory,
            atoms,
            outfile="ts",
            driver="ase-mace",
            total_steps=1000,
            deut=False,
            tol_energy=5.0e-6,
            tol_force=5.0e-6,
            tol_position=1.0e-6):
    """
    Prepares the transition state search XML file for a transition state calculation.

    Parameters:
    directory (str): Directory where the XML file will be saved.
    atoms (ase.Atoms): ASE Atoms object containing the atomic structure.
    outfile (str, optional): Output file name. Default is "ts".
    driver (str, optional): Driver to be used for the transition state calculation. Default is "ase-mace".
    total_steps (int, optional): Total number of steps for the transition state calculation. Default is 1000.
    deut (bool, optional): Whether to use deuterium masses. Default is False.
    tol_energy (float, optional): Energy tolerance for the transition state calculation. Default is 5.0e-6.
    tol_force (float, optional): Force tolerance for the transition state calculation. Default is 5.0e-6.
    tol_position (float, optional): Position tolerance for the transition state calculation. Default is 1.0e-6.

    Returns:
    None
    """
    # Prepare the transition state search xml file
    tree = ET.parse(os.path.abspath("../templates/TS.xml"))
    root = tree.getroot()

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, deut=deut)

    # Change the driver ff
    update_driver(root, atoms, driver)

    # Change the title
    update_title(root, outfile)

    # Update the total_steps
    update_total_steps(root, total_steps)

    # Update the tolerances
    update_tol(root, tol_energy, tol_force, tol_position)

    # Write the file
    write_xml(root, f'{directory}input.xml')
    return None


def run_ts(directory,
           atoms,
           server="i-pi input.xml",
           outfile="ts",
           driver="ase-mace",
           driver_dict=None,
           total_steps=1000,
           deut=False,
           tol_energy=5.0e-6,
           tol_force=5.0e-6,
           tol_position=1.0e-6):
    """
    Runs the transition state calculation by preparing the necessary XML file and executing the i-PI server and driver.

    Parameters:
    directory (str): Directory where the XML file will be saved and the simulation will be run.
    atoms (ase.Atoms): ASE Atoms object containing the atomic structure.
    server (str, optional): Command to start the i-PI server. Default is "i-pi input.xml".
    outfile (str, optional): Output file name. Default is "ts".
    driver (str, optional): Driver to be used for the transition state calculation. Default is "ase-mace".
    driver_dict (dict, optional): Dictionary of driver parameters. Default is None.
    total_steps (int, optional): Total number of steps for the transition state calculation. Default is 1000.
    deut (bool, optional): Whether to use deuterium masses. Default is False.
    tol_energy (float, optional): Energy tolerance for the transition state calculation. Default is 5.0e-6.
    tol_force (float, optional): Force tolerance for the transition state calculation. Default is 5.0e-6.
    tol_position (float, optional): Position tolerance for the transition state calculation. Default is 1.0e-6.

    Returns:
    None
    """
    if driver_dict is None:
        driver_dict = {}
    # Read the transition state and write it to the ts directory
    write_xyz(ase.io.read(f"ts.xyz", "-1"), os.path.join(directory, "init.xyz"))
    # Prepare the ts xml file
    prep_ts(directory, atoms,
            outfile=outfile,
            driver=driver,
            total_steps=total_steps,
            deut=deut,
            tol_energy=tol_energy,
            tol_force=tol_force,
            tol_position=tol_position)
    # Prepare the driver
    driver = prep_driver(driver, driver_dict)
    # Run the ts
    run_ipi(server, driver, outfile + ".out", cwd=directory)


def prep_inst(directory,
              atoms,
              outfile="inst",
              driver="ase-mace",
              total_steps=1000,
              deut=False,
              n_beads=4,
              temperature=300.0,
              tol_energy=5.0e-6,
              tol_force=5.0e-6,
              tol_position=1.0e-6):
    """
    Prepares the instanton calculation XML file.

    Parameters:
    directory (str): Directory where the XML file will be saved.
    atoms (ase.Atoms): ASE Atoms object containing the atomic structure.
    outfile (str, optional): Output file name. Default is "inst".
    driver (str, optional): Driver to be used for the instanton calculation. Default is "ase-mace".
    total_steps (int, optional): Total number of steps for the instanton calculation. Default is 1000.
    deut (bool, optional): Whether to use deuterium masses. Default is False.
    n_beads (int, optional): Number of beads for the instanton calculation. Default is 4.
    temperature (float, optional): Temperature for the instanton calculation in Kelvin. Default is 300.0.
    tol_energy (float, optional): Energy tolerance for the instanton calculation. Default is 5.0e-6.
    tol_force (float, optional): Force tolerance for the instanton calculation. Default is 5.0e-6.
    tol_position (float, optional): Position tolerance for the instanton calculation. Default is 1.0e-6.

    Returns:
    None
    """
    n_atoms = len(atoms)
    n_dof = 3 * n_atoms - 6
    n_doft = 3 * n_atoms

    # Prepare the instanton calculation
    tree = ET.parse(os.path.abspath("../templates/INST.xml"))
    root = tree.getroot()

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, deut=deut)

    # Change the driver ff
    update_driver(root, atoms, driver)

    # Change the title
    update_title(root, outfile)

    # Update the total_steps
    update_total_steps(root, total_steps)

    # Modify the nbeads attribute
    update_nbeads(root, n_beads)

    # Modify the hessian attribute
    update_hessian(root, n_doft, 1)

    # Update the open paths
    update_open_paths(root, n_atoms)

    # Update the temperature
    update_temperature(root, temperature)

    # Update the tolerances
    update_tol(root, tol_energy, tol_force, tol_position)

    # Write the file
    write_xml(root, f'{directory}input.xml')


def run_inst(directory,
             atoms,
             directory_ts,
             server="i-pi input.xml",
             outfile="inst",
             driver="ase-mace",
             driver_dict=None,
             total_steps=1000,
             deut=False,
             n_beads=4,
             temperature=300.0,
             tol_energy=5.0e-6,
             tol_force=5.0e-6,
             tol_position=1.0e-6):
    """
    Runs the instanton calculation by preparing the necessary XML file and executing the i-PI server and driver.

    Parameters:
    directory (str): Directory where the XML file will be saved and the simulation will be run.
    atoms (ase.Atoms): ASE Atoms object containing the atomic structure.
    directory_ts (str): Directory containing the transition state calculation files.
    server (str, optional): Command to start the i-PI server. Default is "i-pi input.xml".
    outfile (str, optional): Output file name. Default is "inst".
    driver (str, optional): Driver to be used for the instanton calculation. Default is "ase-mace".
    driver_dict (dict, optional): Dictionary of driver parameters. Default is None.
    total_steps (int, optional): Total number of steps for the instanton calculation. Default is 1000.
    deut (bool, optional): Whether to use deuterium masses. Default is False.
    n_beads (int, optional): Number of beads for the instanton calculation. Default is 4.
    temperature (float, optional): Temperature for the instanton calculation in Kelvin. Default is 300.0.
    tol_energy (float, optional): Energy tolerance for the instanton calculation. Default is 5.0e-6.
    tol_force (float, optional): Force tolerance for the instanton calculation. Default is 5.0e-6.
    tol_position (float, optional): Position tolerance for the instanton calculation. Default is 1.0e-6.

    Returns:
    None
    """
    if driver_dict is None:
        driver_dict = {}
    # Copy the files from the ts calculation
    copy_xyz(get_final_xyz(directory_ts), directory)
    copy_hess(get_final_hess(directory_ts), directory)
    # Prepare the instanton calculation
    prep_inst(directory, atoms,
              outfile=outfile,
              driver=driver,
              total_steps=total_steps,
              deut=deut,
              n_beads=n_beads,
              temperature=temperature,
              tol_energy=tol_energy,
              tol_force=tol_force,
              tol_position=tol_position)
    # Prepare the driver
    driver = prep_driver(driver, driver_dict)
    # Run the instanton
    run_ipi(server, driver, outfile + ".out", cwd=directory)
    return None

# def inst_rerun():
#     # Copy the files from the 40 beads calculation
#     copy_xyz(get_final_xyz(dir_instanton_1, sub="*instanton*"), dir_instanton_2, file_out = "init0.xyz")
#     copy_hess(get_final_hess(dir_instanton_1, sub="*instanton*"), dir_instanton_2, file_out = "hessian0.dat")
#     # Prepare the interpolation
#     ipi_path = os.path.dirname(ipi.__path__[0])
#     interp_cmd = os.path.join(ipi_path, "tools/py/Instanton_interpolation.py")
#     cmd = f"python3 {interp_cmd} -m -xyz {dir_instanton_2 + "init0.xyz"} -hess {dir_instanton_2 + "hessian0.dat"} -n {n_beads_new}"
#     # Interpolate the hessian
#     interp_proc = subprocess.Popen(cmd.split())
#     interp_proc.wait()
#     # Copy the interpolated files
#     copy_xyz(get_final_xyz(dir_base, sub="new_instanton*"), dir_instanton_2, file_out="init.xyz")
#     copy_hess(get_final_hess(dir_base, sub="new_hessian*"), dir_instanton_2, file_out="hessian.dat")
#     # remove the old files
#     os.remove(get_final_xyz(dir_instanton_2, sub="init0*"))
#     os.remove(get_final_hess(dir_instanton_2, sub="hessian0*"))
#     os.remove(get_final_xyz(dir_base, sub="new_instanton*"))
#     os.remove(get_final_hess(dir_base, sub="new_hessian*"))
#     return None


# def ipi_prep_inst_rerun(directory,
#                         atoms,
#                         outfile="inst",
#                         f_driver="ase",
#                         total_steps=1000,
#                         f_deut=False,
#                         n_beads=4,
#                         temperature=300.0,
#                         tol_energy=5.0e-6,
#                         tol_force=5.0e-6,
#                         tol_position=1.0e-6):
#     n_atoms = len(atoms)
#     n_dof = 3 * n_atoms - 6
#     n_doft = 3 * n_atoms
#
#     # Prepare the instanton calculation
#     inst_str = """
#     <simulation mode="static" verbosity="low">
#         <output prefix='inst'>
#             <properties stride='1' filename='out'>[step, potential{electronvolt}]</properties>
#         </output>
#         <total_steps>20</total_steps>
#         <ffsocket name="cbe" mode="unix">
#             <address>localhost</address>
#         </ffsocket>
#         <system>
#             <initialize nbeads='4'>
#                 <file mode='xyz' units='angstrom'> init.xyz </file>
#                 <cell mode="abc" units='angstrom'>[100 100 100]</cell>
#             </initialize>
#             <forces>
#                 <force forcefield="cbe"></force>
#             </forces>
#             <ensemble>
#                 <temperature units="kelvin">300</temperature>
#             </ensemble>
#             <normal_modes>
#                 <open_paths>[0, 1, 2, 3, 4, 5]</open_paths>
#             </normal_modes>
#             <motion mode='instanton'>
#                 <instanton mode='rate'>
#                     <alt_out>10</alt_out>
#                     <tolerances>
#                         <energy>5e-6</energy>
#                         <force>5e-6</force>
#                         <position>1e-3</position>
#                     </tolerances>
#                     <delta>0.1</delta>
#                     <opt>nichols</opt>
#                     <hessian_update>powell</hessian_update>
#                     <hessian_asr>poly</hessian_asr>
#                     <hessian_final>true</hessian_final>
#                     <biggest_step>0.3</biggest_step>
#                     <hessian mode='file' shape='(18, 18)'>hessian.dat</hessian>
#                 </instanton>
#             </motion>
#         </system>
#     </simulation>
#     """
#
#     # Parse the string
#     root = ET.fromstring(inst_str)
#
#     # Fix the cell
#     ipi_in_update_cell(root, atoms)
#
#     # Fix the masses
#     ipi_in_update_mass(root, atoms, deut=f_deut)
#
#     # Change the driver ff
#     ipi_in_update_driver(root, atoms, f_driver)
#
#     # Change the title
#     ipi_in_update_title(root, outfile)
#
#     # Update the total_steps
#     ipi_in_update_total_steps(root, total_steps)
#
#     # Modify the nbeads attribute
#     ipi_in_update_nbeads(root, n_beads)
#
#     # Modify the hessian attribute
#     ipi_in_update_hessian(root, n_doft, n_beads)
#
#     # Update the open paths
#     ipi_in_update_open_paths(root, n_atoms)
#
#     # Update the temperature
#     ipi_in_update_temperature(root, temperature)
#
#     # Update the tolerances
#     ipi_in_update_tol(root, tol_energy, tol_force, tol_position)
#
#     # Write the file
#     write_xml(root, f'{directory}input.xml')
