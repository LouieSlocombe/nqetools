import os
import subprocess
import time

import ase.io

from .driver import prep_driver
from .io import (write_xml,
                 copy_xyz,
                 get_final_xyz,
                 copy_hess,
                 get_final_hess,
                 write_xyz)
from .plumed import prep_plumed
from .tools import rm_ipi_tmp
from .xml_parse import *


def run_ipi(directory,
            server,
            driver,
            outfile,
            n=1,
            t_sleep=5.0):
    """
    Runs the i-PI server and driver processes for a simulation.

    Parameters:
    directory (str): Directory where the simulation will be run.
    server (str): Command to start the i-PI server.
    driver (str): Command to start the driver processes.
    outfile (str): Output file name to check if the simulation has already been run.
    n (int, optional): Number of driver processes to start. Default is 1.
    t_sleep (float, optional): Time to wait for the i-PI server to start. Default is 5.0 seconds.

    Returns:
    None
    """
    # Get the current directory
    dir_base = os.getcwd()
    # Change to the working directory
    os.chdir(directory)
    # Remove the tmp file if it exists
    rm_ipi_tmp()
    # Start the i-PI server and the driver processes
    if not os.path.exists(outfile):
        # Don't rerun if the outputs already exist
        ipi_proc = subprocess.Popen(server.split())
        # Wait for i-PI to start
        time.sleep(t_sleep)
        # Start the driver processes
        driver_proc = [subprocess.Popen(driver.split()) for _ in range(n)]
    else:
        raise FileExistsError(f"Output file {outfile} already exists. Skipping the run.")
    # Wait for all the simulations to finish
    if ipi_proc is not None:
        ipi_proc.wait()
        for process in driver_proc:
            process.wait()
    # Move back to the original directory
    os.chdir(dir_base)
    return None


def prep_md_xml(directory,
                atoms,
                outfile="md",
                driver="ase-mace",
                total_steps=1000,
                deut=False,
                temperature=300.0,
                timestep=1.0e-3,
                md_type="NVT",
                stride=10,
                checkpoint_stride=1000,
                n_beads=1,
                properties=None):
    # Prepare the MD simulation XML file
    tree = ET.parse(os.path.expanduser(os.path.abspath(f"../templates/{md_type.upper()}.xml")))
    root = tree.getroot()

    # Add in the properties to be tracked
    append_properties(root, properties)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, f_deut=deut)

    # Change the driver ff
    update_driver(root, atoms, driver)

    # Change the title
    update_title(root, outfile)

    # Update the total_steps
    update_total_steps(root, total_steps)

    # Update the temperature
    update_temperature(root, temperature)

    # Update the timestep
    update_timestep(root, timestep)

    # If the md_type has PIMD update
    if 'PIMD' in md_type:
        # Add in the centroid
        add_trajectory_centroid(root)

        # Update the number of beads
        update_nbeads(root, n_beads)

    # Update the stride
    update_stride(root, stride)

    # Update the checkpoint stride
    update_checkpoint_stride(root, checkpoint_stride)

    # Write the file
    write_xml(root, os.path.join(directory, 'input.xml'))
    return None


def run_md(directory,
           atoms,
           server="i-pi input.xml",
           outfile="md",
           driver="ase-mace",
           driver_dict=None,
           total_steps=1000,
           deut=False,
           temperature=300.0,
           timestep=1.0e-3,
           md_type="NVT",
           stride=10,
           checkpoint_stride=1000,
           n_beads=1,
           properties=None):
    md_type = md_type.upper()
    # assert md_type in ["NVT", "NPT"], f"MD type {md_type} not supported"
    if driver_dict is None:
        driver_dict = {}

    # Make the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Set the initial structure
    write_xyz(atoms, os.path.join(directory, "init.xyz"))

    # Prepare the MD xml file
    prep_md_xml(directory, atoms,
                outfile=outfile,
                driver=driver,
                total_steps=total_steps,
                deut=deut,
                temperature=temperature,
                timestep=timestep,
                md_type=md_type,
                stride=stride,
                checkpoint_stride=checkpoint_stride,
                n_beads=n_beads,
                properties=properties)
    # Prepare the driver
    driver = prep_driver(directory, driver, driver_dict)
    # Run the MD
    print(f"Running the MD ({md_type}) with the driver: {driver}", flush=True)
    run_ipi(directory, server, driver, outfile + ".out")
    # Load the structure
    atoms_out = ase.io.read(os.path.join(directory, f"{outfile}.pos_0.xyz"), index=":")
    return atoms_out


def prep_plumed_xml(directory,
                    atoms,
                    outfile="md",
                    driver="ase-mace",
                    total_steps=1000,
                    deut=False,
                    temperature=300.0,
                    timestep=1.0e-3,
                    md_type="NVT",
                    stride=10,
                    checkpoint_stride=1000,
                    n_beads=1,
                    plumed_extras=None,
                    properties=None):
    # Prepare the MD simulation XML file
    tree = ET.parse(os.path.expanduser(os.path.abspath(f"../templates/{md_type.upper()}.xml")))
    root = tree.getroot()

    # Add in the bias to the properties to be tracked
    append_properties(root, ['bias'])
    # If properties is not None, add them to the properties to be tracked
    if properties is not None:
        append_properties(root, properties)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, f_deut=deut)

    # Change the driver ff
    update_driver(root, atoms, driver)

    # Change the title
    update_title(root, outfile)

    # Update the total_steps
    update_total_steps(root, total_steps)

    # Update the temperature
    update_temperature(root, temperature)

    # Update the timestep
    update_timestep(root, timestep)

    # If the md_type has PIMD update
    if 'PIMD' in md_type:
        # Add in the centroid
        add_trajectory_centroid(root)

        # Update the number of beads
        update_nbeads(root, n_beads)

    # Add the plumed input file
    add_plumed_xml(root, plumed_extras=plumed_extras)

    # Update the stride
    update_stride(root, stride)

    # Update the checkpoint stride
    update_checkpoint_stride(root, checkpoint_stride)

    # Write the file
    write_xml(root, os.path.join(directory, 'input.xml'))
    return None


def run_plumed_md(directory,
                  atoms,
                  server="i-pi input.xml",
                  outfile="md",
                  driver="ase-mace",
                  driver_dict=None,
                  total_steps=1000,
                  deut=False,
                  temperature=300.0,
                  timestep=1.0e-3,
                  md_type="NVT",
                  stride=10,
                  checkpoint_stride=1000,
                  n_beads=1,
                  plumed_type="pos",
                  plumed_dict=None,
                  plumed_extras=None,
                  properties=None):
    if plumed_dict is None:
        plumed_dict = {'directory': directory, 'temperature': temperature}

    md_type = md_type.upper()
    if driver_dict is None:
        driver_dict = {}

    # Make the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Set the initial structure
    write_xyz(atoms, os.path.join(directory, "init.xyz"))

    # Prepare the plumed input file
    prep_plumed(plumed_type, atoms, plumed_dict)

    # Prepare the MD xml file
    prep_plumed_xml(directory, atoms,
                    outfile=outfile,
                    driver=driver,
                    total_steps=total_steps,
                    deut=deut,
                    temperature=temperature,
                    timestep=timestep,
                    md_type=md_type,
                    stride=stride,
                    checkpoint_stride=checkpoint_stride,
                    n_beads=n_beads,
                    plumed_extras=plumed_extras,
                    properties=properties)

    # Prepare the driver
    driver = prep_driver(directory, driver, driver_dict)

    # Run the MD
    print(f"Running plumed MD ({md_type}) with the driver: {driver}", flush=True)
    run_ipi(directory, server, driver, outfile + ".out", n=n_beads)

    # Load the structure
    atoms_out = ase.io.read(os.path.join(directory, f"{outfile}.pos_0.xyz"), index=":")
    return atoms_out


def prep_optimise_xml(directory,
                      atoms,
                      outfile="min",
                      driver="ase-mace",
                      total_steps=1000,
                      deut=False,
                      optimizer="cg",
                      tol_energy=5.0e-6,
                      tol_force=5.0e-6,
                      tol_position=1.0e-6,
                      stride=1,
                      checkpoint_stride=10,
                      properties=None):
    # Prepare the minimization xml file
    tree = ET.parse(os.path.expanduser(os.path.abspath("../templates/MIN.xml")))
    root = tree.getroot()

    # Add in the properties to be tracked
    append_properties(root, properties)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, f_deut=deut)

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

    # Update the stride
    update_stride(root, stride)

    # Update the checkpoint stride
    update_checkpoint_stride(root, checkpoint_stride)

    # Write the file
    write_xml(root, os.path.join(directory, 'input.xml'))
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
                 tol_position=1.0e-6,
                 stride=1,
                 checkpoint_stride=1000,
                 properties=None):
    # Prepare the minimization xml file
    if driver_dict is None:
        driver_dict = {}

    # Make the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Set the initial structure
    write_xyz(atoms, os.path.join(directory, "init.xyz"))

    prep_optimise_xml(directory, atoms,
                      outfile=outfile,
                      driver=driver,
                      total_steps=total_steps,
                      deut=deut,
                      optimizer=optimizer,
                      tol_energy=tol_energy,
                      tol_force=tol_force,
                      tol_position=tol_position,
                      stride=stride,
                      checkpoint_stride=checkpoint_stride,
                      properties=properties)
    # Prepare the driver
    driver = prep_driver(directory, driver, driver_dict)

    # Run the minimization
    print(f"Running the minimization with the driver: {driver}", flush=True)
    run_ipi(directory, server, driver, outfile + ".out")
    # Load the structure
    atoms_out = ase.io.read(os.path.join(directory, f"{outfile}.pos.xyz"), index=":")
    return atoms_out


def prep_phonons_xml(directory,
                     atoms,
                     outfile="phonon",
                     driver="ase-mace",
                     total_steps=1000,
                     deut=False,
                     stride=1,
                     checkpoint_stride=1000,
                     properties=None):
    # Prepare the phonon xml file
    tree = ET.parse(os.path.expanduser(os.path.abspath("../templates/PHO.xml")))
    root = tree.getroot()

    # Add in the properties to be tracked
    append_properties(root, properties)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, f_deut=deut)

    # Change the driver ff
    update_driver(root, atoms, driver)

    # Change the title
    update_title(root, outfile)

    # Update the total_steps
    update_total_steps(root, total_steps)

    # Update the stride
    update_stride(root, stride)

    # Update the checkpoint stride
    update_checkpoint_stride(root, checkpoint_stride)

    # Write the file
    write_xml(root, f'{directory}input.xml')
    return None


def run_phonons(directory,
                atoms,
                min_file_path,
                server="i-pi input.xml",
                outfile="phonon",
                driver="ase-mace",
                driver_dict=None,
                total_steps=1000,
                deut=False,
                stride=1,
                checkpoint_stride=1000,
                properties=None):
    if driver_dict is None:
        driver_dict = {}

    # Make the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Set the initial structure
    write_xyz(atoms, os.path.join(directory, "init.xyz"))

    # Get the directory and the file name
    dir_react_min, outfile_min = os.path.split(min_file_path)
    # Prepare the phonon xyz file
    copy_xyz(get_final_xyz(dir_react_min, sub=f"{outfile_min}*"), directory)
    # Prepare the phonon xml file
    prep_phonons_xml(directory, atoms,
                     outfile=outfile,
                     driver=driver,
                     total_steps=total_steps,
                     deut=deut,
                     stride=stride,
                     checkpoint_stride=checkpoint_stride,
                     properties=properties)
    # Prepare the driver
    driver = prep_driver(directory, driver, driver_dict)
    # Run the phonons
    run_ipi(directory, server, driver, outfile + ".out")


def prep_ts_xml(directory,
                atoms,
                outfile="ts",
                driver="ase-mace",
                total_steps=1000,
                deut=False,
                tol_energy=5.0e-6,
                tol_force=5.0e-6,
                tol_position=1.0e-6,
                stride=1,
                checkpoint_stride=1000,
                properties=None):
    # Prepare the transition state search xml file
    tree = ET.parse(os.path.expanduser(os.path.abspath("../templates/TS.xml")))
    root = tree.getroot()

    # Add in the properties to be tracked
    append_properties(root, properties)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, f_deut=deut)

    # Change the driver ff
    update_driver(root, atoms, driver)

    # Change the title
    update_title(root, outfile)

    # Update the total_steps
    update_total_steps(root, total_steps)

    # Update the tolerances
    update_tol(root, tol_energy, tol_force, tol_position)

    # Update the stride
    update_stride(root, stride)

    # Update the checkpoint stride
    update_checkpoint_stride(root, checkpoint_stride)

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
           tol_position=1.0e-6,
           stride=1,
           checkpoint_stride=1000,
           properties=None):
    if driver_dict is None:
        driver_dict = {}
    # Read the transition state and write it to the ts directory
    write_xyz(ase.io.read(f"ts.xyz", "-1"), os.path.join(directory, "init.xyz"))
    # Prepare the ts xml file
    prep_ts_xml(directory, atoms,
                outfile=outfile,
                driver=driver,
                total_steps=total_steps,
                deut=deut,
                tol_energy=tol_energy,
                tol_force=tol_force,
                tol_position=tol_position,
                stride=stride,
                checkpoint_stride=checkpoint_stride,
                properties=properties)
    # Prepare the driver
    driver = prep_driver(directory, driver, driver_dict)
    # Run the ts
    run_ipi(directory, server, driver, outfile + ".out")


def prep_inst_xml(directory,
                  atoms,
                  outfile="inst",
                  driver="ase-mace",
                  total_steps=1000,
                  deut=False,
                  n_beads=4,
                  temperature=300.0,
                  tol_energy=5.0e-6,
                  tol_force=5.0e-6,
                  tol_position=1.0e-6,
                  stride=1,
                  checkpoint_stride=1000,
                  properties=None):
    n_atoms = len(atoms)
    n_dof = 3 * n_atoms - 6
    n_doft = 3 * n_atoms

    # Prepare the instanton calculation
    tree = ET.parse(os.path.expanduser(os.path.abspath("../templates/INST.xml")))
    root = tree.getroot()

    # Add in the properties to be tracked
    append_properties(root, properties)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, f_deut=deut)

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

    # Update the stride
    update_stride(root, stride)

    # Update the checkpoint stride
    update_checkpoint_stride(root, checkpoint_stride)

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
             tol_position=1.0e-6,
             stride=1,
             checkpoint_stride=1000,
             properties=None):
    if driver_dict is None:
        driver_dict = {}

    # Make the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Copy the files from the ts calculation
    copy_xyz(get_final_xyz(directory_ts), directory)
    copy_hess(get_final_hess(directory_ts), directory)

    # Prepare the instanton calculation
    prep_inst_xml(directory, atoms,
                  outfile=outfile,
                  driver=driver,
                  total_steps=total_steps,
                  deut=deut,
                  n_beads=n_beads,
                  temperature=temperature,
                  tol_energy=tol_energy,
                  tol_force=tol_force,
                  tol_position=tol_position,
                  stride=stride,
                  checkpoint_stride=checkpoint_stride,
                  properties=properties)

    # Prepare the driver
    driver = prep_driver(directory, driver, driver_dict)

    # Run the instanton
    run_ipi(directory, server, driver, outfile + ".out")
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
