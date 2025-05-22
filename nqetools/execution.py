import subprocess
import time
from itertools import chain
import xml.etree.ElementTree as et
import ipi
import numpy as np

from .driver import prep_driver
from .io import (write_xml,
                 copy_hess,
                 get_final_hess,
                 write_xyz,
                 read_ipi_xyz,
                 remove_directory)
from .plumed import prep_plumed
from .tools import rm_ipi_tmp, round_sf
from .xml_parse import *


def run_command(command):
    """
    Runs a simple command using the subprocess module.

    Parameters:
    command (str): The command to run.

    Returns:
    str: The output of the command.
    """
    result = subprocess.run(command.split(), shell=True, capture_output=True, text=True)
    return result.stdout


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


def run_plumed_hills(directory, temperature=300.0, bins=100, stride=100, cv=None):
    print(f'Running plumed hills', flush=True)
    if cv is None:
        cv = [[0.21, 0.31], [-1, 1]]

    # Convert the temperature to kBT
    kbt = temperature * 0.0083144621

    # Get the paths for the HILLS and FES files
    hills_str = os.path.join(directory, 'HILLS')
    fes_str = os.path.join(directory, 'FES')

    # Start building the command
    command = f"plumed sum_hills --hills {hills_str} --outfile {fes_str} --stride {stride} --kt {round_sf(kbt)} --mintozero"

    if len(np.shape(cv)) == 2:
        if all(item is None for item in chain.from_iterable(cv)):
            bins_values = ",".join([str(bins)] * len(cv))
            command += f' --bin {bins_values}'
        else:
            # Convert the CV list to a string
            min_values = ",".join(str(c[0]) for c in cv)
            max_values = ",".join(str(c[1]) for c in cv)
            bins_values = ",".join([str(bins)] * len(cv))
            command += f' --min {min_values} --max {max_values} --bin {bins_values}'
    else:
        if cv[0] is None and cv[1] is None:
            bins_values = str(bins)
            command += f' --bin {bins_values}'
        else:
            min_values = str(cv[0])
            max_values = str(cv[1])
            bins_values = str(bins)
            command += f' --min {min_values} --max {max_values} --bin {bins_values}'

    print(f"Running command:\n{command}", flush=True)

    # Run the sum_hills command
    with open(os.path.join(directory, "plumed.dat"), "r") as file:
        subprocess.run(command.split(), stdin=file, text=True)

    print(f"Plumed hills run complete\n", flush=True)
    return None


def run_plumed_hills_opes(directory, temperature=300.0, bins=100, cv=None):
    print(f'Running plumed OPES hills', flush=True)
    if cv is None:
        cv = [[0.21, 0.31], [-1, 1]]

    # Convert the temperature to kBT
    kbt = temperature * 0.0083144621

    # Need to be in the directory of the run
    cwd = os.getcwd()
    os.chdir(directory)
    path_to_opes = os.path.join(find_nqetools_path(), "opes", "fes_from_state.py")
    command = f'{path_to_opes} --kt {round_sf(kbt)}'

    if len(np.shape(cv)) == 2:
        if all(item is None for item in chain.from_iterable(cv)):
            bins_values = ",".join([str(bins)] * len(cv))
            command += f' --bin {bins_values}'
        else:
            # Convert the CV list to a string
            min_values = ",".join(str(c[0]) for c in cv)
            max_values = ",".join(str(c[1]) for c in cv)
            bins_values = ",".join([str(bins)] * len(cv))
            command += f' --min {min_values} --max {max_values} --bin {bins_values}'
    else:
        if cv[0] is None and cv[1] is None:
            bins_values = str(bins)
            command += f' --bin {bins_values}'
        else:
            min_values = str(cv[0])
            max_values = str(cv[1])
            bins_values = str(bins)
            command += f' --min {min_values} --max {max_values} --bin {bins_values}'

    command += f' --all_stored'
    print(f"Working directory: {directory}", flush=True)
    print(f"Running command:\n{command}", flush=True)

    subprocess.run(command.split())

    # change back to the original directory
    os.chdir(cwd)
    print(f"Plumed OPES hills run complete\n", flush=True)
    return None


def run_instanton_post_process(directory,
                               process_type='reactant',
                               temperature=300.0,
                               filter_list=None,
                               ref_energy=None,
                               n_beads=1,
                               outfile='thermo_data.out'):
    print(f'Running the {process_type} thermo/instanton post-processing', flush=True)

    # Need to be in the directory of the run
    cwd = os.getcwd()
    os.chdir(directory)
    path_to_proc = os.path.join(find_nqetools_path(), "instanton_tools", "postproc.py")
    # Build the command
    command = f'python {path_to_proc} RESTART -t {temperature} -c {process_type} -n {n_beads}'
    # Add the filter list if provided
    if filter_list is not None:
        command += f' -f {filter_list}'

    # Add the reference energy if provided
    if ref_energy is not None:
        command += f' -e {ref_energy}'

    print(f"Working directory: {directory}", flush=True)
    print(f"Running command:\n{command}", flush=True)

    # Run the command and write the output to the outfile
    with open(outfile, "w") as file:
        result = subprocess.run(command.split(), stdout=file, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}", flush=True)

    # Change back to the original directory
    os.chdir(cwd)
    print(f"Thermo/Instanton post-processing complete\n", flush=True)
    return None

def run_instanton_interpolation(directory_old, directory_new, new_n_beads):
    # https://github.com/i-pi/piqm2023-tutorial/blob/main/05-RPI/tutorial-4.ipynb
    # Make the folder input/instanton/80
    # Go to the folder input/instanton/80
    # Copy the previous optimized instanton geometry and name it init0
    # Copy the previous optimized instanton hessian and name it hess0
    # python ${ipi_path}/tools/py/Instanton_interpolation.py -m -xyz init0 -hess hess0 -n 80
    # Rename the new hessian and instanton geometry files to hessian.dat and init.xyz, respectively
    # Copy the input.xml file from input/instanton/40/
    # Change the number of beads from 40 to 80 in input.xml
    # Change the hessian shape from (18,18) to (18,1440) in input.xml dof * new_n_beads

    print(f'Running the instanton interpolation', flush=True)
    # Keep track of the current directory
    cwd = os.getcwd()

    # Make the new directory
    os.makedirs(directory_new, exist_ok=True)
    # Change to the new directory
    os.chdir(directory_new)
    # Copy the previous optimized instanton geometry and name it init0
    os.system(f'cp {os.path.join(directory_old, "init.xyz")} init0')
    # Copy the previous optimized instanton hessian and name it hess0
    os.system(f'cp {os.path.join(directory_old, "hessian.dat")} hess0')

    # Get the path to the interpolation script
    path_to_proc = os.path.join(find_nqetools_path(), "instanton_tools", "interpolation.py")
    # Build the command
    command = f'python {path_to_proc} -m -xyz init0 -hess hess0 -n {new_n_beads}'

    print(f"Running command:\n{command}", flush=True)

    # Run the command and write the output to the outfile
    with open("interpolation.out", "w") as file:
        result = subprocess.run(command.split(), stdout=file, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}", flush=True)

    # Rename the new hessian and instanton geometry files to hessian.dat and init.xyz, respectively
    os.system(f'cp hess0 hessian.dat')
    os.system(f'cp init0 init.xyz')

    # Copy the input.xml file from input/instanton/40/
    os.system(f'cp {os.path.join(directory_old, "input.xml")} input.xml')

    # Change the number of beads from 40 to 80 in input.xml
    tree = et.parse('input.xml')
    root = tree.getroot()
    update_nbeads(root, new_n_beads)

    # Get the degrees of freedom
    atoms = read_ipi_xyz(os.path.join(directory_old, "init.xyz"))
    n_doft = 3 * len(atoms)

    # Update the hessian shape from (18,18) to (18,1440) in input.xml dof * new_n_beads
    update_hessian(root, n_doft, new_n_beads)

    # Change back to the original directory
    os.chdir(cwd)
    print(f"Instanton interpolation complete\n", flush=True)
    return None


def prep_optimise_xml(directory,
                      atoms,
                      outfile="min",
                      driver="ase-mace",
                      total_steps=100,
                      deuterate=False,
                      optimiser="lbfgs",
                      tol_energy=1.0e-4,
                      tol_force=1.0e-4,
                      tol_position=1.0e-4,
                      stride=1,
                      checkpoint_stride=10,
                      properties=None,
                      xml_in=None,
                      file_in="init.xyz"):
    # Prepare the minimisation xml file
    if xml_in is not None:
        tree = et.parse(xml_in)
    else:
        tree = et.parse(os.path.join(find_nqetools_path(), "templates/MIN.xml"))
    root = tree.getroot()

    # Add in the properties to be tracked
    if properties is not None:
        append_properties(root, properties)

    # Set the initial structure
    write_xyz(atoms, os.path.join(directory, file_in))

    # Update the structure loading
    update_file(root, file_in)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, f_deut=deuterate)

    # Change the driver ff
    update_driver(root, atoms, driver)

    # Change the title
    update_title(root, outfile)

    # Update the total_steps
    update_total_steps(root, total_steps)

    # Update the optimiser
    update_optimiser(root, optimiser)

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
                 driver_args=None,
                 total_steps=100,
                 deuterate=False,
                 optimiser="lbfgs",
                 tol_energy=1.0e-4,
                 tol_force=1.0e-4,
                 tol_position=1.0e-4,
                 stride=1,
                 checkpoint_stride=1000,
                 properties=None,
                 xml_in=None):
    print(f"Running the minimisation with the driver: {driver}", flush=True)
    # Prepare the minimization xml file
    if driver_args is None:
        driver_args = {}

    # Clean the directory if it exists
    remove_directory(directory)

    # Make the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # If atoms is a list of atoms, take the last one as the structure
    if isinstance(atoms, list):
        atoms = atoms[-1]

    # Prepare the minimisation xml file
    prep_optimise_xml(directory,
                      atoms,
                      outfile=outfile,
                      driver=driver,
                      total_steps=total_steps,
                      deuterate=deuterate,
                      optimiser=optimiser,
                      tol_energy=tol_energy,
                      tol_force=tol_force,
                      tol_position=tol_position,
                      stride=stride,
                      checkpoint_stride=checkpoint_stride,
                      properties=properties,
                      xml_in=xml_in)
    # Prepare the driver
    driver = prep_driver(atoms, directory, driver, driver_args)

    # Run the minimisation
    run_ipi(directory, server, driver, f"{outfile}.out")
    # Load the structure
    atoms_out = read_ipi_xyz(os.path.join(directory, f"{outfile}.pos_0.xyz"))
    output_data, output_desc = ipi.read_output(os.path.join(directory, f"{outfile}.out"))
    print(f"Minimisation complete\n", flush=True)
    return atoms_out, output_data, output_desc


def prep_md_xml(directory,
                atoms,
                outfile="md",
                driver="ase-mace",
                total_steps=1000,
                deuterate=False,
                temperature=300.0,
                timestep=1,
                thermostat=None,
                md_type="NVT",
                splitting="baoab",
                fix_com=False,
                stride=10,
                checkpoint_stride=1000,
                n_beads=1,
                properties=None,
                xml_in=None,
                file_in="init.xyz"):
    # Prepare the MD simulation XML file
    if xml_in is not None:
        tree = et.parse(xml_in)
    else:
        tree = et.parse(os.path.join(find_nqetools_path(), f"templates/{md_type.upper()}.xml"))
    root = tree.getroot()

    # Add in the properties to be tracked
    if properties is not None:
        append_properties(root, properties)

    # Set the initial structure
    write_xyz(atoms, os.path.join(directory, file_in))

    # Update the structure loading
    update_file(root, file_in)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, f_deut=deuterate)

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

    # Updating splitting
    update_dynamics_splitting(root, splitting)

    # Update the fix_com
    update_motion_fix_com(root, fix_com)

    # Add the thermostat section
    if thermostat is not None:
        add_thermostat_section(root, thermostat=thermostat)

    # If the number of beads is greater than 1 running PIMD
    if n_beads > 1:
        # Add in the centroid
        add_trajectory_centroid(root)

        # Update the number of beads
        update_nbeads(root, n_beads)

        # Update the properties to be tracked
        append_properties(root, ['kinetic_cv{electronvolt}', 'pressure_cv{megapascal}'])
    else:
        # Update the properties to be tracked
        append_properties(root, ['kinetic_md{electronvolt}', 'pressure_md{megapascal}'])

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
           driver_args=None,
           total_steps=1000,
           deuterate=False,
           temperature=300.0,
           timestep=1,
           thermostat=None,
           md_type="NVT",
           splitting="baoab",
           fix_com=False,
           stride=10,
           checkpoint_stride=1000,
           n_beads=1,
           properties=None,
           xml_in=None):
    print(f"Running the MD ({md_type}) with the driver: {driver}", flush=True)
    if driver_args is None:
        driver_args = {}

    # Clean the directory if it exists
    remove_directory(directory)

    # Make the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # If atoms is a list of atoms, take the last one as the structure
    if isinstance(atoms, list):
        atoms = atoms[-1]

    # Prepare the MD xml file
    prep_md_xml(directory,
                atoms,
                outfile=outfile,
                driver=driver,
                total_steps=total_steps,
                deuterate=deuterate,
                temperature=temperature,
                timestep=timestep,
                thermostat=thermostat,
                md_type=md_type,
                splitting=splitting,
                fix_com=fix_com,
                stride=stride,
                checkpoint_stride=checkpoint_stride,
                n_beads=n_beads,
                properties=properties,
                xml_in=xml_in)
    # Prepare the driver
    driver = prep_driver(atoms, directory, driver, driver_args)
    # Run the MD
    run_ipi(directory, server, driver, f"{outfile}.out")
    # Load the structure
    if n_beads > 1:
        atoms_out = read_ipi_xyz(os.path.join(directory, f"{outfile}.xc.xyz"))
    else:
        atoms_out = read_ipi_xyz(os.path.join(directory, f"{outfile}.pos_0.xyz"))
    output_data, output_desc = ipi.read_output(os.path.join(directory, f"{outfile}.out"))
    print(f"MD ({md_type}) complete\n", flush=True)
    return atoms_out, output_data, output_desc


def prep_plumed_xml(directory,
                    atoms,
                    outfile="md",
                    driver="ase-mace",
                    total_steps=1000,
                    deuterate=False,
                    temperature=300.0,
                    timestep=1.0,
                    thermostat=None,
                    md_type="NVT",
                    splitting="baoab",
                    fix_com=False,
                    stride=10,
                    checkpoint_stride=1000,
                    n_beads=1,
                    plumed_extras=None,
                    properties=None,
                    xml_in=None,
                    file_in="init.xyz"):
    # Prepare the MD simulation XML file
    if xml_in is not None:
        tree = et.parse(xml_in)
    else:
        tree = et.parse(os.path.join(find_nqetools_path(), f"templates/{md_type.upper()}.xml"))
    root = tree.getroot()

    # Add in the bias to the properties to be tracked
    append_properties(root, ['ensemble_bias{electronvolt}'])

    # If properties is not None, add them to the properties to be tracked
    if properties is not None:
        append_properties(root, properties)

    # Set the initial structure
    write_xyz(atoms, os.path.join(directory, file_in))

    # Update the structure loading
    update_file(root, file_in)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, f_deut=deuterate)

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

    # Updating splitting
    update_dynamics_splitting(root, splitting)

    # Update the fixcom
    update_motion_fix_com(root, fix_com)

    # Add the thermostat section
    if thermostat is not None:
        add_thermostat_section(root, thermostat=thermostat)

    # If the number of beads is greater than 1 running PIMD
    if n_beads > 1:
        # Add in the centroid
        add_trajectory_centroid(root)

        # Update the number of beads
        update_nbeads(root, n_beads)

        # Update the properties to be tracked
        append_properties(root, ['kinetic_cv{electronvolt}', 'pressure_cv{megapascal}'])
    else:
        # Update the properties to be tracked
        append_properties(root, ['kinetic_md{electronvolt}', 'pressure_md{megapascal}'])

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
                  driver_args=None,
                  total_steps=1000,
                  deuterate=False,
                  temperature=300.0,
                  timestep=1.0,
                  thermostat=None,
                  md_type="NVT",
                  splitting="baoab",
                  fix_com=False,
                  stride=10,
                  checkpoint_stride=1000,
                  n_beads=1,
                  plumed_type="mtd-pos",
                  plumed_args=None,
                  properties=None,
                  xml_in=None):
    print(f"Running the MD ({md_type}) with the driver: {driver}", flush=True)
    # Update the plumed dictionary
    if plumed_args is None:
        plumed_args = {'directory': directory,
                       'temperature': temperature}
    else:
        # Add the directory and temperature to the plumed dictionary
        plumed_args['directory'] = directory
        plumed_args['temperature'] = temperature

    # Update the driver dictionary
    if driver_args is None:
        driver_args = {}

    # Clean the directory if it exists
    remove_directory(directory)

    # Make the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # If atoms is a list of atoms, take the last one as the structure
    if isinstance(atoms, list):
        atoms = atoms[-1]

    # Prepare the plumed input file
    plumed_extras = prep_plumed(atoms, plumed_type, plumed_args)

    # Prepare the MD xml file
    prep_plumed_xml(directory,
                    atoms,
                    outfile=outfile,
                    driver=driver,
                    total_steps=total_steps,
                    deuterate=deuterate,
                    temperature=temperature,
                    timestep=timestep,
                    thermostat=thermostat,
                    md_type=md_type,
                    splitting=splitting,
                    fix_com=fix_com,
                    stride=stride,
                    checkpoint_stride=checkpoint_stride,
                    n_beads=n_beads,
                    plumed_extras=plumed_extras,
                    properties=properties,
                    xml_in=xml_in)

    # Prepare the driver
    driver = prep_driver(atoms, directory, driver, driver_args)

    # Run the MD
    run_ipi(directory, server, driver, f"{outfile}.out", n=n_beads)

    # Load the structure
    if n_beads > 1:
        atoms_out = read_ipi_xyz(os.path.join(directory, f"{outfile}.xc.xyz"))
    else:
        atoms_out = read_ipi_xyz(os.path.join(directory, f"{outfile}.pos_0.xyz"))
    output_data, output_desc = ipi.read_output(os.path.join(directory, f"{outfile}.out"))
    print(f"MD ({md_type}) complete\n", flush=True)
    return atoms_out, output_data, output_desc


def prep_phonons_xml(directory,
                     atoms,
                     outfile="phonon",
                     driver="ase-mace",
                     total_steps=1000,
                     deuterate=False,
                     stride=1,
                     checkpoint_stride=1000,
                     properties=None,
                     xml_in=None,
                     file_in="init.xyz"):
    # Prepare the phonon xml file
    if xml_in is not None:
        tree = et.parse(xml_in)
    else:
        tree = et.parse(os.path.join(find_nqetools_path(), f"templates/PHO.xml"))
    root = tree.getroot()

    # Add in the properties to be tracked
    if properties is not None:
        append_properties(root, properties)

    # Set the initial structure
    write_xyz(atoms, os.path.join(directory, file_in))

    # Update the structure loading
    update_file(root, file_in)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, f_deut=deuterate)

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
    write_xml(root, os.path.join(directory, 'input.xml'))
    return None


def run_phonons(directory,
                atoms,
                server="i-pi input.xml",
                outfile="phonon",
                driver="ase-mace",
                driver_args=None,
                total_steps=1000,
                deuterate=False,
                stride=1,
                checkpoint_stride=1000,
                properties=None,
                xml_in=None):
    print(f"Running the phonons with the driver: {driver}", flush=True)
    if driver_args is None:
        driver_args = {}

    # Clean the directory if it exists
    remove_directory(directory)

    # Make the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # If atoms is a list of atoms, take the last one as the structure
    if isinstance(atoms, list):
        atoms = atoms[-1]

    # Prepare the phonon xml file
    prep_phonons_xml(directory,
                     atoms,
                     outfile=outfile,
                     driver=driver,
                     total_steps=total_steps,
                     deuterate=deuterate,
                     stride=stride,
                     checkpoint_stride=checkpoint_stride,
                     properties=properties,
                     xml_in=xml_in)
    # Prepare the driver
    driver = prep_driver(atoms, directory, driver, driver_args)
    # Run the phonons
    run_ipi(directory, server, driver, f"{outfile}.out")
    print(f"Phonons complete\n", flush=True)
    return None


def prep_ts_xml(directory,
                atoms,
                outfile="ts",
                driver="ase-mace",
                total_steps=1000,
                deuterate=False,
                tol_energy=5.0e-6,
                tol_force=5.0e-6,
                tol_position=1.0e-6,
                stride=1,
                checkpoint_stride=1000,
                properties=None,
                xml_in=None,
                file_in="init.xyz"):
    # Prepare the transition state search xml file
    if xml_in is not None:
        tree = et.parse(xml_in)
    else:
        tree = et.parse(os.path.join(find_nqetools_path(), f"templates/TS.xml"))
    root = tree.getroot()

    # Add in the properties to be tracked
    if properties is not None:
        append_properties(root, properties)

    # Set the initial structure
    write_xyz(atoms, os.path.join(directory, file_in))

    # Update the structure loading
    update_file(root, file_in)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, f_deut=deuterate)

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
    write_xml(root, os.path.join(directory, 'input.xml'))
    return None


def run_ts(directory,
           atoms,
           server="i-pi input.xml",
           outfile="ts",
           driver="ase-mace",
           driver_args=None,
           total_steps=1000,
           deuterate=False,
           tol_energy=5.0e-6,
           tol_force=5.0e-6,
           tol_position=1.0e-6,
           stride=1,
           checkpoint_stride=1000,
           properties=None,
           xml_in=None):
    print(f"Running the transition state search with the driver: {driver}", flush=True)
    if driver_args is None:
        driver_args = {}

    # Clean the directory if it exists
    remove_directory(directory)

    # Make the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # If atoms is a list of atoms, take the last one as the structure
    if isinstance(atoms, list):
        atoms = atoms[-1]

    # Prepare the ts xml file
    prep_ts_xml(directory,
                atoms,
                outfile=outfile,
                driver=driver,
                total_steps=total_steps,
                deuterate=deuterate,
                tol_energy=tol_energy,
                tol_force=tol_force,
                tol_position=tol_position,
                stride=stride,
                checkpoint_stride=checkpoint_stride,
                properties=properties,
                xml_in=xml_in)
    # Prepare the driver
    driver = prep_driver(atoms, directory, driver, driver_args)
    # Run the ts
    run_ipi(directory, server, driver, f"{outfile}.out")
    # Load the structure
    atoms_out = read_ipi_xyz(os.path.join(directory, f"{outfile}.pos_0.xyz"))
    output_data, output_desc = ipi.read_output(os.path.join(directory, f"{outfile}.out"))
    print(f"TS search complete\n", flush=True)
    return atoms_out, output_data, output_desc


def prep_instanton_xml(directory,
                       atoms,
                       outfile="instanton",
                       driver="ase-mace",
                       total_steps=1000,
                       deuterate=False,
                       n_beads=4,
                       temperature=300.0,
                       tol_energy=5.0e-6,
                       tol_force=5.0e-6,
                       tol_position=1.0e-6,
                       stride=1,
                       checkpoint_stride=1000,
                       properties=None,
                       xml_in=None,
                       file_in="init.xyz"):
    n_atoms = len(atoms)
    # n_dof = 3 * n_atoms - 6
    n_doft = 3 * n_atoms

    # Prepare the instanton calculation
    if xml_in is not None:
        tree = et.parse(xml_in)
    else:
        tree = et.parse(os.path.join(find_nqetools_path(), f"templates/INST.xml"))
    root = tree.getroot()

    # Add in the properties to be tracked
    if properties is not None:
        append_properties(root, properties)

    # Set the initial structure
    write_xyz(atoms, os.path.join(directory, file_in))

    # Update the structure loading
    update_file(root, file_in)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, f_deut=deuterate)

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
    write_xml(root, os.path.join(directory, 'input.xml'))
    return None


def run_instanton(directory,
                  atoms,
                  directory_ts,
                  server="i-pi input.xml",
                  outfile="instanton",
                  driver="ase-mace",
                  driver_args=None,
                  total_steps=1000,
                  deuterate=False,
                  n_beads=4,
                  temperature=300.0,
                  tol_energy=5.0e-6,
                  tol_force=5.0e-6,
                  tol_position=1.0e-6,
                  stride=1,
                  checkpoint_stride=1000,
                  properties=None,
                  xml_in=None):
    print(f"Running the instanton with the driver: {driver}", flush=True)
    if driver_args is None:
        driver_args = {}

    # Clean the directory if it exists
    remove_directory(directory)

    # Make the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # If atoms is a list of atoms, take the last one as the structure
    if isinstance(atoms, list):
        atoms = atoms[-1]

    # Copy the files from the ts calculation
    copy_hess(get_final_hess(directory_ts), directory)

    # Prepare the instanton calculation
    prep_instanton_xml(directory,
                       atoms,
                       outfile=outfile,
                       driver=driver,
                       total_steps=total_steps,
                       deuterate=deuterate,
                       n_beads=n_beads,
                       temperature=temperature,
                       tol_energy=tol_energy,
                       tol_force=tol_force,
                       tol_position=tol_position,
                       stride=stride,
                       checkpoint_stride=checkpoint_stride,
                       properties=properties,
                       xml_in=xml_in)

    # Prepare the driver
    driver = prep_driver(atoms, directory, driver, driver_args)

    # Run the instanton
    run_ipi(directory, server, driver, f"{outfile}.out")
    print(f"Instanton complete\n", flush=True)
    return None
