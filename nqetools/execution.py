import os
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
                 remove_directory,
                 find_nqetools_path)
from .plumed import prep_plumed
from .tools import rm_ipi_tmp, round_sf
from .xml_parse import (append_properties,
                         update_cell,
                         update_checkpoint_stride,
                         update_driver,
                         update_dynamics_splitting,
                         update_file,
                         update_hessian,
                         update_mass,
                         update_motion_fix_com,
                         update_nbeads,
                         update_open_paths,
                         update_optimiser,
                         update_stride,
                         update_temperature,
                         update_timestep,
                         update_title,
                         update_tol,
                         update_total_steps,
                         add_plumed_xml,
                         add_thermostat_section,
                         add_trajectory_centroid)
import multiprocessing


def run_command(command):
    """Runs a simple command using the subprocess module.

    Parameters
    ----------
    command : str
        The command to run.

    Returns
    -------
    str
        The output of the command.
    """
    result = subprocess.run(command.split(), shell=True, capture_output=True, text=True)
    return result.stdout


def check_driver_processes(n_processes, max_ratio=0.9, warn_only=False):
    """Checks if the requested number of driver processes exceeds available system resources.

    Parameters
    ----------
    n_processes : int
        The requested number of driver processes.
    max_ratio : float, optional
        Maximum ratio of cores to use. Default is 0.9 (90%).
    warn_only : bool, optional
        If True, only warn but don't adjust the number. Default is False.

    Returns
    -------
    int
        Recommended number of processes to use
    """

    available_cores = multiprocessing.cpu_count()

    safe_processes = max(1, int(available_cores * max_ratio))

    if n_processes > safe_processes:
        if warn_only:
            print(f"Warning: Requested {n_processes} driver processes, but only {available_cores} CPU cores available.")
            print(f"Recommended maximum: {safe_processes} processes (using {int(max_ratio * 100)}% of cores).")
            return n_processes
        else:
            print(f"Adjusting driver processes from {n_processes} to {safe_processes} based on available CPU cores.")
            return safe_processes

    return n_processes


def run_ipi(directory,
            server,
            driver,
            outfile,
            n=1,
            t_sleep=5.0):
    """Runs the i-PI server and driver processes for a simulation.

    Parameters
    ----------
    directory : str
        Directory where the simulation will be run.
    server : str
        Command to start the i-PI server.
    driver : str
        Command to start the driver processes.
    outfile : str
        Output file name to check if the simulation has already been run.
    n : int, optional
        Number of driver processes to start. Default is 1.
    t_sleep : float, optional
        Time to wait for the i-PI server to start. Default is 5.0 seconds.

    Returns
    -------
    None
    """
    dir_base = os.getcwd()
    os.chdir(directory)
    rm_ipi_tmp()
    n = check_driver_processes(n)

    if not os.path.exists(outfile):
        ipi_proc = subprocess.Popen(server.split())
        time.sleep(t_sleep)
        driver_proc = [subprocess.Popen(driver.split()) for _ in range(n)]
    else:
        raise FileExistsError(f"Output file {outfile} already exists. Skipping the run.")
    if ipi_proc is not None:
        ipi_proc.wait()
        for process in driver_proc:
            process.wait()
    os.chdir(dir_base)
    return None


def run_plumed_hills(directory, temperature=300.0, bins=100, stride=100, cv=None):
    print('Running plumed hills', flush=True)
    if cv is None:
        cv = [[0.21, 0.31], [-1, 1]]

    kbt = temperature * 0.0083144621

    hills_str = os.path.join(directory, 'HILLS')
    fes_str = os.path.join(directory, 'FES')

    command = f"plumed sum_hills --hills {hills_str} --outfile {fes_str} --stride {stride} --kt {round_sf(kbt)} --mintozero"

    if len(np.shape(cv)) == 2:
        if all(item is None for item in chain.from_iterable(cv)):
            bins_values = ",".join([str(bins)] * len(cv))
            command += f' --bin {bins_values}'
        else:
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

    with open(os.path.join(directory, "plumed.dat"), "r") as file:
        subprocess.run(command.split(), stdin=file, text=True)

    print("Plumed hills run complete\n", flush=True)
    return None


def run_plumed_hills_opes(directory, temperature=300.0, bins=100, cv=None):
    print('Running plumed OPES hills', flush=True)
    if cv is None:
        cv = [[0.21, 0.31], [-1, 1]]

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

    command += ' --all_stored'
    print(f"Working directory: {directory}", flush=True)
    print(f"Running command:\n{command}", flush=True)

    subprocess.run(command.split())

    os.chdir(cwd)
    print("Plumed OPES hills run complete\n", flush=True)
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
    command = f'python {path_to_proc} RESTART -t {temperature} -c {process_type} -n {n_beads}'
    if filter_list is not None:
        command += f' -f {filter_list}'

    if ref_energy is not None:
        command += f' -e {ref_energy}'

    print(f"Working directory: {directory}", flush=True)
    print(f"Running command:\n{command}", flush=True)

    with open(outfile, "w") as file:
        result = subprocess.run(command.split(), stdout=file, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}", flush=True)

    os.chdir(cwd)
    print("Thermo/Instanton post-processing complete\n", flush=True)
    return None


def run_instanton_interpolation(directory_old, directory_new, new_n_beads):
    # Reimplements the manual recipe from:
    # https://github.com/i-pi/piqm2023-tutorial/blob/main/05-RPI/tutorial-4.ipynb
    print('Running the instanton interpolation', flush=True)
    cwd = os.getcwd()

    os.makedirs(directory_new, exist_ok=True)
    os.chdir(directory_new)
    os.system(f'cp {os.path.join(directory_old, "init.xyz")} init0')
    os.system(f'cp {os.path.join(directory_old, "hessian.dat")} hess0')

    path_to_proc = os.path.join(find_nqetools_path(), "instanton_tools", "interpolation.py")
    command = f'python {path_to_proc} -m -xyz init0 -hess hess0 -n {new_n_beads}'

    print(f"Running command:\n{command}", flush=True)

    with open("interpolation.out", "w") as file:
        result = subprocess.run(command.split(), stdout=file, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}", flush=True)

    os.system('cp hess0 hessian.dat')
    os.system('cp init0 init.xyz')

    os.system(f'cp {os.path.join(directory_old, "input.xml")} input.xml')

    tree = et.parse('input.xml')
    root = tree.getroot()
    update_nbeads(root, new_n_beads)

    atoms = read_ipi_xyz(os.path.join(directory_old, "init.xyz"))
    n_doft = 3 * len(atoms)

    update_hessian(root, n_doft, new_n_beads)

    os.chdir(cwd)
    print("Instanton interpolation complete\n", flush=True)
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
    if xml_in is not None:
        tree = et.parse(xml_in)
    else:
        tree = et.parse(os.path.join(find_nqetools_path(), "templates/MIN.xml"))
    root = tree.getroot()

    if properties is not None:
        append_properties(root, properties)

    write_xyz(atoms, os.path.join(directory, file_in))

    update_file(root, file_in)

    update_cell(root, atoms)

    update_mass(root, atoms, f_deut=deuterate)

    update_driver(root, atoms, driver)

    update_title(root, outfile)

    update_total_steps(root, total_steps)

    update_optimiser(root, optimiser)

    update_tol(root, tol_energy, tol_force, tol_position)

    update_stride(root, stride)

    update_checkpoint_stride(root, checkpoint_stride)

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
    if driver_args is None:
        driver_args = {}

    remove_directory(directory)

    os.makedirs(directory, exist_ok=True)

    if isinstance(atoms, list):
        atoms = atoms[-1]

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
    driver = prep_driver(atoms, directory, driver, driver_args)

    run_ipi(directory, server, driver, f"{outfile}.out")
    atoms_out = read_ipi_xyz(os.path.join(directory, f"{outfile}.pos_0.xyz"))
    output_data, output_desc = ipi.read_output(os.path.join(directory, f"{outfile}.out"))
    print("Minimisation complete\n", flush=True)
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
    if xml_in is not None:
        tree = et.parse(xml_in)
    else:
        tree = et.parse(os.path.join(find_nqetools_path(), f"templates/{md_type.upper()}.xml"))
    root = tree.getroot()

    if properties is not None:
        append_properties(root, properties)

    write_xyz(atoms, os.path.join(directory, file_in))

    update_file(root, file_in)

    update_cell(root, atoms)

    update_mass(root, atoms, f_deut=deuterate)

    update_driver(root, atoms, driver)

    update_title(root, outfile)

    update_total_steps(root, total_steps)

    update_temperature(root, temperature)

    update_timestep(root, timestep)

    update_dynamics_splitting(root, splitting)

    update_motion_fix_com(root, fix_com)

    if thermostat is not None:
        add_thermostat_section(root, thermostat=thermostat)

    if n_beads > 1:
        add_trajectory_centroid(root)

        update_nbeads(root, n_beads)

        append_properties(root, ['kinetic_cv{electronvolt}', 'pressure_cv{megapascal}'])
    else:
        append_properties(root, ['kinetic_md{electronvolt}', 'pressure_md{megapascal}'])

    update_stride(root, stride)

    update_checkpoint_stride(root, checkpoint_stride)

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
           n_procs=None,
           properties=None,
           xml_in=None):
    print(f"Running the MD ({md_type}) with the driver: {driver}", flush=True)
    if driver_args is None:
        driver_args = {}
    if n_procs is None:
        n_procs = n_beads
    elif n_procs > n_beads:
        n_procs = n_beads

    remove_directory(directory)

    os.makedirs(directory, exist_ok=True)

    if isinstance(atoms, list):
        atoms = atoms[-1]

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
    driver = prep_driver(atoms, directory, driver, driver_args)
    run_ipi(directory, server, driver, f"{outfile}.out", n=n_procs)
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
    if xml_in is not None:
        tree = et.parse(xml_in)
    else:
        tree = et.parse(os.path.join(find_nqetools_path(), f"templates/{md_type.upper()}.xml"))
    root = tree.getroot()

    append_properties(root, ['ensemble_bias{electronvolt}'])

    if properties is not None:
        append_properties(root, properties)

    write_xyz(atoms, os.path.join(directory, file_in))

    update_file(root, file_in)

    update_cell(root, atoms)

    update_mass(root, atoms, f_deut=deuterate)

    update_driver(root, atoms, driver)

    update_title(root, outfile)

    update_total_steps(root, total_steps)

    update_temperature(root, temperature)

    update_timestep(root, timestep)

    update_dynamics_splitting(root, splitting)

    update_motion_fix_com(root, fix_com)

    if thermostat is not None:
        add_thermostat_section(root, thermostat=thermostat)

    if n_beads > 1:
        add_trajectory_centroid(root)

        update_nbeads(root, n_beads)

        append_properties(root, ['kinetic_cv{electronvolt}', 'pressure_cv{megapascal}'])
    else:
        append_properties(root, ['kinetic_md{electronvolt}', 'pressure_md{megapascal}'])

    add_plumed_xml(root, plumed_extras=plumed_extras)

    update_stride(root, stride)

    update_checkpoint_stride(root, checkpoint_stride)

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
                  n_procs=None,
                  plumed_type="mtd-pos",
                  plumed_args=None,
                  properties=None,
                  xml_in=None):
    print(f"Running the MD ({md_type}) with the driver: {driver}", flush=True)
    if plumed_args is None:
        plumed_args = {'directory': directory,
                       'temperature': temperature}
    else:
        plumed_args['directory'] = directory
        plumed_args['temperature'] = temperature

    if driver_args is None:
        driver_args = {}

    if n_procs is None:
        n_procs = n_beads
    elif n_procs > n_beads:
        n_procs = n_beads

    remove_directory(directory)

    os.makedirs(directory, exist_ok=True)

    if isinstance(atoms, list):
        atoms = atoms[-1]

    plumed_extras = prep_plumed(atoms, plumed_type, plumed_args)

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

    driver = prep_driver(atoms, directory, driver, driver_args)

    run_ipi(directory, server, driver, f"{outfile}.out", n=n_procs)

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
    if xml_in is not None:
        tree = et.parse(xml_in)
    else:
        tree = et.parse(os.path.join(find_nqetools_path(), "templates/PHO.xml"))
    root = tree.getroot()

    if properties is not None:
        append_properties(root, properties)

    write_xyz(atoms, os.path.join(directory, file_in))

    update_file(root, file_in)

    update_cell(root, atoms)

    update_mass(root, atoms, f_deut=deuterate)

    update_driver(root, atoms, driver)

    update_title(root, outfile)

    update_total_steps(root, total_steps)

    update_stride(root, stride)

    update_checkpoint_stride(root, checkpoint_stride)

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

    remove_directory(directory)

    os.makedirs(directory, exist_ok=True)

    if isinstance(atoms, list):
        atoms = atoms[-1]

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
    driver = prep_driver(atoms, directory, driver, driver_args)
    run_ipi(directory, server, driver, f"{outfile}.out")
    print("Phonons complete\n", flush=True)
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
    if xml_in is not None:
        tree = et.parse(xml_in)
    else:
        tree = et.parse(os.path.join(find_nqetools_path(), "templates/TS.xml"))
    root = tree.getroot()

    if properties is not None:
        append_properties(root, properties)

    write_xyz(atoms, os.path.join(directory, file_in))

    update_file(root, file_in)

    update_cell(root, atoms)

    update_mass(root, atoms, f_deut=deuterate)

    update_driver(root, atoms, driver)

    update_title(root, outfile)

    update_total_steps(root, total_steps)

    update_tol(root, tol_energy, tol_force, tol_position)

    update_stride(root, stride)

    update_checkpoint_stride(root, checkpoint_stride)

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

    remove_directory(directory)

    os.makedirs(directory, exist_ok=True)

    if isinstance(atoms, list):
        atoms = atoms[-1]

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
    driver = prep_driver(atoms, directory, driver, driver_args)
    run_ipi(directory, server, driver, f"{outfile}.out")
    atoms_out = read_ipi_xyz(os.path.join(directory, f"{outfile}.pos_0.xyz"))
    output_data, output_desc = ipi.read_output(os.path.join(directory, f"{outfile}.out"))
    print("TS search complete\n", flush=True)
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
    n_doft = 3 * n_atoms

    if xml_in is not None:
        tree = et.parse(xml_in)
    else:
        tree = et.parse(os.path.join(find_nqetools_path(), "templates/INST.xml"))
    root = tree.getroot()

    if properties is not None:
        append_properties(root, properties)

    write_xyz(atoms, os.path.join(directory, file_in))

    update_file(root, file_in)

    update_cell(root, atoms)

    update_mass(root, atoms, f_deut=deuterate)

    update_driver(root, atoms, driver)

    update_title(root, outfile)

    update_total_steps(root, total_steps)

    update_nbeads(root, n_beads)

    update_hessian(root, n_doft, 1)

    update_open_paths(root, n_atoms)

    update_temperature(root, temperature)

    update_tol(root, tol_energy, tol_force, tol_position)

    update_stride(root, stride)

    update_checkpoint_stride(root, checkpoint_stride)

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
                  n_procs=None,
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
    if n_procs is None:
        n_procs = n_beads
    elif n_procs > n_beads:
        n_procs = n_beads

    remove_directory(directory)

    os.makedirs(directory, exist_ok=True)

    if isinstance(atoms, list):
        atoms = atoms[-1]

    copy_hess(get_final_hess(directory_ts), directory)

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

    driver = prep_driver(atoms, directory, driver, driver_args)

    run_ipi(directory, server, driver, f"{outfile}.out", n=n_procs)
    print("Instanton complete\n", flush=True)
    return None
