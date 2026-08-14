"""Preparation and execution of i-PI simulations.

Each calculation type - minimisation, MD, biased MD, phonons, transition
state search and instanton - is exposed as a pair of functions. The
``prep_*_xml`` half loads the matching XML template from ``templates/``,
rewrites it for the system and settings at hand, and writes it alongside
the input geometry. The ``run_*`` half wraps that: it clears and recreates
the run directory, prepares the client driver, launches i-PI, and reads the
results back.

Also provides post-processing entry points that shell out to PLUMED and to
the instanton tools for free energy surfaces, partition functions and
bead interpolation.

Notes
-----
Runs are destructive: every ``run_*`` function removes its target
directory before starting, so results must be read or copied out before
re-running into the same path.
"""

import contextlib
import os
import shutil
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


@contextlib.contextmanager
def working_directory(directory):
    """Temporarily change the working directory, restoring it on the way out.

    The original directory is restored even if the body raises, so a failed run
    does not leave the interpreter pointing at the run directory.

    Parameters
    ----------
    directory : str
        The directory to change into for the duration of the block.

    Yields
    ------
    str
        The directory that was changed into.
    """
    dir_base = os.getcwd()
    os.chdir(directory)
    try:
        yield directory
    finally:
        os.chdir(dir_base)


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
    result = subprocess.run(command.split(), check=False, capture_output=True, text=True)
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
    # Check before changing directory, so a refused run leaves the cwd untouched
    if os.path.exists(os.path.join(directory, outfile)):
        raise FileExistsError(f"Output file {outfile} already exists in {directory}. Skipping the run.")

    n = check_driver_processes(n)

    with working_directory(directory):
        rm_ipi_tmp()
        ipi_proc = subprocess.Popen(server.split())
        time.sleep(t_sleep)

        # Each driver gets its own log; the drivers are not run through a shell,
        # so any redirection has to happen here rather than in the command string
        driver_logs = [open(f"driver_{i}.out", "w") for i in range(n)]
        try:
            driver_proc = [subprocess.Popen(driver.split(), stdout=log, stderr=subprocess.STDOUT)
                           for log in driver_logs]
            ipi_proc.wait()
            for process in driver_proc:
                process.wait()
        finally:
            for log in driver_logs:
                log.close()
    return None


def run_plumed_hills(directory, temperature=300.0, bins=100, stride=100, cv=None):
    """Reconstruct a free energy surface from metadynamics hills.

    Runs ``plumed sum_hills`` over the HILLS file, writing the surface to
    FES in the same directory.

    Parameters
    ----------
    directory : str
        Directory containing HILLS and plumed.dat.
    temperature : float, optional
        Temperature in K, converted to kJ/mol for PLUMED. Default is 300.0.
    bins : int, optional
        Number of grid bins per collective variable. Default is 100.
    stride : int, optional
        Emit an intermediate surface every this many hills, which is what
        makes the convergence plots possible. Default is 100.
    cv : list, optional
        Grid bounds as ``[min, max]`` for one collective variable, or a
        list of such pairs for several. Bounds of None let PLUMED choose
        them. Default is [[0.21, 0.31], [-1, 1]].

    Returns
    -------
    None
    """
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

    with open(os.path.join(directory, "plumed.dat")) as file:
        subprocess.run(command.split(), check=False, stdin=file, text=True)

    print("Plumed hills run complete\n", flush=True)
    return None


def run_plumed_hills_opes(directory, temperature=300.0, bins=100, cv=None):
    """Reconstruct a free energy surface from an OPES state file.

    OPES stores compressed kernels rather than a hill history, so the
    surface is rebuilt with ``opes/fes_from_state.py`` instead of
    ``plumed sum_hills``.

    Parameters
    ----------
    directory : str
        Directory containing the OPES STATE file.
    temperature : float, optional
        Temperature in K, converted to kJ/mol. Default is 300.0.
    bins : int, optional
        Number of grid bins per collective variable. Default is 100.
    cv : list, optional
        Grid bounds as ``[min, max]`` for one collective variable, or a
        list of such pairs for several. Bounds of None let the grid be
        chosen automatically. Default is [[0.21, 0.31], [-1, 1]].

    Returns
    -------
    None

    Notes
    -----
    Every stored state is written out, not just the last, so convergence
    can be checked across the run.
    """
    print('Running plumed OPES hills', flush=True)
    if cv is None:
        cv = [[0.21, 0.31], [-1, 1]]

    kbt = temperature * 0.0083144621

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

    # The FES script takes no path argument, so it finds STATE via the cwd
    with working_directory(directory):
        subprocess.run(command.split(), check=False)

    print("Plumed OPES hills run complete\n", flush=True)
    return None


def run_instanton_post_process(directory,
                               process_type='reactant',
                               temperature=300.0,
                               filter_list=None,
                               ref_energy=None,
                               n_beads=1,
                               outfile='thermo_data.out'):
    """Run the instanton post-processing on a completed calculation.

    Invokes ``instanton_tools/postproc.py`` against the RESTART file to
    extract the partition functions that the rate expressions need.

    Parameters
    ----------
    directory : str
        Directory containing the RESTART file.
    process_type : {"reactant", "TS", "instanton"}, optional
        Which stationary point the calculation describes. Default is
        'reactant'.
    temperature : float, optional
        Temperature in K. Default is 300.0.
    filter_list : list of int, optional
        Zero-based indices of atoms to exclude from the partition
        functions. Default is None.
    ref_energy : float, optional
        Energy zero in eV, normally the reactant energy, so barriers come
        out relative to it. Default is None.
    n_beads : int, optional
        Number of beads in the full polymer, required for the reactant
        case. Default is 1.
    outfile : str, optional
        File to capture stdout into. Default is 'thermo_data.out'.

    Returns
    -------
    None

    Notes
    -----
    Failures are reported on stderr rather than raised, so callers should
    check that the parsed output contains the expected fields.
    """
    print(f'Running the {process_type} thermo/instanton post-processing', flush=True)

    path_to_proc = os.path.join(find_nqetools_path(), "instanton_tools", "postproc.py")
    command = f'python {path_to_proc} RESTART -t {temperature} -c {process_type} -n {n_beads}'
    if filter_list is not None:
        command += f' -f {filter_list}'

    if ref_energy is not None:
        command += f' -e {ref_energy}'

    print(f"Working directory: {directory}", flush=True)
    print(f"Running command:\n{command}", flush=True)

    # RESTART is passed as a bare name, so it resolves against the cwd
    with working_directory(directory), open(outfile, "w") as file:
        result = subprocess.run(command.split(), check=False, stdout=file, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}", flush=True)

    print("Thermo/Instanton post-processing complete\n", flush=True)
    return None


def run_instanton_interpolation(directory_old, directory_new, new_n_beads):
    """Resample a converged instanton onto a larger number of beads.

    Copies the geometry, Hessian and input from a finished run into a new
    directory, interpolates them onto a finer ring polymer, and rewrites
    the XML to match. This is how a bead-convergence sequence is stepped
    up without restarting from the transition state each time.

    Parameters
    ----------
    directory_old : str
        Directory of the completed lower-bead instanton run.
    directory_new : str
        Directory to create and populate for the next run.
    new_n_beads : int
        Number of beads for the new calculation.

    Returns
    -------
    None

    Notes
    -----
    Reimplements the manual recipe from the i-PI PIQM2023 tutorial,
    ``05-RPI/tutorial-4.ipynb``.
    """
    print('Running the instanton interpolation', flush=True)

    os.makedirs(directory_new, exist_ok=True)

    with working_directory(directory_new):
        shutil.copy(os.path.join(directory_old, "init.xyz"), "init0")
        shutil.copy(os.path.join(directory_old, "hessian.dat"), "hess0")

        path_to_proc = os.path.join(find_nqetools_path(), "instanton_tools", "interpolation.py")
        command = f'python {path_to_proc} -m -xyz init0 -hess hess0 -n {new_n_beads}'

        print(f"Running command:\n{command}", flush=True)

        with open("interpolation.out", "w") as file:
            result = subprocess.run(command.split(), check=False, stdout=file, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"Error: {result.stderr}", flush=True)

        shutil.copy("hess0", "hessian.dat")
        shutil.copy("init0", "init.xyz")

        shutil.copy(os.path.join(directory_old, "input.xml"), "input.xml")

        tree = et.parse('input.xml')
        root = tree.getroot()
        update_nbeads(root, new_n_beads)

        atoms = read_ipi_xyz(os.path.join(directory_old, "init.xyz"))
        n_doft = 3 * len(atoms)

        update_hessian(root, n_doft, new_n_beads)

        write_xml(root, 'input.xml')

    print("Instanton interpolation complete\n", flush=True)
    return None


def _prepare_run_directory(directory, atoms, driver_args):
    """Clear the run directory and normalise the arguments every run shares.

    Runs are destructive by design: the directory is removed and recreated so
    a rerun cannot pick up stale output from the last one.

    Parameters
    ----------
    directory : str
        Run directory. Removed and recreated.
    atoms : ase.Atoms or list of ase.Atoms
        Starting structure. A list is taken to be a trajectory, of which the
        last frame is the structure to start from.
    driver_args : dict or None
        Extra keyword arguments for the driver.

    Returns
    -------
    tuple of (ase.Atoms, dict)
        The structure to start from and the driver arguments, with None
        replaced by an empty dict.
    """
    if driver_args is None:
        driver_args = {}

    remove_directory(directory)
    os.makedirs(directory, exist_ok=True)

    if isinstance(atoms, list):
        atoms = atoms[-1]

    return atoms, driver_args


def _collect_ipi_output(directory, outfile, n_beads=1):
    """Read back the trajectory and scalar output i-PI wrote.

    Parameters
    ----------
    directory : str
        Run directory.
    outfile : str
        Prefix i-PI was given for its output files.
    n_beads : int, optional
        Number of beads. Above one the centroid trajectory is read, since the
        individual bead positions are not the configuration of interest.
        Default is 1.

    Returns
    -------
    tuple of (list of ase.Atoms, dict, dict)
        The trajectory, the parsed scalar output, and the descriptions of
        those output columns.
    """
    trajectory = f"{outfile}.xc.xyz" if n_beads > 1 else f"{outfile}.pos_0.xyz"
    atoms_out = read_ipi_xyz(os.path.join(directory, trajectory))
    output_data, output_desc = ipi.read_output(os.path.join(directory, f"{outfile}.out"))
    return atoms_out, output_data, output_desc


def _load_ipi_template(template, xml_in=None):
    """Parse the packaged i-PI template, or `xml_in` in its place.

    Parameters
    ----------
    template : str
        File name of the template in ``templates/``, such as "MIN.xml".
    xml_in : str, optional
        Path to an XML file to use instead of the template. Default is None.

    Returns
    -------
    Element
        Root of the parsed document, ready to be rewritten in place.
    """
    if xml_in is None:
        xml_in = os.path.join(find_nqetools_path(), "templates", template)
    return et.parse(xml_in).getroot()


def _apply_common_updates(root, directory, atoms, file_in, driver, outfile,
                          deuterate, total_steps, properties=None,
                          extra_properties=None):
    """Apply the updates every i-PI input needs, whatever the run type.

    Writes the starting geometry alongside the input, then points the input
    at it and rewrites the cell, masses, driver socket, output prefix and
    step count.

    Parameters
    ----------
    root : Element
        Root of the document to rewrite.
    directory : str
        Directory the structure file is written into.
    atoms : ase.Atoms
        Starting structure.
    file_in : str
        Name to give the structure file.
    driver : str
        Driver name, used to select the force provider.
    outfile : str
        Prefix for i-PI output files.
    deuterate : bool
        If True, replace hydrogen masses with deuterium.
    total_steps : int
        Step limit for the run.
    properties : list of str, optional
        Extra properties to append to the output list. Default is None.
    extra_properties : list of str, optional
        Properties appended before `properties`, for inputs that need one
        regardless of what the caller asked for. Default is None.
    """
    if extra_properties is not None:
        append_properties(root, extra_properties)
    if properties is not None:
        append_properties(root, properties)

    write_xyz(atoms, os.path.join(directory, file_in))

    update_file(root, file_in)
    update_cell(root, atoms)
    update_mass(root, atoms, f_deut=deuterate)
    update_driver(root, atoms, driver)
    update_title(root, outfile)
    update_total_steps(root, total_steps)


def _apply_dynamics_updates(root, temperature, timestep, splitting, fix_com,
                            thermostat, n_beads):
    """Apply the updates shared by the plain and PLUMED-biased MD inputs.

    Parameters
    ----------
    root : Element
        Root of the document to rewrite.
    temperature : float
        Simulation temperature in K.
    timestep : float
        Integration timestep in fs.
    splitting : str
        Dynamics splitting scheme, such as "baoab".
    fix_com : bool
        If True, hold the centre of mass fixed.
    thermostat : str or None
        Name of a thermostat definition in ``thermostats/``, or None to leave
        the template's own thermostat in place.
    n_beads : int
        Number of ring-polymer beads. Above one the centroid trajectory is
        added and the path-integral estimators are used in place of the
        classical ones.
    """
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


def _write_ipi_xml(root, directory, stride, checkpoint_stride):
    """Set the output strides and write the finished input.xml.

    Parameters
    ----------
    root : Element
        Root of the rewritten document.
    directory : str
        Directory to write input.xml into.
    stride : int
        Interval between trajectory writes.
    checkpoint_stride : int
        Interval between checkpoint writes.
    """
    update_stride(root, stride)
    update_checkpoint_stride(root, checkpoint_stride)
    write_xml(root, os.path.join(directory, 'input.xml'))


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
    """Write the i-PI input for a geometry optimisation.

    Parameters
    ----------
    directory : str
        Directory to write input.xml and the structure file into.
    atoms : ase.Atoms
        Starting structure.
    outfile : str, optional
        Prefix for i-PI output files. Default is "min".
    driver : str, optional
        Driver name, used to select the force provider. Default is
        "ase-mace".
    total_steps : int, optional
        Maximum number of optimiser steps. Default is 100.
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    optimiser : str, optional
        Optimisation algorithm. Default is "lbfgs".
    tol_energy : float, optional
        Energy convergence tolerance. Default is 1.0e-4.
    tol_force : float, optional
        Force convergence tolerance. Default is 1.0e-4.
    tol_position : float, optional
        Position convergence tolerance. Default is 1.0e-4.
    stride : int, optional
        Interval between trajectory writes. Default is 1.
    checkpoint_stride : int, optional
        Interval between checkpoint writes. Default is 10.
    properties : list of str, optional
        Extra properties to append to the output list. Default is None.
    xml_in : str, optional
        Path to an XML file to use instead of the MIN.xml template.
        Default is None.
    file_in : str, optional
        Name of the structure file to write. Default is "init.xyz".

    Returns
    -------
    None
    """
    root = _load_ipi_template("MIN.xml", xml_in)

    _apply_common_updates(root, directory, atoms, file_in, driver, outfile,
                          deuterate, total_steps, properties)

    update_optimiser(root, optimiser)
    update_tol(root, tol_energy, tol_force, tol_position)

    _write_ipi_xml(root, directory, stride, checkpoint_stride)


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
    """Run a geometry optimisation and return the relaxed structure.

    Parameters
    ----------
    directory : str
        Run directory. Removed and recreated before the run.
    atoms : ase.Atoms or list of ase.Atoms
        Starting structure. If a list is given, the last frame is used.
    server : str, optional
        Command that starts the i-PI server. Default is "i-pi input.xml".
    outfile : str, optional
        Prefix for i-PI output files. Default is "min".
    driver : str, optional
        Driver name passed to :func:`~nqetools.driver.prep_driver`.
        Default is "ase-mace".
    driver_args : dict, optional
        Extra keyword arguments for the driver. Default is None.
    total_steps : int, optional
        Maximum number of optimiser steps. Default is 100.
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    optimiser : str, optional
        Optimisation algorithm. Default is "lbfgs".
    tol_energy : float, optional
        Energy convergence tolerance. Default is 1.0e-4.
    tol_force : float, optional
        Force convergence tolerance. Default is 1.0e-4.
    tol_position : float, optional
        Position convergence tolerance. Default is 1.0e-4.
    stride : int, optional
        Interval between trajectory writes. Default is 1.
    checkpoint_stride : int, optional
        Interval between checkpoint writes. Default is 1000.
    properties : list of str, optional
        Extra properties to append to the output list. Default is None.
    xml_in : str, optional
        Path to an XML file to use instead of the template. Default is
        None.

    Returns
    -------
    tuple of (list of ase.Atoms, dict, dict)
        The optimisation trajectory, the parsed scalar output, and the
        descriptions of those output columns.
    """
    print(f"Running the minimisation with the driver: {driver}", flush=True)
    atoms, driver_args = _prepare_run_directory(directory, atoms, driver_args)

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

    print("Minimisation complete\n", flush=True)
    return _collect_ipi_output(directory, outfile)


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
    """Write the i-PI input for a molecular dynamics run.

    Parameters
    ----------
    directory : str
        Directory to write input.xml and the structure file into.
    atoms : ase.Atoms
        Starting structure.
    outfile : str, optional
        Prefix for i-PI output files. Default is "md".
    driver : str, optional
        Driver name, used to select the force provider. Default is
        "ase-mace".
    total_steps : int, optional
        Number of MD steps. Default is 1000.
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    temperature : float, optional
        Target temperature in K. Default is 300.0.
    timestep : float, optional
        Integration timestep in fs. Default is 1.
    thermostat : str, optional
        Name of a thermostat XML fragment to splice in, overriding the
        template's own. Default is None.
    md_type : str, optional
        Ensemble, naming the template to load, for example "NVT", "NPT"
        or "NVE". Default is "NVT".
    splitting : str, optional
        Integrator splitting scheme. Default is "baoab".
    fix_com : bool, optional
        If True, remove centre of mass motion. Default is False.
    stride : int, optional
        Interval between trajectory writes. Default is 10.
    checkpoint_stride : int, optional
        Interval between checkpoint writes. Default is 1000.
    n_beads : int, optional
        Ring-polymer beads. Values above 1 select path-integral dynamics,
        which adds centroid output and switches the kinetic and pressure
        estimators to their centroid-virial forms. Default is 1.
    properties : list of str, optional
        Extra properties to append to the output list. Default is None.
    xml_in : str, optional
        Path to an XML file to use instead of the ensemble template.
        Default is None.
    file_in : str, optional
        Name of the structure file to write. Default is "init.xyz".

    Returns
    -------
    None
    """
    root = _load_ipi_template(f"{md_type.upper()}.xml", xml_in)

    _apply_common_updates(root, directory, atoms, file_in, driver, outfile,
                          deuterate, total_steps, properties)

    _apply_dynamics_updates(root, temperature, timestep, splitting, fix_com,
                            thermostat, n_beads)

    _write_ipi_xml(root, directory, stride, checkpoint_stride)


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
    """Run molecular dynamics and return the resulting trajectory.

    Parameters
    ----------
    directory : str
        Run directory. Removed and recreated before the run.
    atoms : ase.Atoms or list of ase.Atoms
        Starting structure. If a list is given, the last frame is used.
    server : str, optional
        Command that starts the i-PI server. Default is "i-pi input.xml".
    outfile : str, optional
        Prefix for i-PI output files. Default is "md".
    driver : str, optional
        Driver name passed to :func:`~nqetools.driver.prep_driver`.
        Default is "ase-mace".
    driver_args : dict, optional
        Extra keyword arguments for the driver. Default is None.
    total_steps : int, optional
        Number of MD steps. Default is 1000.
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    temperature : float, optional
        Target temperature in K. Default is 300.0.
    timestep : float, optional
        Integration timestep in fs. Default is 1.
    thermostat : str, optional
        Name of a thermostat XML fragment to splice in. Default is None.
    md_type : str, optional
        Ensemble, naming the template to load. Default is "NVT".
    splitting : str, optional
        Integrator splitting scheme. Default is "baoab".
    fix_com : bool, optional
        If True, remove centre of mass motion. Default is False.
    stride : int, optional
        Interval between trajectory writes. Default is 10.
    checkpoint_stride : int, optional
        Interval between checkpoint writes. Default is 1000.
    n_beads : int, optional
        Ring-polymer beads. Values above 1 select path-integral dynamics.
        Default is 1.
    n_procs : int, optional
        Number of driver processes. Capped at `n_beads`, since beads are
        the unit of parallel force evaluation. Defaults to `n_beads`.
    properties : list of str, optional
        Extra properties to append to the output list. Default is None.
    xml_in : str, optional
        Path to an XML file to use instead of the template. Default is
        None.

    Returns
    -------
    tuple of (list of ase.Atoms, dict, dict)
        The trajectory, the parsed scalar output, and the descriptions of
        those output columns. For path-integral runs the trajectory holds
        the centroid rather than a single bead.
    """
    print(f"Running the MD ({md_type}) with the driver: {driver}", flush=True)
    atoms, driver_args = _prepare_run_directory(directory, atoms, driver_args)

    if n_procs is None or n_procs > n_beads:
        n_procs = n_beads

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

    print(f"MD ({md_type}) complete\n", flush=True)
    return _collect_ipi_output(directory, outfile, n_beads)


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
    """Write the i-PI input for a PLUMED-biased molecular dynamics run.

    Identical to :func:`prep_md_xml` except that a PLUMED force section is
    spliced in and the bias energy is added to the output list, so the
    accumulated bias can be monitored for convergence.

    Parameters
    ----------
    directory : str
        Directory to write input.xml and the structure file into.
    atoms : ase.Atoms
        Starting structure.
    outfile : str, optional
        Prefix for i-PI output files. Default is "md".
    driver : str, optional
        Driver name, used to select the force provider. Default is
        "ase-mace".
    total_steps : int, optional
        Number of MD steps. Default is 1000.
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    temperature : float, optional
        Target temperature in K. Default is 300.0.
    timestep : float, optional
        Integration timestep in fs. Default is 1.0.
    thermostat : str, optional
        Name of a thermostat XML fragment to splice in. Default is None.
    md_type : str, optional
        Ensemble, naming the template to load. Default is "NVT".
    splitting : str, optional
        Integrator splitting scheme. Default is "baoab".
    fix_com : bool, optional
        If True, remove centre of mass motion. Default is False.
    stride : int, optional
        Interval between trajectory writes. Default is 10.
    checkpoint_stride : int, optional
        Interval between checkpoint writes. Default is 1000.
    n_beads : int, optional
        Ring-polymer beads. Values above 1 select path-integral dynamics.
        Default is 1.
    plumed_extras : dict, optional
        Extra attributes for the PLUMED force section, as returned by
        :func:`~nqetools.plumed.prep_plumed`. Default is None.
    properties : list of str, optional
        Extra properties to append to the output list. Default is None.
    xml_in : str, optional
        Path to an XML file to use instead of the ensemble template.
        Default is None.
    file_in : str, optional
        Name of the structure file to write. Default is "init.xyz".

    Returns
    -------
    None
    """
    root = _load_ipi_template(f"{md_type.upper()}.xml", xml_in)

    # The bias has to be reported whatever else the caller asked for, since
    # it is what the reweighting in post-processing works from.
    _apply_common_updates(root, directory, atoms, file_in, driver, outfile,
                          deuterate, total_steps, properties,
                          extra_properties=['ensemble_bias{electronvolt}'])

    _apply_dynamics_updates(root, temperature, timestep, splitting, fix_com,
                            thermostat, n_beads)

    add_plumed_xml(root, plumed_extras=plumed_extras)

    _write_ipi_xml(root, directory, stride, checkpoint_stride)


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
    """Run PLUMED-biased molecular dynamics and return the trajectory.

    Builds the PLUMED input from `plumed_type` and `plumed_args`, then
    runs MD with that bias applied.

    Parameters
    ----------
    directory : str
        Run directory. Removed and recreated before the run.
    atoms : ase.Atoms or list of ase.Atoms
        Starting structure. If a list is given, the last frame is used.
    server : str, optional
        Command that starts the i-PI server. Default is "i-pi input.xml".
    outfile : str, optional
        Prefix for i-PI output files. Default is "md".
    driver : str, optional
        Driver name passed to :func:`~nqetools.driver.prep_driver`.
        Default is "ase-mace".
    driver_args : dict, optional
        Extra keyword arguments for the driver. Default is None.
    total_steps : int, optional
        Number of MD steps. Default is 1000.
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    temperature : float, optional
        Target temperature in K. Also passed to PLUMED, which needs it to
        set the bias factor. Default is 300.0.
    timestep : float, optional
        Integration timestep in fs. Default is 1.0.
    thermostat : str, optional
        Name of a thermostat XML fragment to splice in. Default is None.
    md_type : str, optional
        Ensemble, naming the template to load. Default is "NVT".
    splitting : str, optional
        Integrator splitting scheme. Default is "baoab".
    fix_com : bool, optional
        If True, remove centre of mass motion. Default is False.
    stride : int, optional
        Interval between trajectory writes. Default is 10.
    checkpoint_stride : int, optional
        Interval between checkpoint writes. Default is 1000.
    n_beads : int, optional
        Ring-polymer beads. Values above 1 select path-integral dynamics.
        Default is 1.
    n_procs : int, optional
        Number of driver processes, capped at `n_beads`. Defaults to
        `n_beads`.
    plumed_type : str, optional
        Which PLUMED input writer to use, for example "mtd-pos" or
        "opes-dist". Default is "mtd-pos".
    plumed_args : dict, optional
        Arguments for that writer. The run directory and temperature are
        always overwritten with the values used here. Default is None.
    properties : list of str, optional
        Extra properties to append to the output list. Default is None.
    xml_in : str, optional
        Path to an XML file to use instead of the template. Default is
        None.

    Returns
    -------
    tuple of (list of ase.Atoms, dict, dict)
        The trajectory, the parsed scalar output, and the descriptions of
        those output columns. For path-integral runs the trajectory holds
        the centroid rather than a single bead.
    """
    print(f"Running the MD ({md_type}) with the driver: {driver}", flush=True)

    if plumed_args is None:
        plumed_args = {}
    plumed_args['directory'] = directory
    plumed_args['temperature'] = temperature

    atoms, driver_args = _prepare_run_directory(directory, atoms, driver_args)

    if n_procs is None or n_procs > n_beads:
        n_procs = n_beads

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

    print(f"MD ({md_type}) complete\n", flush=True)
    return _collect_ipi_output(directory, outfile, n_beads)


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
    """Write the i-PI input for a phonon calculation.

    Parameters
    ----------
    directory : str
        Directory to write input.xml and the structure file into.
    atoms : ase.Atoms
        Structure to displace, which should already be at a minimum.
    outfile : str, optional
        Prefix for i-PI output files. Default is "phonon".
    driver : str, optional
        Driver name, used to select the force provider. Default is
        "ase-mace".
    total_steps : int, optional
        Maximum number of displacement steps. Default is 1000.
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium, which shifts the
        frequencies without changing the underlying force constants.
        Default is False.
    stride : int, optional
        Interval between trajectory writes. Default is 1.
    checkpoint_stride : int, optional
        Interval between checkpoint writes. Default is 1000.
    properties : list of str, optional
        Extra properties to append to the output list. Default is None.
    xml_in : str, optional
        Path to an XML file to use instead of the PHO.xml template.
        Default is None.
    file_in : str, optional
        Name of the structure file to write. Default is "init.xyz".

    Returns
    -------
    None
    """
    root = _load_ipi_template("PHO.xml", xml_in)

    _apply_common_updates(root, directory, atoms, file_in, driver, outfile,
                          deuterate, total_steps, properties)

    _write_ipi_xml(root, directory, stride, checkpoint_stride)


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
    """Run a phonon calculation to obtain the dynamical matrix.

    Parameters
    ----------
    directory : str
        Run directory. Removed and recreated before the run.
    atoms : ase.Atoms or list of ase.Atoms
        Structure to displace, which should already be at a minimum. If a
        list is given, the last frame is used.
    server : str, optional
        Command that starts the i-PI server. Default is "i-pi input.xml".
    outfile : str, optional
        Prefix for i-PI output files. Default is "phonon".
    driver : str, optional
        Driver name passed to :func:`~nqetools.driver.prep_driver`.
        Default is "ase-mace".
    driver_args : dict, optional
        Extra keyword arguments for the driver. Default is None.
    total_steps : int, optional
        Maximum number of displacement steps. Default is 1000.
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    stride : int, optional
        Interval between trajectory writes. Default is 1.
    checkpoint_stride : int, optional
        Interval between checkpoint writes. Default is 1000.
    properties : list of str, optional
        Extra properties to append to the output list. Default is None.
    xml_in : str, optional
        Path to an XML file to use instead of the template. Default is
        None.

    Returns
    -------
    None
        Results are left in the run directory for the post-processing
        step to pick up.
    """
    print(f"Running the phonons with the driver: {driver}", flush=True)
    atoms, driver_args = _prepare_run_directory(directory, atoms, driver_args)

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
    """Write the i-PI input for a transition state search.

    Parameters
    ----------
    directory : str
        Directory to write input.xml and the structure file into.
    atoms : ase.Atoms
        Starting guess for the saddle point.
    outfile : str, optional
        Prefix for i-PI output files. Default is "ts".
    driver : str, optional
        Driver name, used to select the force provider. Default is
        "ase-mace".
    total_steps : int, optional
        Maximum number of search steps. Default is 1000.
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    tol_energy : float, optional
        Energy convergence tolerance. Default is 5.0e-6.
    tol_force : float, optional
        Force convergence tolerance. Default is 5.0e-6.
    tol_position : float, optional
        Position convergence tolerance. Default is 1.0e-6.
    stride : int, optional
        Interval between trajectory writes. Default is 1.
    checkpoint_stride : int, optional
        Interval between checkpoint writes. Default is 1000.
    properties : list of str, optional
        Extra properties to append to the output list. Default is None.
    xml_in : str, optional
        Path to an XML file to use instead of the TS.xml template.
        Default is None.
    file_in : str, optional
        Name of the structure file to write. Default is "init.xyz".

    Returns
    -------
    None

    Notes
    -----
    The tolerances are two orders of magnitude tighter than for a
    minimisation, because the Hessian produced here seeds the instanton
    search and a loosely converged saddle gives a poor starting orbit.
    """
    root = _load_ipi_template("TS.xml", xml_in)

    _apply_common_updates(root, directory, atoms, file_in, driver, outfile,
                          deuterate, total_steps, properties)

    update_tol(root, tol_energy, tol_force, tol_position)

    _write_ipi_xml(root, directory, stride, checkpoint_stride)


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
    """Run a transition state search and return the located saddle point.

    Parameters
    ----------
    directory : str
        Run directory. Removed and recreated before the run.
    atoms : ase.Atoms or list of ase.Atoms
        Starting guess for the saddle point. If a list is given, the last
        frame is used.
    server : str, optional
        Command that starts the i-PI server. Default is "i-pi input.xml".
    outfile : str, optional
        Prefix for i-PI output files. Default is "ts".
    driver : str, optional
        Driver name passed to :func:`~nqetools.driver.prep_driver`.
        Default is "ase-mace".
    driver_args : dict, optional
        Extra keyword arguments for the driver. Default is None.
    total_steps : int, optional
        Maximum number of search steps. Default is 1000.
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    tol_energy : float, optional
        Energy convergence tolerance. Default is 5.0e-6.
    tol_force : float, optional
        Force convergence tolerance. Default is 5.0e-6.
    tol_position : float, optional
        Position convergence tolerance. Default is 1.0e-6.
    stride : int, optional
        Interval between trajectory writes. Default is 1.
    checkpoint_stride : int, optional
        Interval between checkpoint writes. Default is 1000.
    properties : list of str, optional
        Extra properties to append to the output list. Default is None.
    xml_in : str, optional
        Path to an XML file to use instead of the template. Default is
        None.

    Returns
    -------
    tuple of (list of ase.Atoms, dict, dict)
        The search trajectory, the parsed scalar output, and the
        descriptions of those output columns.

    Notes
    -----
    The Hessian left in the run directory is what :func:`run_instanton`
    reads to start its own optimisation.
    """
    print(f"Running the transition state search with the driver: {driver}", flush=True)
    atoms, driver_args = _prepare_run_directory(directory, atoms, driver_args)

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

    print("TS search complete\n", flush=True)
    return _collect_ipi_output(directory, outfile)


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
    """Write the i-PI input for an instanton optimisation.

    Parameters
    ----------
    directory : str
        Directory to write input.xml and the structure file into. A
        Hessian from a transition state search must already be present.
    atoms : ase.Atoms
        Transition state structure, used as the starting orbit.
    outfile : str, optional
        Prefix for i-PI output files. Default is "instanton".
    driver : str, optional
        Driver name, used to select the force provider. Default is
        "ase-mace".
    total_steps : int, optional
        Maximum number of optimiser steps. Default is 1000.
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium, which is how the
        kinetic isotope effect is obtained. Default is False.
    n_beads : int, optional
        Number of ring-polymer beads in the half polymer. Default is 4.
    temperature : float, optional
        Temperature in K, which sets the imaginary-time period and so the
        extent of the instanton orbit. Default is 300.0.
    tol_energy : float, optional
        Energy convergence tolerance. Default is 5.0e-6.
    tol_force : float, optional
        Force convergence tolerance. Default is 5.0e-6.
    tol_position : float, optional
        Position convergence tolerance. Default is 1.0e-6.
    stride : int, optional
        Interval between trajectory writes. Default is 1.
    checkpoint_stride : int, optional
        Interval between checkpoint writes. Default is 1000.
    properties : list of str, optional
        Extra properties to append to the output list. Default is None.
    xml_in : str, optional
        Path to an XML file to use instead of the INST.xml template.
        Default is None.
    file_in : str, optional
        Name of the structure file to write. Default is "init.xyz".

    Returns
    -------
    None

    Notes
    -----
    Open paths are enabled for every atom, since the instanton orbit is a
    linear polymer with free ends rather than a closed ring.
    """
    n_atoms = len(atoms)
    n_doft = 3 * n_atoms

    root = _load_ipi_template("INST.xml", xml_in)

    _apply_common_updates(root, directory, atoms, file_in, driver, outfile,
                          deuterate, total_steps, properties)

    update_nbeads(root, n_beads)
    # The hessian starts as the single-bead one copied in from the transition
    # state search; i-PI grows it to the full ring polymer itself.
    update_hessian(root, n_doft, 1)
    update_open_paths(root, n_atoms)
    update_temperature(root, temperature)
    update_tol(root, tol_energy, tol_force, tol_position)

    _write_ipi_xml(root, directory, stride, checkpoint_stride)


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
    """Run an instanton optimisation, seeded from a transition state.

    Copies the final Hessian out of a completed transition state search
    into the new run directory, then optimises the ring-polymer orbit.

    Parameters
    ----------
    directory : str
        Run directory. Removed and recreated before the run.
    atoms : ase.Atoms or list of ase.Atoms
        Transition state structure. If a list is given, the last frame is
        used.
    directory_ts : str
        Directory of the completed transition state search, supplying the
        starting Hessian.
    server : str, optional
        Command that starts the i-PI server. Default is "i-pi input.xml".
    outfile : str, optional
        Prefix for i-PI output files. Default is "instanton".
    driver : str, optional
        Driver name passed to :func:`~nqetools.driver.prep_driver`.
        Default is "ase-mace".
    driver_args : dict, optional
        Extra keyword arguments for the driver. Default is None.
    total_steps : int, optional
        Maximum number of optimiser steps. Default is 1000.
    deuterate : bool, optional
        If True, replace hydrogen masses with deuterium. Default is False.
    n_beads : int, optional
        Number of ring-polymer beads in the half polymer. Default is 4.
    n_procs : int, optional
        Number of driver processes, capped at `n_beads`. Defaults to
        `n_beads`.
    temperature : float, optional
        Temperature in K. Default is 300.0.
    tol_energy : float, optional
        Energy convergence tolerance. Default is 5.0e-6.
    tol_force : float, optional
        Force convergence tolerance. Default is 5.0e-6.
    tol_position : float, optional
        Position convergence tolerance. Default is 1.0e-6.
    stride : int, optional
        Interval between trajectory writes. Default is 1.
    checkpoint_stride : int, optional
        Interval between checkpoint writes. Default is 1000.
    properties : list of str, optional
        Extra properties to append to the output list. Default is None.
    xml_in : str, optional
        Path to an XML file to use instead of the template. Default is
        None.

    Returns
    -------
    None
        Results are left in the run directory for
        :func:`run_instanton_post_process` to pick up.
    """
    print(f"Running the instanton with the driver: {driver}", flush=True)
    atoms, driver_args = _prepare_run_directory(directory, atoms, driver_args)

    if n_procs is None or n_procs > n_beads:
        n_procs = n_beads

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
