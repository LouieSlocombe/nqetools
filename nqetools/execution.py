import os
import subprocess
import time
import xml.etree.ElementTree as ET

import ase.io

from nqetools.driver import prep_driver
from nqetools.io import write_xml, copy_xyz, get_final_xyz, copy_hess, get_final_hess, write_xyz
from nqetools.tools import rm_ipi_tmp
from nqetools.xml import update_cell, update_mass, update_driver, update_title, \
    update_total_steps, update_optimizer, update_tol, update_nbeads, update_hessian, \
    update_open_paths, update_temperature


def run_ipi(server,
            driver,
            outfile,
            n=1,
            t_sleep=5.0,
            cwd=None):
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


def ipi_prep_optimise(directory,
                      atoms,
                      outfile="min",
                      f_driver="ase-mace",
                      total_steps=1000,
                      f_deut=False,
                      optimizer="cg",
                      tol_energy=5.0e-6,
                      tol_force=5.0e-6,
                      tol_position=1.0e-6):
    # Prepare the minimization xml file
    min_str = """
    <simulation mode="static" verbosity="low">
        <output prefix='min'>
            <properties stride='1' filename='out'>[step, potential{electronvolt}]</properties>
            <trajectory stride="1" filename="pos" cell_units='angstrom' format="xyz">x_centroid{angstrom}</trajectory>
        </output>
        <total_steps>400</total_steps>
        <ffsocket name="cbe" mode="unix">
            <address>localhost</address>
        </ffsocket>
        <system>
            <initialize nbeads='1'>
                <file mode='xyz' units='angstrom'> init.xyz </file>
                <cell mode="abc" units='angstrom'>[100 100 100]</cell>
            </initialize>
            <forces>
                <force forcefield="cbe"></force>
            </forces>
            <motion mode="minimize">
                <optimizer mode="sd">
                    <tolerances>
                        <energy>5e-5</energy>
                        <force>5e-5</force>
                        <position>5e-5</position>
                    </tolerances>
                </optimizer>
            </motion>
        </system>
    </simulation>
    """
    # Parse the string
    root = ET.fromstring(min_str)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, deut=f_deut)

    # Change the driver ff
    update_driver(root, f_driver)

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


def ipi_run_optimise(directory,
                     atoms,
                     server="i-pi input.xml",
                     outfile="min",
                     f_driver="ase-mace",
                     driver_dict=None,
                     total_steps=1000,
                     f_deut=False,
                     optimizer="cg",
                     tol_energy=5.0e-6,
                     tol_force=5.0e-6,
                     tol_position=1.0e-6):
    # Prepare the minimization xml file
    if driver_dict is None:
        driver_dict = {}
    ipi_prep_optimise(directory,
                      atoms,
                      outfile=outfile,
                      f_driver=f_driver,
                      total_steps=total_steps,
                      f_deut=f_deut,
                      optimizer=optimizer,
                      tol_energy=tol_energy,
                      tol_force=tol_force,
                      tol_position=tol_position)
    # Prepare the driver
    driver = prep_driver(f_driver, driver_dict)

    # Run the minimization
    print(f"Running the minimization with the driver: {driver}", flush=True)
    run_ipi(server, driver, outfile + ".out", cwd=directory)

    return None


def ipi_prep_phonons(directory,
                     atoms,
                     outfile="phonon",
                     f_driver="ase-mace",
                     total_steps=1000,
                     f_deut=False):
    # Prepare the phonon xml file
    pho_str = """
    <simulation mode="static" verbosity="low">
        <output prefix='phonon'>
            <properties stride='1' filename='out'>[step, potential{electronvolt}]</properties>
            <trajectory stride="1" filename="pos" cell_units='angstrom' format="xyz">x_centroid{angstrom}</trajectory>
        </output>
        <total_steps>400</total_steps>
        <ffsocket name="cbe" mode="unix">
            <address>localhost</address>
        </ffsocket>
        <system>
            <initialize nbeads='1'>
                <file mode='xyz' units='angstrom'> init.xyz </file>
                <cell mode="abc" units='angstrom'>[100 100 100]</cell>
            </initialize>
            <forces>
                <force forcefield="cbe"></force>
            </forces>
            <motion mode="vibrations">
                <vibrations mode="fd">
                    <pos_shift>0.01</pos_shift>
                    <prefix>phonons</prefix>
                    <asr>poly</asr>
                </vibrations>
            </motion>
        </system>
    </simulation>
    """
    # Parse the string
    root = ET.fromstring(pho_str)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, deut=f_deut)

    # Change the driver ff
    update_driver(root, f_driver)

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
                    f_driver="ase-mace",
                    driver_dict=None,
                    total_steps=1000,
                    f_deut=False):
    if driver_dict is None:
        driver_dict = {}
    # Get the directory and the file name
    dir_react_min, outfile_min = os.path.split(min_file_path)
    # Prepare the phonon xyz file
    copy_xyz(get_final_xyz(dir_react_min, sub=f"{outfile_min}*"), directory)
    # Prepare the phonon xml file
    ipi_prep_phonons(directory,
                     atoms,
                     outfile=outfile,
                     f_driver=f_driver,
                     total_steps=total_steps,
                     f_deut=f_deut)
    # Prepare the driver
    driver = prep_driver(f_driver, driver_dict)
    # Run the phonons
    run_ipi(server, driver, outfile + ".out", cwd=directory)


def ipi_prep_ts(directory,
                atoms,
                outfile="ts",
                f_driver="ase-mace",
                total_steps=1000,
                f_deut=False,
                tol_energy=5.0e-6,
                tol_force=5.0e-6,
                tol_position=1.0e-6):
    # Prepare the transition state search xml file
    ts_str = """
    <simulation mode="static" verbosity="low">
        <output prefix='ts'>
            <properties stride='1' filename='out'>[step, potential{electronvolt}]</properties>
            <trajectory stride='1' filename='pos' cell_units='angstrom' format='xyz'>positions{angstrom}</trajectory>
        </output>
        <total_steps>20</total_steps>
        <ffsocket name="cbe" mode="unix">
            <address>localhost</address>
        </ffsocket>
        <system>
            <initialize nbeads='1'>
                <file mode='xyz' units='angstrom'> init.xyz </file>
                <cell mode="abc" units='angstrom'>[100 100 100]</cell>
            </initialize>
            <forces>
                <force forcefield="cbe"></force>
            </forces>
            <motion mode='instanton'>
                <instanton mode='rate'>
                    <tolerances>
                        <energy>5e-6</energy>
                        <force>5e-6</force>
                        <position>1e-3</position>
                    </tolerances>
                    <alt_out>-1</alt_out>
                    <hessian_update>powell</hessian_update>
                    <hessian_asr>poly</hessian_asr>
                    <hessian_init>true</hessian_init>
                    <hessian_final>true</hessian_final>
                    <biggest_step>0.3</biggest_step>
                </instanton>
            </motion>
        </system>
    </simulation>
    """
    # Parse the string
    root = ET.fromstring(ts_str)

    # fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, deut=f_deut)

    # Change the driver ff
    update_driver(root, f_driver)

    # Change the title
    update_title(root, outfile)

    # Update the total_steps
    update_total_steps(root, total_steps)

    # Update the tolerances
    update_tol(root, tol_energy, tol_force, tol_position)

    # Write the file
    write_xml(root, f'{directory}input.xml')
    return None


def ipi_run_ts(directory,
               atoms,
               server="i-pi input.xml",
               outfile="ts",
               f_driver="ase-mace",
               driver_dict=None,
               total_steps=1000,
               f_deut=False,
               tol_energy=5.0e-6,
               tol_force=5.0e-6,
               tol_position=1.0e-6
               ):
    if driver_dict is None:
        driver_dict = {}
    # Read the transition state and write it to the ts directory
    write_xyz(ase.io.read(f"ts.xyz", "-1"), os.path.join(directory, "init.xyz"))
    # Prepare the ts xml file
    ipi_prep_ts(directory,
                atoms,
                outfile=outfile,
                f_driver=f_driver,
                total_steps=total_steps,
                f_deut=f_deut,
                tol_energy=tol_energy,
                tol_force=tol_force,
                tol_position=tol_position)
    # Prepare the driver
    driver = prep_driver(f_driver, driver_dict)
    # Run the ts
    run_ipi(server, driver, outfile + ".out", cwd=directory)


def ipi_prep_inst(directory,
                  atoms,
                  outfile="inst",
                  f_driver="ase-mace",
                  total_steps=1000,
                  f_deut=False,
                  n_beads=4,
                  temperature=300.0,
                  tol_energy=5.0e-6,
                  tol_force=5.0e-6,
                  tol_position=1.0e-6):
    n_atoms = len(atoms)
    n_dof = 3 * n_atoms - 6
    n_doft = 3 * n_atoms

    # Prepare the instanton calculation
    inst_str = """
    <simulation mode="static" verbosity="low">
        <output prefix='inst'>
            <properties stride='1' filename='out'>[step, potential{electronvolt}]</properties>
        </output>
        <total_steps>20</total_steps>
        <ffsocket name="cbe" mode="unix">
            <address>localhost</address>
        </ffsocket>
        <system>
            <initialize nbeads='4'>
                <file mode='xyz' units='angstrom'> init.xyz </file>
                <cell mode="abc" units='angstrom'>[100 100 100]</cell>
            </initialize>
            <forces>
                <force forcefield="cbe"></force>
            </forces>
            <ensemble>
                <temperature units="kelvin">300</temperature>
            </ensemble>
            <normal_modes>
                <open_paths>[0, 1, 2, 3, 4, 5]</open_paths>
            </normal_modes>
            <motion mode='instanton'>
                <instanton mode='rate'>
                    <alt_out>10</alt_out>
                    <tolerances>
                        <energy>5e-6</energy>
                        <force>5e-6</force>
                        <position>1e-3</position>
                    </tolerances>
                    <delta>0.1</delta>
                    <opt>nichols</opt>
                    <hessian_update>powell</hessian_update>
                    <hessian_asr>poly</hessian_asr>
                    <hessian_final>true</hessian_final>
                    <biggest_step>0.3</biggest_step>
                    <hessian mode='file' shape='(18, 18)'>hessian.dat</hessian>
                </instanton>
            </motion>
        </system>
    </simulation>
    """

    # Parse the string
    root = ET.fromstring(inst_str)

    # Fix the cell
    update_cell(root, atoms)

    # Fix the masses
    update_mass(root, atoms, deut=f_deut)

    # Change the driver ff
    update_driver(root, f_driver)

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


def ipi_run_inst(directory,
                 atoms,
                 directory_ts,
                 server="i-pi input.xml",
                 outfile="inst",
                 f_driver="ase-mace",
                 driver_dict=None,
                 total_steps=1000,
                 f_deut=False,
                 n_beads=4,
                 temperature=300.0,
                 tol_energy=5.0e-6,
                 tol_force=5.0e-6,
                 tol_position=1.0e-6):
    if driver_dict is None:
        driver_dict = {}
    # Copy the files from the ts calculation
    copy_xyz(get_final_xyz(directory_ts), directory)
    copy_hess(get_final_hess(directory_ts), directory)
    # Prepare the instanton calculation
    ipi_prep_inst(directory,
                  atoms,
                  outfile=outfile,
                  f_driver=f_driver,
                  total_steps=total_steps,
                  f_deut=f_deut,
                  n_beads=n_beads,
                  temperature=temperature,
                  tol_energy=tol_energy,
                  tol_force=tol_force,
                  tol_position=tol_position)
    # Prepare the driver
    driver = prep_driver(f_driver, driver_dict)
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
#     ipi_in_update_driver(root, f_driver)
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
