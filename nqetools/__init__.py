"""Tools for nuclear quantum effect calculations.

Ties together i-PI, PLUMED, OpenMM and several quantum chemistry codes
behind one interface, for studying reactions where the nuclei cannot be
treated classically - chiefly proton transfer, where tunnelling and
zero-point energy shift both the rate and its isotope dependence.

The package covers the whole path: preparing structures
(:mod:`~nqetools.io`, :mod:`~nqetools.tools`), choosing a level of theory
(:mod:`~nqetools.calculators`, :mod:`~nqetools.driver`), running
minimisations, dynamics and instanton searches
(:mod:`~nqetools.execution`), biasing them
(:mod:`~nqetools.plumed`), and analysing the results
(:mod:`~nqetools.calcs`, :mod:`~nqetools.instanton`,
:mod:`~nqetools.plotting`).

Rates can be obtained two ways: approximate tunnelling corrections from
:mod:`~nqetools.calcs`, or full ring-polymer instanton theory from
:mod:`~nqetools.instanton`. Condensed-phase path-integral dynamics is
handled separately through :mod:`~nqetools.openmm`.

Reaction-path work is not here. Nudged elastic bands, saddle-point and IRC
searches, ORCA calculators and free-energy surfaces all live in
`reactiontools <https://github.com/LouieSlocombe/reactiontools>`_, which
``nqetools`` installs as a dependency::

    import reactiontools as rt

    calc = rt.orca_calc_preset(**rt.orca_preset_dft_gold)
    fes = rt.as_fes("FES0.dat", source_unit="kJ/mol", energy_unit="eV")

Some of those were previously re-exported from ``nqetools`` itself, as
``nqe.orca_calc_preset``, ``nqe.load_fes_data``, ``nqe.plot_fes_series_1d``
and friends. They are no longer, so that there is one copy of the code
rather than two drifting apart.
"""

from .calcs import (correlate,
                    autocorrelate,
                    moving_average,
                    freq_from_eigvals,
                    calculate_temperature_crossover,
                    calculate_good_nbeads,
                    wigner_correction,
                    bell_correction,
                    eckart_correction)
from .calculators import (nwchem_calc_preset,
                          qchem_calc_preset,
                          calculate_vib_spectrum,
                          calculate_ccsd_energy,
                          calculate_free_energy,
                          calculate_hessian)
from .conversions import (bohr_to_angstrom,
                          A_to_nm,
                          eV_to_J,
                          J_to_kJ,
                          avo_num,
                          eV_to_kJpermol,
                          eVperA2_to_kJpermolpernm2,
                          convert_atom_list_bohr_to_angstrom)
from .driver import (write_ase_mace_driver,
                     write_ase_nwchem_driver,
                     write_ase_orca_driver,
                     prep_driver)
from .execution import (run_command,
                        run_ipi,
                        run_md,
                        run_plumed_md,
                        run_optimise,
                        run_phonons,
                        run_ts,
                        run_instanton,
                        run_plumed_hills,
                        run_plumed_hills_opes,
                        run_instanton_post_process,
                        run_instanton_interpolation)
from .instanton import (parse_inst_thermo_data,
                        parse_ts_thermo_data,
                        calc_instanton_kappa,
                        calc_kappa_full,
                        parse_react_thermo_data,
                        calc_forward_rate,
                        calc_forward_rate_orca,
                        exp_decay,
                        fit_exp_decay,
                        extrapolate_inf_bead_limit)
from .io import (read_ipi_xyz,
                 read_ipi_output,
                 write_xml,
                 write_xyz,
                 remove_directory,
                 copy_and_rename_file,
                 list_files_with_pattern,
                 get_final_xyz,
                 get_final_hess,
                 copy_xyz,
                 copy_hess,
                 find_nqetools_path,
                 xyz_to_sdf,
                 extract_nonstandard_res,
                 get_non_standard_residues,
                 list_non_standard_residues,
                 clean_ions_in_pdb,
                 relabel_residues_in_pdb,
                 remove_residues_in_pdb,
                 remove_water_residues_in_pdb)
from .openmm import (fix_pdb,
                     zero_velocities,
                     write_multimodel_pdb,
                     centroid_positions,
                     init_beads,
                     get_thermal_de_broglie_wavelength,
                     init_beads_scaled,
                     md_workflow,
                     md_analysis,
                     make_sdf,
                     pdb_patcher,
                     combine_sdf_pdb,
                     prepare_lig_system,
                     prepare_ligand_ff,
                     deuterate_system,
                     get_atoms_in_residue,
                     save_pdb_selection,
                     run_openmm_relaxation,
                     run_openmm_relaxation_simple,
                     run_openmm_heating,
                     run_openmm_npt,
                     run_openmm_prod,
                     run_openmm_rpmd_equilibration,
                     run_openmm_rpmd_contracted,
                     run_openmm_rpmd_prod,
                     RPMDQuantumSpreadReporter,
                     RPMDBeadReporter,
                     RPMDCentroidReporter,
                     count_dna_and_estimate_charge)
from .plotting import (plot_step_energy,
                       plot_time_potential_bias,
                       plot_time_temperature,
                       plot_time_energy_conservation,
                       plot_arrhenius,
                       plot_arrhenius_2,
                       plot_kappa_temperature,
                       plot_kappa_temperature_inv,
                       plot_kie_temperature,
                       plot_bead_convergence,
                       plot_plumed_field)
from .plumed import (prep_plumed,
                     write_plumed_mtd_pos,
                     write_plumed_opes_pos,
                     write_plumed_mtd_coord,
                     write_plumed_opes_coord,
                     write_plumed_mtd_dists,
                     write_plumed_opes_dists)
from .qchem_mod import (QChem)
from .tools import (add_ipi_paths,
                    rm_ipi_tmp,
                    has_pbc,
                    add_hydrogen_halfway,
                    add_hydrogen_at_distance,
                    move_atom_halfway,
                    optimise_atom_halfway,
                    round_sf,
                    get_file_extension,
                    cluster_atoms,
                    cluster_non_hydrogen_atoms,
                    move_clusters_to_distance,
                    reindex_atoms_by_cluster,
                    move_com_to_origin,
                    move_to_distances,
                    get_ipi_driver,
                    get_fes_times,
                    make_dimer,
                    convert_code_to_string,
                    align_principal_axis,
                    align_mols,
                    get_distance,
                    closest_corresponding_index,
                    combine_without_overlaps,
                    largest_bonded_cluster_indices)
from .xml_parse import (update_properties,
                        append_properties,
                        update_mass,
                        update_file,
                        update_cell,
                        update_driver,
                        update_nbeads,
                        update_hessian,
                        update_temperature,
                        update_title,
                        update_total_steps,
                        update_optimiser,
                        update_tol,
                        update_open_paths,
                        update_timestep,
                        update_stride,
                        update_checkpoint_stride,
                        add_trajectory_centroid,
                        add_trajectory_plumed_extras,
                        add_plumed_ff_section,
                        add_plumed_bias_section,
                        add_plumed_xml,
                        add_trajectory_file,
                        add_thermostat_section,
                        update_dynamics_splitting,
                        update_motion_fix_com)

__version__ = "0.1.0"
