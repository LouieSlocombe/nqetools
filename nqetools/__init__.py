from .calcs import (correlate,
                    autocorrelate,
                    moving_average,
                    freq_from_eigvals,
                    calculate_temperature_crossover,
                    calculate_good_nbeads,
                    calc_kappa)

from .calculators import (nwchem_calc_preset,
                          orca_calc_preset,
                          qchem_calc_preset,
                          orca_preset_dft_cheap,
                          orca_preset_dft_gold,
                          orca_preset_xtb,
                          orca_preset_mp2_gold,
                          orca_preset_ccsd_gold)

from .conversions import (bohr_to_angstrom,
                          A_to_nm,
                          eV_to_J,
                          J_to_kJ,
                          avo_num,
                          eV_to_kJpermol,
                          eVperA2_to_kJpermolpernm2)

from .driver import (write_ase_mace_driver,
                     write_ase_nwchem_driver,
                     write_ase_orca_driver,
                     prep_driver)

from .execution import (run_ipi,
                        run_md,
                        run_plumed_md,
                        run_optimise,
                        run_phonons,
                        run_ts,
                        run_inst)

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
                 search_fes_files)

from .pathway import (get_neb_path,
                      stitch_path,
                      resample_path,
                      optimise_geom,
                      optimise_reactant_product,
                      prepare_neb,
                      optimise_neb,
                      get_ts_image,
                      optimise_ts,
                      optimise_irc,
                      get_vibrations)

from .plotting import (plot_time_potential_bias,
                       plot_time_temperature,
                       plot_energy_contour_series,
                       plot_energy_contour_compare,
                       plot_energy_sep,
                       n_plot,
                       ax_plot,
                       plot_neb,
                       show_atoms)

from .plumed import (prep_plumed,
                     write_plumed_mtd_pos,
                     write_plumed_opes_pos,
                     write_plumed_mtd_coord,
                     write_plumed_opes_coord,
                     write_plumed_mtd_dists,
                     write_plumed_opes_dists)

from .post_proc import (instanton_postproc)

from .qchem_mod import (QChem)

from .tools import (add_ipi_paths,
                    rm_ipi_tmp,
                    has_pbc,
                    add_hydrogen_halfway,
                    add_hydrogen_at_distance,
                    swap_bonding_configuration,
                    get_fmax,
                    move_atom_halfway,
                    optimise_atom_halfway,
                    round_sf,
                    get_file_extension,
                    cluster_atoms,
                    move_clusters_to_distance,
                    reindex_atoms_by_cluster,
                    move_com_to_origin,
                    move_to_distances,
                    get_ipi_driver,
                    )

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
                        update_optimizer,
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
                        update_motion_fixcom)

__version__ = "0.1.0"
