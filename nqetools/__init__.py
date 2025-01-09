from .calcs import (correlate,
                    autocorrelate,
                    moving_average,
                    freq_from_eigvals,
                    temp_cross,
                    calculate_nbeads,
                    calc_kappa)

from .driver import (write_mace_driver,
                     write_nwchem_driver,
                     write_orca_driver,
                     prep_driver)

from .execution import (run_ipi,
                        run_optimise,
                        ipi_run_phonons,
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
                 copy_hess)

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

from .plotting import (plot_time_energy,
                       plot_time_temperature,
                       plot_energy_contour_series,
                       plot_energy_contour_compare,
                       plot_energy_sep,
                       n_plot,
                       ax_plot,
                       plot_neb,
                       show_atoms)

from .post_proc import (instanton_postproc)

from .tools import (add_ipi_paths,
                    rm_ipi_tmp,
                    has_pbc,
                    add_hydrogen_halfway,
                    add_hydrogen_at_distance,
                    swap_bonding_configuration,
                    get_fmax,
                    move_atom_halfway,
                    optimise_atom_halfway,
                    )

from .xml_parse import (update_mass,
                        update_cell,
                        update_driver,
                        update_nbeads,
                        update_hessian,
                        update_temperature,
                        update_title,
                        update_total_steps,
                        update_optimizer,
                        update_tol,
                        update_open_paths)

from .qchem_mod import (QChem)

from .calculators import (nwchem_calc_preset,
                          orca_calc_preset,
                          qchem_calc_preset,
                          orca_preset_dft_cheap,
                          orca_preset_dft_gold,
                          orca_preset_xtb,
                          orca_preset_mp2_gold,
                          orca_preset_ccsd_gold,
                          )

__version__ = "0.1.0"
