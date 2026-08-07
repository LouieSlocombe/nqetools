import os

import numpy as np

import nqetools as nqe


def test_parse_react_thermo_data():
    # Zundel system
    print(flush=True)
    temperature = 300.0
    directory = 'data/instanton/react_phonon/'
    nqe.run_instanton_post_process(directory,
                                   process_type='reactant',
                                   temperature=temperature)
    data = nqe.parse_react_thermo_data(directory)
    print(data, flush=True)

    os.remove(os.path.join(directory, 'eigenvalues_reactant.dat'))
    os.remove(os.path.join(directory, 'thermo_data.out'))
    os.remove(os.path.join(directory, 'freq.dat'))

    assert np.allclose(data['Qtras'], 32.619, rtol=1e-3)
    assert np.allclose(data['Qrot'], 7585.695, rtol=1e-3)
    assert np.allclose(data['logQvib_rp'], -25.34, rtol=1e-3)
    assert np.allclose(data['V/kBT'], 0.00098229064015049, rtol=1e-3)


def test_parse_ts_thermo_data():
    # Zundel system
    print(flush=True)
    temperature = 300.0
    directory = 'data/instanton/ts/'
    nqe.run_instanton_post_process(directory,
                                   process_type='TS',
                                   temperature=temperature)
    data = nqe.parse_ts_thermo_data(directory)
    print(data, flush=True)

    os.remove(os.path.join(directory, 'thermo_data.out'))
    os.remove(os.path.join(directory, 'freq.dat'))

    assert np.allclose(data['Qtras'], 10.187982156996487, rtol=1e-3)
    assert np.allclose(data['Qrot'], 1206.1509707160842, rtol=1e-3)
    assert np.allclose(data['logQvib'], -44.27838452659755, rtol=1e-3)
    assert np.allclose(data['V/kBT'], 25.165638546469864, rtol=1e-3)


def test_calc_forward_rate():
    # Zundel system
    print(flush=True)
    temperature = 300.0
    react_directory = 'data/instanton/react_phonon/'
    ts_directory = 'data/instanton/ts/'
    k_f = nqe.calc_forward_rate(react_directory, ts_directory, temperature)

    print(k_f, flush=True)

    os.remove(os.path.join(react_directory, 'eigenvalues_reactant.dat'))
    os.remove(os.path.join(react_directory, 'thermo_data.out'))
    os.remove(os.path.join(react_directory, 'freq.dat'))

    os.remove(os.path.join(ts_directory, 'thermo_data.out'))
    os.remove(os.path.join(ts_directory, 'freq.dat'))

    assert np.allclose(k_f, 2.179039420841875e-08, rtol=1e-3)


def test_parse_inst_thermo_data():
    # CBE system
    print(flush=True)
    temperature = 300.0
    n_beads = 40
    directory = f'data/instanton/inst_{n_beads}/'
    nqe.run_instanton_post_process(directory,
                                   process_type='instanton',
                                   temperature=temperature,
                                   n_beads=n_beads)
    data = nqe.parse_inst_thermo_data(directory)
    print(data, flush=True)

    os.remove(os.path.join(directory, 'thermo_data.out'))
    os.remove(os.path.join(directory, 'freq.dat'))

    assert np.allclose(data['Temperature'], 300.0, rtol=1e-3)
    assert np.allclose(data['NBEADS'], 80, rtol=1e-3)
    assert np.allclose(data['1/(betaP*hbar)'], 0.07600356479999999, rtol=1e-3)
    assert np.allclose(data['BN'], 28.575, rtol=1e-3)
    assert np.allclose(data['Qt'], 10.188, rtol=1e-3)
    assert np.allclose(data['Qrot'], 1251.027, rtol=1e-3)
    assert np.allclose(data['log(Qvib*N)'], -43.477, rtol=1e-3)
    assert np.allclose(data['S/hbar'], 25.025, rtol=1e-3)


# Warning this takes 5 minutes to run
def test_calc_instanton_kappa_bead():
    # CBE system
    print(flush=True)
    temperature = 300.0
    n_beads = 40

    directory_ts = 'data/instanton/ts/'
    nqe.run_instanton_post_process(directory_ts,
                                   process_type='TS',
                                   temperature=temperature)
    data_ts = nqe.parse_ts_thermo_data(directory_ts)
    os.remove(os.path.join(directory_ts, 'thermo_data.out'))
    os.remove(os.path.join(directory_ts, 'freq.dat'))

    directory_inst = f'data/instanton/inst_{n_beads}/'
    nqe.run_instanton_post_process(directory_inst,
                                   process_type='instanton',
                                   temperature=temperature,
                                   n_beads=n_beads)
    data_inst = nqe.parse_inst_thermo_data(directory_inst)
    os.remove(os.path.join(directory_inst, 'thermo_data.out'))
    os.remove(os.path.join(directory_inst, 'freq.dat'))

    kappa = nqe.calc_instanton_kappa(data_ts, data_inst)
    print(f'Tunneling factor, kappa = {kappa:5.5f}')
    assert np.allclose(kappa, 10.1278133296990684, rtol=1e-3)

    n_beads = 80
    directory_inst = f'data/instanton/inst_{n_beads}/'
    nqe.run_instanton_post_process(directory_inst,
                                   process_type='instanton',
                                   temperature=temperature,
                                   n_beads=n_beads)
    data_inst = nqe.parse_inst_thermo_data(directory_inst)
    os.remove(os.path.join(directory_inst, 'thermo_data.out'))
    os.remove(os.path.join(directory_inst, 'freq.dat'))

    kappa = nqe.calc_instanton_kappa(data_ts, data_inst)
    print(f'Tunneling factor, kappa = {kappa:5.5f}')
    assert np.allclose(kappa, 9.816, rtol=1e-3)


def test_ch4hcbe_end_to_end():
    # CBE system
    print(flush=True)

    directory_opti = 'opti'
    directory_phonon_react = 'phonon_react'
    directory_ts = 'ts'
    directory_instanton = 'instanton'

    driver_code = 'cbe'

    temperature = 300.0
    tol_energy = 1.0e-6
    tol_force = 1.0e-6
    tol_position = 1.0e-6
    n_beads = 40

    atoms = nqe.read_ipi_xyz("data/ch4hcbe.xyz")[-1]
    n_atoms = len(atoms)
    atoms_ts = nqe.read_ipi_xyz("data/ch4hcbe_ts.xyz")[-1]

    output = nqe.run_optimise(directory_opti,
                              atoms,
                              driver=driver_code,
                              tol_energy=tol_energy,
                              tol_force=tol_force,
                              tol_position=tol_position)
    atoms_opti, output_data_opti, _output_desc_opti = output

    energy_react = output_data_opti['potential'][-1]
    print(f'Final energy of reactant = {energy_react:5.5f}', flush=True)

    nqe.run_phonons(directory_phonon_react,
                    atoms_opti,
                    driver=driver_code)

    output = nqe.run_ts(directory_ts,
                        atoms_ts,
                        driver=driver_code,
                        tol_energy=tol_energy,
                        tol_force=tol_force,
                        tol_position=tol_position)
    atoms_ts, output_data_ts, _output_desc_ts = output

    energy_ts = output_data_ts['potential'][-1]
    print(f'Final energy of TS = {energy_ts:5.5f}', flush=True)

    delta_energy = energy_ts - energy_react
    print(f'Energy difference (TS - Reactant) = {delta_energy:5.5f}', flush=True)

    nqe.run_instanton(directory_instanton,
                      atoms_ts,
                      directory_ts,
                      driver=driver_code,
                      n_beads=n_beads,
                      temperature=temperature,
                      tol_energy=tol_energy,
                      tol_force=tol_force,
                      tol_position=tol_position)

    rate = nqe.calc_forward_rate(directory_phonon_react, directory_ts, temperature, filter_list=n_atoms - 1)

    print(f'Reaction rate = {rate}', flush=True)
    # assert np.allclose(rate, 1.0416486358380245e-08, rtol=1e-3)

    kappa = nqe.calc_kappa_full(directory_phonon_react,
                                directory_ts,
                                directory_instanton,
                                temperature,
                                n_beads, filter_list=n_atoms - 1)
    print(f'Tunneling factor, kappa = {kappa:5.5f}', flush=True)
    assert np.allclose(kappa, 10.1278133296990684, rtol=1e-3)

    nqe.remove_directory(directory_opti)
    nqe.remove_directory(directory_phonon_react)
    nqe.remove_directory(directory_ts)
    nqe.remove_directory(directory_instanton)


def test_ch4hcbe_end_to_end_orca():
    # CBE system
    print(flush=True)

    directory_opti = 'opti'
    directory_phonon_react = 'phonon_react'
    directory_ts = 'ts'
    directory_instanton = 'instanton'

    driver_code = 'ase-orca'
    driver_settings = {'xc': 'blyp',
                       'charge': 0,
                       'multi': 2,
                       'n_procs': 1, }

    temperature = 300.0
    tol_energy = 1.0e-3
    tol_force = 1.0e-3
    tol_position = 1.0e-3
    n_beads = 20

    atoms = nqe.read_ipi_xyz("data/ch4hcbe.xyz")[-1]
    n_atoms = len(atoms)
    atoms_ts = nqe.read_ipi_xyz("data/ch4hcbe_ts.xyz")[-1]

    output = nqe.run_optimise(directory_opti,
                              atoms,
                              driver=driver_code,
                              driver_args=driver_settings,
                              tol_energy=tol_energy,
                              tol_force=tol_force,
                              tol_position=tol_position)
    atoms_opti, output_data_opti, _output_desc_opti = output

    energy_react = output_data_opti['potential'][-1]
    print(f'Final energy of reactant = {energy_react:5.5f}', flush=True)

    nqe.run_phonons(directory_phonon_react,
                    atoms_opti,
                    driver=driver_code,
                    driver_args=driver_settings)

    output = nqe.run_ts(directory_ts,
                        atoms_ts,
                        driver=driver_code,
                        driver_args=driver_settings,
                        tol_energy=tol_energy,
                        tol_force=tol_force,
                        tol_position=tol_position)
    atoms_ts, output_data_ts, _output_desc_ts = output

    energy_ts = output_data_ts['potential'][-1]
    print(f'Final energy of TS = {energy_ts:5.5f}', flush=True)

    delta_energy = energy_ts - energy_react
    print(f'Energy difference (TS - Reactant) = {delta_energy:5.5f}', flush=True)

    rate = nqe.calc_forward_rate(directory_phonon_react, directory_ts, temperature, filter_list=n_atoms - 1)
    print(f'Reaction rate = {rate}', flush=True)
    # assert np.allclose(rate, 1.0416486358380245e-08, rtol=1e-3)

    nqe.run_instanton(directory_instanton,
                      atoms_ts,
                      directory_ts,
                      driver=driver_code,
                      driver_args=driver_settings,
                      n_beads=n_beads,
                      temperature=temperature,
                      tol_energy=tol_energy,
                      tol_force=tol_force,
                      tol_position=tol_position)

    kappa = nqe.calc_kappa_full(directory_phonon_react,
                                directory_ts,
                                directory_instanton,
                                temperature,
                                n_beads, filter_list=n_atoms - 1)
    print(f'Tunneling factor, kappa = {kappa:5.5f}', flush=True)


def test_ch4hcbe_temperature():
    # CBE system
    print(flush=True)

    directory_opti = 'opti'
    directory_phonon_react = 'phonon_react'
    directory_ts = 'ts'
    directory_instanton = 'instanton'

    driver_code = 'cbe'

    tol_energy = 5.0e-4
    tol_force = 5.0e-4
    tol_position = 5.0e-4
    n_beads = 40

    atoms = nqe.read_ipi_xyz("data/ch4hcbe.xyz")[-1]
    n_atoms = len(atoms)
    atoms_ts = nqe.read_ipi_xyz("data/ch4hcbe_ts.xyz")[-1]

    output = nqe.run_optimise(directory_opti,
                              atoms,
                              driver=driver_code,
                              tol_energy=tol_energy,
                              tol_force=tol_force,
                              tol_position=tol_position)
    atoms_opti, _output_data_opti, _output_desc_opti = output

    nqe.run_phonons(directory_phonon_react,
                    atoms_opti,
                    driver=driver_code)

    output = nqe.run_ts(directory_ts,
                        atoms_ts,
                        driver=driver_code,
                        tol_energy=tol_energy,
                        tol_force=tol_force,
                        tol_position=tol_position)
    atoms_ts, _output_data_ts, _output_desc_ts = output

    temperatures = [300.0, 250.0, 200.0]
    kappas = []
    kappa_ref = [10.131, 20.419, 226.685]

    for i, temperature in enumerate(temperatures):
        print(flush=True)
        print(f'{i}, Running instanton for temperature = {temperature} K', flush=True)

        nqe.run_instanton(directory_instanton,
                          atoms_ts,
                          directory_ts,
                          driver=driver_code,
                          n_beads=n_beads,
                          temperature=temperature,
                          tol_energy=tol_energy,
                          tol_force=tol_force,
                          tol_position=tol_position)

        kappa = nqe.calc_kappa_full(directory_phonon_react,
                                    directory_ts,
                                    directory_instanton,
                                    temperature,
                                    n_beads,
                                    filter_list=n_atoms - 1)

        kappas.append(kappa)
        nqe.remove_directory(directory_instanton)

    for kappa in kappas:
        print(f'Tunneling factor, kappa = {kappa:5.5f}', flush=True)

    assert np.allclose(kappas, kappa_ref, rtol=1e-3)

    nqe.remove_directory(directory_ts)


def test_thermo():
    print(flush=True)
    temperature = 300.0
    n_beads = 40
    n_atoms = 6
    directory_phonon_react = 'data/thermo/phonon_react'
    directory_ts = 'data/thermo/ts'
    directory_instanton = 'data/thermo/instanton'

    rate = nqe.calc_forward_rate(directory_phonon_react,
                                 directory_ts,
                                 temperature,
                                 debug=True,
                                 use_part_funcs=True, filter_list=n_atoms - 1)
    print(f'Reaction rate = {rate}', flush=True)

    kappa = nqe.calc_kappa_full(directory_phonon_react,
                                directory_ts,
                                directory_instanton,
                                temperature,
                                n_beads, filter_list=n_atoms - 1)
    print(f'Tunneling factor, kappa = {kappa:5.5f}', flush=True)

    pass
