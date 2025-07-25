import os

import numpy as np

import nqetools as nqe


def test_parse_react_thermo_data():
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


def test_parse_inst_thermo_data():
    print(flush=True)
    temperature = 300.0
    n_beads = 80
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
    assert np.allclose(data['NBEADS'], 160, rtol=1e-3)
    assert np.allclose(data['1/(betaP*hbar)'], 0.15201, rtol=1e-3)
    assert np.allclose(data['BN'], 14.28, rtol=1e-3)
    assert np.allclose(data['Qt'], 10.188, rtol=1e-3)
    assert np.allclose(data['Qrot'], 1251.027, rtol=1e-3)
    assert np.allclose(data['log(Qvib*N)'], -43.477, rtol=1e-3)
    assert np.allclose(data['S/hbar'], 25.026, rtol=1e-3)


def test_calc_instanton_kappa():
    print(flush=True)
    temperature = 300.0
    n_beads = 40

    directory_ts = 'data/instanton/ts/'
    nqe.run_instanton_post_process(directory_ts,
                                   process_type='TS',
                                   temperature=temperature)
    data_ts = nqe.parse_ts_thermo_data(directory_ts)

    directory_inst = f'data/instanton/inst_{n_beads}/'
    nqe.run_instanton_post_process(directory_inst,
                                   process_type='instanton',
                                   temperature=temperature,
                                   n_beads=n_beads)
    data_inst = nqe.parse_inst_thermo_data(directory_inst)

    kappa = nqe.calc_instanton_kappa(data_ts, data_inst)
    print('Tunneling factor, kappa = {:5.3f}'.format(kappa))

    os.remove(os.path.join(directory_ts, 'thermo_data.out'))
    os.remove(os.path.join(directory_ts, 'freq.dat'))
    os.remove(os.path.join(directory_inst, 'thermo_data.out'))
    os.remove(os.path.join(directory_inst, 'freq.dat'))

    assert np.allclose(kappa, 10.1278133296990684, rtol=1e-3)
