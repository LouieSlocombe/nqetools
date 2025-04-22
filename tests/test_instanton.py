import os

import nqetools as nqe


# Fails as the file is not minimized properly
def test_parse_react_thermo_data():
    print(flush=True)
    temperature = 300.0
    directory = 'data/instanton/react_phonon/'
    nqe.run_instanton_post_process(directory,
                                   process_type='reactant',
                                   temperature=temperature)

    data = nqe.parse_react_thermo_data(directory)
    print(data, flush=True)
    os.remove(os.path.join(directory, 'thermo_data.out'))
    os.remove(os.path.join(directory, 'freq.dat'))
    pass


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
    pass


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
    pass


def test_calc_instanton_kappa():
    print(flush=True)
    temperature = 300
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
    assert round(kappa, 3) == round(10.127813329699068, 3)
