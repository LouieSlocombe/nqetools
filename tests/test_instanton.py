import os
import sys
import time
from subprocess import Popen

import ase.build
import numpy as np
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.calculators.nwchem import NWChem
from ase.calculators.socketio import PySocketIOClient, SocketIOCalculator
from ase.io import write
from ase.optimize import BFGS
from mace.calculators import mace_anicc, mace_off

import nqetools as nqe


def test_parse_inst_thermo_data():
    temperature = 300
    n_beads = 80
    directory_instanton = f'data/instanton/inst_{n_beads}/'
    nqe.run_instanton_post_process(directory_instanton,
                                   process_type='instanton',
                                   temperature=temperature,
                                   n_beads=n_beads)

    data = nqe.parse_inst_thermo_data(directory_instanton)

    print(data)
    pass


def test_parse_ts_thermo_data():
    print(flush=True)
    temperature = 300
    directory_instanton = 'data/instanton/ts/'
    nqe.run_instanton_post_process(directory_instanton, process_type='TS', temperature=temperature)

    data = nqe.parse_ts_thermo_data(directory_instanton)
    print(data)
    pass


def test_calc_instanton_kappa():
    print(flush=True)
    temperature = 300
    n_beads = 40

    directory_instanton = 'data/instanton/ts/'
    nqe.run_instanton_post_process(directory_instanton,
                                   process_type='TS',
                                   temperature=temperature)
    data_ts = nqe.parse_ts_thermo_data(directory_instanton)

    directory_ts = f'data/instanton/inst_{n_beads}/'
    nqe.run_instanton_post_process(directory_ts,
                                   process_type='instanton',
                                   temperature=temperature,
                                   n_beads=n_beads)

    data_inst = nqe.parse_inst_thermo_data(directory_ts)

    kappa = nqe.calc_instanton_kappa(data_ts, data_inst)
    print(kappa)
    #     [40  ,  10.127813329699068],
    #     [80  ,   9.809437521247709]])
