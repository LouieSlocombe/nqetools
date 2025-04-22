import os
import sys

import numpy as np
from ipi.engine.motion.instanton import SpringMapper
from ipi.engine.simulation import Simulation
from ipi.utils.hesstools import clean_hessian
from ipi.utils.instools import red2comp
from ipi.utils.messages import verbosity
from ipi.utils.units import unit_to_internal, Constants
from scipy.constants import k, h

from .execution import run_instanton_post_process


def kappa_core(
        q_trn_ts,
        q_rot_ts,
        q_vib_ts,
        q_trn_inst,
        q_rot_inst,
        q_vib_inst,
        bn,
        temperature,
        n,
        s_over_hbar,
        beta_times_v,
        hbar=1.0):
    """
    Computes the tunneling factor (kappa) for a given set of parameters.

    Parameters:
    q_trn_ts (float): Translational partition function at the transition state.
    q_rot_ts (float): Rotational partition function at the transition state.
    q_vib_ts (float): Vibrational partition function at the transition state.
    q_trn_inst (float): Translational partition function at the instanton.
    q_rot_inst (float): Rotational partition function at the instanton.
    q_vib_inst (float): Vibrational partition function at the instanton.
    bn (float): A parameter related to the instanton.
    temperature (float): Temperature in Kelvin.
    n (int): Number of beads.
    s_over_hbar (float): Action divided by reduced Planck's constant.
    beta_times_v (float): Beta times potential energy.
    hbar (float, optional): Reduced Planck's constant. Default is 1.0.

    Returns:
    float: The computed tunneling factor (kappa).
    """
    kelvin2au = 3.1668152e-06
    beta = 1.0 / (temperature * kelvin2au)
    f_trn = q_trn_inst / q_trn_ts
    f_rot = q_rot_inst / q_rot_ts
    f_vib = np.sqrt((2. * np.pi * n * bn) / (beta * hbar ** 2)) * q_vib_inst / q_vib_ts

    kappa = f_trn * f_rot * f_vib * np.exp(-s_over_hbar + beta_times_v)

    # printing out the transmission factor and the relevant contributions.
    print('f_tra               = {:5.3f}'.format(f_trn), flush=True)
    print('f_rot               = {:5.3f}'.format(f_rot), flush=True)
    print('f_vib               = {:5.3f}'.format(f_vib), flush=True)
    print('exp(-S/hbar+V/beta) = {:5.3f}'.format(np.exp(-s_over_hbar + beta_times_v)), flush=True)
    print('=============================', flush=True)
    print('Tunneling factor    = {:5.3f}'.format(kappa), flush=True)
    return kappa


def calc_kappa(ts_path, instanton_path, temperature, n_beads):
    """
    Calculates the tunneling factor (kappa) for a given transition state and instanton path.

    Parameters:
    ts_path (str): Path to the transition state data.
    instanton_path (str): Path to the instanton data.
    temperature (float): Temperature in Kelvin.
    n_beads (int): Number of beads.

    Returns:
    float: The computed tunneling factor (kappa).
    """
    Q_trn_TS, Q_rot_TS, Q_vib_TS, Beta_times_V = instanton_post_process(ts_path,
                                                                        case="TS",
                                                                        temperature=temperature,
                                                                        n_beads_r=n_beads)

    Q_trn_inst, Q_rot_inst, Q_vib_inst, BN, S_over_hbar = instanton_post_process(instanton_path,
                                                                                 case="instanton",
                                                                                 temperature=temperature)

    kappa = kappa_core(Q_trn_TS, Q_rot_TS, Q_vib_TS, Q_trn_inst, Q_rot_inst, Q_vib_inst, BN, temperature, n_beads,
                       S_over_hbar, Beta_times_V)
    return kappa


def instanton_post_process(input_file,
                           case="reactant",
                           temperature=300.0,
                           asr="poly",
                           energy_shift=0.0,
                           filter_list=[],
                           n_beads_r=0,
                           freq_reac=0,
                           quiet=False,
                           save=False,
                           ):
    """
    input_file: Restart file
    case: Type of the calculation to analyse. Options: 'instanton', 'reactant' or 'TS'.
    temperature: Temperature in K.
    asr: Removes the zero frequency vibrational modes depending on the symmerty of the system
    energy_shift: Zero of energy in eV
    filter_list: List of atoms indexes to filter i.e. eliminate its componentes
    n_beads_r: Number of beads (full polymer) to compute the approximate partition function (only reactant case)
    freq_reac: List of frequencies of the minimum. Required for splitting calculation.
    quiet: Avoid the Qvib and Qrot calculation in the instanton case.


    Reads all the information needed from a i-pi RESTART file and compute the partition functions of the reactant, transition state (TS) or
    instanton according to J. Phys. Chem. Lett. 7, 437(2016) (Instanton Rate calculations) or J. Chem. Phys. 134, 054109 (2011) (Tunneling Splitting)
    Instanton Rate: J. Phys. Chem. Lett.  7, 4374(2016)
    Tunneling Splitting: J. Chem. Phys. 134, 054109 (2011)

    Syntax:    python  Instanton_postproc.py  <checkpoint_file> -c <case> -t  <temperature (K)>  (-n <nbeads(full polymer)>) (-freq <freq_reactant.dat>)

    Examples for rate calculation:
               python  Instanton_postproc.py   RESTART  -c  instanton    -t   300
               python  Instanton_postproc.py   RESTART  -c  reactant     -t   300            -n 50
               python  Instanton_postproc.py   RESTART  -c    TS         -t   300

    Examples for splitting  calculation (2 steps):
             i)   python  Instanton_postproc.py   RESTART  -c  reactant   -t   10  -n 32 --->this generate the 'eigenvalues_reactant.dat' file
             ii)  python  Instanton_postproc.py   RESTART  -c  instanton  -t   10  -freq eigenvalues_reactant.dat
    """

    # Units
    K2au = unit_to_internal("temperature", "kelvin", 1.0)
    kb = Constants.kb
    hbar = Constants.hbar
    eV2au = unit_to_internal("energy", "electronvolt", 1.0)
    cal2au = unit_to_internal("energy", "cal/mol", 1.0)
    cm2au = unit_to_internal("frequency", "hertz", 1.0) * 3e10

    temp = temperature * K2au
    input_freq = freq_reac
    Verbosity = verbosity
    Verbosity.level = "quiet"

    if case not in list(["reactant", "TS", "instanton"]):
        raise ValueError("We can not indentify the case.")
    if asr not in list(["poly", "linear", "crystal", "none"]):
        raise ValueError("We can not indentify asr case.")

    if asr == "poly":
        nzeros = 6
    elif asr == "crystal":
        nzeros = 3
    else:
        nzeros = 0
    if asr == "linear":
        raise NotImplementedError("Sum rules for linear molecules is not implemented")

    if temperature == 0.0:
        raise ValueError("The temperature must be specified.'")

    def get_double(q0, nbeads0, natoms, h0):
        """Takes nbeads, positions and hessian (only the 'physical part') of the half polymer and
        returns the equivalent for the full ring polymer."""
        q = np.concatenate((q0, np.flipud(q0)), axis=0)
        nbeads = 2 * nbeads0
        ii = 3 * natoms
        iii = 3 * natoms * nbeads0

        h = np.zeros((iii * 2, iii * 2))
        h[:iii, :iii] = h0

        for i in range(nbeads0):
            x = i * ii + iii
            y = (nbeads0 - 1 - i) * ii
            h[x:x + ii, x:x + ii] = h0[y:y + ii, y:y + ii]

        return q, nbeads, h

    def spring_pot(nbeads, q, omega2, m3):
        e = sum(omega2 * 0.5 * np.dot(m3[0] * (q[i + 1] - q[i]), (q[i + 1] - q[i])) for i in range(nbeads - 1))
        return e

    def filter(pos, h, natoms, m, m3, filt):
        filt3 = [3 * i + j for i in filt for j in range(3)]
        pos = np.delete(pos, filt3, axis=1)
        h = np.delete(np.delete(h, filt3, axis=1), filt3, axis=0)
        m = np.delete(m, filt, axis=0)
        m3 = np.delete(m3, filt3, axis=1)
        natoms -= len(filt)
        return pos, h, natoms, m, m3

    def get_rp_freq(w0, nbeads, temp, mode="rate"):
        """Compute the ring polymer frequencies for multidimensional harmonic potential
        defined by the frequencies w0."""
        hbar, kb = 1.0, 1
        betaP = 1 / (kb * nbeads * temp)
        factor = betaP * hbar
        w, ww = 0.0, []

        if np.amin(w0) < 0.0:
            print("@get_rp_freq: We have a negative frequency, something is going wrong.")
            sys.exit()

        if mode == "rate":
            for n in range(w0.size):
                for k in range(nbeads):
                    if w0[n] == 0 and k == 0:
                        continue
                    w += np.log(
                        factor * np.sqrt(4.0 / (betaP * hbar) ** 2 * np.sin(np.abs(k) * np.pi / nbeads) ** 2 + w0[n]))
            return w

        elif mode == "splitting":
            for n in range(w0.size):
                for k in range(nbeads):
                    ww.append(
                        np.sqrt(4.0 / (betaP * hbar) ** 2 * np.sin((k + 1) * np.pi / (2 * nbeads + 2)) ** 2 + w0[n]))
            return np.array(ww)

        else:
            print("We can't identify the mode")
            sys.exit()

    def save_frequencies(d, nzeros, filename="freq.dat"):
        """Save frequencies in a file."""
        aux = np.zeros(nzeros)
        freq = np.sign(d) * np.abs(d) ** 0.5 / cm2au
        dd = np.concatenate((aux, freq))
        np.savetxt(filename, dd.reshape(dd.size, 1), header="Frequencies (cm^-1)")
        print(f"We saved the frequencies in {filename}", flush=True)

    print("We are ready to start. Reading {}".format(input_file), flush=True)

    simulation = Simulation.load_from_xml(
        input_file, custom_verbosity="quiet", request_banner=False, read_only=True
    )

    beads = simulation.syslist[0].motion.beads.clone()
    m = simulation.syslist[0].motion.beads.m.copy()
    nbeads = simulation.syslist[0].motion.beads.nbeads
    natoms = simulation.syslist[0].motion.beads.natoms

    if case == "reactant":
        if n_beads_r == 0:
            print("Need to specify number of beads for partition function", flush=True)
            sys.exit()

    if case != "instanton" and nbeads > 1:
        print(("Incompatibility between case and nbeads in {}.".format(input_file)), flush=True)
        print(("case {} , beads {}".format(case, nbeads)), flush=True)
        sys.exit()

    if case == "reactant":
        dynmat = simulation.syslist[0].motion.dynmatrix.copy()
        ism = np.sqrt(beads.m3)
        h = dynmat * np.outer(ism, ism)
        pos, m3 = beads.q, beads.m3
        if filter_list:
            pos, h, natoms, m, m3 = filter(pos, h, natoms, m, m3, filter_list)

    elif case == "TS":
        pos = beads.q
        h = simulation.syslist[0].motion.optarrays["hessian"].copy()
        m3 = beads.m3
        pots = simulation.syslist[0].motion.optarrays["old_u"]
        v0 = simulation.syslist[0].motion.optarrays["energy_shift"]

        if energy_shift != 0.0:
            print("Overwriting energy shift with the provided values", flush=True)
            v0 = energy_shift * eV2au
    elif case == "instanton":
        hessian = simulation.syslist[0].motion.optarrays["hessian"].copy()
        mode = simulation.syslist[0].motion.options["mode"]
        temp2 = simulation.syslist[0].ensemble.temp
        pots = simulation.syslist[0].motion.optarrays["old_u"]
        grads = -simulation.syslist[0].motion.optarrays["old_f"]
        v0 = simulation.syslist[0].motion.optarrays["energy_shift"]

        if energy_shift != 0.0:
            print("Overwriting energy shift with the provided values", flush=True)
            v0 = energy_shift * eV2au

        if np.absolute(temp - temp2) / K2au > 2:
            print(
                "\n Mismatch between provided temperature and temperature in the calculation", flush=True
            )
            sys.exit()

        if mode == "rate":
            h0 = red2comp(hessian, nbeads, natoms)
            pos, nbeads, hessian2 = get_double(beads.q, nbeads, natoms, h0)
            hessian = hessian2
            m3 = np.concatenate((beads.m3, beads.m3), axis=0)
            omega2 = (temp * nbeads * kb / hbar) ** 2
            if not quiet:
                spring = SpringMapper.spring_hessian(
                    natoms, nbeads, beads.m3[0], omega2, mode="full"
                )
                h = np.add(hessian, spring)
        elif mode == "splitting":
            if input_freq is None:
                print(
                    'Please provide a name of the file containing the list of the frequencies for the minimum using "-freq" flag',
                    flush=True
                )
                print(" You can generate that file using this script in the case reactant.", flush=True)
                sys.exit()

            print("Our linear polymer has  {}".format(nbeads), flush=True)
            pos = beads.q
            m3 = beads.m3
            omega2 = (temp * nbeads * kb / hbar) ** 2

            if not quiet:
                h0 = red2comp(hessian, nbeads, natoms)
                spring = SpringMapper.spring_hessian(
                    natoms, nbeads, beads.m3[0], omega2, mode="splitting"
                )
                h = np.add(h0, spring)
                if asr != "none":
                    print(
                        "We are changing asr to none since we consider a fixed ended linear polimer for the post-processing",
                        flush=True
                    )
                    asr = "none"
        else:
            print("We can not recognize the mode. STOP HERE", flush=True)
            sys.exit()

    beta = 1.0 / (kb * temp)
    betaP = 1.0 / (kb * (nbeads) * temp)

    print("\nTemperature: {} K".format(temp / K2au), flush=True)
    print("NBEADS: {}".format(nbeads), flush=True)
    print("atoms:  {}".format(natoms), flush=True)
    print("ASR:    {}".format(asr), flush=True)
    print("1/(betaP*hbar) = {:8.5f}".format((1 / (betaP * hbar))), flush=True)

    if not quiet or case == "reactant" or case == "TS":
        print("Diagonalizing ... \n\n", flush=True)
        d, w, detI = clean_hessian(h, pos, natoms, nbeads, m, m3, asr, mofi=True)
        print("Lowest 10 frequencies (cm^-1)", flush=True)
        d10 = np.array2string(
            np.sign(d[0:10]) * np.absolute(d[0:10]) ** 0.5 / cm2au,
            precision=2,
            max_line_width=100,
            formatter={"float_kind": lambda x: "%.2f" % x},
        )
        print("{}".format(d10), flush=True)
        if save:
            save_frequencies(d, nzeros)

    if case == "reactant":
        Qtras = (np.sum(m) / (2 * np.pi * beta * hbar ** 2)) ** 1.5

        if asr == "poly":
            Qrot = (8 * np.pi * detI / (hbar ** 6 * beta ** 3)) ** 0.5
        else:
            Qrot = 1.0

        # logQvib    = -np.sum( np.log( 2*np.sinh( (beta*hbar*np.sqrt(d)/2.0) )  ))   #Limit n->inf
        logQvib_rp = -get_rp_freq(d, n_beads_r, temp)

        # This file is created to use afterward in the splitting calculations
        if save:
            outfile = open("eigenvalues_reactant.dat", "w")
            aux = np.zeros(nzeros)
            dd = np.concatenate((aux, d))
            np.savetxt(outfile, dd.reshape(1, dd.size))
            outfile.close()

        print("We are done. Reactants. Nbeads {}".format(n_beads_r), flush=True)
        print("{:14s} | {:8s} | {:8s}".format("Qtras(bohr^-3)", "Qrot", "logQvib_rp"), flush=True)
        print("{:14.3f} | {:8.3f} |{:8.3f}\n".format(Qtras, Qrot, logQvib_rp), flush=True)
        print("A file with the eigenvalues in atomic units was generated", flush=True)
        return Qtras, Qrot, logQvib_rp

    elif case == "TS":
        Qtras = ((np.sum(m)) / (2 * np.pi * beta * hbar ** 2)) ** 1.5

        if asr == "poly":
            Qrot = (8 * np.pi * detI / (hbar ** 6 * beta ** 3)) ** 0.5
        else:
            Qrot = 1.0

        logQvib = -np.sum(
            np.log(2 * np.sinh((beta * hbar * np.sqrt(np.delete(d, 0)) / 2.0)))
        )

        U = pots.sum() - v0
        u_ev = U / eV2au
        Beta_times_V = U / (kb * temp)
        print("We are done. TS", flush=True)
        print("Partition functions at {} K".format(temp / K2au), flush=True)
        print("Qtras: {}".format(Qtras), flush=True)
        print("Qrot: {}".format(Qrot), flush=True)
        print("logQvib: {}".format(logQvib), flush=True)
        print("Potential energy at TS:  {} eV, V/kBT {}\n".format(u_ev, Beta_times_V), flush=True)
        return Qtras, Qrot, np.exp(logQvib), Beta_times_V

    elif case == "instanton":
        if mode == "rate":
            Qtras = (np.sum(m) / (2 * np.pi * beta * hbar ** 2)) ** 1.5

            if asr == "poly" and not quiet:
                Qrot = (8 * np.pi * detI / (hbar ** 6 * betaP ** 3)) ** 0.5
                Qrot /= nbeads ** 3
            else:
                Qrot = 1.0

            if not quiet:
                del_freq = np.sign(d[1]) * np.absolute(d[1]) ** 0.5 / cm2au
                print("Deleted frequency: {:8.3f} cm^-1".format(del_freq), flush=True)

                if asr != "poly":
                    print("WARNING asr != poly", flush=True)
                    print("First 10 eigenvalues", flush=True)
                    ten_eigv = np.sign(d[0:10]) * np.absolute(d[0:10]) ** 0.5 / cm2au
                    print("{}".format(ten_eigv), flush=True)
                    print(
                        "Please check that this you don't have any unwanted zero frequency", flush=True
                    )
                logQvib = (
                        -np.sum(np.log(betaP * hbar * np.sqrt(np.absolute(np.delete(d, 1)))))
                        + nzeros * np.log(nbeads)
                        + np.log(nbeads)
                )
            else:
                logQvib = 0.0

            BN = 2 * np.sum(beads.m3[1:, :] * (beads.q[1:, :] - beads.q[:-1, :]) ** 2)
            factor = 1.0000  # default
            action1 = (2 * pots.sum() * factor - nbeads * v0) * 1.0 / (temp * nbeads * kb)
            action2 = spring_pot(nbeads, pos, omega2, m3) / (temp * nbeads * kb)
            S_over_hbar = action1 + action2
            print(
                "We are done. Instanton rate. Nbeads {} (diff only {})".format(
                    nbeads, nbeads / 2
                ), flush=True
            )
            print(
                "   {:8s} {:8s}  | {:11s} | {:11s} | {:11s} | {:8s} ( {:8s},{:8s} ) |".format(
                    "BN",
                    "(BN*N)",
                    "Qt(bohr^-3)",
                    "Qrot",
                    "log(Qvib*N)",
                    "S/hbar",
                    "S1/hbar",
                    "S2/hbar",
                ), flush=True
            )
            print(
                "{:8.3f} ( {:8.3f} ) | {:11.3f} | {:11.3f} | {:11.3f} | {:8.3f} ( {:8.3f} {:8.3f} ) |".format(
                    BN,
                    BN * nbeads,
                    Qtras,
                    Qrot,
                    logQvib,
                    (action1 + action2),
                    action1,
                    action2,
                ), flush=True
            )
            return Qtras, Qrot, np.exp(logQvib), BN, S_over_hbar

        elif mode == "splitting":
            out = open(input_freq, "r")
            d_min = np.zeros(natoms * 3)
            aux = out.readline().split()
            if len(aux) != (natoms * 3):
                print(("We are expecting {} frequencies.".format((natoms * 3 - 6))), flush=True)
                print(("instead we have read  {}".format(len(aux))), flush=True)
            for i in range((natoms * 3)):
                d_min[i] = float(aux[i])
            d_min = d_min.reshape((natoms * 3))
            out.close()
            ww = get_rp_freq(np.sign(d_min) * d_min ** 2, nbeads, temp, mode="splitting")
            react = np.sum(np.log(ww))

            action1 = (pots.sum() - nbeads * v0) * 1 / (temp * nbeads * kb)
            action2 = spring_pot(nbeads, pos, omega2, m3) / (temp * nbeads * kb)
            action = action1 + action2
            if action / hbar > 5.0:
                print(
                    "WARNING, S/h seems to big. Probably a proper energy shift is missing.", flush=True
                )

            BN = np.sum(beads.m3[1:, :] * (beads.q[1:, :] - beads.q[:-1, :]) ** 2)

            if not quiet:
                inst = np.sum(np.log(np.sqrt(np.absolute(np.delete(d, [1])))))
                phi = np.exp(inst - react)
            else:
                phi = 1

            tetaphi = (
                    betaP * hbar * np.sqrt(action / (2 * hbar * np.pi)) * np.exp(-action / hbar)
            )
            teta = tetaphi / phi
            h = -teta / betaP
            # cm2au= (2 * np.pi * 3e10 * 2.4188843e-17)  # Handy for debugging

            print("We are done", flush=True)
            print("Nbeads {}, betaP {} a.u.,hbar {} a.u".format(nbeads, betaP, hbar), flush=True)
            print("V0  {} eV ( {} Kcal/mol) ".format(v0 / eV2au, v0 / cal2au / 1000), flush=True)
            print(
                "S1/hbar {} ,S2/hbar {} ,S/hbar {}".format(
                    action1 / hbar, action2 / hbar, action / hbar
                ), flush=True
            )
            print("BN {} a.u.".format(BN), flush=True)
            print(
                "BN/(hbar^2 * betaN)  {}  (should be same as S/hbar) ".format(
                    (BN / ((hbar ** 2) * betaP))
                ), flush=True
            )
            if quiet:
                print("phi is not computed because you specified the quiet option", flush=True)
                print(
                    ("We can provied only Tetaphi which value is {} a.u. ".format(tetaphi)), flush=True
                )
            else:
                print(("phi {} a.u.   Teta {} a.u. ".format(phi, tetaphi / phi)), flush=True)
                print(
                    "Tunnelling splitting matrix element (h)  {} a.u ({} cm^-1)".format(
                        h, h / cm2au
                    ), flush=True
                )
        else:
            print("We can not recognize the mode.", flush=True)
            sys.exit()


def parse_inst_thermo_data(directory, filename='thermo_data.out'):
    """
    Parse the thermo_data.out file and extract thermodynamic values.

    Args:
        directory (str): Directory containing the thermo_data file
        filename (str): Name of the thermo data file (default: 'thermo_data.out')

    Returns:
        dict: Dictionary containing the thermodynamic values including Temperature, NBEADS, and 1/(betaP*hbar)
    """
    filepath = os.path.join(directory, filename)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {}

    # Find needed values from the file
    for i, line in enumerate(lines):
        if 'Temperature:' in line:
            data['Temperature'] = float(line.split()[1])
        elif 'NBEADS:' in line:
            data['NBEADS'] = int(line.split()[1])
        elif '1/(betaP*hbar)' in line:
            data['1/(betaP*hbar)'] = float(line.split('=')[1].strip())
        elif 'BN' in line and 'Qt' in line:
            # Get the next line which contains the values
            data_line = lines[i + 1]
            # Split the line and clean up the values
            values = data_line.split('|')
            bn_parts = values[0].strip().replace('(', '').replace(')', '').split()

            # Add the remaining data
            data.update({
                'BN': float(bn_parts[0]),
                'Qt': float(values[1].strip()),
                'Qrot': float(values[2].strip()),
                'log(Qvib*N)': float(values[3].strip()),
                'S/hbar': float(values[4].split('(')[0].strip()),
            })
            return data

    raise ValueError(f"Could not find thermodynamic data in {filepath}")


def parse_ts_thermo_data(directory, filename='thermo_data.out'):
    """
    Parse the TS thermo_data.out file and extract thermodynamic values.

    Args:
        directory (str): Directory containing the thermo_data file
        filename (str): Name of the thermo data file (default: 'thermo_data.out')

    Returns:
        dict: Dictionary containing Qtras, Qrot, logQvib, and V/kBT values
    """
    filepath = os.path.join(directory, filename)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {}

    for line in lines:
        if 'Qtras:' in line:
            data['Qtras'] = float(line.split(':')[1].strip())
        elif 'Qrot:' in line:
            data['Qrot'] = float(line.split(':')[1].strip())
        elif 'logQvib:' in line:
            data['logQvib'] = float(line.split(':')[1].strip())
        elif 'V/kBT' in line:
            data['V/kBT'] = float(line.split()[-1].strip())

    if not data:
        raise ValueError(f"Could not find thermodynamic data in {filepath}")

    return data


def calc_instanton_kappa(ts_data, inst_data):
    """
    Calculate the tunneling factor (kappa) from TS and instanton thermodynamic data.

    Args:
        ts_data (dict): Dictionary containing TS thermodynamic data
        inst_data (dict): Dictionary containing instanton thermodynamic data

    Returns:
        float: Tunneling factor (kappa)
    """
    # Extract partition functions and other parameters
    f_trn = inst_data['Qt'] / ts_data['Qtras']
    f_rot = inst_data['Qrot'] / ts_data['Qrot']

    # Calculate f_vib using BN and temperature
    beta = 1.0 / (inst_data['Temperature'] * 3.1668152e-06)  # kelvin2au = 3.1668152e-06
    f_vib = np.sqrt((2.0 * np.pi * inst_data['NBEADS'] * inst_data['BN']) / (beta * 1.0 ** 2)) * \
            np.exp(inst_data['log(Qvib*N)'] - ts_data['logQvib'])

    # Calculate kappa
    kappa = f_trn * f_rot * f_vib * np.exp(-inst_data['S/hbar'] + ts_data['V/kBT'])

    return kappa


def calc_kappa_full(
        directory_ts,
        directory_instanton,
        temperature,
        react_energy=0.0,
        n_beads=4,
        filter_list=None):
    run_instanton_post_process(directory_ts,
                               process_type='TS',
                               temperature=temperature,
                               filter_list=filter_list,
                               ref_energy=react_energy)
    data_ts = parse_ts_thermo_data(directory_ts)

    run_instanton_post_process(directory_instanton,
                               process_type='instanton',
                               temperature=temperature,
                               n_beads=n_beads,
                               filter_list=filter_list,
                               ref_energy=react_energy)
    data_inst = parse_inst_thermo_data(directory_instanton)
    return calc_instanton_kappa(data_ts, data_inst)


def parse_react_thermo_data(directory, filename='thermo_data.out'):
    """
    Parse the reactant thermo_data.out file and extract thermodynamic values.

    Args:
        directory (str): Directory containing the thermo_data file
        filename (str): Name of the thermo data file (default: 'thermo_data.out')

    Returns:
        dict: Dictionary containing Qtras, Qrot, logQvib_rp, and potential energy values
    """
    filepath = os.path.join(directory, filename)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {}

    for i, line in enumerate(lines):
        if 'Qtras(bohr^-3) | Qrot     | logQvib_rp' in line:
            # Get the next line which contains the values
            data_line = lines[i + 1].strip()
            values = data_line.split('|')

            data['Qtras'] = float(values[0].strip())
            data['Qrot'] = float(values[1].strip())
            data['logQvib_rp'] = float(values[2].strip())

    if not data:
        raise ValueError(f"Could not find reactant thermodynamic data in {filepath}")

    return data


def calculate_forward_rate(T,
                           Q_trans_react,
                           Q_rot_react,
                           logQ_vib_react,
                           Q_trans_TS,
                           Q_rot_TS,
                           logQ_vib_TS,
                           vkBT):
    """
    Calculate the forward rate constant using Transition State Theory.

    Parameters:
    T (float): Temperature in Kelvin
    Q_trans_react (float): Translational partition function of the reactant
    Q_rot_react (float): Rotational partition function of the reactant
    logQ_vib_react (float): Logarithm of vibrational partition function of the reactant
    Q_trans_TS (float): Translational partition function of the transition state
    q_rot_ts (float): Rotational partition function of the transition state
    logQ_vib_TS (float): Logarithm of vibrational partition function of the transition state
    V_TS (float): Potential energy difference between TS and reactant (in Joules)

    Returns:
    float: Forward rate constant (in s⁻¹)
    """

    # Convert log partition functions to normal partition functions
    Q_vib_react = np.exp(logQ_vib_react)
    Q_vib_TS = np.exp(logQ_vib_TS)

    # Calculate the ratio of partition functions
    partition_function_ratio = (Q_trans_TS * Q_rot_TS * Q_vib_TS) / (Q_trans_react * Q_rot_react * Q_vib_react)

    # Calculate the exponential Boltzmann factor
    boltzmann_factor = np.exp(-vkBT)

    # Calculate the forward rate constant
    k_f = (k * T / h) * partition_function_ratio * boltzmann_factor

    return k_f


def calc_forward_rate(n_atoms, ts_directory, react_directory, temperature, react_energy):
    """
    Calculate the forward rate constant using data from TS and reactant thermo files.

    Args:
        ts_directory (str): Directory containing TS thermo_data file
        react_directory (str): Directory containing reactant thermo_data file
        temperature (float): Temperature in Kelvin

    Returns:
        float: Forward rate constant (in s⁻¹)
    """

    run_instanton_post_process(react_directory,
                               process_type='reactant',
                               temperature=temperature,
                               filter_list=n_atoms - 1)

    run_instanton_post_process(ts_directory,
                               process_type='TS',
                               temperature=temperature,
                               ref_energy=react_energy)

    kelvin2au = 3.1668152e-06
    beta = 1.0

    # Get TS data
    ts_data = parse_ts_thermo_data(ts_directory)

    # Get reactant data
    react_data = parse_react_thermo_data(react_directory)

    # Calculate forward rate using TST
    k_f = calculate_forward_rate(
        T=temperature,
        Q_trans_react=react_data['Qtras'],
        Q_rot_react=react_data['Qrot'],
        logQ_vib_react=react_data['logQvib_rp'],
        Q_trans_TS=ts_data['Qtras'],
        Q_rot_TS=ts_data['Qrot'],
        logQ_vib_TS=ts_data['logQvib'],
        vkBT=ts_data['V/kBT'] - (react_energy / (temperature * kelvin2au))
    )

    return k_f
