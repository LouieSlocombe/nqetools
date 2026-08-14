"""Generation of PLUMED input files for enhanced sampling.

Each writer emits a plumed.dat for one biasing scheme and returns the
names of the COLVAR columns it produces, which i-PI needs in order to read
the results back. :func:`prep_plumed` dispatches between them by name.

Schemes come in metadynamics ('mtd-') and OPES ('opes-') variants that
differ only in the bias they apply, and are built from a small set of
collective variables: interatomic distances, antisymmetric stretches,
coordination-number differences, and centre-of-mass separations. The
coordination-based variables are bounded by a switching function, which
makes them less sensitive to the donor-acceptor distance than raw
distances are.

Notes
-----
Arguments throughout are given in ASE units (eV, Angstrom) and converted
to the kJ/mol and nm that PLUMED expects. Atom indices are zero-based on
the way in and shifted to PLUMED's one-based convention.
"""

import os

from .conversions import (A_to_nm,
                          eV_to_kJpermol,
                          eVperA2_to_kJpermolpernm2)
from .tools import round_sf, get_distance


def prep_plumed(atoms, plumed_type, plumed_args):
    """Dispatch to the PLUMED input writer named by `plumed_type`.

    Single entry point used by :func:`~nqetools.execution.run_plumed_md`,
    mapping a short scheme name onto the function that writes the
    corresponding plumed.dat.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the run will start from. Only passed to the writers that
        need to enumerate atoms, such as the coordination-number schemes.
    plumed_type : str
        Name of the biasing scheme. Names beginning 'mtd-' use
        metadynamics and 'opes-' use OPES. The special value 'custom'
        writes ``plumed_args['input']`` verbatim.
    plumed_args : dict
        Keyword arguments forwarded to the selected writer. For 'custom',
        must contain 'directory', 'input' and 'output'.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by the selected input, for
        i-PI to read back.

    Raises
    ------
    ValueError
        If `plumed_type` is not a recognised scheme.
    """
    if plumed_type == 'mtd-pos':
        return write_plumed_mtd_pos(**plumed_args)
    elif plumed_type == 'opes-pos':
        return write_plumed_opes_pos(**plumed_args)
    elif plumed_type == 'mtd-coord':
        return write_plumed_mtd_coord(atoms, **plumed_args)
    elif plumed_type == 'opes-coord':
        return write_plumed_opes_coord(atoms, **plumed_args)
    elif plumed_type == 'mtd-dists':
        return write_plumed_mtd_dists(**plumed_args)
    elif plumed_type == 'opes-dists':
        return write_plumed_opes_dists(**plumed_args)
    elif plumed_type == 'mtd-dist':
        return write_plumed_mtd_dist(**plumed_args)
    elif plumed_type == 'opes-dist':
        return write_plumed_opes_dist(**plumed_args)
    elif plumed_type == 'mtd-diff1':
        return write_plumed_mtd_diff1(**plumed_args)
    elif plumed_type == 'opes-diff1':
        return write_plumed_opes_diff1(**plumed_args)
    elif plumed_type == 'mtd-diff2':
        return write_plumed_mtd_diff2(**plumed_args)
    elif plumed_type == 'opes-diff2':
        return write_plumed_opes_diff2(**plumed_args)
    elif plumed_type == 'mtd-pt1':
        return write_plumed_mtd_pt1(atoms, **plumed_args)
    elif plumed_type == 'opes-pt1':
        return write_plumed_opes_pt1(atoms, **plumed_args)
    elif plumed_type == 'mtd-pt2_a':
        return write_plumed_mtd_pt2_a(atoms, **plumed_args)
    elif plumed_type == 'opes-pt2_a':
        return write_plumed_opes_pt2_a(atoms, **plumed_args)

    elif plumed_type == 'mtd-pt-wob':
        return write_plumed_mtd_pt_wob(atoms, **plumed_args)
    elif plumed_type == 'opes-pt-wob':
        return write_plumed_opes_pt_wob(atoms, **plumed_args)
    elif plumed_type == 'mtd-pt-wob-sep':
        return write_plumed_mtd_pt_wob_sep(atoms, **plumed_args)
    elif plumed_type == 'opes-pt-wob-sep':
        return write_plumed_opes_pt_wob_sep(atoms, **plumed_args)
    elif plumed_type == 'opes-pt-wob-dist':
        return write_plumed_opes_pt_wob_dist(atoms, **plumed_args)

    elif plumed_type == 'opes_com':
        return write_plumed_opes_com(**plumed_args)
    elif plumed_type == 'opes_1pt':
        return write_plumed_opes_1pt(**plumed_args)
    elif plumed_type == 'opes_1pt_coord':
        return write_plumed_opes_1pt_coord(**plumed_args)
    elif plumed_type == 'opes_1pt_3donor_coord':
        return write_plumed_opes_1pt_3donor_coord(**plumed_args)


    elif plumed_type == 'opes_2pt_2d':
        return write_plumed_opes_2pt_2d(**plumed_args)
    elif plumed_type == 'opes_2pt_2d_coord':
        return write_plumed_opes_2pt_2d_coord(**plumed_args)
    elif plumed_type == 'opes_2pt_1d':
        return write_plumed_opes_2pt_1d(**plumed_args)
    elif plumed_type == 'opes_2pt_1d_coord':
        return write_plumed_opes_2pt_1d_coord(**plumed_args)
    elif plumed_type == 'opes_2pt_1d_coord_com':
        return write_plumed_opes_2pt_1d_coord_com(**plumed_args)

    elif plumed_type == 'custom':
        with open(os.path.join(plumed_args['directory'], "plumed.dat"), "w") as f:
            f.write(plumed_args['input'])
        return plumed_args['output']

    else:
        raise ValueError(f'Unknown plumed type: {plumed_type}')


def write_plumed_mtd_pos(directory=None,
                         idx_atom=0,
                         pace=20,
                         sigma=0.01,
                         height=1.0,
                         bias=2.5,
                         temperature=300,
                         stride=10,
                         ):
    """Write metadynamics biasing the x position of a single atom.

    The simplest possible scheme, used mainly for testing the i-PI to
    PLUMED coupling rather than for production runs.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx_atom : int, optional
        Index of the biased atom. Default is 0.
    pace : int, optional
        Steps between bias depositions. Default is 20.
    sigma : float, optional
        Gaussian width in nm. Default is 0.01.
    height : float, optional
        Initial hill height in eV. Default is 1.0.
    bias : float, optional
        Well-tempered bias factor. Larger values flatten the surface more
        aggressively but converge more slowly. Default is 2.5.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.
    Atom indices are zero-based on the way in and shifted to PLUMED's
    one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    height = round_sf(height * eV_to_kJpermol)

    idx_atom += 1
    impt = f"""
q: POSITION ATOM={idx_atom}
mtd: METAD ARG=q.x PACE={pace} SIGMA={sigma} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['q.x', 'mtd.bias']


def write_plumed_opes_pos(directory=None,
                          idx_atom=0,
                          pace=20,
                          barrier=1.0,
                          temperature=300,
                          stride=10,
                          stride_hills=100,
                          explore=False
                          ):
    """Write OPES biasing the x position of a single atom.

    OPES counterpart of :func:`write_plumed_mtd_pos`.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx_atom : int, optional
        Index of the biased atom. Default is 0.
    pace : int, optional
        Steps between bias depositions. Default is 20.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 1.0.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.
    Atom indices are zero-based on the way in and shifted to PLUMED's
    one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    idx_atom += 1

    barrier = round_sf(barrier * eV_to_kJpermol)

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
q: POSITION ATOM={idx_atom}
opes: {opes_command} ARG=q.x PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['q.x', 'opes.bias']


def write_plumed_mtd_coord(atoms,
                           directory=None,
                           idx1=0,
                           idx2=1,
                           temperature=300,
                           sigma=None,
                           d_low=1.4,
                           d_upper=4.0,
                           kappa=0.026,
                           pace=10,
                           stride=10,
                           height=0.041,
                           bias=10,
                           ):
    """Write metadynamics on the donor-acceptor distance and a coordination difference.

    Two collective variables are biased together: the heavy-atom
    separation, and the difference between the donor and acceptor
    coordination numbers, which tracks which of the two the shared atom
    is bound to. An upper wall on the separation stops the two heavy
    atoms drifting apart.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the run will start from, used to enumerate the atoms that
        form the coordination shell.
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        Index of the donor atom. Default is 0.
    idx2 : int, optional
        Index of the acceptor atom. Default is 1.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    sigma : list of float, optional
        Gaussian widths for the distance and coordination difference.
        Default is [0.005, 0.05].
    d_low : float, optional
        Switching function midpoint in Angstrom. Default is 1.4.
    d_upper : float, optional
        Position of the upper wall in Angstrom. Default is 4.0.
    kappa : float, optional
        Upper wall force constant in eV/A^2. Default is 0.026.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    height : float, optional
        Initial hill height in eV. Default is 0.041.
    bias : float, optional
        Well-tempered bias factor. Larger values flatten the surface more
        aggressively but converge more slowly. Default is 10.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.

    The coordination shell is every atom except the donor and acceptor
    themselves.
    """
    if directory is None:
        directory = os.getcwd()

    if sigma is None:
        sigma = [0.005, 0.05]

    d_low = d_low * A_to_nm
    d_upper = d_upper * A_to_nm

    kappa = round_sf(kappa * eVperA2_to_kJpermolpernm2)

    height = round_sf(height * eV_to_kJpermol)

    group_idx = list(range(len(atoms)))

    group_idx.remove(idx1)
    group_idx.remove(idx2)

    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]

    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    impt = f"""
d: DISTANCE ATOMS={idx1},{idx2} 
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
mtd:   METAD ARG=d,dc PACE={pace} SIGMA={sigma[0]},{sigma[1]} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}
uwall: UPPER_WALLS ARG=d AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d', 'c1.lessthan', 'c2.lessthan', 'dc', 'mtd.bias']


def write_plumed_opes_coord(atoms,
                            directory=None,
                            idx1=0,
                            idx2=1,
                            temperature=300,
                            d_low=1.4,
                            d_upper=4.0,
                            kappa=0.026,
                            pace=10,
                            stride=10,
                            barrier=0.041,
                            stride_hills=100,
                            explore=False,
                            ):
    """Write OPES on the donor-acceptor distance and a coordination difference.

    Two collective variables are biased together: the heavy-atom
    separation, and the difference between the donor and acceptor
    coordination numbers, which tracks which of the two the shared atom
    is bound to. An upper wall on the separation stops the two heavy
    atoms drifting apart.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the run will start from, used to enumerate the atoms that
        form the coordination shell.
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        Index of the donor atom. Default is 0.
    idx2 : int, optional
        Index of the acceptor atom. Default is 1.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    d_low : float, optional
        Switching function midpoint in Angstrom. Default is 1.4.
    d_upper : float, optional
        Position of the upper wall in Angstrom. Default is 4.0.
    kappa : float, optional
        Upper wall force constant in eV/A^2. Default is 0.026.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.041.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.

    The coordination shell is every atom except the donor and acceptor
    themselves.
    """
    if directory is None:
        directory = os.getcwd()

    d_low = d_low * A_to_nm
    d_upper = d_upper * A_to_nm

    kappa = round_sf(kappa * eVperA2_to_kJpermolpernm2)

    barrier = round_sf(barrier * eV_to_kJpermol)

    group_idx = list(range(len(atoms)))

    group_idx.remove(idx1)
    group_idx.remove(idx2)

    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]

    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d: DISTANCE ATOMS={idx1},{idx2} 
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=d,dc PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES 
uwall: UPPER_WALLS ARG=d AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d', 'c1.lessthan', 'c2.lessthan', 'dc', 'opes.bias']


def write_plumed_mtd_dists(directory=None,
                           idx1=0,
                           idx2=1,
                           idx3=2,
                           idx4=3,
                           temperature=300,
                           sigma=None,
                           pace=10,
                           stride=10,
                           height=0.041,
                           bias=10,
                           ):
    """Write metadynamics on two independent interatomic distances.

    Biases two unrelated atom pairs at once, giving a two-dimensional
    free energy surface in those distances.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        First atom of the first pair. Default is 0.
    idx2 : int, optional
        Second atom of the first pair. Default is 1.
    idx3 : int, optional
        First atom of the second pair. Default is 2.
    idx4 : int, optional
        Second atom of the second pair. Default is 3.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    sigma : list of float, optional
        Gaussian widths for the two distances in nm. Default is
        [0.05, 0.05].
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    height : float, optional
        Initial hill height in eV. Default is 0.041.
    bias : float, optional
        Well-tempered bias factor. Larger values flatten the surface more
        aggressively but converge more slowly. Default is 10.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.
    Atom indices are zero-based on the way in and shifted to PLUMED's
    one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    if sigma is None:
        sigma = [0.05, 0.05]

    height = round_sf(height * eV_to_kJpermol)

    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx3},{idx4}
mtd: METAD ARG=d1,d2 PACE={pace} SIGMA={sigma[0]},{sigma[1]} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'd2', 'mtd.bias']


def write_plumed_opes_dists(directory=None,
                            idx1=0,
                            idx2=1,
                            idx3=2,
                            idx4=3,
                            temperature=300,
                            pace=10,
                            stride=10,
                            barrier=0.041,
                            stride_hills=100,
                            explore=False,
                            ):
    """Write OPES on two independent interatomic distances.

    Biases two unrelated atom pairs at once, giving a two-dimensional
    free energy surface in those distances.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        First atom of the first pair. Default is 0.
    idx2 : int, optional
        Second atom of the first pair. Default is 1.
    idx3 : int, optional
        First atom of the second pair. Default is 2.
    idx4 : int, optional
        Second atom of the second pair. Default is 3.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.041.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.
    Atom indices are zero-based on the way in and shifted to PLUMED's
    one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    barrier = round_sf(barrier * eV_to_kJpermol)

    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx3},{idx4}
opes: {opes_command} ARG=d1,d2 PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'd2', 'opes.bias']


def write_plumed_mtd_dist(directory=None,
                          idx1=0,
                          idx2=1,
                          temperature=300,
                          sigma=0.05,
                          pace=10,
                          stride=10,
                          height=0.041,
                          bias=10,
                          ):
    """Write metadynamics on a single interatomic distance.

    The one-dimensional case: a bond length or heavy-atom separation
    used directly as the collective variable.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        First atom of the pair. Default is 0.
    idx2 : int, optional
        Second atom of the pair. Default is 1.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    sigma : float, optional
        Gaussian width in nm. Default is 0.05.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    height : float, optional
        Initial hill height in eV. Default is 0.041.
    bias : float, optional
        Well-tempered bias factor. Larger values flatten the surface more
        aggressively but converge more slowly. Default is 10.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.
    Atom indices are zero-based on the way in and shifted to PLUMED's
    one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    height = round_sf(height * eV_to_kJpermol)

    idx1 += 1
    idx2 += 1

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
mtd: METAD ARG=d1 PACE={pace} SIGMA={sigma} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'mtd.bias']


def write_plumed_opes_dist(directory=None,
                           idx1=0,
                           idx2=1,
                           temperature=300,
                           pace=10,
                           stride=10,
                           barrier=0.041,
                           stride_hills=100,
                           explore=False,
                           ):
    """Write OPES on a single interatomic distance.

    The one-dimensional case: a bond length or heavy-atom separation
    used directly as the collective variable.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        First atom of the pair. Default is 0.
    idx2 : int, optional
        Second atom of the pair. Default is 1.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.041.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.
    Atom indices are zero-based on the way in and shifted to PLUMED's
    one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    barrier = round_sf(barrier * eV_to_kJpermol)

    idx1 += 1
    idx2 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
opes: {opes_command} ARG=d1 PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'opes.bias']


def write_plumed_mtd_diff1(directory=None,
                           idx1=0,
                           idx2=1,
                           idx3=2,
                           temperature=300,
                           sigma=0.05,
                           pace=10,
                           stride=10,
                           height=0.041,
                           bias=10,
                           ):
    """Write metadynamics on an antisymmetric stretch coordinate.

    Biases the difference between the donor-hydrogen and
    hydrogen-acceptor distances. This coordinate is negative in the
    reactant well, zero at the barrier and positive in the product well,
    so a single collective variable spans the whole transfer.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        Index of the donor atom. Default is 0.
    idx2 : int, optional
        Index of the transferring atom. Default is 1.
    idx3 : int, optional
        Index of the acceptor atom. Default is 2.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    sigma : float, optional
        Gaussian width in nm. Default is 0.05.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    height : float, optional
        Initial hill height in eV. Default is 0.041.
    bias : float, optional
        Well-tempered bias factor. Larger values flatten the surface more
        aggressively but converge more slowly. Default is 10.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.
    Atom indices are zero-based on the way in and shifted to PLUMED's
    one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    height = round_sf(height * eV_to_kJpermol)

    idx1 += 1
    idx2 += 1
    idx3 += 1

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx2},{idx3}
diff: COMBINE ARG=d1,d2 COEFFICIENTS=1,-1 PERIODIC=NO
mtd: METAD ARG=diff PACE={pace} SIGMA={sigma} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'd2', 'diff', 'mtd.bias']


def write_plumed_opes_diff1(directory=None,
                            idx1=0,
                            idx2=1,
                            idx3=2,
                            temperature=300,
                            pace=10,
                            stride=10,
                            barrier=0.041,
                            stride_hills=100,
                            explore=False,
                            ):
    """Write OPES on an antisymmetric stretch coordinate.

    Biases the difference between the donor-hydrogen and
    hydrogen-acceptor distances. This coordinate is negative in the
    reactant well, zero at the barrier and positive in the product well,
    so a single collective variable spans the whole transfer.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        Index of the donor atom. Default is 0.
    idx2 : int, optional
        Index of the transferring atom. Default is 1.
    idx3 : int, optional
        Index of the acceptor atom. Default is 2.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.041.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.
    Atom indices are zero-based on the way in and shifted to PLUMED's
    one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    barrier = round_sf(barrier * eV_to_kJpermol)

    idx1 += 1
    idx2 += 1
    idx3 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx2},{idx3}
diff: COMBINE ARG=d1,d2 COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=diff PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'd2', 'diff', 'opes.bias']


def write_plumed_mtd_diff2(directory=None,
                           idx1=0,
                           idx2=1,
                           idx3=2,
                           idx4=3,
                           idx5=4,
                           idx6=5,
                           temperature=300,
                           sigma=None,
                           pace=10,
                           stride=10,
                           height=0.041,
                           bias=10,
                           ):
    """Write metadynamics on two antisymmetric stretch coordinates.

    The double-transfer analogue of
    :func:`write_plumed_mtd_diff1`: two independent transfers are
    biased at once, so the resulting surface distinguishes stepwise from
    concerted mechanisms.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        Donor of the first transfer. Default is 0.
    idx2 : int, optional
        Transferring atom of the first transfer. Default is 1.
    idx3 : int, optional
        Acceptor of the first transfer. Default is 2.
    idx4 : int, optional
        Donor of the second transfer. Default is 3.
    idx5 : int, optional
        Transferring atom of the second transfer. Default is 4.
    idx6 : int, optional
        Acceptor of the second transfer. Default is 5.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    sigma : list of float, optional
        Gaussian widths for the two coordinates in nm. Default is
        [0.05, 0.05].
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    height : float, optional
        Initial hill height in eV. Default is 0.041.
    bias : float, optional
        Well-tempered bias factor. Larger values flatten the surface more
        aggressively but converge more slowly. Default is 10.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.
    Atom indices are zero-based on the way in and shifted to PLUMED's
    one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    if sigma is None:
        sigma = [0.05, 0.05]

    height = round_sf(height * eV_to_kJpermol)

    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1
    idx5 += 1
    idx6 += 1

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx2},{idx3}

d3: DISTANCE ATOMS={idx4},{idx5}
d4: DISTANCE ATOMS={idx5},{idx6}

diff1: COMBINE ARG=d1,d2 COEFFICIENTS=1,-1 PERIODIC=NO
diff2: COMBINE ARG=d3,d4 COEFFICIENTS=1,-1 PERIODIC=NO

mtd: METAD ARG=diff1,diff2 PACE={pace} SIGMA={sigma[0]},{sigma[1]} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'd2', 'd3', 'd4', 'diff1', 'diff2', 'mtd.bias']


def write_plumed_opes_diff2(directory=None,
                            idx1=0,
                            idx2=1,
                            idx3=2,
                            idx4=3,
                            idx5=4,
                            idx6=5,
                            temperature=300,
                            pace=10,
                            stride=10,
                            barrier=0.041,
                            stride_hills=100,
                            explore=False,
                            ):
    """Write OPES on two antisymmetric stretch coordinates.

    The double-transfer analogue of
    :func:`write_plumed_opes_diff1`: two independent transfers are
    biased at once, so the resulting surface distinguishes stepwise from
    concerted mechanisms.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        Donor of the first transfer. Default is 0.
    idx2 : int, optional
        Transferring atom of the first transfer. Default is 1.
    idx3 : int, optional
        Acceptor of the first transfer. Default is 2.
    idx4 : int, optional
        Donor of the second transfer. Default is 3.
    idx5 : int, optional
        Transferring atom of the second transfer. Default is 4.
    idx6 : int, optional
        Acceptor of the second transfer. Default is 5.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.041.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.
    Atom indices are zero-based on the way in and shifted to PLUMED's
    one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    barrier = round_sf(barrier * eV_to_kJpermol)

    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1
    idx5 += 1
    idx6 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx2},{idx3}

d3: DISTANCE ATOMS={idx4},{idx5}
d4: DISTANCE ATOMS={idx5},{idx6}

diff1: COMBINE ARG=d1,d2 COEFFICIENTS=1,-1 PERIODIC=NO
diff2: COMBINE ARG=d3,d4 COEFFICIENTS=1,-1 PERIODIC=NO

opes: {opes_command} ARG=diff1,diff2 PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d1', 'd2', 'd3', 'd4', 'diff1', 'diff2', 'opes.bias']


def write_plumed_mtd_pt1(atoms,
                         directory=None,
                         idx1=0,
                         idx2=1,
                         temperature=300,
                         sigma=0.005,
                         d_low=1.4,
                         pace=10,
                         stride=10,
                         height=0.041,
                         bias=10,
                         ):
    """Write metadynamics on a single coordination-number difference.

    Biases the difference between the donor and acceptor coordination
    numbers alone, without the heavy-atom distance that
    :func:`write_plumed_mtd_coord` also biases. Suits cases where the
    donor-acceptor separation is already constrained.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the run will start from, used to enumerate the atoms that
        form the coordination shell.
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        Index of the donor atom. Default is 0.
    idx2 : int, optional
        Index of the acceptor atom. Default is 1.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    sigma : float, optional
        Gaussian width for the coordination difference. Default is 0.005.
    d_low : float, optional
        Switching function midpoint in Angstrom. Default is 1.4.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    height : float, optional
        Initial hill height in eV. Default is 0.041.
    bias : float, optional
        Well-tempered bias factor. Larger values flatten the surface more
        aggressively but converge more slowly. Default is 10.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    d_low = d_low * A_to_nm

    height = round_sf(height * eV_to_kJpermol)

    group_idx = list(range(len(atoms)))

    group_idx.remove(idx1)
    group_idx.remove(idx2)

    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]

    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    impt = f"""
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
mtd:   METAD ARG=dc PACE={pace} SIGMA={sigma} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['c1.lessthan', 'c2.lessthan', 'dc', 'mtd.bias']


def write_plumed_opes_pt1(atoms,
                          directory=None,
                          idx1=0,
                          idx2=1,
                          temperature=300,
                          d_low=1.4,
                          pace=10,
                          stride=10,
                          barrier=0.041,
                          stride_hills=100,
                          explore=False,
                          ):
    """Write OPES on a single coordination-number difference.

    Biases the difference between the donor and acceptor coordination
    numbers alone, without the heavy-atom distance that
    :func:`write_plumed_opes_coord` also biases. Suits cases where the
    donor-acceptor separation is already constrained.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the run will start from, used to enumerate the atoms that
        form the coordination shell.
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        Index of the donor atom. Default is 0.
    idx2 : int, optional
        Index of the acceptor atom. Default is 1.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    d_low : float, optional
        Switching function midpoint in Angstrom. Default is 1.4.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.041.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    d_low = d_low * A_to_nm

    barrier = round_sf(barrier * eV_to_kJpermol)

    group_idx = list(range(len(atoms)))

    group_idx.remove(idx1)
    group_idx.remove(idx2)

    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]

    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=dc PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES 

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['c1.lessthan', 'c2.lessthan', 'dc', 'opes.bias']


def write_plumed_mtd_pt2_a(atoms,
                           directory=None,
                           idx1=0,
                           idx2=1,
                           idx3=2,
                           idx4=3,
                           temperature=300,
                           sigma=0.005,
                           d_low=1.4,
                           pace=10,
                           stride=10,
                           height=0.041,
                           bias=10,
                           ):
    """Write metadynamics on two coordination-number differences.

    The double-transfer analogue of
    :func:`write_plumed_mtd_pt1`, biasing one coordination difference
    per proton so the two transfers can proceed independently.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the run will start from, used to enumerate the atoms that
        form the coordination shell.
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        Donor of the first transfer. Default is 0.
    idx2 : int, optional
        Acceptor of the first transfer. Default is 1.
    idx3 : int, optional
        Donor of the second transfer. Default is 2.
    idx4 : int, optional
        Acceptor of the second transfer. Default is 3.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    sigma : float, optional
        Gaussian width for both coordination differences. Default is 0.005.
    d_low : float, optional
        Switching function midpoint in Angstrom. Default is 1.4.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    height : float, optional
        Initial hill height in eV. Default is 0.041.
    bias : float, optional
        Well-tempered bias factor. Larger values flatten the surface more
        aggressively but converge more slowly. Default is 10.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    d_low = d_low * A_to_nm

    height = round_sf(height * eV_to_kJpermol)

    group_idx = list(range(len(atoms)))

    group_idx.remove(idx1)
    group_idx.remove(idx2)
    group_idx.remove(idx3)
    group_idx.remove(idx4)

    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1
    group_idx = [x + 1 for x in group_idx]

    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    impt = f"""
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c3: DISTANCES GROUPA={idx3} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c4: DISTANCES GROUPA={idx4} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
dc2: COMBINE ARG=c3.lessthan,c4.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
mtd:   METAD ARG=dc1,dc2 PACE={pace} SIGMA={sigma} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['c1.lessthan', 'c2.lessthan', 'c3.lessthan', 'c4.lessthan', 'dc1', 'dc2', 'mtd.bias']


def write_plumed_opes_pt2_a(atoms,
                            directory=None,
                            idx1=0,
                            idx2=1,
                            idx3=2,
                            idx4=3,
                            temperature=300,
                            d_low=1.4,
                            pace=10,
                            stride=10,
                            barrier=0.041,
                            stride_hills=100,
                            explore=False,
                            ):
    """Write OPES on two coordination-number differences.

    The double-transfer analogue of
    :func:`write_plumed_opes_pt1`, biasing one coordination difference
    per proton so the two transfers can proceed independently.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the run will start from, used to enumerate the atoms that
        form the coordination shell.
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        Donor of the first transfer. Default is 0.
    idx2 : int, optional
        Acceptor of the first transfer. Default is 1.
    idx3 : int, optional
        Donor of the second transfer. Default is 2.
    idx4 : int, optional
        Acceptor of the second transfer. Default is 3.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    d_low : float, optional
        Switching function midpoint in Angstrom. Default is 1.4.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.041.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    d_low = d_low * A_to_nm

    barrier = round_sf(barrier * eV_to_kJpermol)

    group_idx = list(range(len(atoms)))

    group_idx.remove(idx1)
    group_idx.remove(idx2)
    group_idx.remove(idx3)
    group_idx.remove(idx4)

    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1
    group_idx = [x + 1 for x in group_idx]

    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c3: DISTANCES GROUPA={idx3} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c4: DISTANCES GROUPA={idx4} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
dc2: COMBINE ARG=c3.lessthan,c4.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=dc1,dc2  PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES 

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['c1.lessthan', 'c2.lessthan', 'c3.lessthan', 'c4.lessthan', 'dc1', 'dc2', 'opes.bias']


def write_plumed_mtd_pt_wob(atoms,
                            directory=None,
                            idx1=0,
                            idx2=1,
                            temperature=300,
                            sigma=0.005,
                            d_low=1.4,
                            pace=10,
                            stride=10,
                            height=0.041,
                            bias=10,
                            ):
    """Write metadynamics on a coordination difference, labelling the CV 'dc1'.

    Produces the same biasing scheme as :func:`write_plumed_mtd_pt1` - same
    collective variable, same METAD directive, no wall on either - and
    differs only in naming the combined coordinate ``dc1`` rather than
    ``dc``, so the COLVAR column it returns is named to match. Kept as a
    separate entry point because existing analysis scripts read that column
    by name.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the run will start from, used to enumerate the atoms that
        form the coordination shell.
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        Index of the donor atom. Default is 0.
    idx2 : int, optional
        Index of the acceptor atom. Default is 1.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    sigma : float, optional
        Gaussian width for the coordination difference. Default is 0.005.
    d_low : float, optional
        Switching function midpoint in Angstrom. Default is 1.4.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    height : float, optional
        Initial hill height in eV. Default is 0.041.
    bias : float, optional
        Well-tempered bias factor. Larger values flatten the surface more
        aggressively but converge more slowly. Default is 10.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    d_low = d_low * A_to_nm

    height = round_sf(height * eV_to_kJpermol)

    group_idx = list(range(len(atoms)))

    group_idx.remove(idx1)
    group_idx.remove(idx2)

    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]

    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    impt = f"""
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
mtd: METAD ARG=dc1 PACE={pace} SIGMA={sigma} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['c1.lessthan', 'c2.lessthan', 'dc1', 'mtd.bias']


def write_plumed_opes_pt_wob(atoms,
                             directory=None,
                             idx1=0,
                             idx2=1,
                             temperature=300,
                             d_low=1.4,
                             pace=10,
                             stride=10,
                             barrier=0.041,
                             stride_hills=100,
                             explore=False,
                             ):
    """Write OPES on a coordination difference, labelling the CV 'dc1'.

    Produces the same biasing scheme as :func:`write_plumed_opes_pt1` - same
    collective variable, same OPES directive, no wall on either - and
    differs only in naming the combined coordinate ``dc1`` rather than
    ``dc``, so the COLVAR column it returns is named to match. Kept as a
    separate entry point because existing analysis scripts read that column
    by name.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the run will start from, used to enumerate the atoms that
        form the coordination shell.
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        Index of the donor atom. Default is 0.
    idx2 : int, optional
        Index of the acceptor atom. Default is 1.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    d_low : float, optional
        Switching function midpoint in Angstrom. Default is 1.4.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.041.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    d_low = d_low * A_to_nm

    barrier = round_sf(barrier * eV_to_kJpermol)

    group_idx = list(range(len(atoms)))

    group_idx.remove(idx1)
    group_idx.remove(idx2)

    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]

    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=dc1  PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES 

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['c1.lessthan', 'c2.lessthan', 'dc1', 'opes.bias']


def write_plumed_mtd_pt_wob_sep(atoms,
                                directory=None,
                                idx1=0,
                                idx2=1,
                                list_1=None,
                                list_2=None,
                                temperature=300,
                                sigma=None,
                                d_low=1.4,
                                d_upper=4.0,
                                kappa=0.1,
                                pace=10,
                                stride=10,
                                height=0.041,
                                bias=10,
                                ):
    """Write metadynamics on a coordination difference and a centre-of-mass distance.

    Biases the proton coordination difference together with the
    separation of two molecular fragments, so the transfer and the
    approach of the fragments are sampled jointly. An upper wall keeps
    the fragments from dissociating.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the run will start from, used to enumerate the atoms that
        form the coordination shell.
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        Index of the donor atom. Default is 0.
    idx2 : int, optional
        Index of the acceptor atom. Default is 1.
    list_1 : list of int, optional
        Indices of the atoms making up the first fragment, whose centre of
        mass defines one end of the distance.
    list_2 : list of int, optional
        Indices of the atoms making up the second fragment.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    sigma : list of float, optional
        Gaussian widths for the coordination difference and the distance.
        Default is [0.005, 0.05].
    d_low : float, optional
        Switching function midpoint in Angstrom. Default is 1.4.
    d_upper : float, optional
        Position of the upper wall in Angstrom. Default is 4.0.
    kappa : float, optional
        Upper wall force constant in eV/A^2. Default is 0.1.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    height : float, optional
        Initial hill height in eV. Default is 0.041.
    bias : float, optional
        Well-tempered bias factor. Larger values flatten the surface more
        aggressively but converge more slowly. Default is 10.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    if sigma is None:
        sigma = [0.005, 0.05]

    d_low = d_low * A_to_nm
    d_upper = d_upper * A_to_nm

    height = round_sf(height * eV_to_kJpermol)

    group_idx = list(range(len(atoms)))

    group_idx.remove(idx1)
    group_idx.remove(idx2)

    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]
    list_1 = [x + 1 for x in list_1]
    list_2 = [x + 1 for x in list_2]

    idx_group = ",".join([str(x) for x in group_idx])
    idx_list_1 = ",".join([str(x) for x in list_1])
    idx_list_2 = ",".join([str(x) for x in list_2])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    impt = f"""
com1: COM ATOMS={idx_list_1}
com2: COM ATOMS={idx_list_2}

c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}

d: DISTANCE ATOMS=com1,com2

dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
mtd: METAD ARG=d,dc1 PACE={pace} SIGMA={sigma[0]},{sigma[1]} HEIGHT={height} FILE=HILLS BIASFACTOR={bias} TEMP={temperature}
uwall: UPPER_WALLS ARG=d AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d', 'c1.lessthan', 'c2.lessthan', 'dc1', 'mtd.bias']


def write_plumed_opes_pt_wob_sep(atoms,
                                 directory=None,
                                 idx1=0,
                                 idx2=1,
                                 list_1=None,
                                 list_2=None,
                                 temperature=300,
                                 d_low=1.4,
                                 d_upper=4.0,
                                 kappa=0.1,
                                 pace=10,
                                 stride=10,
                                 barrier=0.041,
                                 stride_hills=100,
                                 explore=False,
                                 ):
    """Write OPES on a coordination difference and a centre-of-mass distance.

    Biases the proton coordination difference together with the
    separation of two molecular fragments, so the transfer and the
    approach of the fragments are sampled jointly. An upper wall keeps
    the fragments from dissociating.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the run will start from, used to enumerate the atoms that
        form the coordination shell.
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        Index of the donor atom. Default is 0.
    idx2 : int, optional
        Index of the acceptor atom. Default is 1.
    list_1 : list of int, optional
        Indices of the atoms making up the first fragment, whose centre of
        mass defines one end of the distance.
    list_2 : list of int, optional
        Indices of the atoms making up the second fragment.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    d_low : float, optional
        Switching function midpoint in Angstrom. Default is 1.4.
    d_upper : float, optional
        Position of the upper wall in Angstrom. Default is 4.0.
    kappa : float, optional
        Upper wall force constant in eV/A^2. Default is 0.1.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.041.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    d_low = d_low * A_to_nm
    d_upper = d_upper * A_to_nm

    barrier = round_sf(barrier * eV_to_kJpermol)

    group_idx = list(range(len(atoms)))

    group_idx.remove(idx1)
    group_idx.remove(idx2)

    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]
    list_1 = [x + 1 for x in list_1]
    list_2 = [x + 1 for x in list_2]

    idx_group = ",".join([str(x) for x in group_idx])
    idx_list_1 = ",".join([str(x) for x in list_1])
    idx_list_2 = ",".join([str(x) for x in list_2])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
com1: COM ATOMS={idx_list_1}
com2: COM ATOMS={idx_list_2}

c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}

d: DISTANCE ATOMS=com1,com2

dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=d,dc1  PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES 
uwall: UPPER_WALLS ARG=d AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d', 'c1.lessthan', 'c2.lessthan', 'dc1', 'opes.bias']


def write_plumed_opes_pt_wob_dist(atoms,
                                  directory=None,
                                  idx1=0,
                                  idx2=1,
                                  idx3=2,
                                  idx4=3,
                                  idx5=4,
                                  idx6=5,
                                  idx7=6,
                                  idx8=7,
                                  temperature=300,
                                  d_low=1.4,
                                  pace=10,
                                  stride=10,
                                  barrier=0.041,
                                  stride_hills=100,
                                  explore=False,
                                  ):
    """Write OPES on a difference of summed distances.

    Four distances are measured, summed in pairs, and their difference
    used as a single collective variable spanning a concerted double
    transfer.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the run will start from, used to enumerate the atoms that
        form the coordination shell.
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx1 : int, optional
        First atom of the first distance. Default is 0.
    idx2 : int, optional
        Second atom of the first distance. Default is 1.
    idx3 : int, optional
        First atom of the second distance. Default is 2.
    idx4 : int, optional
        Second atom of the second distance. Default is 3.
    idx5 : int, optional
        First atom of the third distance. Default is 4.
    idx6 : int, optional
        Second atom of the third distance. Default is 5.
    idx7 : int, optional
        First atom of the fourth distance. Default is 6.
    idx8 : int, optional
        Second atom of the fourth distance. Default is 7.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.
    d_low : float, optional
        Switching function midpoint in Angstrom. Default is 1.4.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.041.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.

    Only `idx1` and `idx2` are shifted to PLUMED's one-based indexing;
    `idx3` through `idx8` are written out unchanged, and `atoms`, `d_low`
    and the coordination groups are computed but unused. An earlier
    coordination-based input is also built and then discarded before the
    distance-based one above is written.
    """
    if directory is None:
        directory = os.getcwd()

    d_low = d_low * A_to_nm

    barrier = round_sf(barrier * eV_to_kJpermol)

    group_idx = list(range(len(atoms)))

    group_idx.remove(idx1)
    group_idx.remove(idx2)

    idx1 += 1
    idx2 += 1
    group_idx = [x + 1 for x in group_idx]

    idx_group = ",".join([str(x) for x in group_idx])

    d_low_line = f"RATIONAL R_0={round_sf(d_low)}"

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{d_low_line}}}
dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=dc1  PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES 

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """

    impt = f"""
# Compute distances between specified atoms
d1: DISTANCE ATOMS={idx1},{idx2}
d2: DISTANCE ATOMS={idx3},{idx4}
d3: DISTANCE ATOMS={idx5},{idx6}
d4: DISTANCE ATOMS={idx7},{idx8}

# Compute sums of distances
sum1: COMBINE ARG=d1,d3 COEFFICIENTS=1,1 PERIODIC=NO
sum2: COMBINE ARG=d2,d4 COEFFICIENTS=1,1 PERIODIC=NO

# Compute the final difference
dc1: COMBINE ARG=sum1,sum2 COEFFICIENTS=1,-1 PERIODIC=NO

opes: {opes_command} ARG=dc1  PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES 

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1

    """

    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['dc1', 'opes.bias']


def write_plumed_opes_com(directory=None,
                          group_1=None,
                          group_2=None,
                          temperature=300.0,
                          pace=10,
                          stride=10,
                          barrier=0.5,
                          d_upper=5.0,
                          kappa=500.0,
                          stride_hills=100,
                          explore=False,
                          ):
    """Write OPES on the distance between two centres of mass.

    Biases the separation of two molecular fragments, with an upper wall
    to stop them drifting apart once unbound.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    group_1 : list of int, optional
        Indices of the atoms in the first fragment.
    group_2 : list of int, optional
        Indices of the atoms in the second fragment.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.0.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.5.
    d_upper : float, optional
        Position of the upper wall in Angstrom. Default is 5.0.
    kappa : float, optional
        Upper wall force constant in eV/A^2. Default is 500.0.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.
    Atom indices are zero-based on the way in and shifted to PLUMED's
    one-based convention.
    """
    if directory is None:
        directory = os.getcwd()
    if group_1 is None:
        group_1 = [0]
    if group_2 is None:
        group_2 = [1]

    barrier = round_sf(barrier * eV_to_kJpermol)
    d_upper = round_sf(d_upper * A_to_nm)

    group_1 = [x + 1 for x in group_1]
    group_2 = [x + 1 for x in group_2]

    group_1 = ",".join([str(x) for x in group_1])
    group_2 = ",".join([str(x) for x in group_2])

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
com1: COM ATOMS={group_1}
com2: COM ATOMS={group_2}
d12: DISTANCE ATOMS=com1,com2
opes: {opes_command} ARG=d12 PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES
upperwall: UPPER_WALLS ARG=d12 AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d12', 'opes.bias', 'upperwall.bias']


def write_plumed_opes_1pt(directory=None,
                          idx_d=0,
                          idx_h=1,
                          idx_a=2,
                          temperature=300.0,
                          pace=10,
                          stride=10,
                          barrier=0.5,
                          d_upper=3.5,
                          kappa=500.0,
                          stride_hills=100,
                          explore=False,
                          ):
    """Write OPES on a single proton transfer, using distances.

    Biases the antisymmetric stretch built from the donor-hydrogen and
    acceptor-hydrogen distances, with an upper wall on the
    donor-hydrogen distance to keep the proton from leaving.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx_d : int, optional
        Index of the donor atom. Default is 0.
    idx_h : int, optional
        Index of the transferring hydrogen. Default is 1.
    idx_a : int, optional
        Index of the acceptor atom. Default is 2.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.0.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.5.
    d_upper : float, optional
        Position of the upper wall in Angstrom. Default is 3.5.
    kappa : float, optional
        Upper wall force constant in eV/A^2. Default is 500.0.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.
    Atom indices are zero-based on the way in and shifted to PLUMED's
    one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    barrier = round_sf(barrier * eV_to_kJpermol)
    d_upper = round_sf(d_upper * A_to_nm)

    idx_d += 1
    idx_h += 1
    idx_a += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d_dh: DISTANCE ATOMS={idx_d},{idx_h}
d_ah: DISTANCE ATOMS={idx_a},{idx_h}
d_da: DISTANCE ATOMS={idx_d},{idx_a}

diff: COMBINE ARG=d_dh,d_ah COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=diff PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

upperwall: UPPER_WALLS ARG=d_da AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d_dh', 'd_ah', 'diff', 'opes.bias', 'upperwall.bias']


def write_plumed_opes_1pt_coord(directory=None,
                                idx_d=0,
                                idx_h=1,
                                idx_a=2,
                                temperature=300.0,
                                pace=10,
                                stride=10,
                                barrier=0.5,
                                r0=1.5,
                                stride_hills=100,
                                explore=False):
    """Write OPES on a single proton transfer, using coordination numbers.

    As :func:`write_plumed_opes_1pt`, but built from switching functions
    rather than raw distances, which bounds the collective variable and
    makes it less sensitive to the heavy-atom separation.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx_d : int, optional
        Index of the donor atom. Default is 0.
    idx_h : int, optional
        Index of the transferring hydrogen. Default is 1.
    idx_a : int, optional
        Index of the acceptor atom. Default is 2.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.0.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.5.
    r0 : float, optional
        Switching function midpoint in Angstrom. Default is 1.5.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    barrier = round_sf(barrier * eV_to_kJpermol)
    r0 = round_sf(r0 * A_to_nm)

    idx_d += 1
    idx_h += 1
    idx_a += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
c_dh: COORDINATION GROUPA={idx_d} GROUPB={idx_h} R_0={r0}
c_ah: COORDINATION GROUPA={idx_a} GROUPB={idx_h} R_0={r0}
c_da: COORDINATION GROUPA={idx_d} GROUPB={idx_a} R_0={r0}

diff: COMBINE ARG=c_dh,c_ah COEFFICIENTS=1,-1 PERIODIC=NO
opes: {opes_command} ARG=diff PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['c_dh', 'c_ah', 'diff', 'opes.bias']


def write_plumed_opes_1pt_3donor_coord(atoms,
                                       directory=None,
                                       idx_n3=0,
                                       idx_h3=1,
                                       idx_o6=2,
                                       idx_o4=2,
                                       idx_n1=3,
                                       idx_h1=4,
                                       idx_o2=5,
                                       idx_n2=6,
                                       temperature=300.0,
                                       pace=10,
                                       stride=10,
                                       barrier=0.5,
                                       r0=1.1,
                                       stride_hills=100,
                                       explore=False,
                                       d_upper=4.0,
                                       kappa=500.0):
    """Write OPES on a proton transfer with three candidate donors.

    Built for a nucleobase pair, where a proton may be shared between
    several heteroatoms. The coordination numbers over all candidate
    sites are combined into one collective variable, so the bias does
    not presuppose which pathway the proton takes.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the run will start from, used to enumerate the atoms that
        form the coordination shell.
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx_n3 : int, optional
        Index of the N3 nitrogen. Default is 0.
    idx_h3 : int, optional
        Index of the hydrogen bound to N3. Default is 1.
    idx_o6 : int, optional
        Index of the O6 oxygen. Default is 2.
    idx_o4 : int, optional
        Index of the O4 oxygen. Default is 2.
    idx_n1 : int, optional
        Index of the N1 nitrogen. Default is 3.
    idx_h1 : int, optional
        Index of the hydrogen bound to N1. Default is 4.
    idx_o2 : int, optional
        Index of the O2 oxygen. Default is 5.
    idx_n2 : int, optional
        Index of the N2 nitrogen. Default is 6.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.0.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.5.
    r0 : float, optional
        Switching function midpoint in Angstrom. Default is 1.1.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.
    d_upper : float, optional
        Position of the upper wall in Angstrom. Default is 4.0.
    kappa : float, optional
        Upper wall force constant in eV/A^2. Default is 500.0.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    barrier = round_sf(barrier * eV_to_kJpermol)

    d_upper = round_sf(d_upper * A_to_nm)

    r_1 = round_sf(get_distance(atoms, idx_n3, idx_h3) * r0 * A_to_nm)
    r_2 = round_sf(get_distance(atoms, idx_o6, idx_h3) * r0 * A_to_nm)
    r_3 = round_sf(get_distance(atoms, idx_o6, idx_h3) * r0 * A_to_nm)
    r_4 = round_sf(get_distance(atoms, idx_o4, idx_h3) * r0 * A_to_nm)
    r_5 = round_sf(get_distance(atoms, idx_n1, idx_h1) * r0 * A_to_nm)
    r_6 = round_sf(get_distance(atoms, idx_n3, idx_h1) * r0 * A_to_nm)
    r_7 = round_sf(get_distance(atoms, idx_n1, idx_o2) * r0 * A_to_nm)
    r_8 = round_sf(get_distance(atoms, idx_n2, idx_o2) * r0 * A_to_nm)
    r_9 = round_sf(get_distance(atoms, idx_n2, idx_o2) * r0 * A_to_nm)
    r_10 = round_sf(get_distance(atoms, idx_n1, idx_n3) * r0 * A_to_nm)

    idx_n3 += 1
    idx_h3 += 1
    idx_o6 += 1
    idx_o4 += 1
    idx_n1 += 1
    idx_h1 += 1
    idx_o2 += 1
    idx_h1 += 1
    idx_n2 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
# z1: top PT reaction coordinate
c_1: COORDINATION GROUPA={idx_n3} GROUPB={idx_h3} R_0={r_1}
c_2: COORDINATION GROUPA={idx_o6} GROUPB={idx_h3} R_0={r_2}

# c_1: DISTANCE ATOMS={idx_n3},{idx_h3}
# c_2: DISTANCE ATOMS={idx_o6},{idx_h3}

z1: COMBINE ARG=c_1,c_2 COEFFICIENTS=1,-1 PERIODIC=NO

# z2: top PT reaction coordinate
c_3: COORDINATION GROUPA={idx_o6} GROUPB={idx_h3} R_0={r_3}
c_4: COORDINATION GROUPA={idx_o4} GROUPB={idx_h3} R_0={r_4}

# c_3: DISTANCE ATOMS={idx_o6},{idx_h3}
# c_4: DISTANCE ATOMS={idx_o4},{idx_h3}

z2: COMBINE ARG=c_3,c_4 COEFFICIENTS=1,-1 PERIODIC=NO

# z3: second PT reaction coordinate
c_5: COORDINATION GROUPA={idx_n1} GROUPB={idx_h1} R_0={r_5}
c_6: COORDINATION GROUPA={idx_n3} GROUPB={idx_h1} R_0={r_6}

# c_5: DISTANCE ATOMS={idx_n1},{idx_h1}
# c_6: DISTANCE ATOMS={idx_n3},{idx_h1}

z3: COMBINE ARG=c_5,c_6 COEFFICIENTS=1,-1 PERIODIC=NO
    
# z4
c_7: COORDINATION GROUPA={idx_n1} GROUPB={idx_o2} R_0={r_7}
c_8: COORDINATION GROUPA={idx_n2} GROUPB={idx_o2} R_0={r_8}

# c_7: DISTANCE ATOMS={idx_n1},{idx_o2}
# c_8: DISTANCE ATOMS={idx_n2},{idx_o2}

z4: COMBINE ARG=c_7,c_8 COEFFICIENTS=1,-1 PERIODIC=NO

# z5
c_9: COORDINATION GROUPA={idx_n2} GROUPB={idx_o2} R_0={r_9}
c_10: COORDINATION GROUPA={idx_n1} GROUPB={idx_n3} R_0={r_10}

# c_9: DISTANCE ATOMS={idx_n2},{idx_o2} 
# c_10: DISTANCE ATOMS={idx_n1},{idx_n3}

z5: COMBINE ARG=c_9,c_10 COEFFICIENTS=1,-1 PERIODIC=NO

z:   COMBINE ARG=z1,z2,z3,z4,z5 COEFFICIENTS=1,1,1,1,1 PERIODIC=NO

opes: {opes_command} ARG=z PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

d1: DISTANCE ATOMS={idx_o6},{idx_o4} 
d2: DISTANCE ATOMS={idx_n1},{idx_n3}
d3: DISTANCE ATOMS={idx_n2},{idx_o2}


uw1: UPPER_WALLS ARG=d1 AT={d_upper} KAPPA={kappa}
uw2: UPPER_WALLS ARG=d2 AT={d_upper} KAPPA={kappa}
uw3: UPPER_WALLS ARG=d3 AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['z', 'opes.bias']


def write_plumed_opes_2pt_2d(directory=None,
                             idx_d1=0,
                             idx_h1=1,
                             idx_a1=2,
                             idx_d2=3,
                             idx_h2=4,
                             idx_a2=5,
                             temperature=300.0,
                             pace=10,
                             stride=10,
                             barrier=0.5,
                             d_upper=3.5,
                             kappa=500.0,
                             stride_hills=100,
                             explore=False,
                             ):
    """Write OPES on a double proton transfer, two dimensions.

    Biases both antisymmetric stretch coordinates independently, giving
    a two-dimensional surface in which stepwise and concerted pathways
    appear as distinct routes between the wells.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx_d1 : int, optional
        Donor of the first transfer. Default is 0.
    idx_h1 : int, optional
        Hydrogen of the first transfer. Default is 1.
    idx_a1 : int, optional
        Acceptor of the first transfer. Default is 2.
    idx_d2 : int, optional
        Donor of the second transfer. Default is 3.
    idx_h2 : int, optional
        Hydrogen of the second transfer. Default is 4.
    idx_a2 : int, optional
        Acceptor of the second transfer. Default is 5.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.0.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.5.
    d_upper : float, optional
        Position of the upper wall in Angstrom. Default is 3.5.
    kappa : float, optional
        Upper wall force constant in eV/A^2. Default is 500.0.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.
    Atom indices are zero-based on the way in and shifted to PLUMED's
    one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    barrier = round_sf(barrier * eV_to_kJpermol)
    d_upper = round_sf(d_upper * A_to_nm)

    idx_d1 += 1
    idx_h1 += 1
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1
    idx_a2 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d_dh1: DISTANCE ATOMS={idx_d1},{idx_h1}
d_ah1: DISTANCE ATOMS={idx_a1},{idx_h1}
d_da1: DISTANCE ATOMS={idx_d1},{idx_a1}

d_dh2: DISTANCE ATOMS={idx_d2},{idx_h2}
d_ah2: DISTANCE ATOMS={idx_a2},{idx_h2}
d_da2: DISTANCE ATOMS={idx_d2},{idx_a2}

diff1: COMBINE ARG=d_dh1,d_ah1 COEFFICIENTS=1,-1 PERIODIC=NO
diff2: COMBINE ARG=d_dh2,d_ah2 COEFFICIENTS=1,-1 PERIODIC=NO

opes: {opes_command} ARG=diff1,diff2 PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

upperwall1: UPPER_WALLS ARG=d_da1 AT={d_upper} KAPPA={kappa}
upperwall2: UPPER_WALLS ARG=d_da2 AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d_dh1', 'd_ah1', 'diff1', 'd_dh2', 'd_ah2', 'diff2', 'opes.bias', 'upperwall1.bias', 'upperwall2.bias']


def write_plumed_opes_2pt_2d_coord(directory=None,
                                   idx_d1=0,
                                   idx_h1=1,
                                   idx_a1=2,
                                   idx_d2=3,
                                   idx_h2=4,
                                   idx_a2=5,
                                   temperature=300.0,
                                   pace=10,
                                   stride=10,
                                   barrier=0.5,
                                   stride_hills=100,
                                   explore=False,
                                   r0=1.5,
                                   ):
    """Write OPES on a double proton transfer, two dimensions, coordination based.

    As :func:`write_plumed_opes_2pt_2d`, but built from switching
    functions rather than raw distances.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx_d1 : int, optional
        Donor of the first transfer. Default is 0.
    idx_h1 : int, optional
        Hydrogen of the first transfer. Default is 1.
    idx_a1 : int, optional
        Acceptor of the first transfer. Default is 2.
    idx_d2 : int, optional
        Donor of the second transfer. Default is 3.
    idx_h2 : int, optional
        Hydrogen of the second transfer. Default is 4.
    idx_a2 : int, optional
        Acceptor of the second transfer. Default is 5.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.0.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.5.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.
    r0 : float, optional
        Switching function midpoint in Angstrom. Default is 1.5.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    barrier = round_sf(barrier * eV_to_kJpermol)
    r0 = round_sf(r0 * A_to_nm)

    idx_d1 += 1
    idx_h1 += 1
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1
    idx_a2 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
# Coordination numbers for each donor–H–acceptor triplet (1-based indices)
c_dh1: COORDINATION GROUPA={idx_d1} GROUPB={idx_h1} R_0={r0}
c_ah1: COORDINATION GROUPA={idx_a1} GROUPB={idx_h1} R_0={r0}
c_da1: COORDINATION GROUPA={idx_d1} GROUPB={idx_a1} R_0={r0} 

c_dh2: COORDINATION GROUPA={idx_d2} GROUPB={idx_h2} R_0={r0} 
c_ah2: COORDINATION GROUPA={idx_a2} GROUPB={idx_h2} R_0={r0} 
c_da2: COORDINATION GROUPA={idx_d2} GROUPB={idx_a2} R_0={r0} 

# Proton-transfer-like coordinates (donor–H minus acceptor–H)
diff1: COMBINE ARG=c_dh1,c_ah1 COEFFICIENTS=1,-1 PERIODIC=NO
diff2: COMBINE ARG=c_dh2,c_ah2 COEFFICIENTS=1,-1 PERIODIC=NO

# 2D OPES bias on (diff1, diff2)
opes: {opes_command} ARG=diff1,diff2 PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
"""
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)

    return ['c_dh1', 'c_ah1', 'diff1', 'c_dh2', 'c_ah2', 'diff2', 'opes.bias']


def write_plumed_opes_2pt_1d(directory=None,
                             idx_d1=0,
                             idx_h1=1,
                             idx_a1=2,
                             idx_d2=3,
                             idx_h2=4,
                             idx_a2=5,
                             temperature=300.0,
                             pace=10,
                             stride=10,
                             barrier=0.5,
                             d_upper=3.5,
                             kappa=500.0,
                             stride_hills=100,
                             explore=False,
                             ):
    """Write OPES on a double proton transfer, one dimension.

    Combines both antisymmetric stretch coordinates into a single
    collective variable. Cheaper to converge than the two-dimensional
    form, at the cost of projecting stepwise and concerted pathways onto
    the same axis.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx_d1 : int, optional
        Donor of the first transfer. Default is 0.
    idx_h1 : int, optional
        Hydrogen of the first transfer. Default is 1.
    idx_a1 : int, optional
        Acceptor of the first transfer. Default is 2.
    idx_d2 : int, optional
        Donor of the second transfer. Default is 3.
    idx_h2 : int, optional
        Hydrogen of the second transfer. Default is 4.
    idx_a2 : int, optional
        Acceptor of the second transfer. Default is 5.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.0.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.5.
    d_upper : float, optional
        Position of the upper wall in Angstrom. Default is 3.5.
    kappa : float, optional
        Upper wall force constant in eV/A^2. Default is 500.0.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Energies are given in eV and converted to the kJ/mol PLUMED expects.
    Atom indices are zero-based on the way in and shifted to PLUMED's
    one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    barrier = round_sf(barrier * eV_to_kJpermol)
    d_upper = round_sf(d_upper * A_to_nm)

    idx_d1 += 1
    idx_h1 += 1
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1
    idx_a2 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
d_dh1: DISTANCE ATOMS={idx_d1},{idx_h1}
d_ah1: DISTANCE ATOMS={idx_a1},{idx_h1}
d_da1: DISTANCE ATOMS={idx_d1},{idx_a1}

d_dh2: DISTANCE ATOMS={idx_d2},{idx_h2}
d_ah2: DISTANCE ATOMS={idx_a2},{idx_h2}
d_da2: DISTANCE ATOMS={idx_d2},{idx_a2}

diff1: COMBINE ARG=d_dh1,d_ah1 COEFFICIENTS=1,-1 PERIODIC=NO
diff2: COMBINE ARG=d_dh2,d_ah2 COEFFICIENTS=1,-1 PERIODIC=NO
pt_cv: COMBINE ARG=diff1,diff2 COEFFICIENTS=0.5,0.5 PERIODIC=NO

opes: {opes_command} ARG=pt_cv PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

upperwall1: UPPER_WALLS ARG=d_da1 AT={d_upper} KAPPA={kappa}
upperwall2: UPPER_WALLS ARG=d_da2 AT={d_upper} KAPPA={kappa}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
    """
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)
    return ['d_dh1', 'd_ah1', 'diff1', 'd_dh2', 'd_ah2', 'diff2', 'pt_cv', 'opes.bias', 'upperwall1.bias',
            'upperwall2.bias']


def write_plumed_opes_2pt_1d_coord(directory=None,
                                   idx_d1=0,
                                   idx_h1=1,
                                   idx_a1=2,
                                   idx_d2=3,
                                   idx_h2=4,
                                   idx_a2=5,
                                   temperature=300.0,
                                   pace=10,
                                   stride=10,
                                   barrier=0.5,
                                   stride_hills=100,
                                   explore=False,
                                   r0=1.5,
                                   ):
    """Write OPES on a double proton transfer, one dimension, coordination based.

    As :func:`write_plumed_opes_2pt_1d`, but built from switching
    functions rather than raw distances.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx_d1 : int, optional
        Donor of the first transfer. Default is 0.
    idx_h1 : int, optional
        Hydrogen of the first transfer. Default is 1.
    idx_a1 : int, optional
        Acceptor of the first transfer. Default is 2.
    idx_d2 : int, optional
        Donor of the second transfer. Default is 3.
    idx_h2 : int, optional
        Hydrogen of the second transfer. Default is 4.
    idx_a2 : int, optional
        Acceptor of the second transfer. Default is 5.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.0.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.5.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.
    r0 : float, optional
        Switching function midpoint in Angstrom. Default is 1.5.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.
    """
    if directory is None:
        directory = os.getcwd()

    barrier = round_sf(barrier * eV_to_kJpermol)
    r0 = round_sf(r0 * A_to_nm)

    idx_d1 += 1
    idx_h1 += 1
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1
    idx_a2 += 1

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
# Coordination numbers (donor–H, acceptor–H, donor–acceptor) for both paths
c_dh1: COORDINATION GROUPA={idx_d1} GROUPB={idx_h1} R_0={r0}
c_ah1: COORDINATION GROUPA={idx_a1} GROUPB={idx_h1} R_0={r0} 
c_da1: COORDINATION GROUPA={idx_d1} GROUPB={idx_a1} R_0={r0} 

c_dh2: COORDINATION GROUPA={idx_d2} GROUPB={idx_h2} R_0={r0} 
c_ah2: COORDINATION GROUPA={idx_a2} GROUPB={idx_h2} R_0={r0}
c_da2: COORDINATION GROUPA={idx_d2} GROUPB={idx_a2} R_0={r0} 

# Two proton-transfer-like coordinates
diff1: COMBINE ARG=c_dh1,c_ah1 COEFFICIENTS=1,-1 PERIODIC=NO
diff2: COMBINE ARG=c_dh2,c_ah2 COEFFICIENTS=1,-1 PERIODIC=NO

# 1D collective variable (average of the two)
pt_cv: COMBINE ARG=diff1,diff2 COEFFICIENTS=0.5,0.5 PERIODIC=NO

# OPES on the 1D CV
opes: {opes_command} ARG=pt_cv PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
"""
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)

    return [
        'c_dh1', 'c_ah1', 'diff1',
        'c_dh2', 'c_ah2', 'diff2',
        'pt_cv', 'opes.bias',
    ]


def write_plumed_opes_2pt_1d_coord_com(directory=None,
                                       idx_d1=0,
                                       idx_h1=1,
                                       idx_a1=2,
                                       idx_d2=3,
                                       idx_h2=4,
                                       idx_a2=5,
                                       group_1=None,
                                       group_2=None,
                                       d_upper=5.0,
                                       kappa=500.0,
                                       temperature=300.0,
                                       pace=10,
                                       stride=10,
                                       barrier=0.5,
                                       stride_hills=100,
                                       explore=False,
                                       r0=1.5,
                                       ):
    """Write OPES on a double proton transfer and a fragment separation.

    Extends :func:`write_plumed_opes_2pt_1d_coord` with the distance
    between two fragment centres of mass, so the transfer is sampled
    alongside the approach of the two molecules. An upper wall stops the
    fragments dissociating.

    Parameters
    ----------
    directory : str, optional
        Directory to write plumed.dat into. Defaults to the current
        working directory.
    idx_d1 : int, optional
        Donor of the first transfer. Default is 0.
    idx_h1 : int, optional
        Hydrogen of the first transfer. Default is 1.
    idx_a1 : int, optional
        Acceptor of the first transfer. Default is 2.
    idx_d2 : int, optional
        Donor of the second transfer. Default is 3.
    idx_h2 : int, optional
        Hydrogen of the second transfer. Default is 4.
    idx_a2 : int, optional
        Acceptor of the second transfer. Default is 5.
    group_1 : list of int, optional
        Indices of the atoms in the first fragment.
    group_2 : list of int, optional
        Indices of the atoms in the second fragment.
    d_upper : float, optional
        Position of the upper wall in Angstrom. Default is 5.0.
    kappa : float, optional
        Upper wall force constant in eV/A^2. Default is 500.0.
    temperature : float, optional
        Simulation temperature in K, used to set the well-tempered bias
        factor. Default is 300.0.
    pace : int, optional
        Steps between bias depositions. Default is 10.
    stride : int, optional
        Steps between COLVAR writes. Default is 10.
    barrier : float, optional
        Expected barrier height in eV, which sets how far OPES will push.
        Default is 0.5.
    stride_hills : int, optional
        STATE is written every ``pace * stride_hills`` steps. Default is 100.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE, which samples the biased
        distribution more broadly at the cost of slower convergence.
        Default is False.
    r0 : float, optional
        Switching function midpoint in Angstrom. Default is 1.5.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input, for i-PI to
        read back.

    Notes
    -----
    Arguments are given in ASE units (eV, Angstrom) and converted to the
    kJ/mol and nm that PLUMED expects. Atom indices are zero-based on the
    way in and shifted to PLUMED's one-based convention.
    """
    if directory is None:
        directory = os.getcwd()
    if group_1 is None:
        group_1 = [0]
    if group_2 is None:
        group_2 = [1]

    barrier = round_sf(barrier * eV_to_kJpermol)
    r0 = round_sf(r0 * A_to_nm)
    d_upper = round_sf(d_upper * A_to_nm)

    idx_d1 += 1
    idx_h1 += 1
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1
    idx_a2 += 1

    group_1 = [x + 1 for x in group_1]
    group_2 = [x + 1 for x in group_2]

    group_1 = ",".join([str(x) for x in group_1])
    group_2 = ",".join([str(x) for x in group_2])

    opes_command = 'OPES_METAD'
    if explore:
        opes_command += '_EXPLORE'

    impt = f"""
# Coordination numbers (donor–H, acceptor–H, donor–acceptor) for both paths
c_dh1: COORDINATION GROUPA={idx_d1} GROUPB={idx_h1} R_0={r0}
c_ah1: COORDINATION GROUPA={idx_a1} GROUPB={idx_h1} R_0={r0} 
c_da1: COORDINATION GROUPA={idx_d1} GROUPB={idx_a1} R_0={r0} 

c_dh2: COORDINATION GROUPA={idx_d2} GROUPB={idx_h2} R_0={r0} 
c_ah2: COORDINATION GROUPA={idx_a2} GROUPB={idx_h2} R_0={r0}
c_da2: COORDINATION GROUPA={idx_d2} GROUPB={idx_a2} R_0={r0} 

# Two proton-transfer-like coordinates
diff1: COMBINE ARG=c_dh1,c_ah1 COEFFICIENTS=1,-1 PERIODIC=NO
diff2: COMBINE ARG=c_dh2,c_ah2 COEFFICIENTS=1,-1 PERIODIC=NO

# 1D collective variable (average of the two)
pt_cv: COMBINE ARG=diff1,diff2 COEFFICIENTS=0.5,0.5 PERIODIC=NO

# Center of mass for the two groups
com1: COM ATOMS={group_1}
com2: COM ATOMS={group_2}
d12: DISTANCE ATOMS=com1,com2

# OPES on the 1D CV
opes: {opes_command} ARG=pt_cv,d12 PACE={pace} BARRIER={barrier} TEMP={temperature} STATE_WFILE=STATE STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES
upperwall: UPPER_WALLS ARG=d12 AT={d_upper} KAPPA={kappa}
PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
"""
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(impt)

    return [
        'c_dh1', 'c_ah1', 'diff1',
        'c_dh2', 'c_ah2', 'diff2',
        'pt_cv', 'opes.bias',
        'd12', 'upperwall.bias',
    ]
