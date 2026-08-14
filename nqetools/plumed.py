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

import inspect
import os

from .conversions import A_to_nm, eV_to_kJpermol, eVperA2_to_kJpermolpernm2
from .tools import round_sf, get_distance


def _metad(args, pace, sigma, height, bias, temperature):
    """The METAD directive that drives a well-tempered metadynamics scheme.

    Parameters
    ----------
    args : str
        Comma-separated names of the collective variables to bias.
    pace : int
        Steps between hill depositions.
    sigma : float or str
        Gaussian width, or a comma-separated pair for a two-dimensional bias.
    height : float
        Hill height in kJ/mol.
    bias : float
        Well-tempered bias factor.
    temperature : float
        Simulation temperature in K.

    Returns
    -------
    str
        One plumed.dat line, labelled ``mtd``.
    """
    return (
        f"mtd: METAD ARG={args} PACE={pace} SIGMA={sigma} HEIGHT={height} "
        f"FILE=HILLS BIASFACTOR={bias} TEMP={temperature}"
    )


def _opes(args, pace, barrier, temperature, stride_hills, explore=False):
    """The OPES directive, in its EXPLORE variant when `explore` is set.

    Parameters
    ----------
    args : str
        Comma-separated names of the collective variables to bias.
    pace : int
        Steps between bias updates.
    barrier : float
        Expected barrier height in kJ/mol.
    temperature : float
        Simulation temperature in K.
    stride_hills : int
        STATE is written every ``pace * stride_hills`` steps.
    explore : bool, optional
        If True, use OPES_METAD_EXPLORE. Default is False.

    Returns
    -------
    str
        One plumed.dat line, labelled ``opes``.
    """
    command = "OPES_METAD_EXPLORE" if explore else "OPES_METAD"
    return (
        f"opes: {command} ARG={args} PACE={pace} BARRIER={barrier} "
        f"TEMP={temperature} STATE_WFILE=STATE "
        f"STATE_WSTRIDE={pace}*{stride_hills} STORE_STATES"
    )


def _upper_wall(name, arg, at, kappa):
    """A one-sided harmonic restraint holding `arg` below `at`.

    Parameters
    ----------
    name : str
        Label for the restraint, which also names its COLVAR bias column.
    arg : str
        Name of the collective variable to restrain.
    at : float
        Position of the wall, in nm.
    kappa : float
        Force constant, in kJ/mol/nm^2.

    Returns
    -------
    str
        One plumed.dat line.
    """
    return f"{name}: UPPER_WALLS ARG={arg} AT={at} KAPPA={kappa}"


def _switching(d_low):
    """The rational switching function the coordination-based schemes use.

    Parameters
    ----------
    d_low : float
        Switching midpoint in Angstrom, converted to nm here.

    Returns
    -------
    str
        A RATIONAL specification, for use inside a ``LESS_THAN={...}`` block.
    """
    return f"RATIONAL R_0={round_sf(d_low * A_to_nm)}"


def _atom_list(indices):
    """Render zero-based atom indices as PLUMED's one-based comma-separated list.

    Parameters
    ----------
    indices : list of int
        Zero-based atom indices.

    Returns
    -------
    str
        One-based indices, comma separated.
    """
    return ",".join(str(i + 1) for i in indices)


def _coordination_group(atoms, *exclude):
    """The atoms forming a coordination shell: everything except `exclude`.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure the run will start from.
    *exclude : int
        Zero-based indices to leave out, typically the donor and acceptor.

    Returns
    -------
    str
        One-based indices of the remaining atoms, comma separated, ready to
        use as a GROUPB.

    Raises
    ------
    ValueError
        If an excluded index is not in the structure.
    """
    group = list(range(len(atoms)))
    for index in exclude:
        group.remove(index)
    return _atom_list(group)


def _write_plumed(directory, lines, stride, colvars):
    """Write plumed.dat from its directive lines, with the standard tail.

    Every scheme ends the same way - print all collective variables to COLVAR
    on a fixed stride, and flush each step so a running job can be watched.

    Parameters
    ----------
    directory : str or None
        Directory to write plumed.dat into. The current working directory is
        used when None.
    lines : list of str
        Directive lines, in the order they should appear. Empty strings become
        blank lines, which PLUMED ignores but which keep the file readable.
    stride : int
        Steps between COLVAR writes.
    colvars : list of str
        Names of the COLVAR columns this input produces.

    Returns
    -------
    list of str
        `colvars` unchanged, for the writer to return to its caller.
    """
    if directory is None:
        directory = os.getcwd()

    body = "\n".join(lines)
    text = f"""
{body}

PRINT ARG=* STRIDE={stride} FILE=COLVAR
FLUSH STRIDE=1
"""
    with open(os.path.join(directory, "plumed.dat"), "w") as f:
        f.write(text)
    return colvars


def _mtd_coordination_difference(
    atoms,
    directory,
    idx1,
    idx2,
    temperature,
    sigma,
    d_low,
    pace,
    stride,
    height,
    bias,
    cv,
):
    """Metadynamics on the donor-acceptor coordination-number difference.

    Shared by :func:`write_plumed_mtd_pt1` and
    :func:`write_plumed_mtd_pt_wob`, which differ only in `cv`.

    Parameters
    ----------
    cv : str
        Label for the combined coordinate, which becomes a COLVAR column name.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input.

    See Also
    --------
    write_plumed_mtd_pt1 : documents the remaining parameters.
    """
    idx_group = _coordination_group(atoms, idx1, idx2)
    switch = _switching(d_low)
    idx1 += 1
    idx2 += 1

    return _write_plumed(
        directory,
        [
            f"c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            f"c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            f"{cv}: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO",
            _metad(
                cv, pace, sigma, round_sf(height * eV_to_kJpermol), bias, temperature
            ),
        ],
        stride,
        ["c1.lessthan", "c2.lessthan", cv, "mtd.bias"],
    )


def _opes_coordination_difference(
    atoms,
    directory,
    idx1,
    idx2,
    temperature,
    d_low,
    pace,
    stride,
    barrier,
    stride_hills,
    explore,
    cv,
):
    """OPES on the donor-acceptor coordination-number difference.

    Shared by :func:`write_plumed_opes_pt1` and
    :func:`write_plumed_opes_pt_wob`, which differ only in `cv`.

    Parameters
    ----------
    cv : str
        Label for the combined coordinate, which becomes a COLVAR column name.

    Returns
    -------
    list of str
        Names of the COLVAR columns written by this input.

    See Also
    --------
    write_plumed_opes_pt1 : documents the remaining parameters.
    """
    idx_group = _coordination_group(atoms, idx1, idx2)
    switch = _switching(d_low)
    idx1 += 1
    idx2 += 1

    return _write_plumed(
        directory,
        [
            f"c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            f"c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            f"{cv}: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO",
            _opes(
                cv,
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
        ],
        stride,
        ["c1.lessthan", "c2.lessthan", cv, "opes.bias"],
    )


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

    See Also
    --------
    SCHEMES : the full mapping of scheme names to writers.
    """
    if plumed_type == "custom":
        with open(os.path.join(plumed_args["directory"], "plumed.dat"), "w") as f:
            f.write(plumed_args["input"])
        return plumed_args["output"]

    try:
        writer = SCHEMES[plumed_type]
    except KeyError:
        raise ValueError(f"Unknown plumed type: {plumed_type}") from None

    # The coordination-number schemes enumerate the structure and so take an
    # Atoms object; the rest work from atom indices alone. Reading that off the
    # signature keeps the two in step without a second table to maintain.
    if next(iter(inspect.signature(writer).parameters)) == "atoms":
        return writer(atoms, **plumed_args)
    return writer(**plumed_args)


def write_plumed_mtd_pos(
    directory=None,
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
    idx_atom += 1

    return _write_plumed(
        directory,
        [
            f"q: POSITION ATOM={idx_atom}",
            _metad(
                "q.x", pace, sigma, round_sf(height * eV_to_kJpermol), bias, temperature
            ),
        ],
        stride,
        ["q.x", "mtd.bias"],
    )


def write_plumed_opes_pos(
    directory=None,
    idx_atom=0,
    pace=20,
    barrier=1.0,
    temperature=300,
    stride=10,
    stride_hills=100,
    explore=False,
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
    idx_atom += 1

    return _write_plumed(
        directory,
        [
            f"q: POSITION ATOM={idx_atom}",
            _opes(
                "q.x",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
        ],
        stride,
        ["q.x", "opes.bias"],
    )


def write_plumed_mtd_coord(
    atoms,
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
    if sigma is None:
        sigma = [0.005, 0.05]

    idx_group = _coordination_group(atoms, idx1, idx2)
    switch = _switching(d_low)
    idx1 += 1
    idx2 += 1

    return _write_plumed(
        directory,
        [
            f"d: DISTANCE ATOMS={idx1},{idx2}",
            f"c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            f"c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            "dc: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO",
            _metad(
                "d,dc",
                pace,
                f"{sigma[0]},{sigma[1]}",
                round_sf(height * eV_to_kJpermol),
                bias,
                temperature,
            ),
            _upper_wall(
                "uwall",
                "d",
                d_upper * A_to_nm,
                round_sf(kappa * eVperA2_to_kJpermolpernm2),
            ),
        ],
        stride,
        ["d", "c1.lessthan", "c2.lessthan", "dc", "mtd.bias"],
    )


def write_plumed_opes_coord(
    atoms,
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
    idx_group = _coordination_group(atoms, idx1, idx2)
    switch = _switching(d_low)
    idx1 += 1
    idx2 += 1

    return _write_plumed(
        directory,
        [
            f"d: DISTANCE ATOMS={idx1},{idx2}",
            f"c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            f"c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            "dc: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO",
            _opes(
                "d,dc",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
            _upper_wall(
                "uwall",
                "d",
                d_upper * A_to_nm,
                round_sf(kappa * eVperA2_to_kJpermolpernm2),
            ),
        ],
        stride,
        ["d", "c1.lessthan", "c2.lessthan", "dc", "opes.bias"],
    )


def write_plumed_mtd_dists(
    directory=None,
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
    if sigma is None:
        sigma = [0.05, 0.05]

    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1

    return _write_plumed(
        directory,
        [
            f"d1: DISTANCE ATOMS={idx1},{idx2}",
            f"d2: DISTANCE ATOMS={idx3},{idx4}",
            _metad(
                "d1,d2",
                pace,
                f"{sigma[0]},{sigma[1]}",
                round_sf(height * eV_to_kJpermol),
                bias,
                temperature,
            ),
        ],
        stride,
        ["d1", "d2", "mtd.bias"],
    )


def write_plumed_opes_dists(
    directory=None,
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
    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1

    return _write_plumed(
        directory,
        [
            f"d1: DISTANCE ATOMS={idx1},{idx2}",
            f"d2: DISTANCE ATOMS={idx3},{idx4}",
            _opes(
                "d1,d2",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
        ],
        stride,
        ["d1", "d2", "opes.bias"],
    )


def write_plumed_mtd_dist(
    directory=None,
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
    idx1 += 1
    idx2 += 1

    return _write_plumed(
        directory,
        [
            f"d1: DISTANCE ATOMS={idx1},{idx2}",
            _metad(
                "d1", pace, sigma, round_sf(height * eV_to_kJpermol), bias, temperature
            ),
        ],
        stride,
        ["d1", "mtd.bias"],
    )


def write_plumed_opes_dist(
    directory=None,
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
    idx1 += 1
    idx2 += 1

    return _write_plumed(
        directory,
        [
            f"d1: DISTANCE ATOMS={idx1},{idx2}",
            _opes(
                "d1",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
        ],
        stride,
        ["d1", "opes.bias"],
    )


def write_plumed_mtd_diff1(
    directory=None,
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
    idx1 += 1
    idx2 += 1
    idx3 += 1

    return _write_plumed(
        directory,
        [
            f"d1: DISTANCE ATOMS={idx1},{idx2}",
            f"d2: DISTANCE ATOMS={idx2},{idx3}",
            "diff: COMBINE ARG=d1,d2 COEFFICIENTS=1,-1 PERIODIC=NO",
            _metad(
                "diff",
                pace,
                sigma,
                round_sf(height * eV_to_kJpermol),
                bias,
                temperature,
            ),
        ],
        stride,
        ["d1", "d2", "diff", "mtd.bias"],
    )


def write_plumed_opes_diff1(
    directory=None,
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
    idx1 += 1
    idx2 += 1
    idx3 += 1

    return _write_plumed(
        directory,
        [
            f"d1: DISTANCE ATOMS={idx1},{idx2}",
            f"d2: DISTANCE ATOMS={idx2},{idx3}",
            "diff: COMBINE ARG=d1,d2 COEFFICIENTS=1,-1 PERIODIC=NO",
            _opes(
                "diff",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
        ],
        stride,
        ["d1", "d2", "diff", "opes.bias"],
    )


def write_plumed_mtd_diff2(
    directory=None,
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
    if sigma is None:
        sigma = [0.05, 0.05]

    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1
    idx5 += 1
    idx6 += 1

    return _write_plumed(
        directory,
        [
            f"d1: DISTANCE ATOMS={idx1},{idx2}",
            f"d2: DISTANCE ATOMS={idx2},{idx3}",
            "",
            f"d3: DISTANCE ATOMS={idx4},{idx5}",
            f"d4: DISTANCE ATOMS={idx5},{idx6}",
            "",
            "diff1: COMBINE ARG=d1,d2 COEFFICIENTS=1,-1 PERIODIC=NO",
            "diff2: COMBINE ARG=d3,d4 COEFFICIENTS=1,-1 PERIODIC=NO",
            "",
            _metad(
                "diff1,diff2",
                pace,
                f"{sigma[0]},{sigma[1]}",
                round_sf(height * eV_to_kJpermol),
                bias,
                temperature,
            ),
        ],
        stride,
        ["d1", "d2", "d3", "d4", "diff1", "diff2", "mtd.bias"],
    )


def write_plumed_opes_diff2(
    directory=None,
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
    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1
    idx5 += 1
    idx6 += 1

    return _write_plumed(
        directory,
        [
            f"d1: DISTANCE ATOMS={idx1},{idx2}",
            f"d2: DISTANCE ATOMS={idx2},{idx3}",
            "",
            f"d3: DISTANCE ATOMS={idx4},{idx5}",
            f"d4: DISTANCE ATOMS={idx5},{idx6}",
            "",
            "diff1: COMBINE ARG=d1,d2 COEFFICIENTS=1,-1 PERIODIC=NO",
            "diff2: COMBINE ARG=d3,d4 COEFFICIENTS=1,-1 PERIODIC=NO",
            "",
            _opes(
                "diff1,diff2",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
        ],
        stride,
        ["d1", "d2", "d3", "d4", "diff1", "diff2", "opes.bias"],
    )


def write_plumed_mtd_pt1(
    atoms,
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
    return _mtd_coordination_difference(
        atoms,
        directory,
        idx1,
        idx2,
        temperature,
        sigma,
        d_low,
        pace,
        stride,
        height,
        bias,
        cv="dc",
    )


def write_plumed_opes_pt1(
    atoms,
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
    return _opes_coordination_difference(
        atoms,
        directory,
        idx1,
        idx2,
        temperature,
        d_low,
        pace,
        stride,
        barrier,
        stride_hills,
        explore,
        cv="dc",
    )


def write_plumed_mtd_pt2_a(
    atoms,
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
    idx_group = _coordination_group(atoms, idx1, idx2, idx3, idx4)
    switch = _switching(d_low)
    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1

    return _write_plumed(
        directory,
        [
            f"c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            f"c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            f"c3: DISTANCES GROUPA={idx3} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            f"c4: DISTANCES GROUPA={idx4} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            "dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO",
            "dc2: COMBINE ARG=c3.lessthan,c4.lessthan COEFFICIENTS=1,-1 PERIODIC=NO",
            _metad(
                "dc1,dc2",
                pace,
                sigma,
                round_sf(height * eV_to_kJpermol),
                bias,
                temperature,
            ),
        ],
        stride,
        [
            "c1.lessthan",
            "c2.lessthan",
            "c3.lessthan",
            "c4.lessthan",
            "dc1",
            "dc2",
            "mtd.bias",
        ],
    )


def write_plumed_opes_pt2_a(
    atoms,
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
    idx_group = _coordination_group(atoms, idx1, idx2, idx3, idx4)
    switch = _switching(d_low)
    idx1 += 1
    idx2 += 1
    idx3 += 1
    idx4 += 1

    return _write_plumed(
        directory,
        [
            f"c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            f"c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            f"c3: DISTANCES GROUPA={idx3} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            f"c4: DISTANCES GROUPA={idx4} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            "dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO",
            "dc2: COMBINE ARG=c3.lessthan,c4.lessthan COEFFICIENTS=1,-1 PERIODIC=NO",
            _opes(
                "dc1,dc2",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
        ],
        stride,
        [
            "c1.lessthan",
            "c2.lessthan",
            "c3.lessthan",
            "c4.lessthan",
            "dc1",
            "dc2",
            "opes.bias",
        ],
    )


def write_plumed_mtd_pt_wob(
    atoms,
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
    return _mtd_coordination_difference(
        atoms,
        directory,
        idx1,
        idx2,
        temperature,
        sigma,
        d_low,
        pace,
        stride,
        height,
        bias,
        cv="dc1",
    )


def write_plumed_opes_pt_wob(
    atoms,
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
    return _opes_coordination_difference(
        atoms,
        directory,
        idx1,
        idx2,
        temperature,
        d_low,
        pace,
        stride,
        barrier,
        stride_hills,
        explore,
        cv="dc1",
    )


def write_plumed_mtd_pt_wob_sep(
    atoms,
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
    if sigma is None:
        sigma = [0.005, 0.05]

    idx_group = _coordination_group(atoms, idx1, idx2)
    switch = _switching(d_low)
    idx1 += 1
    idx2 += 1
    idx_list_1 = ",".join(str(x + 1) for x in list_1)
    idx_list_2 = ",".join(str(x + 1) for x in list_2)

    return _write_plumed(
        directory,
        [
            f"com1: COM ATOMS={idx_list_1}",
            f"com2: COM ATOMS={idx_list_2}",
            "",
            f"c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            f"c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            "",
            "d: DISTANCE ATOMS=com1,com2",
            "",
            "dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO",
            _metad(
                "d,dc1",
                pace,
                f"{sigma[0]},{sigma[1]}",
                round_sf(height * eV_to_kJpermol),
                bias,
                temperature,
            ),
            # Unlike write_plumed_mtd_coord, this scheme has always passed kappa
            # through in eV/A^2 rather than converting it to PLUMED's units.
            _upper_wall("uwall", "d", d_upper * A_to_nm, kappa),
        ],
        stride,
        ["d", "c1.lessthan", "c2.lessthan", "dc1", "mtd.bias"],
    )


def write_plumed_opes_pt_wob_sep(
    atoms,
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
    idx_group = _coordination_group(atoms, idx1, idx2)
    switch = _switching(d_low)
    idx1 += 1
    idx2 += 1
    idx_list_1 = ",".join(str(x + 1) for x in list_1)
    idx_list_2 = ",".join(str(x + 1) for x in list_2)

    return _write_plumed(
        directory,
        [
            f"com1: COM ATOMS={idx_list_1}",
            f"com2: COM ATOMS={idx_list_2}",
            "",
            f"c1: DISTANCES GROUPA={idx1} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            f"c2: DISTANCES GROUPA={idx2} GROUPB={idx_group} LESS_THAN={{{switch}}}",
            "",
            "d: DISTANCE ATOMS=com1,com2",
            "",
            "dc1: COMBINE ARG=c1.lessthan,c2.lessthan COEFFICIENTS=1,-1 PERIODIC=NO",
            _opes(
                "d,dc1",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
            # Unlike write_plumed_opes_coord, this scheme has always passed kappa
            # through in eV/A^2 rather than converting it to PLUMED's units.
            _upper_wall("uwall", "d", d_upper * A_to_nm, kappa),
        ],
        stride,
        ["d", "c1.lessthan", "c2.lessthan", "dc1", "opes.bias"],
    )


def write_plumed_opes_pt_wob_dist(
    atoms,
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
    idx1 += 1
    idx2 += 1
    # idx3 to idx8 are deliberately left as passed, reproducing this scheme's
    # long-standing behaviour: only the first pair was ever shifted to PLUMED's
    # one-based convention. Every other writer in this module shifts all of its
    # indices, so this is very likely an oversight - but changing it would move
    # the bias onto different atoms, so it is left for a deliberate decision.

    return _write_plumed(
        directory,
        [
            "# Compute distances between specified atoms",
            f"d1: DISTANCE ATOMS={idx1},{idx2}",
            f"d2: DISTANCE ATOMS={idx3},{idx4}",
            f"d3: DISTANCE ATOMS={idx5},{idx6}",
            f"d4: DISTANCE ATOMS={idx7},{idx8}",
            "",
            "# Compute sums of distances",
            "sum1: COMBINE ARG=d1,d3 COEFFICIENTS=1,1 PERIODIC=NO",
            "sum2: COMBINE ARG=d2,d4 COEFFICIENTS=1,1 PERIODIC=NO",
            "",
            "# Compute the final difference",
            "dc1: COMBINE ARG=sum1,sum2 COEFFICIENTS=1,-1 PERIODIC=NO",
            "",
            _opes(
                "dc1",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
        ],
        stride,
        ["dc1", "opes.bias"],
    )


def write_plumed_opes_com(
    directory=None,
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
    if group_1 is None:
        group_1 = [0]
    if group_2 is None:
        group_2 = [1]

    return _write_plumed(
        directory,
        [
            f"com1: COM ATOMS={_atom_list(group_1)}",
            f"com2: COM ATOMS={_atom_list(group_2)}",
            "d12: DISTANCE ATOMS=com1,com2",
            _opes(
                "d12",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
            _upper_wall("upperwall", "d12", round_sf(d_upper * A_to_nm), kappa),
        ],
        stride,
        ["d12", "opes.bias", "upperwall.bias"],
    )


def write_plumed_opes_1pt(
    directory=None,
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
    idx_d += 1
    idx_h += 1
    idx_a += 1

    return _write_plumed(
        directory,
        [
            f"d_dh: DISTANCE ATOMS={idx_d},{idx_h}",
            f"d_ah: DISTANCE ATOMS={idx_a},{idx_h}",
            f"d_da: DISTANCE ATOMS={idx_d},{idx_a}",
            "",
            "diff: COMBINE ARG=d_dh,d_ah COEFFICIENTS=1,-1 PERIODIC=NO",
            _opes(
                "diff",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
            "",
            _upper_wall("upperwall", "d_da", round_sf(d_upper * A_to_nm), kappa),
        ],
        stride,
        ["d_dh", "d_ah", "diff", "opes.bias", "upperwall.bias"],
    )


def write_plumed_opes_1pt_coord(
    directory=None,
    idx_d=0,
    idx_h=1,
    idx_a=2,
    temperature=300.0,
    pace=10,
    stride=10,
    barrier=0.5,
    r0=1.5,
    stride_hills=100,
    explore=False,
):
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
    r0 = round_sf(r0 * A_to_nm)

    idx_d += 1
    idx_h += 1
    idx_a += 1

    return _write_plumed(
        directory,
        [
            f"c_dh: COORDINATION GROUPA={idx_d} GROUPB={idx_h} R_0={r0}",
            f"c_ah: COORDINATION GROUPA={idx_a} GROUPB={idx_h} R_0={r0}",
            f"c_da: COORDINATION GROUPA={idx_d} GROUPB={idx_a} R_0={r0}",
            "",
            "diff: COMBINE ARG=c_dh,c_ah COEFFICIENTS=1,-1 PERIODIC=NO",
            _opes(
                "diff",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
        ],
        stride,
        ["c_dh", "c_ah", "diff", "opes.bias"],
    )


def write_plumed_opes_1pt_3donor_coord(
    atoms,
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
    kappa=500.0,
):
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
    d_upper = round_sf(d_upper * A_to_nm)

    def cutoff(first, second):
        """Switching radius scaled from the distance the pair starts at."""
        return round_sf(get_distance(atoms, first, second) * r0 * A_to_nm)

    r_1 = cutoff(idx_n3, idx_h3)
    r_2 = cutoff(idx_o6, idx_h3)
    r_3 = cutoff(idx_o6, idx_h3)
    r_4 = cutoff(idx_o4, idx_h3)
    r_5 = cutoff(idx_n1, idx_h1)
    r_6 = cutoff(idx_n3, idx_h1)
    r_7 = cutoff(idx_n1, idx_o2)
    r_8 = cutoff(idx_n2, idx_o2)
    r_9 = cutoff(idx_n2, idx_o2)
    r_10 = cutoff(idx_n1, idx_n3)

    idx_n3 += 1
    idx_h3 += 1
    idx_o6 += 1
    idx_o4 += 1
    idx_n1 += 1
    idx_h1 += 1
    idx_o2 += 1
    # idx_h1 is incremented a second time here, as it always has been, leaving
    # the H1 index one higher than the one-based convention gives. Almost
    # certainly a copy-paste slip - but correcting it would move the bias onto
    # a different atom, so it is left for a deliberate decision.
    idx_h1 += 1
    idx_n2 += 1

    return _write_plumed(
        directory,
        [
            "# z1: top PT reaction coordinate",
            f"c_1: COORDINATION GROUPA={idx_n3} GROUPB={idx_h3} R_0={r_1}",
            f"c_2: COORDINATION GROUPA={idx_o6} GROUPB={idx_h3} R_0={r_2}",
            "",
            f"# c_1: DISTANCE ATOMS={idx_n3},{idx_h3}",
            f"# c_2: DISTANCE ATOMS={idx_o6},{idx_h3}",
            "",
            "z1: COMBINE ARG=c_1,c_2 COEFFICIENTS=1,-1 PERIODIC=NO",
            "",
            "# z2: top PT reaction coordinate",
            f"c_3: COORDINATION GROUPA={idx_o6} GROUPB={idx_h3} R_0={r_3}",
            f"c_4: COORDINATION GROUPA={idx_o4} GROUPB={idx_h3} R_0={r_4}",
            "",
            f"# c_3: DISTANCE ATOMS={idx_o6},{idx_h3}",
            f"# c_4: DISTANCE ATOMS={idx_o4},{idx_h3}",
            "",
            "z2: COMBINE ARG=c_3,c_4 COEFFICIENTS=1,-1 PERIODIC=NO",
            "",
            "# z3: second PT reaction coordinate",
            f"c_5: COORDINATION GROUPA={idx_n1} GROUPB={idx_h1} R_0={r_5}",
            f"c_6: COORDINATION GROUPA={idx_n3} GROUPB={idx_h1} R_0={r_6}",
            "",
            f"# c_5: DISTANCE ATOMS={idx_n1},{idx_h1}",
            f"# c_6: DISTANCE ATOMS={idx_n3},{idx_h1}",
            "",
            "z3: COMBINE ARG=c_5,c_6 COEFFICIENTS=1,-1 PERIODIC=NO",
            "",
            "# z4",
            f"c_7: COORDINATION GROUPA={idx_n1} GROUPB={idx_o2} R_0={r_7}",
            f"c_8: COORDINATION GROUPA={idx_n2} GROUPB={idx_o2} R_0={r_8}",
            "",
            f"# c_7: DISTANCE ATOMS={idx_n1},{idx_o2}",
            f"# c_8: DISTANCE ATOMS={idx_n2},{idx_o2}",
            "",
            "z4: COMBINE ARG=c_7,c_8 COEFFICIENTS=1,-1 PERIODIC=NO",
            "",
            "# z5",
            f"c_9: COORDINATION GROUPA={idx_n2} GROUPB={idx_o2} R_0={r_9}",
            f"c_10: COORDINATION GROUPA={idx_n1} GROUPB={idx_n3} R_0={r_10}",
            "",
            f"# c_9: DISTANCE ATOMS={idx_n2},{idx_o2}",
            f"# c_10: DISTANCE ATOMS={idx_n1},{idx_n3}",
            "",
            "z5: COMBINE ARG=c_9,c_10 COEFFICIENTS=1,-1 PERIODIC=NO",
            "",
            "z: COMBINE ARG=z1,z2,z3,z4,z5 COEFFICIENTS=1,1,1,1,1 PERIODIC=NO",
            "",
            _opes(
                "z",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
            "",
            f"d1: DISTANCE ATOMS={idx_o6},{idx_o4}",
            f"d2: DISTANCE ATOMS={idx_n1},{idx_n3}",
            f"d3: DISTANCE ATOMS={idx_n2},{idx_o2}",
            "",
            _upper_wall("uw1", "d1", d_upper, kappa),
            _upper_wall("uw2", "d2", d_upper, kappa),
            _upper_wall("uw3", "d3", d_upper, kappa),
        ],
        stride,
        ["z", "opes.bias"],
    )


def write_plumed_opes_2pt_2d(
    directory=None,
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
    d_upper = round_sf(d_upper * A_to_nm)

    idx_d1 += 1
    idx_h1 += 1
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1
    idx_a2 += 1

    return _write_plumed(
        directory,
        [
            f"d_dh1: DISTANCE ATOMS={idx_d1},{idx_h1}",
            f"d_ah1: DISTANCE ATOMS={idx_a1},{idx_h1}",
            f"d_da1: DISTANCE ATOMS={idx_d1},{idx_a1}",
            "",
            f"d_dh2: DISTANCE ATOMS={idx_d2},{idx_h2}",
            f"d_ah2: DISTANCE ATOMS={idx_a2},{idx_h2}",
            f"d_da2: DISTANCE ATOMS={idx_d2},{idx_a2}",
            "",
            "diff1: COMBINE ARG=d_dh1,d_ah1 COEFFICIENTS=1,-1 PERIODIC=NO",
            "diff2: COMBINE ARG=d_dh2,d_ah2 COEFFICIENTS=1,-1 PERIODIC=NO",
            "",
            _opes(
                "diff1,diff2",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
            "",
            _upper_wall("upperwall1", "d_da1", d_upper, kappa),
            _upper_wall("upperwall2", "d_da2", d_upper, kappa),
        ],
        stride,
        [
            "d_dh1",
            "d_ah1",
            "diff1",
            "d_dh2",
            "d_ah2",
            "diff2",
            "opes.bias",
            "upperwall1.bias",
            "upperwall2.bias",
        ],
    )


def write_plumed_opes_2pt_2d_coord(
    directory=None,
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
    r0 = round_sf(r0 * A_to_nm)

    idx_d1 += 1
    idx_h1 += 1
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1
    idx_a2 += 1

    return _write_plumed(
        directory,
        [
            "# Coordination numbers for each donor–H–acceptor triplet (1-based indices)",
            f"c_dh1: COORDINATION GROUPA={idx_d1} GROUPB={idx_h1} R_0={r0}",
            f"c_ah1: COORDINATION GROUPA={idx_a1} GROUPB={idx_h1} R_0={r0}",
            f"c_da1: COORDINATION GROUPA={idx_d1} GROUPB={idx_a1} R_0={r0}",
            "",
            f"c_dh2: COORDINATION GROUPA={idx_d2} GROUPB={idx_h2} R_0={r0}",
            f"c_ah2: COORDINATION GROUPA={idx_a2} GROUPB={idx_h2} R_0={r0}",
            f"c_da2: COORDINATION GROUPA={idx_d2} GROUPB={idx_a2} R_0={r0}",
            "",
            "# Proton-transfer-like coordinates (donor–H minus acceptor–H)",
            "diff1: COMBINE ARG=c_dh1,c_ah1 COEFFICIENTS=1,-1 PERIODIC=NO",
            "diff2: COMBINE ARG=c_dh2,c_ah2 COEFFICIENTS=1,-1 PERIODIC=NO",
            "",
            "# 2D OPES bias on (diff1, diff2)",
            _opes(
                "diff1,diff2",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
        ],
        stride,
        ["c_dh1", "c_ah1", "diff1", "c_dh2", "c_ah2", "diff2", "opes.bias"],
    )


def write_plumed_opes_2pt_1d(
    directory=None,
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
    d_upper = round_sf(d_upper * A_to_nm)

    idx_d1 += 1
    idx_h1 += 1
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1
    idx_a2 += 1

    return _write_plumed(
        directory,
        [
            f"d_dh1: DISTANCE ATOMS={idx_d1},{idx_h1}",
            f"d_ah1: DISTANCE ATOMS={idx_a1},{idx_h1}",
            f"d_da1: DISTANCE ATOMS={idx_d1},{idx_a1}",
            "",
            f"d_dh2: DISTANCE ATOMS={idx_d2},{idx_h2}",
            f"d_ah2: DISTANCE ATOMS={idx_a2},{idx_h2}",
            f"d_da2: DISTANCE ATOMS={idx_d2},{idx_a2}",
            "",
            "diff1: COMBINE ARG=d_dh1,d_ah1 COEFFICIENTS=1,-1 PERIODIC=NO",
            "diff2: COMBINE ARG=d_dh2,d_ah2 COEFFICIENTS=1,-1 PERIODIC=NO",
            "pt_cv: COMBINE ARG=diff1,diff2 COEFFICIENTS=0.5,0.5 PERIODIC=NO",
            "",
            _opes(
                "pt_cv",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
            "",
            _upper_wall("upperwall1", "d_da1", d_upper, kappa),
            _upper_wall("upperwall2", "d_da2", d_upper, kappa),
        ],
        stride,
        [
            "d_dh1",
            "d_ah1",
            "diff1",
            "d_dh2",
            "d_ah2",
            "diff2",
            "pt_cv",
            "opes.bias",
            "upperwall1.bias",
            "upperwall2.bias",
        ],
    )


def write_plumed_opes_2pt_1d_coord(
    directory=None,
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
    r0 = round_sf(r0 * A_to_nm)

    idx_d1 += 1
    idx_h1 += 1
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1
    idx_a2 += 1

    return _write_plumed(
        directory,
        [
            "# Coordination numbers (donor–H, acceptor–H, donor–acceptor) for both paths",
            f"c_dh1: COORDINATION GROUPA={idx_d1} GROUPB={idx_h1} R_0={r0}",
            f"c_ah1: COORDINATION GROUPA={idx_a1} GROUPB={idx_h1} R_0={r0}",
            f"c_da1: COORDINATION GROUPA={idx_d1} GROUPB={idx_a1} R_0={r0}",
            "",
            f"c_dh2: COORDINATION GROUPA={idx_d2} GROUPB={idx_h2} R_0={r0}",
            f"c_ah2: COORDINATION GROUPA={idx_a2} GROUPB={idx_h2} R_0={r0}",
            f"c_da2: COORDINATION GROUPA={idx_d2} GROUPB={idx_a2} R_0={r0}",
            "",
            "# Two proton-transfer-like coordinates",
            "diff1: COMBINE ARG=c_dh1,c_ah1 COEFFICIENTS=1,-1 PERIODIC=NO",
            "diff2: COMBINE ARG=c_dh2,c_ah2 COEFFICIENTS=1,-1 PERIODIC=NO",
            "",
            "# 1D collective variable (average of the two)",
            "pt_cv: COMBINE ARG=diff1,diff2 COEFFICIENTS=0.5,0.5 PERIODIC=NO",
            "",
            "# OPES on the 1D CV",
            _opes(
                "pt_cv",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
        ],
        stride,
        ["c_dh1", "c_ah1", "diff1", "c_dh2", "c_ah2", "diff2", "pt_cv", "opes.bias"],
    )


def write_plumed_opes_2pt_1d_coord_com(
    directory=None,
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
    if group_1 is None:
        group_1 = [0]
    if group_2 is None:
        group_2 = [1]

    r0 = round_sf(r0 * A_to_nm)
    d_upper = round_sf(d_upper * A_to_nm)

    idx_d1 += 1
    idx_h1 += 1
    idx_a1 += 1
    idx_d2 += 1
    idx_h2 += 1
    idx_a2 += 1

    return _write_plumed(
        directory,
        [
            "# Coordination numbers (donor–H, acceptor–H, donor–acceptor) for both paths",
            f"c_dh1: COORDINATION GROUPA={idx_d1} GROUPB={idx_h1} R_0={r0}",
            f"c_ah1: COORDINATION GROUPA={idx_a1} GROUPB={idx_h1} R_0={r0}",
            f"c_da1: COORDINATION GROUPA={idx_d1} GROUPB={idx_a1} R_0={r0}",
            "",
            f"c_dh2: COORDINATION GROUPA={idx_d2} GROUPB={idx_h2} R_0={r0}",
            f"c_ah2: COORDINATION GROUPA={idx_a2} GROUPB={idx_h2} R_0={r0}",
            f"c_da2: COORDINATION GROUPA={idx_d2} GROUPB={idx_a2} R_0={r0}",
            "",
            "# Two proton-transfer-like coordinates",
            "diff1: COMBINE ARG=c_dh1,c_ah1 COEFFICIENTS=1,-1 PERIODIC=NO",
            "diff2: COMBINE ARG=c_dh2,c_ah2 COEFFICIENTS=1,-1 PERIODIC=NO",
            "",
            "# 1D collective variable (average of the two)",
            "pt_cv: COMBINE ARG=diff1,diff2 COEFFICIENTS=0.5,0.5 PERIODIC=NO",
            "",
            "# Center of mass for the two groups",
            f"com1: COM ATOMS={_atom_list(group_1)}",
            f"com2: COM ATOMS={_atom_list(group_2)}",
            "d12: DISTANCE ATOMS=com1,com2",
            "",
            "# OPES on the 1D CV",
            _opes(
                "pt_cv,d12",
                pace,
                round_sf(barrier * eV_to_kJpermol),
                temperature,
                stride_hills,
                explore,
            ),
            _upper_wall("upperwall", "d12", d_upper, kappa),
        ],
        stride,
        [
            "c_dh1",
            "c_ah1",
            "diff1",
            "c_dh2",
            "c_ah2",
            "diff2",
            "pt_cv",
            "opes.bias",
            "d12",
            "upperwall.bias",
        ],
    )


# Every biasing scheme :func:`prep_plumed` accepts, mapped to the writer that
# emits it. Defined after the writers so it can reference them directly. The
# 'custom' scheme is handled separately, since it writes a caller-supplied
# input rather than generating one.
SCHEMES = {
    "mtd-pos": write_plumed_mtd_pos,
    "opes-pos": write_plumed_opes_pos,
    "mtd-coord": write_plumed_mtd_coord,
    "opes-coord": write_plumed_opes_coord,
    "mtd-dists": write_plumed_mtd_dists,
    "opes-dists": write_plumed_opes_dists,
    "mtd-dist": write_plumed_mtd_dist,
    "opes-dist": write_plumed_opes_dist,
    "mtd-diff1": write_plumed_mtd_diff1,
    "opes-diff1": write_plumed_opes_diff1,
    "mtd-diff2": write_plumed_mtd_diff2,
    "opes-diff2": write_plumed_opes_diff2,
    "mtd-pt1": write_plumed_mtd_pt1,
    "opes-pt1": write_plumed_opes_pt1,
    "mtd-pt2_a": write_plumed_mtd_pt2_a,
    "opes-pt2_a": write_plumed_opes_pt2_a,
    "mtd-pt-wob": write_plumed_mtd_pt_wob,
    "opes-pt-wob": write_plumed_opes_pt_wob,
    "mtd-pt-wob-sep": write_plumed_mtd_pt_wob_sep,
    "opes-pt-wob-sep": write_plumed_opes_pt_wob_sep,
    "opes-pt-wob-dist": write_plumed_opes_pt_wob_dist,
    "opes_com": write_plumed_opes_com,
    "opes_1pt": write_plumed_opes_1pt,
    "opes_1pt_coord": write_plumed_opes_1pt_coord,
    "opes_1pt_3donor_coord": write_plumed_opes_1pt_3donor_coord,
    "opes_2pt_2d": write_plumed_opes_2pt_2d,
    "opes_2pt_2d_coord": write_plumed_opes_2pt_2d_coord,
    "opes_2pt_1d": write_plumed_opes_2pt_1d,
    "opes_2pt_1d_coord": write_plumed_opes_2pt_1d_coord,
    "opes_2pt_1d_coord_com": write_plumed_opes_2pt_1d_coord_com,
}
