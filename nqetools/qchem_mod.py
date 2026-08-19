"""Q-Chem calculator with nuclear-electronic orbital (NEO) support.

Extends ASE's Q-Chem ``FileIOCalculator`` to handle NEO calculations, in
which selected nuclei - normally protons - are treated quantum
mechanically alongside the electrons rather than as fixed point charges.
This requires an extra protonic basis set, built here either from explicit
exponents or from one of the standard PB4/PB5/PB6 presets, and a separate
parser for the NEO-SCF energies and gradients Q-Chem reports.
"""

import ase.units
import numpy as np
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import SCFError


def make_neo_basis(neo_basis, neo_idx, neo_exp, atoms):
    """Make the neo basis for a given set of atoms and indices.

    This constructs the string for the neo basis.

    Parameters
    ----------
    neo_basis : str
        The basis to use for the neo basis.
    neo_idx : list
        The indices of the atoms to apply the neo basis to.
    neo_exp : list
        The exponents to use for the neo basis.
    atoms : ase.Atoms
        The atoms object to apply the neo basis to.

    Returns
    -------
    str
        The string for the neo basis.
    """
    neo_basis_list = neo_basis.split("_")
    ele = atoms.get_chemical_symbols()
    out_str = "$neo_basis \n"
    for idx in neo_idx:
        out_str += ele[idx] + " " + str(idx + 1) + "\n"
        for i in range(len(neo_basis_list)):
            for j in range(int(neo_basis_list[i][0])):
                out_str += neo_basis_list[i][-1].upper() + " 1 1.0\n"
                out_str += "  " + str(neo_exp[j]) + " 1.0\n"
        out_str += "****\n"
    out_str += "$end"
    return out_str


def make_neo_basis_presets(neo_basis_name, neo_idx, atoms):
    """Make the neo basis for a given set of atoms and indices.

    This constructs the string for the neo basis using a preset.

    Parameters
    ----------
    neo_basis_name : str
        The name of the neo basis preset to use.
    neo_idx : list
        The indices of the atoms to apply the neo basis to.
    atoms : ase.Atoms
        The atoms object to apply the neo basis to.

    Returns
    -------
    str
        The string for the neo basis.
    """

    # Shell composition of each preset
    neo_basis_dict = {
        "PB4-D": "4s_3p_2d",
        "PB4-F1": "4s_3p_2d_1f",
        "PB4-F2": "4s_3p_2d_2f",
        "PB5-D": "5s_4p_3d",
        "PB5-F": "5s_4p_3d_2f",
        "PB5-G": "5s_4p_3d_2f_1g",
        "PB6-D": "6s_5p_4d",
        "PB6-F": "6s_5p_4d_3f",
        "PB6-G": "6s_5p_4d_3f_2g",
        "PB6-H": "6s_5p_4d_3f_2g_1h",
    }
    # Exponents in shell order, consumed sequentially against the composition
    neo_basis_terms_dict = {
        "PB4-D": [1.957, 8.734, 16.010, 31.997, 9.438, 13.795, 24.028, 10.524, 19.016],
        "PB4-F1": [
            5.973,
            10.645,
            17.943,
            28.950,
            7.604,
            14.701,
            23.308,
            9.011,
            19.787,
            10.914,
        ],
        "PB4-F2": [
            5.973,
            10.645,
            17.943,
            28.950,
            7.604,
            14.701,
            23.308,
            9.011,
            19.787,
            10.914,
            20.985,
        ],
        "PB5-D": [
            1.908,
            9.051,
            15.051,
            29.766,
            40.135,
            4.907,
            10.088,
            15.893,
            25.774,
            10.352,
            18.358,
            36.366,
        ],
        "PB5-F": [
            4.189,
            6.231,
            14.624,
            20.481,
            48.509,
            2.349,
            7.597,
            18.521,
            30.596,
            8.971,
            17.956,
            21.299,
            10.321,
            26.910,
        ],
        "PB5-G": [
            3.283,
            7.613,
            16.077,
            20.457,
            47.011,
            3.167,
            7.785,
            18.985,
            29.425,
            10.239,
            17.298,
            20.682,
            10.930,
            25.972,
            10.513,
        ],
        "PB6-D": [
            2.513,
            4.840,
            9.088,
            16.231,
            31.110,
            38.754,
            2.174,
            5.897,
            9.996,
            16.347,
            25.716,
            2.930,
            11.015,
            17.256,
            25.541,
        ],
        "PB6-F": [
            2.812,
            5.703,
            10.202,
            15.450,
            29.135,
            39.753,
            1.417,
            6.884,
            10.450,
            16.187,
            26.447,
            4.483,
            9.870,
            18.313,
            25.347,
            6.001,
            10.322,
            24.409,
        ],
        "PB6-G": [
            2.016,
            3.049,
            6.350,
            8.910,
            19.500,
            29.296,
            2.638,
            3.716,
            5.978,
            16.252,
            25.114,
            2.205,
            3.654,
            9.104,
            24.039,
            3.845,
            9.921,
            22.570,
            10.062,
            24.267,
        ],
        "PB6-H": [
            1.386,
            3.249,
            9.562,
            12.485,
            21.429,
            36.931,
            1.399,
            4.240,
            9.930,
            18.376,
            24.119,
            2.607,
            5.141,
            7.750,
            20.768,
            0.509,
            9.129,
            26.408,
            9.445,
            28.407,
            10.193,
        ],
    }

    neo_basis_list = neo_basis_dict[neo_basis_name].split("_")
    ele = atoms.get_chemical_symbols()
    out_str = "$neo_basis \n"
    for idx in neo_idx:
        out_str += ele[idx] + " " + str(idx + 1) + "\n"
        iter_exp = iter(neo_basis_terms_dict[neo_basis_name])
        for i in range(len(neo_basis_list)):
            for _j in range(int(neo_basis_list[i][0])):
                out_str += neo_basis_list[i][-1].upper() + " 1 1.0\n"
                out_str += "  " + str(next(iter_exp)) + " 1.0\n"
        out_str += "****\n"
    out_str += "$end"
    return out_str


class QChem(FileIOCalculator):
    """ASE file-IO calculator for Q-Chem, extended for NEO calculations.

    Any keyword argument not listed below is written straight into the
    ``$rem`` block of the input file, upper-cased to match Q-Chem
    convention. Setting ``neo='TRUE'`` additionally emits a ``$neo_basis``
    block built from `neo_preset`, or from `neo_basis` and `neo_exp`.

    Attributes
    ----------
    name : str
        Calculator name used by ASE.
    implemented_properties : list of str
        Properties this calculator can return: 'energy' and 'forces'.
    default_parameters : dict
        Minimal ``$rem`` settings required for a valid Q-Chem job.
    """

    name = "QChem"
    implemented_properties = ["energy", "forces"]
    _legacy_default_command = "qchem PREFIX.inp PREFIX.out"

    # Minimal requirements per the Q-Chem manual, sect-METHOD
    default_parameters = {
        "method": "hf",
        "basis": "6-31G*",
        "jobtype": None,
        "charge": 0,
    }

    def __init__(
        self,
        restart=None,
        ignore_bad_restart_file=FileIOCalculator._deprecated,
        label="qchem",
        scratch=None,
        n_p=1,
        n_t=1,
        pbs=False,
        basisfile=None,
        ecpfile=None,
        atoms=None,
        neo_idx=None,
        neo_preset=None,
        neo_exp=None,
        neo_basis=None,
        solv_extra=None,
        **kwargs,
    ):
        """Initialise the calculator and assemble the Q-Chem command line.

        Parameters
        ----------
        restart : str, optional
            Prefix of a previous calculation to restart from. Default is
            None.
        ignore_bad_restart_file : bool, optional
            Deprecated ASE argument, retained for signature compatibility.
        label : str, optional
            Prefix for the generated .inp, .out and .log files. Default is
            'qchem'.
        scratch : str, optional
            Path of the Q-Chem scratch directory. Default is None.
        n_p : int, optional
            Number of processors for the -np command line flag. Default is 1.
        n_t : int, optional
            Number of threads for the -nt command line flag. Default is 1.
        pbs : bool, optional
            If True, add the -pbs scheduler flag. Default is False.
        basisfile : str, optional
            Path to a file containing the basis. Use together with
            ``basis='gen'``. Default is None.
        ecpfile : str, optional
            Path to a file containing the effective core potential. Use
            together with ``ecp='gen'``. Default is None.
        atoms : ase.Atoms, optional
            Structure to attach the calculator to. Default is None.
        neo_idx : list of int, optional
            Indices of the nuclei to treat quantum mechanically. Default
            is [0].
        neo_preset : str, optional
            Name of a standard protonic basis such as 'PB4-D'. Takes
            precedence over `neo_basis` and `neo_exp`. Default is None.
        neo_exp : list of float, optional
            Explicit protonic basis exponents. Default is [4.0, 8.0].
        neo_basis : str, optional
            Shell composition of the protonic basis, for example "1s_1p".
            Default is "1s_1p".
        solv_extra : str, optional
            Extra text appended to the solvent block. Default is None.
        **kwargs
            Further options written verbatim into the ``$rem`` block,
            upper-cased.
        """

        FileIOCalculator.__init__(
            self, restart, ignore_bad_restart_file, label, atoms, **kwargs
        )

        if neo_idx is None:
            neo_idx = [0]
        if neo_exp is None:
            neo_exp = [4.0, 8.0]
        if neo_basis is None:
            neo_basis = "1s_1p"

        if pbs:
            self.command = "qchem -pbs "
        else:
            self.command = "qchem "
        if n_p != 1:
            self.command += f"-np {n_p} "
        if n_t != 1:
            self.command += f"-nt {n_t} "
        self.command += "PREFIX.inp PREFIX.out"
        if scratch is not None:
            self.command += f" {scratch}"

        self.command += " >> PREFIX.log"

        self.basisfile = basisfile
        self.ecpfile = ecpfile
        self.neo_idx = neo_idx
        self.neo_exp = neo_exp
        self.neo_basis = neo_basis
        self.neo_preset = neo_preset
        self.solv_extra = solv_extra

    def read(self, label):
        """Restore a calculation from previous output files.

        Parameters
        ----------
        label : str
            Prefix of the calculation to read.

        Raises
        ------
        NotImplementedError
            Always. Restarting from Q-Chem output is not supported.
        """
        raise NotImplementedError

    def read_results(self):
        """Parse energy and forces from the Q-Chem output file.

        Handles both conventional SCF and NEO-SCF output, which report
        their energies and gradients under different headings and in
        different layouts. Results are stored in ``self.results``.

        Raises
        ------
        ase.calculators.calculator.SCFError
            If the SCF failed to converge.
        """
        filename = self.label + ".out"

        with open(filename) as fileobj:
            lineiter = iter(fileobj)
            n_atoms = self.atoms.get_global_number_of_atoms()
            # Q-Chem reports atomic units; ASE expects eV and eV/Angstrom
            e_conv = ase.units.Hartree
            f_conv = ase.units.Hartree / ase.units.Bohr

            for line in lineiter:
                if "SCF failed to converge" in line:
                    raise SCFError()
                elif "ERROR: alpha_min" in line:
                    # Not strictly an SCF failure, but equally unusable
                    raise SCFError()
                elif " Total energy in the final basis set =" in line:  # Legacy SCF
                    self.results["energy"] = float(line.split()[8]) * e_conv
                elif " Total energy =" in line:  # SCF
                    self.results["energy"] = float(line.split()[3]) * e_conv
                elif " E(NEO-SCF) =" in line:  # NEO-SCF
                    self.results["energy"] = float(line.split("=")[1]) * e_conv
                elif " Gradient of SCF Energy" in line:
                    # Accumulated as 3 by N, transposed on assignment
                    gradient = [[] for _ in range(3)]
                    next(lineiter)  # Atom numbering header
                    while True:
                        for i in range(3):
                            line = next(lineiter)[
                                5:
                            ].rstrip()  # Drop the component index
                            # Fixed 12-character fields: split() would merge
                            # adjacent columns once gradients grow wide enough
                            # to fill the whitespace between them
                            gradient[i].extend(
                                list(
                                    map(
                                        float,
                                        [
                                            line[i : i + 12]
                                            for i in range(0, len(line), 12)
                                        ],
                                    )
                                )
                            )

                        # Each block of three components is followed by either a
                        # separator line to skip, or the end-of-matrix marker
                        if " Max gradient component" in next(lineiter):
                            self.results["forces"] = -np.array(gradient).T * f_conv
                            break
                elif " NEO-SCF Analytic gradient:" in line:
                    # NEO reports N by 3, one atom per line
                    gradient = np.zeros((n_atoms, 3))
                    next(lineiter)  # Atom numbering header
                    for i in range(n_atoms):
                        gradient[i, :] = np.fromstring(
                            next(lineiter), dtype=float, sep=" "
                        )[1:]
                    self.results["forces"] = -np.array(gradient) * f_conv

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write the Q-Chem input deck.

        Parameters
        ----------
        atoms : ase.Atoms
            Structure to write into the ``$molecule`` block.
        properties : list of str, optional
            Properties requested by ASE. The presence of 'forces' selects
            a FORCE job rather than a single point. Default is None.
        system_changes : list of str, optional
            Changes since the last calculation, passed through to ASE.
            Default is None.
        """
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        filename = self.label + ".inp"

        with open(filename, "w") as fileobj:
            fileobj.write("$comment\n   ASE generated input file\n$end\n\n")
            fileobj.write("$rem\n")
            if self.parameters["jobtype"] is None:
                if "forces" in properties:
                    fileobj.write("   %-25s   %s\n" % ("JOBTYPE", "FORCE"))
                else:
                    fileobj.write("   %-25s   %s\n" % ("JOBTYPE", "SP"))

            for prm in self.parameters:
                if prm not in ["charge", "multiplicity"]:
                    if self.parameters[prm] is not None:
                        fileobj.write(
                            "   %-25s   %s\n"
                            % (prm.upper(), self.parameters[prm].upper())
                        )

            # Forced, not exposed as a parameter: symmetry reorientation would
            # break the correspondence between Q-Chem and ASE atom ordering
            fileobj.write("   %-25s   %s\n" % ("SYM_IGNORE", "TRUE"))
            fileobj.write("$end\n\n")

            fileobj.write("$molecule\n")

            if "multiplicity" not in self.parameters:
                tot_magmom = atoms.get_initial_magnetic_moments().sum()
                mult = tot_magmom + 1
            else:
                mult = self.parameters["multiplicity"]
            fileobj.write("   %d %d\n" % (self.parameters["charge"], mult))
            for a in atoms:
                fileobj.write(f"   {a.symbol}  {a.x:f}  {a.y:f}  {a.z:f}\n")
            fileobj.write("$end\n\n")

            if self.basisfile is not None:
                with open(self.basisfile) as f_in:
                    basis = f_in.readlines()
                fileobj.write("$basis\n")
                fileobj.writelines(basis)
                fileobj.write("$end\n\n")

            if self.ecpfile is not None:
                with open(self.ecpfile) as f_in:
                    ecp = f_in.readlines()
                fileobj.write("$ecp\n")
                fileobj.writelines(ecp)
                fileobj.write("$end\n\n")

            if "solvent_method" in self.parameters:
                if self.solv_extra is not None:
                    fileobj.write("\n")
                    fileobj.write(self.solv_extra)
                    fileobj.write("\n")

            if "neo" in self.parameters:
                if self.parameters["neo"].upper() == "TRUE":
                    if self.neo_preset is None:
                        out_str = make_neo_basis(
                            self.neo_basis, self.neo_idx, self.neo_exp, atoms
                        )
                    else:
                        out_str = make_neo_basis_presets(
                            self.neo_preset, self.neo_idx, atoms
                        )
                    fileobj.write("\n")
                    fileobj.write(out_str)
                    fileobj.write("\n")
