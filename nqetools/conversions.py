from scipy.constants import physical_constants as const
from ase import Atoms

# Conversion factor from Bohr to Angstrom
bohr_to_angstrom = const["Bohr radius"][0] * 1e10
# Conversion factor between Angstrom and nm
A_to_nm = 1.0e-1  # Å to nm
# Conversion factor from eV to J
eV_to_J = const["electron volt-joule relationship"][0]
# Conversion factor from J to kJ
J_to_kJ = 1.0e-3  # 1 J = 0.001 kJ
# Avogadro's number
avo_num = const["Avogadro constant"][0]  # mol^-1

# Conversion factor from eV to 1 kJ/mol
eV_to_kJpermol = eV_to_J * J_to_kJ * avo_num
# Conversion factor from eV/Å² to 1 kJ/(mol·nm²)
eVperA2_to_kJpermolpernm2 = eV_to_kJpermol / A_to_nm ** 2


def convert_atom_list_bohr_to_angstrom(atoms_list: list[Atoms]) -> list[Atoms]:
    """
    Convert positions and cell parameters from Bohr to Angstrom for a list of ASE Atoms objects.

    Parameters:
    atoms_list (list[ase.Atoms]): List of ASE Atoms objects with positions and cells in Bohr

    Returns:
    list[ase.Atoms]: List of ASE Atoms objects with positions and cells in Angstrom
    """

    converted_atoms = []
    for atoms in atoms_list:
        atoms_copy = atoms.copy()
        atoms_copy.positions *= bohr_to_angstrom
        atoms_copy.cell *= bohr_to_angstrom
        converted_atoms.append(atoms_copy)

    return converted_atoms
