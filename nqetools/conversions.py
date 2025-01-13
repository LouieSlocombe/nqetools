from scipy.constants import physical_constants as const

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

# Conversion factor from eV/Å² to 1 kJ/(mol·nm²)
eV_to_kJpermol = eV_to_J * J_to_kJ * avo_num
# Conversion factor from eV/Å² to 1 kJ/(mol·nm²)
eVperA2_to_kJpermolpernm2 = eV_to_kJpermol / A_to_nm ** 2
