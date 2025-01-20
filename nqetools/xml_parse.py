import xml.etree.ElementTree as ET
from .tools import has_pbc


def get_masses(atoms, f_deut=False, m_d=2.0141):
    """
    Retrieves the masses of atoms, optionally using deuterium masses for hydrogen atoms.

    Parameters:
    atoms (object): An object representing the atoms, which must have `get_chemical_symbols` and `get_masses` methods.
    f_deut (bool, optional): Flag to indicate whether to use deuterium masses for hydrogen atoms. Default is False.
    m_d (float, optional): The mass of deuterium to use if `f_deut` is True. Default is 2.0141.

    Returns:
    list of str: A list of masses as strings.
    """
    if f_deut:
        print("Using deuterium masses.", flush=True)
        masses = [m_d if symbol == 'H' else mass for symbol, mass in
                  zip(atoms.get_chemical_symbols(), atoms.get_masses())]
    else:
        masses = atoms.get_masses()
    return [str(m) for m in masses]


def update_mass(root, atoms, deut=False, m_d=2.0141):
    """
    Updates the XML tree with the masses of atoms, optionally using deuterium masses for hydrogen atoms.

    Parameters:
    root (Element): The root element of the XML tree.
    atoms (object): An object representing the atoms, which must have `get_chemical_symbols` and `get_masses` methods.
    deut (bool, optional): Flag to indicate whether to use deuterium masses for hydrogen atoms. Default is False.
    m_d (float, optional): The mass of deuterium to use if `deut` is True. Default is 2.0141.

    Returns:
    list of str: A list of masses as strings.
    """
    masses = get_masses(atoms, deut, m_d)
    if deut:
        masses_element = ET.Element('masses', {'mode': 'manual', 'units': 'ase'})
        masses_element.text = "[" + ", ".join(masses) + "]"
        for rank in root.iter('initialize'):
            rank.append(masses_element)
    return masses


def update_cell(root, atoms):
    """
    Updates the XML tree with the cell dimensions of the atoms.

    Parameters:
    root (Element): The root element of the XML tree.
    atoms (object): An object representing the atoms, which must have a `get_cell` method that returns an object with a `lengths` method.

    Returns:
    None
    """
    cell_l = atoms.get_cell().lengths()
    for rank in root.iter('initialize'):
        for child in rank:
            if child.tag == "cell":
                child.attrib['units'] = 'angstrom'
                child.text = f"[{cell_l[0]}, {cell_l[1]}, {cell_l[2]}]"
    return None


def update_driver(root, atoms, f_driver):
    """
    Updates the XML tree with driver information based on the specified driver type.

    Parameters:
    root (Element): The root element of the XML tree.
    atoms (object): An object representing the atoms, which must have periodic boundary conditions (PBC).
    f_driver (str): The type of driver to use.

    Returns:
    None
    """
    if f_driver in ["ase-mace", "ase-nwchem", "ase-orca"]:
        f_pbcs = has_pbc(atoms)
        for rank in root.iter('ffsocket'):
            rank.attrib.update({'name': 'driver', 'mode': 'unix', 'pbc': str(f_pbcs)})
            for child in rank:
                if child.tag == "address":
                    child.text = "driver"
        for rank in root.iter('forces'):
            for child in rank:
                if child.tag == "force":
                    child.attrib['forcefield'] = "driver"
    return None


def update_nbeads(root, n_beads):
    """
    Updates the XML tree to set the number of beads in the 'initialize' elements.

    Parameters:
    root (Element): The root element of the XML tree.
    n_beads (int): The number of beads to set in the 'initialize' elements.

    Returns:
    None
    """
    for rank in root.iter('initialize'):
        rank.set('nbeads', str(n_beads))
    return None


def update_hessian(root, n_doft, n_beads):
    """
    Updates the shape attribute of the hessian elements in the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    n_doft (int): The number of degrees of freedom.
    n_beads (int): The number of beads.

    Returns:
    None
    """
    for subchild in root.iterfind('.//motion/instanton/hessian'):
        subchild.attrib['shape'] = f"({int(n_doft)}, {int(n_doft * n_beads)})"
    return None


def update_temperature(root, temperature):
    """
    Updates the temperature element in the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    temperature (float): The temperature value to set.

    Returns:
    None
    """
    # update the temperature in the ensemble section
    for child in root.iterfind('.//ensemble/temperature'):
        child.text = str(temperature)
    # update the temperature in the velocity section
    for child in root.iterfind('.//initialize/velocities'):
        child.text = str(temperature)
    return None


def update_title(root, title):
    """
    Updates the XML tree to set the title in the 'output' elements.

    Parameters:
    root (Element): The root element of the XML tree.
    title (str): The title to set in the 'output' elements.

    Returns:
    None
    """
    # Change the title
    for rank in root.iter('output'):
        rank.set('prefix', title)
    return None


def update_total_steps(root, total_steps):
    """
    Updates the total_steps element in the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    total_steps (int): The total number of steps to set.

    Returns:
    None
    """
    # Find the total_steps element and update its text
    for rank in root.iter('total_steps'):
        rank.text = str(total_steps)
    return None


def update_optimizer(root, optimizer_mode):
    """
    Updates the optimizer element in the XML tree to set the mode attribute.

    Parameters:
    root (Element): The root element of the XML tree.
    optimizer_mode (str): The mode to set in the optimizer element.

    Returns:
    None
    """
    # Find the optimizer element and update its mode attribute
    for rank in root.iter('optimizer'):
        rank.set('mode', optimizer_mode)
    return None


def update_tol(root, energy, force, position):
    """
    Updates the tolerances elements in the XML tree for both optimizer and instanton elements.

    Parameters:
    root (Element): The root element of the XML tree.
    energy (float): The energy tolerance value to set.
    force (float): The force tolerance value to set.
    position (float): The position tolerance value to set.

    Returns:
    None
    """
    for rank in root.iter('optimizer'):
        for child in rank.iter('tolerances'):
            child.find('energy').text = str(energy)
            child.find('force').text = str(force)
            child.find('position').text = str(position)
    for rank in root.iter('instanton'):
        for child in rank.iter('tolerances'):
            child.find('energy').text = str(energy)
            child.find('force').text = str(force)
            child.find('position').text = str(position)
    return None


def update_open_paths(root, n_atoms):
    """
    Updates the open_paths elements in the XML tree within normal_modes.

    Parameters:
    root (Element): The root element of the XML tree.
    n_atoms (int): The number of atoms to generate the open paths list.

    Returns:
    None
    """
    open_paths = list(range(n_atoms))
    for child in root.iterfind('.//normal_modes/open_paths'):
        child.text = str(open_paths)
    return None


def update_timestep(root, timestep):
    """
    Updates the timestep element in the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    timestep (float): The timestep value to set.

    Returns:
    None
    """
    for child in root.iterfind('.//motion/dynamics/timestep'):
        child.text = str(timestep)
    return None


def update_stride(root, stride):
    """
    Updates all instances of the stride element in the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    stride (float): The new stride value to set.

    Returns:
    None
    """
    for child in root.iterfind('.//stride'):
        child.text = str(stride)
    return None


def add_ffplumed_section(root, name="plumed", file_mode="xyz", file_name="init.xyz", plumed_dat="plumed/plumed.dat"):
    """
    Adds a section for ffplumed to the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    name (str, optional): The name attribute for the ffplumed section. Default is "plumed".
    file_mode (str, optional): The mode attribute for the file element. Default is "xyz".
    file_name (str, optional): The file name for the file element. Default is "init.xyz".
    plumed_dat (str, optional): The text for the plumeddat element. Default is "plumed/plumed.dat".

    Returns:
    None
    """
    ffplumed = ET.Element('ffplumed', {'name': name})
    file_element = ET.SubElement(ffplumed, 'file', {'mode': file_mode})
    file_element.text = file_name
    plumed_dat_element = ET.SubElement(ffplumed, 'plumeddat')
    plumed_dat_element.text = plumed_dat
    root.append(ffplumed)
    return None


def add_bias_section(root, forcefield="plumed", nbeads="1"):
    """
    Adds a bias section under the ensemble tag in the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    forcefield (str, optional): The forcefield attribute for the force element. Default is "plumed".
    nbeads (str, optional): The nbeads attribute for the force element. Default is "1".

    Returns:
    None
    """
    bias = ET.Element('bias')
    force_element = ET.SubElement(bias, 'force', {'forcefield': forcefield, 'nbeads': nbeads})
    for ensemble in root.iter('ensemble'):
        ensemble.append(bias)
    return None


def add_smotion_section(root, mode="metad", metaff="[ plumed ]"):
    """
    Adds a smotion section under the simulation tag in the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    mode (str, optional): The mode attribute for the smotion element. Default is "metad".
    metaff (str, optional): The text for the metaff element. Default is "[ plumed ]".

    Returns:
    None
    """
    smotion = ET.Element('smotion', {'mode': mode})
    metad = ET.SubElement(smotion, 'metad')
    metaff_element = ET.SubElement(metad, 'metaff')
    metaff_element.text = metaff
    for simulation in root.iter('simulation'):
        simulation.append(smotion)
    return None
