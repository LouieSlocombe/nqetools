import os
import xml.etree.ElementTree as ET

from .io import find_nqetools_path
from .tools import has_pbc, get_file_extension


def list_to_string(l):
    """
    Converts a list to a string with brackets and commas.

    Parameters:
    l (list): The list to convert to a string.

    Returns:
    str: The string representation of the list.
    """
    return '[' + ', '.join(l) + ']'


def update_properties(root, prop_list):
    """
    Updates the XML tree with properties from a given list.

    Parameters:
    root (Element): The root element of the XML tree.
    prop_list (list of str): A list of properties to update in the XML tree.

    Returns:
    None
    """
    for properties in root.iter('properties'):
        properties.text = list_to_string(prop_list)
    return None


def append_properties(root, prop_list):
    """
    Loads the current list of properties in the given root and appends a new input prop_list.

    Parameters:
    root (Element): The root element of the XML tree.
    prop_list (list of str): A list of properties to append to the existing properties in the XML tree.

    Returns:
    None
    """
    for properties in root.iter('properties'):
        # Load current properties
        current_props = properties.text.split(',')
        current_props = [prop.strip().strip('[').strip(']').strip('\n').strip() for prop in current_props]
        # Append new properties
        updated_props = current_props + prop_list
        # Update the properties text
        properties.text = list_to_string(updated_props)
    return None


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


def update_mass(root, atoms, f_deut=False, m_d=2.0141):
    """
    Updates the XML tree with the masses of atoms, optionally using deuterium masses for hydrogen atoms.

    Parameters:
    root (Element): The root element of the XML tree.
    atoms (object): An object representing the atoms, which must have `get_chemical_symbols` and `get_masses` methods.
    f_deut (bool, optional): Flag to indicate whether to use deuterium masses for hydrogen atoms. Default is False.
    m_d (float, optional): The mass of deuterium to use if `f_deut` is True. Default is 2.0141.

    Returns:
    list of str: A list of masses as strings.
    """
    masses = get_masses(atoms, f_deut, m_d)
    if f_deut:
        masses_element = ET.Element('masses', {'mode': 'manual', 'units': 'ase'})
        masses_element.text = list_to_string(masses)
        for rank in root.iter('initialize'):
            rank.append(masses_element)
    return masses


def update_file(root, filename='init.xyz', units='angstrom'):
    """
    Updates the 'file' element within 'initialize' elements in the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    filename (str, optional): The new filename to set for the 'file' element. Default is 'init.xyz'.
    units (str, optional): The units to set for the 'file' element. Default is 'angstrom'.

    Returns:
    None
    """
    for rank in root.iter('initialize'):
        for child in rank:
            if child.tag == "file":
                child.text = filename
                child.set('mode', get_file_extension(filename).replace('.', ''))
                child.set('units', units)
    return None


def update_cell(root, atoms):
    """
    Updates the XML tree with the cell dimensions of the atoms. If no cell tag is present, it adds it.

    Parameters:
    root (Element): The root element of the XML tree.
    atoms (object): An object representing the atoms, which must have a `get_cell` method that returns an object with a `lengths` method.

    Returns:
    None
    """
    cell_l = atoms.get_cell().lengths()
    for rank in root.iter('initialize'):
        cell_element = rank.find('cell')
        if cell_element is None:
            cell_element = ET.SubElement(rank, 'cell')
        cell_element.attrib['units'] = 'angstrom'
        cell_element.attrib['mode'] = 'abc'
        cell_element.text = f'[{cell_l[0]}, {cell_l[1]}, {cell_l[2]}]'
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
    # Check if the driver is valid
    assert f_driver in ["zundel", "cbe", "ase-mace", "ase-nwchem", "ase-orca", "nwchem"]
    f_pbcs = has_pbc(atoms)
    if f_driver == "zundel":
        for rank in root.iter('ffsocket'):
            rank.attrib.update({'name': 'driver', 'mode': 'unix'})
            rank.attrib.pop('pbc', None)
            for child in rank:
                if child.tag == 'address':
                    child.text = 'zundel'
        for rank in root.iter('forces'):
            for child in rank:
                if child.tag == 'force':
                    child.attrib['forcefield'] = 'driver'
    elif f_driver == "cbe":
        for rank in root.iter('ffsocket'):
            rank.attrib.update({'name': 'cbe', 'mode': 'unix'})
            rank.attrib.pop('pbc', None)
            for child in rank:
                if child.tag == 'address':
                    child.text = 'localhost'
        for rank in root.iter('forces'):
            for child in rank:
                if child.tag == 'force':
                    child.attrib['forcefield'] = 'cbe'
    elif f_driver in ["ase-mace", "ase-nwchem", "ase-orca", "nwchem"]:
        for rank in root.iter('ffsocket'):
            rank.attrib.update({'name': 'driver', 'mode': 'unix', 'pbc': str(f_pbcs)})
            for child in rank:
                if child.tag == 'address':
                    child.text = 'driver'
        for rank in root.iter('forces'):
            for child in rank:
                if child.tag == 'force':
                    child.attrib['forcefield'] = 'driver'
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
        subchild.attrib['shape'] = f'({int(n_doft)}, {int(n_doft * n_beads)})'
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
    Updates the stride attribute for 'properties' and 'trajectory' elements in the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    stride (int): The new stride value to set for the elements.

    Returns:
    None
    """
    for element in root.iter():
        if element.tag in ['properties', 'trajectory']:
            element.set('stride', str(stride))
    return None


def update_checkpoint_stride(root, stride):
    """
    Updates the stride attribute of the checkpoint elements within the output tags in the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    stride (int): The new stride value to set for the checkpoint elements.

    Returns:
    None
    """
    for output in root.iter('output'):
        for checkpoint in output.iter('checkpoint'):
            checkpoint.set('stride', str(stride))


def find_parent(root, child):
    """
    Finds the parent element of a given child element in an XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    child (Element): The child element for which to find the parent.

    Returns:
    Element: The parent element of the given child, or None if no parent is found.
    """
    for parent in root.iter():
        if child in parent:
            return parent
    return None


def add_plumed_smotion_section(root):
    """
    Adds a smotion section under the simulation tag in the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.

    Returns:
    None
    """
    smotion = ET.Element('smotion', {'mode': 'metad'})
    metad = ET.SubElement(smotion, 'metad')
    metaff_element = ET.SubElement(metad, 'metaff')
    metaff_element.text = '[ plumed ]'
    for simulation in root.iter('simulation'):
        simulation.append(smotion)
    return None


def add_trajectory_centroid(root, stride="10", filename="xc", text="x_centroid{angstrom}"):
    """
    Adds a <trajectory> element to the <output> tag in the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    stride (str, optional): The stride attribute for the trajectory element. Default is "10".
    filename (str, optional): The filename attribute for the trajectory element. Default is "xc".
    text (str, optional): The text content for the trajectory element. Default is "x_centroid{angstrom}".

    Returns:
    None
    """
    trajectory = ET.Element('trajectory', {
        'stride': str(stride),
        'filename': filename,
        'format': 'xyz'
    })
    trajectory.text = text
    for output in root.iter('output'):
        output.append(trajectory)
    return None


def add_trajectory_plumed_extras(root, plumed_extras, stride=10):
    """
    Adds a <trajectory> element to the <output> tag in the XML tree with specified plumed extras.

    Parameters:
    root (Element): The root element of the XML tree.
    plumed_extras (list of str): A list of extra types to include in the trajectory element.
    stride (int, optional): The stride attribute for the trajectory element. Default is 10.

    Returns:
    None
    """
    trajectory = ET.Element('trajectory', {
        'stride': str(stride),
        'filename': 'colvar',
        'bead': '0',
        'extra_type': ','.join(plumed_extras)
    })
    trajectory.text = 'extras_bias'
    for output in root.iter('output'):
        output.append(trajectory)
    return None


def add_plumed_ff_section(root, plumed_extras=None, file_name="init.xyz", plumed_dat="plumed.dat"):
    """
    Adds a PLUMED force field section to the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    plumed_extras (list of str, optional): A list of extra types to include in the PLUMED section. Default is None.
    file_name (str, optional): The name of the file to use in the PLUMED section. Default is "init.xyz".
    plumed_dat (str, optional): The name of the PLUMED data file to use. Default is "plumed.dat".

    Returns:
    None
    """
    # Get the ffplumed section
    ffplumed = ET.Element('ffplumed', {'name': 'plumed'})

    # Add the file element
    file_element = ET.SubElement(ffplumed, 'file', {'mode': 'xyz'})
    file_element.text = file_name

    # Add the plumedplumed_datdat element
    plumed_dat_element = ET.SubElement(ffplumed, 'plumed_dat')
    plumed_dat_element.text = plumed_dat

    # Add the plumed_extras element
    if plumed_extras is not None:
        plumed_plumed_extras_element = ET.SubElement(ffplumed, 'plumed_extras')
        plumed_plumed_extras_element.text = '[' + ','.join(plumed_extras) + ']'

    # Insert the ffplumed section after the ffsocket element
    for ffsocket in root.iter('ffsocket'):
        parent = find_parent(root, ffsocket)
        if parent is not None:
            index = list(parent).index(ffsocket)
            parent.insert(index + 1, ffplumed)
    return None


def add_plumed_bias_section(root, plumed_extras=None, nbeads=1):
    """
    Adds a bias section to the XML tree under the ensemble tag.

    Parameters:
    root (Element): The root element of the XML tree.
    plumed_extras (list of str, optional): A list of extra types to include in the interpolate_extras element. Default is None.
    nbeads (int, optional): The number of beads to set in the force element. Default is 1.

    Returns:
    None
    """
    # Get the bias section
    bias = ET.Element('bias')

    # Create the force sub-element with the specified attributes
    force = ET.SubElement(bias, 'force', {'forcefield': 'plumed', 'nbeads': str(nbeads)})
    if plumed_extras is not None:
        # Add the plumed_extras element
        plumed_extras_element = ET.SubElement(force, 'interpolate_extras')
        plumed_extras_element.text = list_to_string(plumed_extras)

    # Properly insert the bias section
    for ensemble in root.iter('ensemble'):
        for temperature in ensemble.iter('temperature'):
            index = list(ensemble).index(temperature)
            ensemble.insert(index + 1, bias)
    return None


def add_plumed_xml(root, plumed_extras=None, file_name="init.xyz", plumed_dat="plumed.dat"):
    """
    Adds PLUMED-related sections to the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    plumed_extras (list of str, optional): A list of extra types to include in the PLUMED sections. Default is None.
    file_name (str, optional): The name of the file to use in the PLUMED sections. Default is "init.xyz".
    plumed_dat (str, optional): The name of the PLUMED data file to use. Default is "plumed.dat".

    Returns:
    None
    """
    # Update the plumed file
    add_plumed_ff_section(root,
                          plumed_extras=plumed_extras,
                          file_name=file_name,
                          plumed_dat=plumed_dat)
    # Add the bias section
    add_plumed_bias_section(root,
                            plumed_extras=plumed_extras,
                            nbeads=1)
    # Add the smotion section
    add_plumed_smotion_section(root)

    # Add the trajectory element
    add_trajectory_plumed_extras(root, plumed_extras)
    return None


def add_trajectory_file(root, filename='pos', stride=20, text='positions'):
    """
    Adds a <trajectory> element to the <output> tag in the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    filename (str, optional): The filename attribute for the trajectory element. Default is 'pos'.
    stride (int, optional): The stride attribute for the trajectory element. Default is 20.
    text (str, optional): The text content for the trajectory element. Default is 'positions'.

    Returns:
    None
    """
    # Create the new trajectory element
    new_trajectory = ET.Element('trajectory', {'filename': filename, 'stride': str(stride)})
    new_trajectory.text = text

    # Find the output element and append the new trajectory element
    output_element = root.find(".//output")
    if output_element is not None:
        output_element.append(new_trajectory)
    else:
        raise ValueError("The output element was not found in the XML file.")
    return None


def add_thermostat_section(root, thermostat="smart_sampling_1ps_n6_w2", xml_path=None):
    """
    Adds or updates the <thermostat> section in the XML tree.

    Parameters:
    root (Element): The root element of the XML tree.
    thermostat (str, optional): The name of the thermostat file (without extension) to use. Default is "smart_sampling_1ps_n6_w2".
    xml_path (str, optional): The path to the directory containing the thermostat XML files. Default is None.

    Returns:
    None
    """
    # Get the path to the thermostat XML files
    if xml_path is None:
        xml_path = os.path.join(find_nqetools_path(), "thermostats")

    # Parse the XML file
    tree = ET.parse(os.path.join(xml_path, thermostat + ".xml"))

    # Remove existing thermostat section
    for thermostat in root.iter('thermostat'):
        parent = find_parent(root, thermostat)
        if parent is not None:
            parent.remove(thermostat)

    # Add thermostat section to the root
    for thermostat in tree.iter('thermostat'):
        for rank in root.iter('dynamics'):
            rank.append(thermostat)
    return None
