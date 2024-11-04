import xml.etree.ElementTree as ET


def get_masses(atoms, f_deut=False, m_d=2.0141):
    if f_deut:
        print("Using deuterium masses.", flush=True)
        masses = [m_d if symbol == 'H' else mass for symbol, mass in
                  zip(atoms.get_chemical_symbols(), atoms.get_masses())]
        masses = [str(m) for m in masses]
    else:
        masses = atoms.get_masses()
        masses = [str(m) for m in masses]
    return masses


def update_mass(root, atoms, deut=False, m_d=2.0141):
    masses = get_masses(atoms, deut, m_d)
    if deut:
        for rank in root.iter('initialize'):
            masses_element = ET.Element('masses', {'mode': 'manual', 'units': 'ase'})
            masses_element.text = "[" + ", ".join(masses) + "]"
            rank.append(masses_element)
    return masses


def update_cell(root, atoms):
    # Get the cell length
    cell = atoms.get_cell()
    cell_l = cell.lengths()
    for rank in root.iter('initialize'):
        for child in rank:
            if child.tag == "cell":
                child.attrib['units'] = 'angstrom'
                child.text = f"[{cell_l[0]}, {cell_l[1]}, {cell_l[2]}]"
    return None


def update_driver(root, f_driver):
    if f_driver == "ase-mace":
        for rank in root.iter('ffsocket'):
            rank.attrib['name'] = "driver"
            rank.attrib['mode'] = "unix"
            rank.attrib['pbc'] = "False"
            for child in rank:
                if child.tag == "address":
                    child.text = "driver"
        for rank in root.iter('forces'):
            for child in rank:
                if child.tag == "force":
                    child.attrib['forcefield'] = "driver"
    return None


def update_nbeads(root, n_beads):
    for rank in root.iter('initialize'):
        rank.attrib['nbeads'] = str(n_beads)
    return None


def update_hessian(root, n_doft, n_beads):
    for rank in root.iter('motion'):
        for child in rank:
            if child.tag == "instanton":
                for subchild in child:
                    if subchild.tag == "hessian":
                        subchild.attrib['shape'] = f"({int(n_doft)}, {int(n_doft * n_beads)})"
    return None


def update_temperature(root, temperature):
    # Change the temperature
    for rank in root.iter('ensemble'):
        for child in rank:
            if child.tag == "temperature":
                child.text = str(temperature)
    return None


def update_title(root, title):
    # Change the title
    for rank in root.iter('output'):
        rank.set('prefix', title)
    return None


def update_total_steps(root, total_steps):
    # Find the total_steps element and update its text
    for rank in root.iter('total_steps'):
        rank.text = str(total_steps)
    return None


def update_optimizer(root, optimizer_mode):
    # Find the optimizer element and update its mode attribute
    for rank in root.iter('optimizer'):
        rank.set('mode', optimizer_mode)
    return None


def update_tol(root, energy, force, position):
    # Find the optimizer tolerances elements and update their text
    for rank in root.iter('optimizer'):
        for tolerances in rank.iter('tolerances'):
            for child in tolerances:
                if child.tag == "energy":
                    child.text = str(energy)
                elif child.tag == "force":
                    child.text = str(force)
                elif child.tag == "position":
                    child.text = str(position)
    # Find the instanton tolerances elements and update their text
    for rank in root.iter('instanton'):
        for tolerances in rank.iter('tolerances'):
            for child in tolerances:
                if child.tag == "energy":
                    child.text = str(energy)
                elif child.tag == "force":
                    child.text = str(force)
                elif child.tag == "position":
                    child.text = str(position)
    return None


def update_open_paths(root, n_atoms):
    open_paths = list(range(n_atoms))
    # Find the open_paths element and update its text
    for rank in root.iter('normal_modes'):
        for child in rank:
            if child.tag == "open_paths":
                child.text = str(open_paths)
    return None
