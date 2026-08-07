import os
import re
import xml.etree.ElementTree as et

import ase.build

import nqetools as nqe

base_dir = nqe.find_nqetools_path()


def vis_xml(root):
    """Writes the XML tree to a file named 'input.xml' in the current working directory.

    Parameters
    ----------
    root : Element
        The root element of the XML tree.

    Returns
    -------
    None
    """
    nqe.write_xml(root, os.path.join(os.getcwd(), 'input.xml'))


def count_matching_words(input_string, target_word):
    """Counts the number of occurrences of a target word in an input string.

    Parameters
    ----------
    input_string : str
        The string in which to search for the target word.
    target_word : str
        The word to search for in the input string.

    Returns
    -------
    int
        The number of times the target word appears in the input string.
    """
    pattern = re.escape(target_word)
    return len(re.findall(pattern, input_string))


def test_update_file():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    nqe.update_file(root, "test.pdb")

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), "test.pdb") == 1
    assert count_matching_words(str(et.tostring(root)), "pdb") == 2


def test_update_cell():
    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=20.0)
    cell_in = atoms.get_cell()
    if nqe.has_pbc(atoms):
        print("The system has periodic boundary conditions.")

    directory = "opti"
    nqe.remove_directory(directory)
    atoms_out = nqe.run_optimise(directory, atoms, driver='ase-mace', total_steps=1)
    atoms = atoms_out[-1]
    if nqe.has_pbc(atoms):
        print("The system has periodic boundary conditions.")

    atoms.set_pbc(False)

    cell_out = atoms.get_cell()
    print(cell_in)
    print(cell_out)
    assert all(all((a - b) < 0.1 for a, b in zip(cell_in[i], cell_out[i], strict=True)) for i in range(3))


def test_update_properties():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    properties = ["-99", "-98", "-97"]
    nqe.update_properties(root, properties)

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), "-99") == 1
    assert count_matching_words(str(et.tostring(root)), "-98") == 1
    assert count_matching_words(str(et.tostring(root)), "-97") == 1


def test_append_properties():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    properties = ["-99", "-98", "-97"]
    nqe.append_properties(root, properties)

    # Write to file for visual inspection
    vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), "-99") == 1
    assert count_matching_words(str(et.tostring(root)), "-98") == 1
    assert count_matching_words(str(et.tostring(root)), "-97") == 1


def test_update_temperature():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    temperature = -999
    nqe.update_temperature(root, temperature)
    assert count_matching_words(str(et.tostring(root)), str(temperature)) == 1

    root = et.parse(os.path.abspath("../templates/NVE.xml")).getroot()
    temperature = -999
    nqe.update_temperature(root, temperature)

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), str(temperature)) == 2


def test_update_timestep():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    timestep = 0.002
    nqe.update_timestep(root, timestep)

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), str(timestep)) == 1


def test_update_stride():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    stride = -99
    nqe.update_stride(root, stride)

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), str(stride)) == 2


def test_update_nbeads():
    root = et.parse(os.path.join(base_dir, "templates/NVT-PIMD.xml")).getroot()

    nqe.add_plumed_xml(root)

    n_beads = -99
    nqe.update_nbeads(root, n_beads)

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), str(n_beads)) == 1


def test_update_checkpoint_stride():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    checkpoint_stride = -99
    nqe.update_checkpoint_stride(root, checkpoint_stride)

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), str(checkpoint_stride)) == 1


def test_add_trajectory_centroid():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    nqe.add_trajectory_centroid(root)

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), "centroid") == 1


def test_add_trajectory_plumed_extras():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    nqe.add_trajectory_plumed_extras(root, plumed_extras=["doo", "dc", "mtd.bias"])

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), "extras_bias") == 1


def test_add_plumed_bias_section():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    nqe.add_plumed_bias_section(root, plumed_extras=["doo", "dc", "mtd.bias"])

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), "interpolate_extras") == 2
    assert count_matching_words(str(et.tostring(root)), "doo") == 1


def test_add_plumed_ff_section():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    nqe.add_plumed_ff_section(root, plumed_extras=["doo", "dc", "mtd.bias"])

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), "plumed_extras") == 2
    assert count_matching_words(str(et.tostring(root)), "doo") == 1


def test_add_plumed_xml():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    nqe.add_plumed_xml(root)

    # Write to file for visual inspection
    # vis_xml(root)

    # Check that ffplumed is in the xml
    assert count_matching_words(str(et.tostring(root)), "ffplumed") == 2

    # Check that bias has been added to the xml
    assert count_matching_words(str(et.tostring(root)), "bias") == 2

    # Check that smotion has been added to the xml
    assert count_matching_words(str(et.tostring(root)), "smotion") == 2


def test_add_trajectory_file():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    nqe.add_trajectory_file(root, filename="kin", stride=1, text='kinetic_cv')

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), "kinetic_cv") == 1


def test_add_thermostat_section():
    root = et.parse(os.path.join(base_dir, "templates/NVT.xml")).getroot()
    nqe.add_thermostat_section(root)

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), "thermostat") == 2


def test_update_dynamics_splitting():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    nqe.update_dynamics_splitting(root, splitting="baoab")

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), "baoab") == 1


def test_update_motion_fixcom():
    root = et.parse(os.path.join(base_dir, "templates/NVE.xml")).getroot()
    nqe.update_motion_fix_com(root, fix_com=True)

    # Write to file for visual inspection
    # vis_xml(root)

    assert count_matching_words(str(et.tostring(root)), "True") == 1
