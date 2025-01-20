import os
import nqetools as nqe
import xml.etree.ElementTree as ET
import re


def vis_xml(root):
    nqe.write_xml(root, os.path.join(os.getcwd(), 'input.xml'))


def count_matching_words(input_string, target_word):
    pattern = re.escape(target_word)
    return len(re.findall(pattern, input_string))


def test_temperature():
    root = ET.parse(os.path.abspath("../templates/INST.xml")).getroot()
    # Set the temperature
    temperature = -999
    # Update the temperature
    nqe.update_temperature(root, temperature)
    # Check that the temperature is set
    assert count_matching_words(str(ET.tostring(root)), str(temperature)) == 1

    root = ET.parse(os.path.abspath("../templates/NVE.xml")).getroot()
    # Set the temperature
    temperature = -999
    # Update the temperature
    nqe.update_temperature(root, temperature)
    # Check that the temperature is set
    assert count_matching_words(str(ET.tostring(root)), str(temperature)) == 2


def test_timestep():
    root = ET.parse(os.path.abspath("../templates/NVE.xml")).getroot()
    # Set the timestep
    timestep = 0.002
    # Update the timestep
    nqe.update_timestep(root, timestep)

    # Check that the timestep is set
    assert count_matching_words(str(ET.tostring(root)), str(timestep)) == 1


def test_plumed():
    root = ET.parse(os.path.abspath("../templates/NVE.xml")).getroot()
    # Update the plumed file
    nqe.add_plumed(root)

    # Write to file for visual inspection
    # vis_xml(root)

    # Check that ffplumed is in the xml
    assert count_matching_words(str(ET.tostring(root)), "ffplumed") == 2

    # Check that bias has been added to the xml
    assert count_matching_words(str(ET.tostring(root)), "bias") == 2

    # Check that smotion has been added to the xml
    assert count_matching_words(str(ET.tostring(root)), "smotion") == 2
