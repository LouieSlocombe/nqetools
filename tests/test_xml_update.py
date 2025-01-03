import os
import nqetools as nqe
import xml.etree.ElementTree as ET
import re


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
