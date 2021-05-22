import os

import numpy as np
from yaml import load, Loader


# Calibration helper functions for SenseGlove
# Refer to scripts/calibrate_senseglove.py for generation of calibration file


def load_calibration(c_file):
    """Load and return SenseGlove calibration from specified calibration file

    Args:
        c_file (str): Absolute or Relative path (from the current working directory) to the calibration file

    Returns:
        dict: A dictionary containing the calibration values
    """
    # Set the path for the calibration file
    if not os.path.isabs(c_file):
        c_file = os.path.join(os.getcwd(), c_file)

    with open(c_file, 'r') as calibration_file:
        calibration = load(calibration_file, Loader=Loader)
    return calibration


def apply_calibration(fingers_dict, calibration, output_range=(-1, 1)):
    """Applies calibration limits to raw values, to obtain output in a specified range

    Args:
        fingers_dict (dict): Dictionary containing raw sensor values ([prox med dist]) for joints in each finger
        calibration (dict): Dictionary of calibration values for the SenseGlove (refer to load_calibration())

    Returns:
        dict: Dictionary with values for each joint scaled to the output range
    """
    fingers_dict = fingers_dict.copy()
    for finger, values in fingers_dict.items():
        for i in range(len(values)):
            fingers_dict[finger][i] = np.interp(values[i],
                                                (calibration[finger]['min'][i], calibration[finger]['max'][i]),
                                                output_range)
    return fingers_dict
