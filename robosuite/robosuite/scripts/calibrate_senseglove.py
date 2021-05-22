import argparse
import os
import sys
from time import time
import yaml

from robosuite.devices import SGDeviceManager, SenseGlove


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration (in seconds) to collect data for calibration")
    parser.add_argument("--right", action="store_true", help="Use right hand SenseGlove")
    parser.add_argument("--output-file", type=str, default="senseglove_calibration.yaml",
                        help="Output path for calibration file")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing calibration file with same name")
    args = parser.parse_args()

    # Set the path for the calibration file
    if os.path.isabs(args.output_file):
        output_file = args.output_file
    else:
        output_file = os.path.join(os.getcwd(), args.output_file)

    if os.path.exists(output_file) and not args.overwrite:
        print("Calibration file '{}' already exists. Please provide a new path or use the --overwrite flag",
              output_file)
        sys.exit(1)
    else:
        # Open file for writing
        calibration_file = open(output_file, 'w')

    # Initialize the SenseGlove Device Manager to scan for SenseGloves
    sg_dev_manager = SGDeviceManager()
    # Get the required glove
    sg_glove = sg_dev_manager.get_glove(right=args.right)

    # If not found...exit
    if sg_glove is None:
        print("Failed to detect {} glove! Please ensure it is connected and try again...".
              format('right' if args.right else 'left'))
        sys.exit(1)

    # Initialize the SenseGlove object
    glove = SenseGlove(sg_glove)

    # Print warning to user to put on the appropriate glove and wait for user input before continuing
    input('Please wear the {} glove and then click [ENTER] to start the calibration process...'.
          format('right' if args.right else 'left'))

    print('Open and close hand repeatedly over desired behaviour to calibrate the device')

    # Initialize dictionary to hold calibration values
    # The format will be as follows
    # {
    #     'THUMB': {'min': [-0.4, -0.1, -0.3], 'max': [0.9, 1.0, 0.7]},
    #     'INDEX': {'min': [0.0, -0.4, 0.2], 'max': [0.8, 0.9, 0.85]},
    #     ...
    # }
    # The min/max list for each finger are of length 3 with values following the order of
    # proximal, medial and distal flexions
    calibration_dict = {}
    start_time = time()
    first_reading = True
    while (time() - start_time) <= args.duration:
        # Get the angles for the fingers
        finger_angles_dict = glove.get_grasp_state()

        if first_reading:
            for finger, values in finger_angles_dict.items():
                calibration_dict[finger] = {}
                calibration_dict[finger]['min'] = values
                calibration_dict[finger]['max'] = values
            first_reading = False
        else:
            # Check if we have attained the minimum or maximum value for any reading
            # Override the calibration values if we have
            for finger, values in finger_angles_dict.items():
                calibration_dict[finger]['min'] = \
                    [min(v1, v2) for v1, v2 in zip(calibration_dict[finger]['min'], values)]
                calibration_dict[finger]['max'] = \
                    [max(v1, v2) for v1, v2 in zip(calibration_dict[finger]['max'], values)]

        print('Time Elapsed (s): {:.1f}/{:.1f}'.format(time() - start_time, args.duration,
                                                       end='\r', flush=True))

    print('Calibration complete!')
    print("================================================")
    print("Calibrated Values: {}".format(calibration_dict))
    print("================================================")

    yaml.dump(calibration_dict, calibration_file)
    calibration_file.close()
    print('Saved calibration at {}'.format(output_file))