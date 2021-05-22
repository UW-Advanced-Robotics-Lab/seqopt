import os
import numpy as np
from robosuite.devices import GraspDevice
import signal
import sys
import time

import cppyy

from robosuite.models.grippers.hand_model import Finger, FingerJoint
from robosuite.utils.calibration_utils import load_calibration, apply_calibration

# Set the paths for the C++ headers and compiled libraries
SGCONNECT_HEADER_PATH = 'SenseGlove-API/Core/SGConnect/incl'
SGCONNECT_LIB = 'SenseGlove-API/Core/SGConnect/lib/Linux/SGConnect.so'
SGCORE_HEADER_PATH = 'SenseGlove-API/Core/SGCoreCpp/incl'
SGCORE_LIB = 'SenseGlove-API/Core/SGCoreCpp/lib/Linux/SGCore.so'

# Load required header files
cppyy.include(os.path.join(os.path.dirname(__file__), SGCONNECT_HEADER_PATH, 'SGConnect.h'))
cppyy.include(os.path.join(os.path.dirname(__file__), SGCORE_HEADER_PATH, 'SenseGlove.h'))

# Load the C/C++ Shared Libraries
cppyy.load_library(os.path.join(os.path.dirname(__file__), SGCONNECT_LIB))
cppyy.load_library(os.path.join(os.path.dirname(__file__), SGCORE_LIB))


class SenseGlove(GraspDevice):

    def __init__(self,
                 glove,
                 fingers=('THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY'),
                 calibration_file=None,
                 output_range=(-1,1)):
        self._glove = glove
        self._fingers = fingers
        self._finger_idx_map = {
            'THUMB': cppyy.gbl.SGCore.Finger.Thumb,
            'INDEX': cppyy.gbl.SGCore.Finger.Index,
            'MIDDLE': cppyy.gbl.SGCore.Finger.Middle,
            'RING': cppyy.gbl.SGCore.Finger.Ring,
            'PINKY': cppyy.gbl.SGCore.Finger.Pinky
        }

        # Load calibration values if given
        if calibration_file is not None:
            self._calibration = load_calibration(calibration_file)
        else:
            self._calibration = None

        self._output_range = output_range

    def __del__(self):
        self._glove.SendHaptics(cppyy.gbl.SGCore.Haptics.SG_BuzzCmd.off)

    def get_grasp_state(self):
        '''
        Get grasp state of all 5 fingers of the SenseGlove
        '''
        # Create placeholder to read sensor data for the SenseGlove
        sensor_data = cppyy.gbl.SGCore.SG.SG_SensorData()

        # Read sensor data from device
        read_success = False
        # Occasionally, the read fails, retrying usually works
        while not read_success:
            read_success = self._glove.GetSensorData(sensor_data)

        #  The number of readings per finger returned by the SDK
        num_angles_per_finger = 4

        # To prevent excessive reads, we read all the joint angles at once
        # We split the list to get the 3 primary angles for each (specified) finger (from the base to the tip)
        finger_joint_angles = list(sensor_data.GetAngleSequence())
        angles_dict = {}
        for finger in self._fingers:
            finger_idx = self._finger_idx_map[finger]
            angles_dict[finger] = finger_joint_angles[finger_idx * 4: finger_idx * 4 + 3]

        # If we have calibration values, apply them to the raw joint angles read and map the values
        # to the desired output range
        if self._calibration is not None:
            angles_dict = apply_calibration(angles_dict, self._calibration, self._output_range)

        return angles_dict

    def _to_senseglove_finger(self, finger: Finger):
        sg_finger = None
        if finger == Finger.THUMB:
            sg_finger = self._finger_idx_map['THUMB']
        elif finger == Finger.INDEX:
            sg_finger = self._finger_idx_map['INDEX']
        elif finger == Finger.MIDDLE:
            sg_finger = self._finger_idx_map['MIDDLE']
        elif finger == Finger.RING:
            sg_finger = self._finger_idx_map['RING']
        elif finger == Finger.PINKY:
            sg_finger = self._finger_idx_map['PINKY']
        else:
            raise ValueError("Unsupported finger ''".format(finger))

        return sg_finger

    def get_grasp_action(self, grasp_state, hand_model):
        action = []
        finger_mappings = {Finger.THUMB: 'THUMB',
                           Finger.INDEX: 'INDEX',
                           Finger.MIDDLE: 'MIDDLE',
                           Finger.RING: 'RING',
                           Finger.PINKY: 'PINKY'}
        joint_mappings = {FingerJoint.PROXIMAL: 0,
                          FingerJoint.MEDIAL: 1,
                          FingerJoint.DISTAL: 2}
        for key, values in hand_model.items():
            for value in values:
                action.append(grasp_state[finger_mappings[key]][joint_mappings[value]])

        return action

    def set_haptic_feedback(self, feedback_dict, force_scale=10.0):
        # We need to send commands for all fingers at once. So we have to compile all the finger commands
        force_values = []
        for finger in (Finger.THUMB, Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY):
            if finger in feedback_dict.keys():
                force_values.append(int(feedback_dict[finger] * force_scale))
            else:
                force_values.append(0)

        self._glove.SendHaptics(cppyy.gbl.SGCore.Haptics.SG_BuzzCmd(*force_values))
        # for finger, force in feedback_dict.items():
        #     cmd = cppyy.gbl.SGCore.Haptics.SG_Cmd(self._to_senseglove_finger(finger), int(force * force_scale))
        #     print("Force for finger {} = {}".format(finger, int(force * force_scale)))
        #     self._glove.SendHaptics(cmd)

    def start_control(self):
        '''
        Not needed as we do not continuously stream reads
        '''
        pass


class SGDeviceManager:

    def __init__(self):
        # Scan for gloves
        cppyy.gbl.SGConnect.Init()

        # Allow up to 10 seconds for connection to establish
        poll_timeout_s = 10
        poll_start_time = time.time()
        while cppyy.gbl.SGConnect.ActiveDevices() < 1 and \
                (time.time() - poll_start_time) < poll_timeout_s:
            continue

        # Ensure at least one glove is seen
        num_gloves = cppyy.gbl.SGConnect.ActiveDevices()
        print('Number of Gloves Detected: {}'.format(num_gloves))
        if num_gloves == 0:
            print("Please connect at least 1 SenseGlove!")

    def __del__(self):
        cppyy.gbl.SGConnect.Dispose()

    def get_num_gloves(self):
        return cppyy.gbl.SGConnect.ActiveDevices()

    def get_glove(self, right):
        glove = cppyy.gbl.SGCore.SG.SenseGlove()
        success = cppyy.gbl.SGCore.SG.SenseGlove.GetSenseGlove(right, glove)
        if success:
            return glove
        else:
            return None


if __name__ == "__main__":
    # Initialize device manager for sensegloves
    sg_dev_manager = SGDeviceManager()

    # Get the right glove
    glove = sg_dev_manager.get_glove(True)

    if glove is None:
        print("Failed to retrieve SenseGlove from SGDeviceManager!")
        sys.exit(1)

    # Initialize object of class SenseGlove
    glove = SenseGlove(glove)

    # Enable control
    glove.start_control()

    # Register shutdown signal for CTRL+C keypress
    def shutdown_signal_handler(signal, frame):
        print('Shutting down...')
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_signal_handler)

    # Print the grasp states at a fixed interval
    last_update_time = 0
    update_interval_s = 3.0
    while True:
        if (time.time() - last_update_time) >= update_interval_s:
            print(glove.get_grasp_state())
            last_update_time = time.time()
