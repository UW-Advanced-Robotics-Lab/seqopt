import glfw

from robosuite.devices import Device
from robosuite.utils.angles import wrap_to_pi
from robosuite.utils.transform_utils import euler2mat, mat2euler
from robosuite.vr.triad_openvr import triad_openvr

import numpy as np


class HTCViveTracker(Device):
    """
    Base class for all robot controllers.
    Defines basic interface for all controllers to adhere to.
    """

    def __init__(self):
        # Initialize handles to the VR hardware
        self._vr = triad_openvr.triad_openvr()

        # Print the VR hardware detected
        self._vr.print_discovered_objects()

        # Ensure that a HTC Vive Tracker is detected
        assert 'tracker_1' in self._vr.devices.keys(),\
            "HTC Vive Tracker not found! Ensure SteamVR is running and the tracker is recognized!"

        self._tracker = self._vr.devices['tracker_1']

        # The position of the origin
        self._origin = np.zeros(3)
        self._initial_ori = np.array([[0.087, -0.995, 0.045],
                                      [-0.996, -0.087, -0.000],
                                      [0.004, -0.044, -0.999]])

        self._last_pose = np.append(self._origin, mat2euler(self._initial_ori))

        # Reset state
        self._reset_state = 0

    def start_control(self):
        """
        Method that should be called externally before controller can 
        start receiving commands. 
        """
        self._reset_state = 0

    def set_origin(self, pos):
        self._origin = pos

    def get_controller_state(self):
        """Returns the current state of the device, a dictionary of pos, orn, grasp, and reset."""
        # Read 6 DOF pose from the tracker
        pose = self._tracker.get_pose_euler()

        if pose is None:
            pose = self._last_pose
        else:
            self._last_pose = pose

        # The position and orientation needs a coordinate transformation due to conventions used
        # by the HTC Vive System and SteamVR
        position = np.array([-pose[2], -pose[0], pose[1]]) - self._origin

        # Note orientations are reported in degrees
        orientation_rpy = np.deg2rad(pose[3:])
        orientation_rpy = orientation_rpy[[0, 2, 1]]
        orientation_rpy[0] *= -1
        orientation_rpy[1] = wrap_to_pi(orientation_rpy[1] - np.pi / 2)
        orientation_rpy[1] *= -1
        rotation = euler2mat(orientation_rpy)

        # For the reset state, we get the signal from the keyboard
        return dict(
            dpos=position,
            rotation=rotation,
            raw_drotation=orientation_rpy,
            grasp=0,  # We don't control the grasp using the Vive Tracker
            reset=self._reset_state
        )

    def on_release(self, window, key, scancode, action, mods):
        """
        Key handler for key releases.

        Args:
            window: [NOT USED]
            key (int): keycode corresponding to the key that was pressed
            scancode: [NOT USED]
            action: [NOT USED]
            mods: [NOT USED]
        """
        # user-commanded reset
        if key == glfw.KEY_Q:
            self._reset_state = 1
