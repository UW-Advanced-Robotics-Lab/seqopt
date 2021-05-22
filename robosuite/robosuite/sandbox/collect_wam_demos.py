import argparse
import numpy as np

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.devices import HTCViveTracker, SenseGlove, SGDeviceManager
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="PickPlaceMilk")
    parser.add_argument("--log-path", type=str, default='demonstrations')
    args = parser.parse_args()

    ############################## ENVIRONMENT INITIALIZATION ##########################################

    # Get controller config for the Operational Space Controller (OSC)
    controller_config = load_controller_config(default_controller='OSC_POSE')

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": 'WAM',
        "controller_configs": controller_config,
    }

    # Create the Environment with the required controllers
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualizations=True,
        reward_shaping=True,
        control_freq=20
    )

    # Wrap the environment in the DataCollection wrapper for logging
    env = DataCollectionWrapper(env, args.log_path, keypress_for_start=True, keypress_for_save=True)

    ################################# DEVICE INITIALIZATION #############################################

    # We use the HTC Vive Tracker for pose tracking
    device = HTCViveTracker()
    # Allow keybindings to control simulation using keyboard
    env.viewer.add_keyup_callback("any", device.on_release)

    # Adjust origin of the position tracking system such that
    # the absolute coordinates agree with the initial pose of the end-effector
    tracker_pos = device.get_controller_state()['dpos']
    device.set_origin(tracker_pos)

    # Initialize SenseGlove for grasping
    sg_device_manager = SGDeviceManager()
    # Get the right glove
    sg_glove = sg_device_manager.get_glove(True)
    if sg_glove:
        grasp_device = SenseGlove(sg_glove, calibration_file='../devices/config/right_senseglove_calibration.yaml')
    else:
        raise Exception(
            "Right SenseGlove could not be detected! Please ensure it is properly connected"
        )

    ################################# COLLECT DEMONSTRATIONS #############################################

    while True:
        # Reset the environment
        obs = env.reset()

        # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        env.render()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()

        # Set active robot
        active_robot = env.robots[0]

        while True:
            # Get the newest action
            action, grasp = input2action(
                device=device,
                grasp_device=grasp_device,
                robot=active_robot,
                active_arm='right',
                env_configuration='single-arm-opposed'
            )

            # If action is none, then this a reset so we should break
            if action is None:
                break

            # Fill out the rest of the action space if necessary
            rem_action_dim = env.action_dim - action.size
            if rem_action_dim > 0:
                # Initialize remaining action space
                rem_action = np.zeros(rem_action_dim)
                action = np.concatenate([action, rem_action])
            elif rem_action_dim < 0:
                # We're in an environment with no gripper action space, so trim the action space to be the action dim
                action = action[:env.action_dim]

            # Send Haptic Feedback
            if grasp_device is not None and hasattr(active_robot.gripper, "haptic_sensors"):
                feedback_dict = {}
                for finger, touch_sensor in active_robot.gripper.haptic_sensors.items():
                    feedback_dict[finger] = env.sim.data.get_sensor('gripper0_' + touch_sensor)
                grasp_device.set_haptic_feedback(feedback_dict)

            # Step through the simulation and render
            obs, reward, done, info = env.step(action)
            env.render()
