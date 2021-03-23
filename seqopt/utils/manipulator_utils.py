"""Helper Functions for dm_control manipulator environment"""
from typing import Union

from colorama import Fore, Style

from numba import jit
import numpy as np
import torch as th

# Important Constants
_HAND_GRASP_OFFSET = 0.065

# Subtask success criteria
_REACH_CLOSE = 0.015
_GRASP_CLOSE = 0.0375
_PLACE_CLOSE = 0.015

# Reward multipliers (note rewards for different subtasks are of different scales, so this is an effort to normalize
# them)
_REACH_REWARD_COEF = 25.0
_GRASP_REWARD_COEF = 30.0
_PLACE_REWARD_COEF = 250.0

# Define indexes of important elements for easy reference
# Index 0 - 15: Pairs of sin(angle), cos(angle) for 8 joints (first 4 arms joints, then 4 gripper joints)
# Index 16 - 23: Joint velocities for the 8 joints
# Index 24 - 28: Touch sensor values (all sensors are contained in the gripper area)
# Index 29 - 32: Pose of hand (x,z,qw,qy)
# Index 33 - 36: Pose of object (x,z,qw,qy)
# Index 37 - 39: Velocity of object (x, z, y)
# Index 40 - 43: Pose of target location (x,z,qw,qy)
INDEX_DICT = dict(
    arm_joints_pos=slice(0, 8),
    gripper_joints_pos=slice(8, 16),
    arm_joints_vel=slice(16, 20),
    gripper_joints_vel=slice(20, 24),
    touch=slice(24, 29),
    hand_pose=slice(29, 33),
    object_pose=slice(33, 37),
    object_vel=slice(37, 40),
    target_pose=slice(40, 44)
)


def get_grasp_pose(state: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
    # The grasp orientation and the hand orientation are the same
    grasp_pose = get_state_item(state, 'hand_pose')
    qw, qy = grasp_pose[2:]

    # Compute the sin and cosine of the orientation (quaternions use half-angle formulas)
    sin_angle = 2 * qw * qy
    cos_angle = 1 - 2 * (qy ** 2)

    # There is an offset between the hand position and grasp position
    if isinstance(state, th.Tensor):
        grasp_pose[:2] += _HAND_GRASP_OFFSET * th.Tensor([sin_angle, cos_angle]).to(state.device)
    elif isinstance(state, np.ndarray):
        grasp_pose[:2] += _HAND_GRASP_OFFSET * np.array([sin_angle, cos_angle])
    else:
        raise ValueError(f'Unsupport input type: {type(state)}')

    return grasp_pose

def quat_to_yaw(qw, qy):
    if isinstance(qw, th.Tensor):
        return th.atan2(2 * qw * qy, 1 - 2 * (qy ** 2))
    else:
        return np.arctan2(2 * qw * qy, 1 - 2 * (qy ** 2))


def wrap_to_pi(angle, upper_eq: bool = False):
    if upper_eq:
        while angle <= -np.pi:
            angle += 2 * np.pi
        while angle > np.pi:
            angle -= 2 * np.pi
    else:
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle >= np.pi:
            angle -= 2 * np.pi

    return angle


def copy_object(object: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
    if isinstance(object, np.ndarray):
        return object.copy()
    elif isinstance(object, th.Tensor):
        return object.clone()


def get_state_item(state: Union[np.ndarray, th.Tensor], item: str):
    if item not in INDEX_DICT.keys():
        raise KeyError(f'No key {item} in state dictionary!')

    return copy_object(state[INDEX_DICT[item]])


def grasp_to_obj_pose(state: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
    # Calculate the yaw angle for the grasp
    grasp_pose = get_grasp_pose(state)
    grasp_qw, grasp_qy = grasp_pose[2:]
    grasp_angle = quat_to_yaw(grasp_qw, grasp_qy)

    # Calculate the yaw angle for the object
    object_pose = get_state_item(state, 'object_pose')
    obj_qw, obj_qy = object_pose[2:]
    obj_angle = 0  # Ball has no concept of angle, quat_to_yaw(obj_qw, obj_qy)

    # Get the relative angle
    relative_angle = wrap_to_pi(grasp_angle - obj_angle)

    if isinstance(state, np.ndarray):
        # Get the distance between the grasp site and the object
        dist = np.linalg.norm(grasp_pose[:2] - object_pose[:2])
        return np.array([dist, relative_angle])
    elif isinstance(state, th.Tensor):
        # Get the distance between the grasp site and the object
        dist = th.norm(grasp_pose[:2] - object_pose[:2])
        return th.Tensor([dist, relative_angle]).type(state.dtype).to(state.device)
    else:
        raise ValueError(f'Unsupported input type: {type(state)}')


def obj_to_target_dist(state: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
    obj_pos = get_state_item(state, 'object_pose')[:2]
    target_pos = get_state_item(state, 'target_pose')[:2]

    if isinstance(state, np.ndarray):
        dist = np.linalg.norm(target_pos - obj_pos)
        return np.array([dist])
    elif isinstance(state, th.Tensor):
        dist = th.norm(target_pos - obj_pos)
        return th.Tensor([dist]).type(state.dtype).to(state.device)
    else:
        raise ValueError(f'Unsupported input type: {type(state)}')


def fingertip_dist(state: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
    # Extract the (trigonometric) joint angles for the 4 gripper joints
    # Note: Joint values in Mujoco follow a Left-Hand system
    #       Additionally, the thumb transformations get mirrored, so it effectively uses a Right-Hand system
    gripper_joints_pos = get_state_item(state, 'gripper_joints_pos')

    # Not quite sure why this angle offset is needed. Without this there seems to be an error with a
    # roughly constant value of 0.02
    angle = -10
    sin_offset, cos_offset = np.sin(angle * np.pi / 180.0), np.cos(angle * np.pi / 180.0)
    sin_finger, cos_finger = gripper_joints_pos[0], gripper_joints_pos[1]
    sin_thumb, cos_thumb = gripper_joints_pos[4], gripper_joints_pos[5]
    gripper_joints_pos[0] = sin_finger * cos_offset + cos_finger * sin_offset
    gripper_joints_pos[1] = cos_finger * cos_offset - sin_finger * sin_offset
    gripper_joints_pos[4] = sin_thumb * cos_offset + cos_thumb * sin_offset
    gripper_joints_pos[5] = cos_thumb * cos_offset - sin_thumb * sin_offset

    if isinstance(state, np.ndarray):
        finger_rot_mat = np.array([[gripper_joints_pos[1], gripper_joints_pos[0]],
                                  [-gripper_joints_pos[0], gripper_joints_pos[1]]])
        # fingertip_rot_mat = np.array([[gripper_joints_pos[3], gripper_joints_pos[2]],
        #                              [-gripper_joints_pos[2], gripper_joints_pos[3]]])
        thumb_rot_mat = np.array([[gripper_joints_pos[5], -gripper_joints_pos[4]],
                                  [gripper_joints_pos[4], gripper_joints_pos[5]]])
        # thumbtip_rot_mat = np.array([[gripper_joints_pos[7], -gripper_joints_pos[6]],
        #                              [gripper_joints_pos[6], gripper_joints_pos[7]]])

        # Calculate positions of the fingertip and thumbtip (in local hand coordinates)
        fingertip_pos = finger_rot_mat @ np.array([-0.01, 0.05]) + np.array([-0.03, 0.045])
                        # finger_rot_mat @ np.array([[0, 1], [1, 0]]) @ fingertip_rot_mat @ np.array([0.05, -0.01])
        thumbtip_pos = thumb_rot_mat @ np.array([-0.01, 0.05]) + np.array([0.03, 0.045])
                       # thumb_rot_mat @ np.array([[0, -1], [1, 0]]) @ thumbtip_rot_mat @ np.array([0.05, -0.01])

        # Calculate distance between the fingertip and thumbtip
        dist = np.linalg.norm(fingertip_pos - thumbtip_pos)
        return np.array([dist])
    else:
        device = state.device
        finger_rot_mat = th.Tensor([[gripper_joints_pos[1], gripper_joints_pos[0]],
                                    [-gripper_joints_pos[0], gripper_joints_pos[1]]]).to(device)
        # fingertip_rot_mat = th.Tensor([[gripper_joints_pos[3], gripper_joints_pos[2]],
        #                                [-gripper_joints_pos[2], gripper_joints_pos[3]]]).to(device)
        thumb_rot_mat = th.Tensor([[gripper_joints_pos[5], -gripper_joints_pos[4]],
                                   [gripper_joints_pos[4], gripper_joints_pos[5]]]).to(device)
        # thumbtip_rot_mat = th.Tensor([[gripper_joints_pos[7], -gripper_joints_pos[6]],
        #                               [gripper_joints_pos[6], gripper_joints_pos[7]]]).to(device)
        fingertip_pos = (finger_rot_mat @ th.Tensor([-0.01, .05]) + th.Tensor([-0.03, .045])).to(device)
                        # fingertip_rot_mat @ th.Tensor([0.053, -0.007]).to(device)
        thumbtip_pos = (thumb_rot_mat @ th.Tensor([-0.01, 0.05]) + th.Tensor([0.03, .045])).to(device)
                       # thumbtip_rot_mat @ th.Tensor([0.053, -0.007]).to(device)

        return th.norm(fingertip_pos - thumbtip_pos)


def scaled_dist(dist: np.ndarray):
    return 1.0 - np.tanh(np.arctanh(np.sqrt(0.95)) / 0.8 * dist)


def reward(last_obs: np.ndarray,
           obs: np.ndarray,
           action: np.ndarray,
           option_id: np.ndarray):
    # Get the distance of the grasp site to the object for the current observation(s) and last observation(s)
    grasp_to_obj_distance = np.apply_along_axis(lambda state: grasp_to_obj_pose(state)[0], axis=-1, arr=obs)
    last_grasp_to_obj_distance = np.apply_along_axis(lambda state: grasp_to_obj_pose(state)[0], axis=-1, arr=last_obs)

    # Get the distance of the object from the target site for the current observation(s) and last observation(s)
    obj_to_target_distance = np.apply_along_axis(lambda state: obj_to_target_dist(state)[0], axis=-1, arr=obs)
    last_obj_to_target_distance = np.apply_along_axis(lambda state: obj_to_target_dist(state)[0], axis=-1, arr=last_obs)

    gripper_close_dist = np.apply_along_axis(lambda state: fingertip_dist(state)[0], axis=-1, arr=obs)
    last_gripper_close_dist = np.apply_along_axis(lambda state: fingertip_dist(state)[0], axis=-1, arr=last_obs)

    # Check success criteria
    reach_success, last_reach_success = (grasp_to_obj_distance <= _REACH_CLOSE),\
                                        (last_grasp_to_obj_distance <= _REACH_CLOSE)
    grasp_success, last_grasp_success = np.logical_and(reach_success, gripper_close_dist <= _GRASP_CLOSE),\
                                        np.logical_and(last_reach_success, last_gripper_close_dist <= _GRASP_CLOSE)

    # print(f"Reach Success: {reach_success}, Grasp Success: {grasp_success}, Grasp to obj dist: {grasp_to_obj_distance}")
    # print(f"Grasp to obj dist: {grasp_to_obj_distance}, Gripper Close Dist: {gripper_close_dist}")

    # Assign rewards based on the option engaged
    # Consider the following 'progress-based' rewards
    # reach_reward = _REACH_REWARD_COEF * (last_grasp_to_obj_distance - grasp_to_obj_distance)
    reach_reward = _REACH_REWARD_COEF * (scaled_dist(grasp_to_obj_distance) - scaled_dist(last_grasp_to_obj_distance))


    # Dense grasp reward
    # grasp_reward = 50.0 * reach_success.astype(int) * (last_gripper_close_dist - gripper_close_dist)
    grasp_reward = _GRASP_REWARD_COEF * (grasp_success.astype(np.int) - last_grasp_success.astype(np.int))
    # delta_obs_to_target_distance = last_obj_to_target_distance - obj_to_target_distance
    # encouragement_rew = np.where(delta_obs_to_target_distance > 1e-3, 1e-1, 0.)
    place_reward = _PLACE_REWARD_COEF * grasp_success.astype(np.int) * (scaled_dist(obj_to_target_distance) -
                                                                        scaled_dist(last_obj_to_target_distance))
    # place_reward = _PLACE_REWARD_COEF * grasp_success.astype(np.int) * (last_obj_to_target_distance - obj_to_target_distance)
    # place_reward = _PLACE_REWARD_COEF * (grasp_success.astype(np.int) * scaled_dist(obj_to_target_distance) -
    #                                      last_grasp_success.astype(np.int) * scaled_dist(last_obj_to_target_distance)) \
    #                 + grasp_success.astype(np.int) * encouragement_rew
                   # (_PLACE_REWARD_COEF * (scaled_dist(obj_to_target_distance) - scaled_dist(last_obj_to_target_distance)))

    # Print if grasp is successful
    # if last_grasp_success:
    #     print(f"{Fore.GREEN}Last Grasp: Successful!{Style.RESET_ALL}")
    # else:
    #     print(f"{Fore.RED}Last Grasp: Unsuccessful!{Style.RESET_ALL}")
    # if grasp_success:
    #     print(f"{Fore.GREEN}Grasp: Successful!{Style.RESET_ALL}, Rew: {grasp_reward}")
    # else:
    #     print(f"{Fore.RED}Grasp: Unsuccessful!{Style.RESET_ALL}")

    # Options will be penalized for inducing negative progress in other options
    # However, they are not rewarded for making positive progress by doing the work of other options
    # print(f"Delta dist (reach): {last_grasp_to_obj_distance - grasp_to_obj_distance}")
    # print(f"Delta dist (place): {delta_obs_to_target_distance}")
    rew = \
        np.where(option_id == 0,
                 reach_reward + np.clip(grasp_reward, None, 0.) + np.clip(place_reward, None, 0.),
                 0.) +\
        np.where(option_id == 1,
                 grasp_reward + np.clip(reach_reward, None, 0.) + np.clip(place_reward, None, 0.),
                 0.) +\
        np.where(option_id == 2,
                 place_reward + np.clip(grasp_reward, None, 0.) + np.clip(reach_reward, None, 0.),
                 0.)
    # gripper_vel = get_state_item(obs[0], 'gripper_joints_vel')
    # print(f"Gripper to obj dist: {grasp_to_obj_distance}")

    # Task success reward
    # last_success = (last_grasp_success and last_obj_to_target_distance <= _PLACE_CLOSE)
    # success = (grasp_success and obj_to_target_distance <= _PLACE_CLOSE)
    # rew += 100 * (success.astype(np.int) - last_success.astype(np.int))

    # Terminal Reward
    # rew = np.where(obj_to_target_distance <= _PLACE_CLOSE, 100.0, rew)

    return rew
