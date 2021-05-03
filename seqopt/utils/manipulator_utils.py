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
# _REACH_REWARD_COEF = 25.0
# _GRASP_REWARD_COEF = 30.0
# _PLACE_REWARD_COEF = 250.0
_REACH_REWARD_COEF = 25.0
_GRASP_REWARD_COEF = 100.0
_PLACE_REWARD_COEF = 100.0

# Define indexes of important elements for easy reference
# Index 0 - 15: Pairs of sin(angle), cos(angle) for 8 joints (first 4 arms joints, then 4 gripper joints)
# Index 16 - 23: Joint velocities for the 8 joints
# Index 24 - 28: Touch sensor values (all sensors are contained in the gripper area)
# Index 29 - 32: Pose of hand (x,z,qw,qy)
# Index 33 - 36: Pose of object (x,z,qw,qy)
# Index 37 - 39: Velocity of object (x, z, y)
# Index 40 - 43: Pose of target location (x,z,qw,qy)
INDEX_DICT = dict(
    arm_joints_pos=np.arange(8),
    gripper_joints_pos=np.arange(8, 16),
    arm_joints_vel=np.arange(16, 20),
    gripper_joints_vel=np.arange(20, 24),
    touch=np.arange(24, 29),
    hand_pose=np.arange(29, 33),
    object_pose=np.arange(33, 37),
    object_vel=np.arange(37, 40),
    target_pose=np.arange(40, 44),
    # IMPORTANT NOTE: Do not use the following observations in any neural networks. These are custom observations
    #                 that are not part of the original environment
    reach_dist=np.arange(44, 45),
    grasped=np.arange(45, 46),
    place_dist=np.arange(46, 47),
    fingertip_dist=np.arange(47, 48),
    grasp_angle=np.arange(48, 49)
)


FETCH_INDEX_DICT = dict(
    object_pos=np.arange(3),
    target_pos=np.arange(3, 6),
    fingertip_dist=np.arange(6, 7),     # DO NOT USE IN NEURAL NETWORKS
    grasped=np.arange(7, 8),            # DO NOT USE IN NEURAL NETWORKS
    grip_pos=np.arange(8, 11),
    object_pos_copy=np.arange(11, 14),  # We have two copies of this observation
    rel_object_pos=np.arange(14, 17),
    gripper_state=np.arange(17, 19),
    object_rot=np.arange(19, 22),
    object_velp=np.arange(22, 25),
    object_velr=np.arange(25, 28),
    gripper_velp=np.arange(28, 31),
    gripper_vel=np.arange(31, 33),
    place_dist=np.arange(33, 34),       # DO NOT USE IN NEURAL NETWORKS
    reach_dist=np.arange(34, 35)        # DO NOT USE IN NEURAL NETWORKS
)


def scaled_dist(dist: np.ndarray):
    return 1.0 - np.tanh(np.arctanh(np.sqrt(0.95)) / 0.8 * dist)

def reward(last_obs: np.ndarray,
           obs: np.ndarray,
           action: np.ndarray,
           option_id: np.ndarray):
    last_reach_dist, reach_dist = last_obs[..., INDEX_DICT['reach_dist']], obs[..., INDEX_DICT['reach_dist']]
    last_grasp_success, grasp_success = last_obs[..., INDEX_DICT['grasped']], obs[..., INDEX_DICT['grasped']]
    last_place_dist, place_dist = last_obs[..., INDEX_DICT['place_dist']], obs[..., INDEX_DICT['place_dist']]

    # last_reach_dist, reach_dist = last_obs[..., FETCH_INDEX_DICT['reach_dist']], obs[..., FETCH_INDEX_DICT['reach_dist']]
    # last_grasp_success, grasp_success = last_obs[..., FETCH_INDEX_DICT['grasped']], obs[..., FETCH_INDEX_DICT['grasped']]
    # last_place_dist, place_dist = last_obs[..., FETCH_INDEX_DICT['place_dist']], obs[..., FETCH_INDEX_DICT['place_dist']]

    # Assign rewards based on the option engaged
    reach_reward = _REACH_REWARD_COEF * (scaled_dist(reach_dist) - scaled_dist(last_reach_dist))
    grasp_reward = _GRASP_REWARD_COEF * (grasp_success - last_grasp_success) * scaled_dist(place_dist)
    place_reward = _PLACE_REWARD_COEF * last_grasp_success * (scaled_dist(place_dist) - scaled_dist(last_place_dist))

    if len(option_id.shape) < 2:
        option_id = np.expand_dims(option_id, axis=-1)

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

    return rew

# def reward(last_obs: np.ndarray,
#            obs: np.ndarray,
#            action: np.ndarray,
#            option_id: np.ndarray):
#     last_reach_dist, reach_dist = last_obs[..., INDEX_DICT['reach_dist']], obs[..., INDEX_DICT['reach_dist']]
#     last_grasp_success, grasp_success = last_obs[..., INDEX_DICT['grasped']], obs[..., INDEX_DICT['grasped']]
#     last_place_dist, place_dist = last_obs[..., INDEX_DICT['place_dist']], obs[..., INDEX_DICT['place_dist']]
#
#     # last_reach_dist, reach_dist = last_obs[..., FETCH_INDEX_DICT['reach_dist']], obs[..., FETCH_INDEX_DICT['reach_dist']]
#     # last_grasp_success, grasp_success = last_obs[..., FETCH_INDEX_DICT['grasped']], obs[..., FETCH_INDEX_DICT['grasped']]
#     # last_place_dist, place_dist = last_obs[..., FETCH_INDEX_DICT['place_dist']], obs[..., FETCH_INDEX_DICT['place_dist']]
#
#     # Assign rewards based on the option engaged
#     reach_reward = _REACH_REWARD_COEF * (scaled_dist(reach_dist) - scaled_dist(last_reach_dist))
#     grasp_reward = _GRASP_REWARD_COEF * (grasp_success - last_grasp_success)
#     place_reward = _PLACE_REWARD_COEF * grasp_success * (scaled_dist(place_dist) - scaled_dist(last_place_dist))
#     place_reward_penalty = _PLACE_REWARD_COEF * (grasp_success * scaled_dist(place_dist) -
#                                                  last_grasp_success * scaled_dist(last_place_dist))
#
#     if len(option_id.shape) < 2:
#         option_id = np.expand_dims(option_id, axis=-1)
#
#     rew = \
#         np.where(option_id == 0,
#                  reach_reward + np.clip(grasp_reward, None, 0.) + np.clip(place_reward_penalty, None, 0.),
#                  0.) +\
#         np.where(option_id == 1,
#                  grasp_reward + np.clip(reach_reward, None, 0.) + np.clip(place_reward_penalty, None, 0.),
#                  0.) +\
#         np.where(option_id == 2,
#                  place_reward + np.clip(grasp_reward, None, 0.) + np.clip(reach_reward, None, 0.),
#                  0.)
#
#     return rew
