"""Helper Functions for dm_control manipulator environment"""
from typing import Union

from colorama import Fore, Style

from numba import jit
import numpy as np
import torch as th


# Reward multipliers (note rewards for different subtasks are of different scales, so this is an effort to normalize
# them)
_REACH_REWARD_COEF = 25.0
_GRASP_REWARD_COEF = 100.0
_SLIDE_REWARD_COEF = 100.0

# Define indexes of important elements for easy reference
INDEX_DICT = dict(
    joints_pos=np.arange(9),
    slide_qpos=np.arange(9, 10),
    handle_pos=np.arange(10, 13),       # Can use this in the neural network, although not part of the original obs
    fingertip_dist=np.arange(13, 14),   # DO NOT USE IN NEURAL NETWORKS
    grasped=np.arange(14, 15),          # DO NOT USE IN NEURAL NETWORKS
    slide_dist=np.arange(15, 16),       # DO NOT USE IN NEURAL NETWORKS
    reach_dist=np.arange(16, 17)        # DO NOT USE IN NEURAL NETWORKS
)


def scaled_dist(dist: np.ndarray, scale: float = 0.8):
    return 1.0 - np.tanh(np.arctanh(np.sqrt(0.95)) / scale * dist)


def reward(last_obs: np.ndarray,
           obs: np.ndarray,
           action: np.ndarray,
           option_id: np.ndarray):
    last_reach_dist, reach_dist = last_obs[..., INDEX_DICT['reach_dist']], obs[..., INDEX_DICT['reach_dist']]
    last_grasp_success, grasp_success = last_obs[..., INDEX_DICT['grasped']], obs[..., INDEX_DICT['grasped']]
    last_slide_dist, slide_dist = last_obs[..., INDEX_DICT['slide_dist']], obs[..., INDEX_DICT['slide_dist']]

    # Assign rewards based on the option engaged
    reach_reward = _REACH_REWARD_COEF * (scaled_dist(reach_dist) - scaled_dist(last_reach_dist))
    grasp_reward = _GRASP_REWARD_COEF * (grasp_success - last_grasp_success) * scaled_dist(slide_dist, scale=0.65)
    slide_reward = _SLIDE_REWARD_COEF * last_grasp_success * (scaled_dist(slide_dist, scale=0.65) -
                                                              scaled_dist(last_slide_dist, scale=0.65))

    if len(option_id.shape) < 2:
        option_id = np.expand_dims(option_id, axis=-1)

    rew = \
        np.where(option_id == 0,
                 reach_reward + np.clip(grasp_reward, None, 0.) + np.clip(slide_reward, None, 0.),
                 0.) +\
        np.where(option_id == 1,
                 grasp_reward + np.clip(reach_reward, None, 0.) + np.clip(slide_reward, None, 0.),
                 0.) +\
        np.where(option_id == 2,
                 slide_reward + np.clip(grasp_reward, None, 0.) + np.clip(reach_reward, None, 0.),
                 0.)

    return rew
