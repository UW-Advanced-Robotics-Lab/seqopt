import numpy as np
import torch as th

from seqopt.common.env_config import EnvConfig
from seqopt.common.types import (
    ActorParams,
    CriticParams,
    TerminatorParams,
    ExplorationParams
)
from seqopt.utils.reward_utils import scaled_dist


def get_env_config():
    # Define seed
    seed = 0

    # Should count-based exploration be enabled
    count_based_exploration = True

    # Gym Environment ID
    env = 'kitchen_relax-v1'

    # Define the indexes for each observation
    obs_dict = dict(
        joints_pos=np.arange(9),
        arm_joints_vel=np.arange(9, 16),
        gripper_joints_vel=np.arange(16, 18),
        knob_qpos=np.arange(18, 19),
        handle_pos=np.arange(32, 35),  # Can use this in the neural network, although not part of the original obs
        fingertip_dist=np.arange(35, 36),  # DO NOT USE IN NEURAL NETWORKS
        grasped=np.arange(36, 37),  # DO NOT USE IN NEURAL NETWORKS
        knob_turn_angle=np.arange(37, 38),  # DO NOT USE IN NEURAL NETWORKS
        reach_dist=np.arange(38, 39)  # DO NOT USE IN NEURAL NETWORKS
    )

    # Define the reward function
    _REACH_REWARD_COEF = 25.0
    _GRASP_REWARD_COEF = _TURN_REWARD_COEF = 100.0

    def reward_func(last_obs: np.ndarray,
                    obs: np.ndarray,
                    action: np.ndarray,
                    option_id: np.ndarray):
        last_reach_dist, reach_dist = last_obs[..., obs_dict['reach_dist']], obs[..., obs_dict['reach_dist']]
        last_grasp_success, grasp_success = last_obs[..., obs_dict['grasped']], obs[..., obs_dict['grasped']]
        last_turn_angle, turn_angle = last_obs[..., obs_dict['knob_turn_angle']], obs[..., obs_dict['knob_turn_angle']]

        # Assign rewards based on the option engaged
        reach_reward = _REACH_REWARD_COEF * (scaled_dist(reach_dist, scale=0.8) -
                                             scaled_dist(last_reach_dist, scale=0.8))
        grasp_reward = _GRASP_REWARD_COEF * (grasp_success - last_grasp_success) * scaled_dist(turn_angle, scale=3.0)
        turn_reward = _TURN_REWARD_COEF * last_grasp_success * (scaled_dist(turn_angle, scale=3.0) -
                                                                scaled_dist(last_turn_angle, scale=3.0))

        if len(option_id.shape) < 2:
            option_id = np.expand_dims(option_id, axis=-1)

        rew = \
            np.where(option_id == 0,
                     reach_reward + np.clip(grasp_reward, None, 0.) + np.clip(turn_reward, None, 0.),
                     0.) +\
            np.where(option_id == 1,
                     grasp_reward + np.clip(reach_reward, None, 0.) + np.clip(turn_reward, None, 0.),
                     0.) +\
            np.where(option_id == 2,
                     turn_reward + np.clip(grasp_reward, None, 0.) + np.clip(reach_reward, None, 0.),
                     0.)

        return rew

    # Define option level information
    n_options = 3

    # Option 1: REACHING
    reach_default_action = np.zeros(9, dtype=np.float32)
    reach_default_action[7:] = 1.
    reach_actor_params = ActorParams(default_action=th.as_tensor(reach_default_action))
    reach_actor_params.net_arch = [200, 200]
    reach_actor_params.observation_mask = \
        np.concatenate([
            obs_dict['joints_pos'],
            obs_dict['arm_joints_vel'],
            obs_dict['handle_pos'],
        ])
    reach_actor_params.action_mask = np.arange(7)

    reach_critic_params = CriticParams()
    reach_critic_params.net_arch = [300, 200]
    reach_critic_params.observation_mask = np.concatenate([
        obs_dict['joints_pos'],
        obs_dict['arm_joints_vel'],
        obs_dict['gripper_joints_vel'],
        obs_dict['knob_qpos'],
        obs_dict['handle_pos']
    ])

    reach_terminator_params = TerminatorParams()
    reach_terminator_params.net_arch = [200, 200]
    reach_terminator_params.observation_mask = np.concatenate([
        obs_dict['joints_pos'],
        obs_dict['handle_pos']
    ])

    reach_exploration_params = None

    # Option 2: GRASPING
    grasp_default_action = np.zeros(9, dtype=np.float32)
    grasp_default_action[7:] = -1.
    grasp_actor_params = ActorParams(default_action=th.as_tensor(grasp_default_action))
    grasp_actor_params.action_mask = np.array([])

    grasp_critic_params = CriticParams()
    grasp_critic_params.net_arch = [300, 200]
    grasp_critic_params.observation_mask = np.concatenate([
        obs_dict['joints_pos'],
        obs_dict['arm_joints_vel'],
        obs_dict['gripper_joints_vel'],
        obs_dict['knob_qpos'],
        obs_dict['handle_pos']
    ])

    grasp_terminator_params = TerminatorParams()
    grasp_terminator_params.net_arch = [200, 200]
    grasp_terminator_params.observation_mask = np.concatenate([
        obs_dict['joints_pos'],
        obs_dict['arm_joints_vel'],
        obs_dict['knob_qpos'],
        obs_dict['handle_pos']
    ])

    if count_based_exploration:
        def features_extractor(states: th.Tensor):
            # Get the distance, relative angles between the grasper and object and the gripper closure
            features = states[..., np.concatenate([obs_dict['reach_dist'],
                                                   obs_dict['fingertip_dist']])]
            return features

        # For distance-based boundary, we will create evenly-spaced values on the log distance scale between
        # 0.01 - 0.1 m (values outside these ranges are pretty meaningless w.r.t the task, so we can bin them together)
        dist_boundaries = th.pow(10, th.linspace(-2, -1, steps=20))

        # Gripper bounds
        step_size = 0.1 / 10
        gripper_boundaries = th.arange(step_size, 0.01 + step_size, step_size)
        features_boundaries = [dist_boundaries, gripper_boundaries]

        # Define the intrinsic reward function
        def intrinsic_reward_func(observation: np.ndarray, count: int) -> np.float32:
            return 1.0 / np.sqrt(count)

        grasp_exploration_params = ExplorationParams(features_extractor=features_extractor,
                                                     feature_boundaries=features_boundaries,
                                                     reward_func=intrinsic_reward_func,
                                                     scale=5.0)
    else:
        grasp_exploration_params = None

    # Option 3: Turning (the knob)
    turn_default_action = np.zeros(9, dtype=np.float32)
    turn_default_action[7:] = -1.
    turn_actor_params = ActorParams(default_action=th.as_tensor(turn_default_action))
    turn_actor_params.net_arch = [100, 50]
    turn_actor_params.observation_mask = np.concatenate([
        obs_dict['joints_pos'],
        obs_dict['knob_qpos'],
        obs_dict['handle_pos']
    ])
    turn_actor_params.action_mask = np.array([6])

    turn_critic_params = CriticParams()
    turn_critic_params.net_arch = [300, 200]
    turn_critic_params.observation_mask = np.concatenate([
        obs_dict['joints_pos'],
        obs_dict['arm_joints_vel'],
        obs_dict['gripper_joints_vel'],
        obs_dict['knob_qpos'],
        obs_dict['handle_pos']
    ])

    turn_terminator_params = TerminatorParams()
    turn_terminator_params.net_arch = [200, 200]
    turn_terminator_params.observation_mask = np.concatenate([
        obs_dict['joints_pos'],
        obs_dict['knob_qpos'],
        obs_dict['handle_pos']
    ])

    turn_exploration_params = None

    # Store all params
    actor_params = [reach_actor_params, grasp_actor_params, turn_actor_params]
    critic_params = [reach_critic_params, grasp_critic_params, turn_critic_params]
    terminator_params = [reach_terminator_params, grasp_terminator_params, turn_terminator_params]
    exploration_params = [reach_exploration_params, grasp_exploration_params, turn_exploration_params]

    config = EnvConfig(
        env=env,
        seed=seed,
        obs_dict=obs_dict,
        reward_func=reward_func,
        n_options=n_options,
        actor_params=actor_params,
        critic_params=critic_params,
        terminator_params=terminator_params,
        exploration_params=exploration_params
    )

    return config
