import dmc2gym
import numpy as np
import torch as th

from seqopt.common.env_config import EnvConfig
from seqopt.common.types import (
    ActorParams,
    CriticParams,
    ExplorationParams
)
import seqopt.environments
from seqopt.utils.reward_utils import scaled_dist


def get_env_config():
    # Define seed
    seed = 0

    # Should count-based exploration be enabled
    count_based_exploration = True

    # Generate the environment (since this is a dm_control environment)
    # Use the dmc2gym library to create a dm_control environment in OpenAI gym format
    # The library registers the environment with gym in the process of making it
    # We discard this environment, since this is a hack to get the environment registered with gym
    env = dmc2gym.make(domain_name='manipulator',
                       task_name='bring_ball',
                       seed=seed,
                       episode_length=500)
    env = env.spec.id

    # Define the indexes for each observation
    obs_dict = dict(
        arm_joints_pos=np.arange(8),
        gripper_joints_pos=np.arange(8, 16),
        arm_joints_vel=np.arange(16, 20),
        gripper_joints_vel=np.arange(20, 24),
        touch=np.arange(24, 29),
        hand_pose=np.arange(29, 33),
        object_pos=np.arange(33, 35),
        object_quat=np.arange(35, 37),
        object_vel=np.arange(37, 40),
        target_pos=np.arange(40, 42),
        target_quat=np.arange(42, 44),
        # IMPORTANT NOTE: Do not use the following observations in any neural networks. These are custom observations
        #                 that are not part of the original environment
        reach_dist=np.arange(44, 45),
        grasped=np.arange(45, 46),
        place_dist=np.arange(46, 47),
        fingertip_dist=np.arange(47, 48),
        grasp_angle=np.arange(48, 49)
    )

    # Define the reward function
    _REACH_REWARD_COEF = 25.0
    _GRASP_REWARD_COEF = _PLACE_REWARD_COEF = 100.0

    def composite_task_potential(obs: np.ndarray):
        reach_dist = obs[..., obs_dict['reach_dist']]
        grasped = obs[..., obs_dict['grasped']]
        place_dist = obs[..., obs_dict['place_dist']]

        task_potential = _REACH_REWARD_COEF * scaled_dist(reach_dist, scale=0.8) + \
                         _PLACE_REWARD_COEF * grasped * scaled_dist(place_dist, scale=0.8)

        return task_potential

    def reward_func(last_obs: np.ndarray,
                    obs: np.ndarray,
                    action: np.ndarray,
                    option_id: np.ndarray):
        reward = composite_task_potential(obs) - composite_task_potential(last_obs)
        return reward

    # Define option level information
    # We only use 1 option to mimic a conventional RL method
    n_options = 1
    option_names = ['Universal']

    actor_params = ActorParams(default_action=th.Tensor([0., 0., 0., 0., -1.]))
    actor_params.net_arch = [200, 100]
    actor_params.observation_mask =\
        np.concatenate([
                obs_dict['arm_joints_pos'],
                obs_dict['gripper_joints_pos'],
                obs_dict['arm_joints_vel'],
                obs_dict['hand_pose'],
                obs_dict['object_pos'],
                obs_dict['target_pos']
        ])
    actor_params.action_mask = None

    critic_params = CriticParams()
    critic_params.net_arch = [300, 200]
    critic_params.observation_mask = np.concatenate([
        obs_dict['arm_joints_pos'],
        obs_dict['gripper_joints_pos'],
        obs_dict['arm_joints_vel'],
        obs_dict['hand_pose'],
        obs_dict['object_pos'],
        obs_dict['target_pos']
    ])

    if count_based_exploration:
        def features_extractor(states: th.Tensor):
            # Get the distance, relative angles between the grasper and object and the gripper closure
            features = states[..., np.concatenate([obs_dict['reach_dist'],
                                                   obs_dict['grasp_angle'],
                                                   obs_dict['fingertip_dist']])]
            return features


        # For distance-based boundary, we will create evenly-spaced values on the log distance scale between
        # 0.01 - 0.1 m (values outside these ranges are pretty meaningless w.r.t the task, so we can bin them together)
        dist_boundaries = th.pow(10, th.linspace(-2, -1, steps=20))
        # The angle bounds are evenly spread in 10 degree intervals
        step_size = np.pi / 18
        angle_boundaries = th.arange(-np.pi + step_size, np.pi, step_size)
        # Gripper bounds
        step_size = 0.08 / 10
        gripper_boundaries = th.arange(step_size, 0.08 + step_size, step_size)
        features_boundaries = [dist_boundaries, angle_boundaries, gripper_boundaries]

        # Define the intrinsic reward function
        def intrinsic_reward_func(observation: np.ndarray, count: int) -> np.float32:
            return 1. / np.sqrt(count)

        exploration_params = ExplorationParams(features_extractor=features_extractor,
                                                     feature_boundaries=features_boundaries,
                                                     reward_func=intrinsic_reward_func,
                                                     scale=5.0)
    else:
        exploration_params = None

    config = EnvConfig(
        env=env,
        seed=seed,
        obs_dict=obs_dict,
        reward_func=reward_func,
        task_potential_func=composite_task_potential,
        n_options=n_options,
        option_names=option_names,
        actor_params=[actor_params],
        critic_params=[critic_params],
        terminator_params=[None],
        exploration_params=[exploration_params]
    )

    return config
