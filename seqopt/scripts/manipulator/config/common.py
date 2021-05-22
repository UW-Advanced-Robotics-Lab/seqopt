import dmc2gym
import numpy as np
import torch as th

from seqopt.common.env_config import EnvConfig
from seqopt.common.types import (
    ActorParams,
    CriticParams,
    TerminatorParams,
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

    def reward_func(last_obs: np.ndarray,
                    obs: np.ndarray,
                    action: np.ndarray,
                    option_id: np.ndarray):
        last_reach_dist, reach_dist = last_obs[..., obs_dict['reach_dist']], obs[..., obs_dict['reach_dist']]
        last_grasp_success, grasp_success = last_obs[..., obs_dict['grasped']], obs[..., obs_dict['grasped']]
        last_place_dist, place_dist = last_obs[..., obs_dict['place_dist']], obs[..., obs_dict['place_dist']]

        # Assign rewards based on the option engaged
        reach_reward = _REACH_REWARD_COEF * (scaled_dist(reach_dist, scale=0.8) -
                                             scaled_dist(last_reach_dist, scale=0.8))
        grasp_reward = _GRASP_REWARD_COEF * (grasp_success - last_grasp_success) * scaled_dist(place_dist, scale=0.8)
        place_reward = _PLACE_REWARD_COEF * last_grasp_success * (scaled_dist(place_dist, scale=0.8) -
                                                                  scaled_dist(last_place_dist, scale=0.8))

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

    # Define option level information
    n_options = 3

    # Option 1: REACHING
    reach_actor_params = ActorParams(default_action=th.Tensor([0., 0., 0., 0., -1.]))
    reach_actor_params.net_arch = [200, 100]
    reach_actor_params.observation_mask =\
        np.concatenate([
                obs_dict['arm_joints_pos'],
                obs_dict['gripper_joints_pos'],
                obs_dict['arm_joints_vel'],
                obs_dict['hand_pose'],
                obs_dict['object_pos']
        ])
    reach_actor_params.action_mask = np.arange(4)

    reach_critic_params = CriticParams()
    reach_critic_params.net_arch = [300, 200]
    reach_critic_params.observation_mask = np.concatenate([
        obs_dict['arm_joints_pos'],
        obs_dict['gripper_joints_pos'],
        obs_dict['arm_joints_vel'],
        obs_dict['hand_pose'],
        obs_dict['object_pos'],
        obs_dict['target_pos']
    ])

    reach_terminator_params = TerminatorParams()
    reach_terminator_params.net_arch = [200, 200]
    reach_terminator_params.observation_mask = np.concatenate([
        obs_dict['gripper_joints_pos'],
        obs_dict['arm_joints_vel'],
        obs_dict['gripper_joints_vel'],
        obs_dict['hand_pose'],
        obs_dict['object_pos'],
    ])

    reach_exploration_params = None

    # Option 2: GRASPING
    grasp_actor_params = ActorParams(default_action=th.Tensor([0., 0., 0., 0., 1.]))
    grasp_actor_params.action_mask = np.array([])

    grasp_critic_params = CriticParams()
    grasp_critic_params.net_arch = [300, 200]
    grasp_critic_params.observation_mask = np.concatenate([
        obs_dict['gripper_joints_pos'],
        obs_dict['hand_pose'],
        obs_dict['object_pos'],
        obs_dict['target_pos']
    ])

    grasp_terminator_params = TerminatorParams()
    grasp_terminator_params.net_arch = [200, 200]
    grasp_terminator_params.observation_mask = np.concatenate([
        obs_dict['gripper_joints_pos'],
        obs_dict['hand_pose'],
        obs_dict['object_pos'],
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
            reach_potential = scaled_dist(observation[..., obs_dict['reach_dist']], scale=0.8)
            reach_potential_max = 1.0
            goodness_decay = np.log(2) / 0.1
            goodness = np.exp((reach_potential - reach_potential_max) * goodness_decay)
            count_decay = np.log(2) / 1e4
            count_factor = np.exp(-count_decay * count)
            intrinsic_reward = goodness * count_factor
            return intrinsic_reward.item()

        grasp_exploration_params = ExplorationParams(features_extractor=features_extractor,
                                                     feature_boundaries=features_boundaries,
                                                     reward_func=intrinsic_reward_func,
                                                     scale=5.0)
    else:
        grasp_exploration_params = None

    # Option 3: PLACING
    def place_default_action(observations: th.Tensor):
        return th.where(observations[..., obs_dict['grasped']].unsqueeze(dim=-1) == 1.0,
                        th.Tensor([0., 0., 0., 0., 0.5]),
                        th.Tensor([0., 0., 0., 0., 0.])).squeeze()

    place_actor_params = ActorParams(default_action=place_default_action)
    place_actor_params.net_arch = [200, 100]
    place_actor_params.observation_mask = np.concatenate([
        obs_dict['arm_joints_pos'],
        obs_dict['gripper_joints_pos'],
        obs_dict['arm_joints_vel'],
        obs_dict['object_pos'],
        obs_dict['target_pos']
    ])
    place_actor_params.action_mask = np.arange(4)

    place_critic_params = CriticParams()
    place_critic_params.net_arch = [300, 200]
    place_critic_params.observation_mask = np.concatenate([
        obs_dict['gripper_joints_pos'],
        obs_dict['arm_joints_vel'],
        obs_dict['hand_pose'],
        obs_dict['object_pos'],
        obs_dict['target_pos']
    ])

    place_terminator_params = TerminatorParams()
    place_terminator_params.net_arch = [200, 200]
    place_terminator_params.observation_mask = np.concatenate([
        obs_dict['gripper_joints_pos'],
        obs_dict['hand_pose'],
        obs_dict['object_pos'],
    ])

    place_exploration_params = None

    # Store all params
    actor_params = [reach_actor_params, grasp_actor_params, place_actor_params]
    critic_params = [reach_critic_params, grasp_critic_params, place_critic_params]
    terminator_params = [reach_terminator_params, grasp_terminator_params, place_terminator_params]
    exploration_params = [reach_exploration_params, grasp_exploration_params, place_exploration_params]

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
