import numpy as np
import torch as th

from robosuite import load_controller_config
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

    # This is a Robosuite environment
    # We are going to do relative pose control using an Operational Space Controller (OSC) with the Jaco arm
    osc_pose_config = load_controller_config(default_controller='OSC_POSE')
    env = dict(
        env_name='Door',
        robots='Jaco',
        controller_configs=osc_pose_config,
        ignore_done=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=10,
        gripper_visualizations=True,
        horizon=500
    )

    # Define the observations
    obs_dict = dict(
        arm_joints_pos=np.arange(14),   # sin/cos representation for the 7 arm joints
        arm_joints_vel=np.arange(14, 21),
        eef_pos=np.arange(21, 24),
        eef_quat=np.arange(24, 28),
        gripper_joints_pos=np.arange(28, 34),
        gripper_joints_vel=np.arange(34, 40),
        door_pos=np.arange(40, 43),
        handle_pos=np.arange(43, 46),
        door_to_eef_pos=np.arange(46, 49),
        handle_to_eef_pos=np.arange(49, 52),
        hinge_qpos=np.arange(52, 53),
        # Some custom observations
        # XXXXXXXXXXXXXXXXXXXXXXXXX
        grasped=np.arange(53, 54),  # DO NOT USE THIS OBSERVATION IN NEURAL NETWORKS; THIS IS FOR CALCULATING REWARDS
        fingertip_dist=np.arange(54, 55),
        reach_dist=np.arange(55, 56),
        gripper_euler_angles=np.arange(56, 59),
        # XXXXXXXXXXXXXXXXXXXXXXXXX
        handle_qpos=np.arange(59, 60),
    )

    # Define the reward function
    # We will have 3 options
    #   - One to reach to the handle
    #   - One to push the handle down
    #   - One to pull the door
    _REACH_REWARD_COEF = 25.0
    _GRASP_REWARD_COEF = _OPEN_DOOR_REWARD_COEF = 100.0

    def reward_func(last_obs: np.ndarray,
                    obs: np.ndarray,
                    action: np.ndarray,
                    option_id: np.ndarray):
        # Calculate distance to eef for the last and current observation
        last_reach_dist, reach_dist = last_obs[..., obs_dict['reach_dist']], \
                                      obs[..., obs_dict['reach_dist']]
        last_grasped, grasped = last_obs[..., obs_dict['grasped']], obs[..., obs_dict['grasped']]
        last_handle_angle, handle_angle = last_obs[..., obs_dict['handle_qpos']], obs[..., obs_dict['handle_qpos']]
        last_door_angle, door_angle = last_obs[..., obs_dict['hinge_qpos']], obs[..., obs_dict['hinge_qpos']]

        # We want to represent the angles as distances to go
        last_handle_angle, handle_angle = 1.57 - last_handle_angle, 1.57 - handle_angle
        last_door_angle, door_angle = 0.4 - last_door_angle, 0.4 - door_angle

        # Assign rewards based on the option engaged
        reach_reward = _REACH_REWARD_COEF * (scaled_dist(reach_dist, scale=0.8) -
                                             scaled_dist(last_reach_dist, scale=0.8))
        grasp_reward = _GRASP_REWARD_COEF * (grasped - last_grasped) * (scaled_dist(last_handle_angle, scale=3.0) +
                                                                        scaled_dist(last_door_angle, scale=0.8))
        door_reward = _OPEN_DOOR_REWARD_COEF * grasped * ((scaled_dist(handle_angle, scale=3.0) +
                                                           scaled_dist(door_angle, scale=0.8)) -
                                                          (scaled_dist(last_handle_angle, scale=3.0) +
                                                           scaled_dist(last_door_angle, scale=0.8)))
        if len(option_id.shape) < 2:
            option_id = np.expand_dims(option_id, axis=-1)

        rew = \
            np.where(option_id == 0,
                     reach_reward + np.clip(grasp_reward, None, 0.) + np.clip(door_reward, None, 0.),
                     0.) +\
            np.where(option_id == 1,
                     grasp_reward + np.clip(reach_reward, None, 0.) + np.clip(door_reward, None, 0.),
                     0.) +\
            np.where(option_id == 2,
                     door_reward + np.clip(grasp_reward, None, 0.) + np.clip(reach_reward, None, 0.),
                     0.)

        return rew

    # Define option level information
    n_options = 3

    # NOTE: Action space is (dx, dy, dz, dax, day, daz, gripper_ctrl)
    # gripper_ctrl = -1 => Open gripper
    # gripper_ctrl = 1 => Close gripper
    # Option 1: REACHING
    reach_default_action = np.zeros(7, dtype=np.float32)
    reach_default_action[-1] = -1.
    reach_actor_params = ActorParams(default_action=th.as_tensor(reach_default_action))
    reach_actor_params.net_arch = [200, 200]
    reach_actor_params.observation_mask = \
        np.concatenate([
            obs_dict['arm_joints_pos'],
            obs_dict['arm_joints_vel'],
            obs_dict['handle_pos'],
            obs_dict['handle_to_eef_pos'],
            obs_dict['handle_qpos']
        ])
    reach_actor_params.action_mask = np.arange(6)

    reach_critic_params = CriticParams()
    reach_critic_params.net_arch = [300, 200]
    reach_critic_params.observation_mask = np.concatenate([
        obs_dict['arm_joints_pos'],
        obs_dict['arm_joints_vel'],
        obs_dict['gripper_joints_pos'],
        obs_dict['handle_pos'],
        obs_dict['handle_to_eef_pos'],
        obs_dict['hinge_qpos'],
        obs_dict['handle_qpos']
    ])

    reach_terminator_params = TerminatorParams()
    reach_terminator_params.net_arch = [200, 200]
    reach_terminator_params.observation_mask = np.concatenate([
        obs_dict['arm_joints_pos'],
        obs_dict['arm_joints_vel'],
        obs_dict['handle_pos'],
        obs_dict['handle_to_eef_pos']
    ])

    reach_exploration_params = None

    # Option 2: Turn the handle (same as the grasp option)
    handle_default_action = np.zeros(7, dtype=np.float32)
    # We should try to close the gripper
    handle_default_action[-1] = 1.
    handle_actor_params = ActorParams(default_action=th.as_tensor(handle_default_action))
    # handle_actor_params.net_arch = [200, 200]
    # handle_actor_params.observation_mask = \
    #     np.concatenate([
    #         obs_dict['arm_joints_pos'],
    #         obs_dict['arm_joints_vel'],
    #         obs_dict['gripper_joints_pos'],
    #         obs_dict['gripper_joints_vel'],
    #         obs_dict['handle_to_eef_pos'],
    #         obs_dict['handle_qpos']
    #     ])
    # handle_actor_params.action_mask = np.arange(6)
    handle_actor_params.action_mask = np.array([])

    handle_critic_params = CriticParams()
    handle_critic_params.net_arch = [300, 200]
    handle_critic_params.observation_mask = np.concatenate([
        obs_dict['arm_joints_pos'],
        obs_dict['arm_joints_vel'],
        obs_dict['gripper_joints_pos'],
        obs_dict['handle_pos'],
        obs_dict['handle_to_eef_pos'],
        obs_dict['hinge_qpos'],
        obs_dict['handle_qpos']
    ])

    handle_terminator_params = TerminatorParams()
    handle_terminator_params.net_arch = [200, 200]
    handle_terminator_params.observation_mask = np.concatenate([
        obs_dict['gripper_joints_pos'],
        obs_dict['gripper_joints_vel'],
        obs_dict['handle_to_eef_pos'],
        obs_dict['handle_qpos']
    ])

    if count_based_exploration:
        def features_extractor(states: th.Tensor):
            # Get the distance, relative angles between the grasper and object and the gripper closure
            features = states[..., np.concatenate([obs_dict['reach_dist'],
                                                   obs_dict['fingertip_dist']])]
            return features


        # For distance-based boundary, we will create evenly-spaced values on the log distance scale between
        # 0.01 - 0.1 m (values outside these ranges are pretty meaningless w.r.t the task, so we can bin them together)
        dist_boundaries = th.pow(2, th.linspace(-5, -3, steps=10))

        # The angle bounds are evenly spread in 10 degree intervals (it's the same for all of roll, pitch and yaw angles)
        # step_size = np.pi / 18
        # angle_boundaries = th.arange(-np.pi + step_size, np.pi, step_size)

        # Gripper bounds (for fingertip dist)
        step_size = 1.51 / 10
        gripper_boundaries = th.arange(step_size, 1.51 + step_size, step_size)
        features_boundaries = [dist_boundaries, gripper_boundaries]

        # Define the intrinsic reward function
        def intrinsic_reward_func(observation: np.ndarray, count: int) -> np.float32:
            return 1.0 / np.sqrt(count)
            # reach_potential = scaled_dist(observation[..., obs_dict['reach_dist']], scale=0.8)
            # reach_potential_max = 1.0
            # reach_decay = np.log(2) / 0.2
            # decay_factor = np.exp((reach_potential - reach_potential_max) * reach_decay)
            # return decay_factor / np.sqrt(count)

            # reach_potential = scaled_dist(observation[..., obs_dict['reach_dist']], scale=0.8)
            # reach_potential_max = 1.0
            # goodness_decay = np.log(2) / 0.1
            # goodness = np.exp((reach_potential - reach_potential_max) * goodness_decay)
            # count_decay = np.log(2) / 1e4
            # count_factor = np.exp(-count_decay * count)
            # intrinsic_reward = goodness * count_factor
            # return intrinsic_reward.item()

        handle_exploration_params = ExplorationParams(features_extractor=features_extractor,
                                                      feature_boundaries=features_boundaries,
                                                      reward_func=intrinsic_reward_func,
                                                      scale=5.0)
    else:
        handle_exploration_params = None

    # Option 3: Pull the door
    def pull_default_action(observations: th.Tensor):
        return th.where(observations[..., obs_dict['grasped']].unsqueeze(dim=-1) == 1.0,
                        th.Tensor([0., 0., 0., 0., 0., 0., 1.]),
                        th.Tensor([0., 0., 0., 0., 0., 0., 0.])).squeeze()
    # pull_default_action = np.zeros(7, dtype=np.float32)
    # pull_default_action[-1] = 1.
    # pull_actor_params = ActorParams(default_action=th.as_tensor(pull_default_action))
    pull_actor_params = ActorParams(default_action=pull_default_action)
    pull_actor_params.net_arch = [200, 200]
    pull_actor_params.observation_mask = np.concatenate([
        obs_dict['arm_joints_pos'],
        obs_dict['arm_joints_vel'],
        obs_dict['gripper_joints_pos'],
        obs_dict['gripper_joints_vel'],
        obs_dict['handle_pos'],
        obs_dict['handle_to_eef_pos'],
        obs_dict['hinge_qpos'],
        obs_dict['handle_qpos']
    ])
    pull_actor_params.action_mask = np.arange(6)

    pull_critic_params = CriticParams()
    pull_critic_params.net_arch = [300, 200]
    pull_critic_params.observation_mask = np.concatenate([
        obs_dict['arm_joints_pos'],
        obs_dict['arm_joints_vel'],
        obs_dict['gripper_joints_pos'],
        obs_dict['handle_pos'],
        obs_dict['handle_to_eef_pos'],
        obs_dict['hinge_qpos'],
        obs_dict['handle_qpos']
    ])

    pull_terminator_params = TerminatorParams()
    pull_terminator_params.net_arch = [200, 200]
    pull_terminator_params.observation_mask = np.concatenate([
        obs_dict['gripper_joints_pos'],
        obs_dict['handle_to_eef_pos'],
        obs_dict['handle_qpos']
    ])

    pull_exploration_params = None

    # Store all params
    actor_params = [reach_actor_params, handle_actor_params, pull_actor_params]
    critic_params = [reach_critic_params, handle_critic_params, pull_critic_params]
    terminator_params = [reach_terminator_params, handle_terminator_params, pull_terminator_params]
    exploration_params = [reach_exploration_params, handle_exploration_params, pull_exploration_params]

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
