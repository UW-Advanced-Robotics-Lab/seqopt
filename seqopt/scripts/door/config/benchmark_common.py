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
        horizon=500,
        use_latch=False
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
        eef_cos_angle=np.arange(59, 60),
        # XXXXXXXXXXXXXXXXXXXXXXXXX
        # handle_qpos=np.arange(60, 61),
    )

    # Define the composite task potential and the reward function
    # We will have 3 options
    # We will have 3 options
    #   - One to reach to the handle
    #   - One to push the handle down
    #   - One to pull the door
    _REACH_REWARD_COEF = 25.0
    _GRASP_REWARD_COEF = _OPEN_DOOR_REWARD_COEF = 100.0

    def composite_task_potential(obs: np.ndarray):
        reach_dist = obs[..., obs_dict['reach_dist']]
        grasped = obs[..., obs_dict['grasped']]
        door_angle = 0.4 - obs[..., obs_dict['hinge_qpos']]

        task_potential = _REACH_REWARD_COEF * scaled_dist(reach_dist, scale=0.8) + \
                         _OPEN_DOOR_REWARD_COEF * grasped * scaled_dist(door_angle, scale=0.8)

        return task_potential

    def reward_func(last_obs: np.ndarray,
                    obs: np.ndarray,
                    action: np.ndarray,
                    option_id: np.ndarray):
        # Compute reward by summing over all achieved subtask rewards
        # reach_dist = obs[..., obs_dict['reach_dist']]
        # door_angle = 0.4 - obs[..., obs_dict['hinge_qpos']]
        #
        # reach_reward = scaled_dist(reach_dist, scale=0.8)
        # grasp_reward = obs[..., obs_dict['grasped']]
        # pull_reward = scaled_dist(door_angle, scale=0.8)
        #
        # total_reward = reach_reward + grasp_reward + pull_reward
        #
        # return total_reward
        reward = composite_task_potential(obs) - composite_task_potential(last_obs)

        return reward

    # Define option level information
    # We only use 1 option to mimic a conventional RL method
    n_options = 1
    # Define the names of the options (mostly for plotting purposes)
    option_names = ['Universal']

    # If object is grasped, make sure to close the gripper
    def default_action(observations: th.Tensor):
        return th.where(observations[..., obs_dict['grasped']].unsqueeze(dim=-1) == 1.0,
                        th.Tensor([0., 0., 0., 0., 0., 0., 1.]),
                        th.Tensor([0., 0., 0., 0., 0., 0., 0.])).squeeze()
    actor_params = ActorParams(default_action=default_action)
    actor_params.net_arch = [300, 200]
    actor_params.observation_mask = \
        np.concatenate([
            obs_dict['arm_joints_pos'],
            obs_dict['arm_joints_vel'],
            obs_dict['gripper_joints_pos'],
            obs_dict['gripper_joints_vel'],
            obs_dict['handle_pos'],
            obs_dict['handle_to_eef_pos'],
            obs_dict['hinge_qpos']
        ])
    actor_params.action_mask = None

    critic_params = CriticParams()
    critic_params.net_arch = [300, 200]
    critic_params.observation_mask =\
        np.concatenate([
            obs_dict['arm_joints_pos'],
            obs_dict['arm_joints_vel'],
            obs_dict['gripper_joints_pos'],
            obs_dict['handle_pos'],
            obs_dict['handle_to_eef_pos'],
            obs_dict['hinge_qpos'],
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
        step_size = 1.0 / 8
        angle_boundaries = th.arange(-1 + step_size, 1, step_size)

        # Gripper bounds (for fingertip dist)
        step_size = 1.51 / 10
        gripper_boundaries = th.arange(step_size, 1.51 + step_size, step_size)
        features_boundaries = [dist_boundaries, gripper_boundaries]

        # Define the intrinsic reward function
        def intrinsic_reward_func(observation: np.ndarray, count: int) -> np.float32:
            return 1.0 / np.sqrt(count)

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
