import argparse
import functools
import os
from typing import Optional

from dm_control.utils import rewards
import dmc2gym
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv
import torch as th

from seqopt import environments
from seqopt.algorithms import SequenceSAC
from seqopt.common.types import ActorParams, CriticParams, TerminatorParams, ExplorationParams
import seqopt.utils.manipulator_utils as manipulator_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-steps', type=int, default=3e6, help='Total steps for learning')
    parser.add_argument('--max-episode-steps', type=int, default=500, help='Max time horizon for episode')
    parser.add_argument('--buffer-size', type=int, default=1e6, help='Replay Buffer size')
    parser.add_argument('--continue-training', type=str, default='', help='Path to model zip file')
    parser.add_argument('--n-eval-episodes', type=int, default=3, help='Number of eval episodes per evaluation')
    parser.add_argument('--eval-freq', type=int, default=15000, help='Number of rollout steps per evaluation')
    parser.add_argument('--save-freq', type=int, default=15000, help='Number of rollout steps per model checkpoint')
    parser.add_argument('--train-freq', type=int, default=1, help='Number of rollout steps per train step')
    parser.add_argument('--gradient-steps', type=int, default=1, help='Number of gradient steps for training')
    parser.add_argument('--n-episodes-rollout', type=int, default=-1, help='Number of rollout episodes per train step')
    parser.add_argument('--learning-starts', type=int, default=100, help='Random action steps for initial data')
    parser.add_argument('--gamma', type=int, default=0.99, help='Discount factor')
    parser.add_argument('--batch-size', type=int, default=256, help='Maximum batch size for gradient updates')
    parser.add_argument('--eval-log-path', type=str, default='', help='Path for evaluation logs')
    parser.add_argument('--tensorboard-log-path', type=str, default='', help='Path for tensorboard logs')
    parser.add_argument('--count-based-exploration', action='store_true',
                        help='Add intrinsic rewards for count-based exploration')
    parser.add_argument('--demos-path', type=str, default='',
                        help='Path to folder with demonstrations for pretraining. Set only when pretraining!')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generators')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Verbosity level
    verbose = 1 if args.verbose else 0
    eval_log_path = args.eval_log_path if args.eval_log_path != '' else None
    tensorboard_log_path = args.tensorboard_log_path if args.tensorboard_log_path != '' else None

    # Use the dmc2gym library to create a dm_control environment in OpenAI gym format
    # The library registers the environment with gym in the process of making it
    # We discard this environment, since this is a hack to get the environment registered with gym
    env = dmc2gym.make(domain_name='manipulator',
                       task_name='bring_ball',
                       seed=args.seed,
                       episode_length=args.max_episode_steps)

    # Get the id of the environment that was registered with gym
    env_id = env.spec.id

    # Discard the environment
    # We wouldn't have to go through this procedure if we were directory working with native Gym environments
    # The conversion from a dm_control to gym environment is not yet a seamless experience
    del env

    # Now we actually create the (vectorized) environment using Stable Baselines 3
    # We have the ability to create multiple environments that may be trained in parallel
    vec_env = make_vec_env(env_id=env_id,
                           n_envs=1,
                           seed=args.seed)

    # Make another copy of the environment for evaluation usage
    eval_vec_env = make_vec_env(env_id=env_id,
                                n_envs=1,
                                seed=args.seed)

    if args.continue_training != '':
        algorithm = SequenceSAC.load(path=args.continue_training,
                                     env=vec_env,
                                     device='cpu')
        # Create Checkpoint Callback to regularly save model
        if eval_log_path:
            checkpoint_callback = CheckpointCallback(save_freq=args.save_freq,
                                                     save_path=os.path.join(eval_log_path, 'checkpoints'),
                                                     name_prefix='seqsac',
                                                     verbose=verbose)
        else:
            print("No eval_log_path specified...Disabling checkpoint saving of model!")
            checkpoint_callback = None

        # Train
        algorithm.learn(
            total_timesteps=args.total_steps,
            reward_func=manipulator_utils.reward,
            callback=checkpoint_callback,
            eval_env=eval_vec_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            eval_log_path=eval_log_path,
            reset_num_timesteps=False,
            demo_path=None
        )
    else:

        # Initialize the SequenceSAC algorithm
        if tensorboard_log_path is None:
            print("No tensorboard log path specified through --tensorboard-log-path. Disabling tensorboard logging!")

        algorithm = SequenceSAC(
            env=vec_env,
            buffer_size=int(args.buffer_size),
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            n_episodes_rollout=args.n_episodes_rollout,
            tensorboard_log=tensorboard_log_path,
            seed=args.seed,
            device='cpu',
            verbose=verbose
        )

        # Add options for task
        # Summary of observation space parameters (total 44 parameters)
        # Index 0 - 15: Pairs of sin(angle), cos(angle) for 8 joints (first 4 arms joints, then 4 gripper joints)
        # Index 16 - 23: Joint velocities for the 8 joints
        # Index 24 - 28: Touch sensor values (all sensors are contained in the gripper area)
        # Index 29 - 32: Pose of hand (x,z,qw,qy)
        # Index 33 - 36: Pose of object (x,z,qw,qy)
        # Index 37 - 39: Velocity of object (x, z, y)
        # Index 40 - 43: Pose of target location (x,z,qw,qy)

        if args.count_based_exploration:
            def features_extractor(states: th.Tensor):
                # Get the distance, relative angles between the grasper and object and the gripper closure
                features = states[..., np.concatenate([manipulator_utils.INDEX_DICT['reach_dist'],
                                                       manipulator_utils.INDEX_DICT['grasp_angle'],
                                                       manipulator_utils.INDEX_DICT['fingertip_dist']])]
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
        else:
            features_extractor = None
            features_boundaries = None

        # Option 0: REACHING
        # ------------------
        reach_actor_params = ActorParams(default_action=th.Tensor([0., 0., 0., 0., -1.]))
        reach_critic_params = CriticParams()
        reach_terminator_params = TerminatorParams()
        reach_exploration_params = None

        # Set actor parameters
        reach_actor_params.net_arch = [200, 100]
        reach_actor_params.observation_mask = np.concatenate([
            np.arange(8),
            np.arange(8, 16),
            np.arange(29, 33),
            np.arange(33, 35)
        ])
        reach_actor_params.action_mask = np.arange(4)

        # Set critic parameters
        reach_critic_params.net_arch = [300, 200]
        reach_critic_params.observation_mask = np.concatenate([
            np.arange(8),
            np.arange(8, 16),
            np.arange(29, 33),
            np.arange(33, 35),
            np.arange(40, 42)
        ])

        # Set terminator parameters
        reach_terminator_params.net_arch = [200, 200]
        reach_terminator_params.observation_mask = np.concatenate([
            np.arange(8, 16),  # sin/cos representation of joint angles for the 4 gripper joints
            np.arange(16, 24),  # joint velocities of all joints
            np.arange(29, 33),  # Pose of hand
            np.arange(33, 35)  # Position of object
        ])

        algorithm.add_option(reach_actor_params,
                             reach_critic_params,
                             reach_terminator_params,
                             reach_exploration_params)

        # Option 1: GRASPING
        # ------------------
        grasp_actor_params = ActorParams(default_action=th.Tensor([0., 0., 0., 0., 1.]))
        grasp_critic_params = CriticParams()
        grasp_terminator_params = TerminatorParams()
        if args.count_based_exploration:
            grasp_exploration_params = ExplorationParams(features_extractor=features_extractor,
                                                         feature_boundaries=features_boundaries,
                                                         scale=5.0)
        else:
            grasp_exploration_params = None

        # Set actor parameters
        grasp_actor_params.action_mask = np.array([])

        # Set critic parameters
        grasp_critic_params.net_arch = [300, 200]
        grasp_critic_params.observation_mask = np.concatenate([
            np.arange(8, 16),    # sin/cos representation of joint angles for the 4 gripper joints
            np.arange(29, 33),   # Pose of hand
            np.arange(33, 35),   # Position of object
            np.arange(40, 42)    # Position of target
        ])

        # Set terminator parameters
        grasp_terminator_params.net_arch = [200, 200]
        grasp_terminator_params.observation_mask = np.concatenate([
            np.arange(8, 16),   # sin/cos representation of joint angles for the 4 gripper joints
            np.arange(29, 33),  # Pose of hand
            np.arange(33, 35),  # Position of object
        ])

        algorithm.add_option(grasp_actor_params,
                             grasp_critic_params,
                             grasp_terminator_params,
                             grasp_exploration_params)

        # Option 2: PLACING
        # -----------------
        # For placing, we want the default action to be dependent on whether the object is grasped or not
        def place_default_action(observations: th.Tensor):
            return th.where(observations[..., manipulator_utils.INDEX_DICT['grasped']].unsqueeze(dim=-1) == 1.0,
                            th.Tensor([0., 0., 0., 0., 0.5]),
                            th.Tensor([0., 0., 0., 0., 0.])).squeeze()

        place_actor_params = ActorParams(default_action=place_default_action)
        place_critic_params = CriticParams()
        place_terminator_params = TerminatorParams()
        place_exploration_params = None

        # Set actor parameters
        place_actor_params.net_arch = [200, 100]
        place_actor_params.observation_mask = np.concatenate([
            np.arange(8),       # sin/cos representation of joint angles for the 4 arm joints
            np.arange(8, 16),   # sin/cos representation of joint angles for the 4 gripper joints
            np.arange(16, 20),  # joint velocities of arm joints
            np.arange(33, 35),  # Position of object
            np.arange(40, 42)   # Position of target location
        ])
        place_actor_params.action_mask = np.arange(4)

        # Set critic parameters
        place_critic_params.net_arch = [300, 200]
        place_critic_params.observation_mask = np.concatenate([
            np.arange(8, 16),   # sin/cos representation of joint angles for the 4 gripper joints
            np.arange(29, 33),  # Pose of hand
            np.arange(33, 35),  # Position of object
            np.arange(40, 42)   # Position of target location
        ])

        # Set terminator parameters
        place_terminator_params.net_arch = [200, 200]
        place_terminator_params.observation_mask = np.concatenate([
            np.arange(8, 16),   # sin/cos representation of joint angles for the 4 gripper joints
            np.arange(29, 33),  # Pose of hand
            np.arange(33, 35),  # Position of object
        ])

        algorithm.add_option(place_actor_params,
                             place_critic_params,
                             place_terminator_params,
                             place_exploration_params)

        # Create Checkpoint Callback to regularly save model
        if eval_log_path:
            checkpoint_callback = CheckpointCallback(save_freq=args.save_freq,
                                                     save_path=os.path.join(eval_log_path, 'checkpoints'),
                                                     name_prefix='seqsac',
                                                     verbose=verbose)
        else:
            print("No eval_log_path specified...Disabling checkpoint saving of model!")
            checkpoint_callback = None

        # Define schedule for demo learning
        def demo_schedule(progress_remaining):
            return max(2 * progress_remaining - 1, 0)

        # Train
        algorithm.learn(
            total_timesteps=args.total_steps,
            reward_func=manipulator_utils.reward,
            callback=checkpoint_callback,
            log_interval=3,
            eval_env=eval_vec_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            eval_log_path=eval_log_path,
            demo_path=None if args.demos_path == '' else args.demos_path,
            demo_learning_schedule=get_schedule_fn(demo_schedule)
        )
