import argparse
import os

import dmc2gym
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, VectorizedActionNoise
from stable_baselines3.ddpg import DDPG

from btopt import environments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-steps', type=int, default=2e8, help='Total steps for learning')
    parser.add_argument('--max-episode-steps', type=int, default=500, help='Max time horizon for episode')
    parser.add_argument('--n-eval-episodes', type=int, default=3, help='Number of eval episodes per evaluation')
    parser.add_argument('--eval-freq', type=int, default=10000, help='Number of rollout steps per evaluation')
    parser.add_argument('--save-freq', type=int, default=10000, help='Number of rollout steps per model checkpoint')
    parser.add_argument('--train-freq', type=int, default=2000, help='Number of rollout steps per training cycle')
    parser.add_argument('--eval-log-path', type=str, default='', help='Path for evaluation logs')
    parser.add_argument('--tensorboard-log-path', type=str, default='', help='Path for tensorboard logs')
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

    # Create the noise process for DDPG
    base_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros_like(vec_env.action_space.sample()),
                                              sigma=0.3*np.ones_like(vec_env.action_space.sample()),
                                              theta=0.15,
                                              dt=1e-3)
    noise = VectorizedActionNoise(base_noise=base_noise, n_envs=1)

    if tensorboard_log_path is None:
        print("No tensorboard log path specified through --tensorboard-log-path. Disabling tensorboard logging!")

    # Initialize the DDPG algorithm
    algorithm = DDPG(policy='MlpPolicy',
                     env=vec_env,
                     learning_rate=1e-4,
                     buffer_size=int(1e6),
                     learning_starts=100,
                     batch_size=64,
                     tau=1e-3,
                     gamma=0.99,
                     train_freq=args.train_freq,
                     n_episodes_rollout=-1,
                     action_noise=noise,
                     seed=args.seed,
                     policy_kwargs=dict(net_arch=dict(qf=[400, 300], pi=[300, 200])),
                     tensorboard_log=tensorboard_log_path,
                     device='auto')

    # Create Checkpoint Callback to regularly save model
    if eval_log_path:
        checkpoint_callback = CheckpointCallback(save_freq=args.save_freq,
                                                 save_path=os.path.join(eval_log_path, 'checkpoints'),
                                                 name_prefix='ddpg',
                                                 verbose=verbose)
    else:
        print("No eval_log_path specified...Disabling checkpoint saving of model!")
        checkpoint_callback = None

    # Start learning
    algorithm.learn(
        total_timesteps=args.total_steps,
        callback=checkpoint_callback,
        eval_env=eval_vec_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        eval_log_path=eval_log_path
    )
