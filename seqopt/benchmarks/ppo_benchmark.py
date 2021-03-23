import argparse
import os

import dmc2gym
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import PPO

from btopt import environments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-steps', type=int, default=3e6, help='Total steps for learning')
    parser.add_argument('--max-episode-steps', type=int, default=500, help='Max time horizon for episode')
    parser.add_argument('--n-eval-episodes', type=int, default=3, help='Number of eval episodes per evaluation')
    parser.add_argument('--eval-freq', type=int, default=10000, help='Number of rollout steps per evaluation')
    parser.add_argument('--save-freq', type=int, default=10000, help='Number of rollout steps per model checkpoint')
    parser.add_argument('--train-freq', type=int, default=2000, help='Number of rollout steps per training cycle')
    parser.add_argument('--eval-log-path', type=str, default='', help='Path for evaluation logs')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generators')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Verbosity level
    verbose = 1 if args.verbose else 0
    eval_log_path = args.eval_log_path if args.eval_log_path != '' else None

    # Use the dmc2gym library to create a dm_control environment in OpenAI gym format
    # The library registers the environment with gym in the process of making it
    # We discard this environment, since this is a hack to get the environment registered with gym
    env = dmc2gym.make(domain_name='manipulator',
                       task_name='bring_block',
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

    # Initialize the DDPG algorithm
    algorithm = PPO(policy='MlpPolicy',
                    env=vec_env,
                    learning_rate=3e-4,
                    n_steps=args.train_freq,
                    batch_size=256,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=1e-4,
                    vf_coef=0.001,
                    max_grad_norm=0.5,
                    use_sde=False,
                    sde_sample_freq=-1,
                    target_kl=0.03,
                    tensorboard_log='ppo_manipulator',
                    verbose=verbose,
                    seed=args.seed,
                    device='auto',
                    policy_kwargs=dict(net_arch=[dict(pi=[400, 300], vf=[200, 200])]))

    # Create Checkpoint Callback to regularly save model
    if eval_log_path:
        checkpoint_callback = CheckpointCallback(save_freq=args.save_freq,
                                                 save_path=os.path.join(eval_log_path, 'checkpoints'),
                                                 name_prefix='ppo',
                                                 verbose=verbose)
    else:
        print("No eval_log_path specified...Disabling checkpoint saving of model!")
        checkpoint_callback = None

    # Start learning
    algorithm.learn(
        total_timesteps=int(args.total_steps),
        callback=checkpoint_callback,
        eval_env=eval_vec_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        eval_log_path=eval_log_path
    )
