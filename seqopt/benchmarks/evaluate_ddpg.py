import argparse
import functools

import dmc2gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ddpg import DDPG

from btopt import environments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Path to model .zip file')
    parser.add_argument('--stochastic-actions', action='store_true', help='Use stochastic policy for actions')
    parser.add_argument('--n-eval-episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--max-episode-steps', type=int, default=500, help='Max time horizon for episode')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generators')
    args = parser.parse_args()

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
    # This feeds into the regeneration of the model (technically, we shouldn't really need this, but its only
    # required here due to the nature of the loading api)
    vec_env = make_vec_env(env_id=env_id,
                           n_envs=1,
                           seed=args.seed)

    # Make another copy of the environment for evaluation usage
    eval_vec_env = make_vec_env(env_id=env_id,
                                n_envs=1,
                                seed=args.seed)

    # Replace the default render function to be comptible with dm_control
    plt.ion()
    plt.show()
    image = plt.imshow(np.zeros((480, 640), dtype=np.uint8), animated=True)

    def render(image, self, mode='rgb_array'):
        frame = self.envs[0].env.physics.render(height=480, width=640, camera_id=0)
        image.set_data(frame)
        plt.pause(0.001)

    render = functools.partial(render, image, eval_vec_env)
    eval_vec_env.render = render

    # Initialize the BTPPO algorithm
    model = DDPG.load(path=args.model,
                      device='auto')
    # Evaluate
    episode_rewards, episode_lengths = evaluate_policy(model=model,
                                                       env=eval_vec_env,
                                                       n_eval_episodes=args.n_eval_episodes,
                                                       deterministic=not args.stochastic_actions,
                                                       render=True,
                                                       return_episode_rewards=True)

    # Print statistics
    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    min_reward, max_reward = np.min(episode_rewards), np.max(episode_rewards)
    mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

    print(f'Episode Rewards: {episode_rewards}, Episode Lengths: {episode_lengths}')
    print(f'Min Reward: {min_reward:.2f}, Max Reward: {max_reward:.2f}')
    print(f'Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}')
