import argparse

import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from seqopt import environments
from seqopt.algorithms import SequenceSAC
from seqopt.fsm.evaluation import evaluate_policy
import seqopt.utils.kitchen_utils as kitchen_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Path to model .zip file')
    parser.add_argument('--stochastic-actions', action='store_true', help='Use stochastic policy for actions')
    parser.add_argument('--stochastic-terminations', action='store_true', help='Use stochastic policy for terminations')
    parser.add_argument('--n-eval-episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--max-episode-steps', type=int, default=500, help='Max time horizon for episode')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generators')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering of evaluation')
    parser.add_argument('--gpu', action='store_true', help='Run models on GPU')
    args = parser.parse_args()

    # Get the id of the environment that was registered with gym
    env_id = 'kitchen_relax-v1'

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

    # Initialize the BTPPO algorithm
    model = SequenceSAC.load(path=args.model,
                             env=vec_env,
                             device='cuda' if args.gpu else 'cpu')

    # Evaluate
    episode_rewards, episode_lengths = evaluate_policy(
        model=model,
        env=eval_vec_env,
        reward_func=kitchen_utils.reward,
        n_eval_episodes=args.n_eval_episodes,
        deterministic_actions=not args.stochastic_actions,
        deterministic_terminations=not args.stochastic_terminations,
        render=not args.no_render,
        return_episode_rewards=True
    )

    # Print statistics
    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    min_reward, max_reward = np.min(episode_rewards), np.max(episode_rewards)
    mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

    print(f'Episode Rewards: {episode_rewards}, Episode Lengths: {episode_lengths}')
    print(f'Min Reward: {min_reward:.2f}, Max Reward: {max_reward:.2f}')
    print(f'Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}')
