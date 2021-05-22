import argparse

import numpy as np
from stable_baselines3.common.env_util import make_vec_env

import robosuite as suite
from robosuite.wrappers import GymWrapper
from seqopt.fsm.evaluation import evaluate_policy
from seqopt.seqppo import SequencePPO
from seqopt.seqsac import SequenceSAC

from .config_map import CONFIG_MAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', type=str.lower, choices=['ppo', 'sac'], help="One of 'ppo' or 'sac'")
    parser.add_argument('environment', type=str.lower, choices=CONFIG_MAP.keys(), help=f"One of {CONFIG_MAP.keys()}")
    parser.add_argument('model', type=str, help='Path to model .zip file')
    parser.add_argument('--stochastic-actions', action='store_true', help='Use stochastic policy for actions')
    parser.add_argument('--stochastic-terminations', action='store_true', help='Use stochastic policy for terminations')
    parser.add_argument('--n-eval-episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generators')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering of evaluation')
    parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'auto'], help="One of ['cpu','cuda','auto']")
    args = parser.parse_args()

    # Check which algorithm needs to be run
    if args.algorithm == 'ppo':
        algo_cls = SequencePPO
    elif args.algorithm == 'sac':
        algo_cls = SequenceSAC
    else:
        raise ValueError(f"Invalid algorithm: '{args.algorithm}'")

    # Load configuration
    env_config, algo_config = CONFIG_MAP[args.environment][args.algorithm]

    # Create training and evaluation environments
    # NOTE: If the env_id passed in is a dict, we want to create a robosuite environment
    # This requires a hack
    if isinstance(env_config.env, dict):
        env_config.env.update(dict(has_renderer=True))
        env_generator = lambda: GymWrapper(suite.make(**env_config.env))
        env_id = env_generator
    else:
        env_id = env_config.env

    vec_env = make_vec_env(env_id=env_id,
                           n_envs=1,
                           seed=env_config.seed)
    eval_vec_env = make_vec_env(env_id=env_id,
                                n_envs=1,
                                seed=env_config.seed)

    model = algo_cls.load(args.model, vec_env, args.device)

    # Evaluate the model
    # Evaluate
    episode_rewards, episode_lengths = evaluate_policy(
        model=model,
        env=eval_vec_env,
        reward_func=env_config.reward_func,
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
