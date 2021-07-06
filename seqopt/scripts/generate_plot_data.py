import argparse
import os
import sys

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
    parser.add_argument('checkpoint_dir', type=str, help='Path to checkpoint model(s)')
    parser.add_argument('filename', type=str, help='Filename for generated data')
    parser.add_argument('--stochastic-actions', action='store_true', help='Use stochastic policy for actions')
    parser.add_argument('--stochastic-terminations', action='store_true', help='Use stochastic policy for terminations')
    parser.add_argument('--n-eval-episodes', type=int, default=5,
                        help='Number of evaluation episodes per model for mean/std calculations')
    parser.add_argument('--skip-n-models', type=int, default=0,
                        help='Number of checkpoint models to skip per evaluation of any model')
    parser.add_argument('--max-model-step', type=int, default=1e7,
                        help='Maximum number of training steps for models (models over this are not evaluated)')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generators')
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
        env_config.env.update(dict(has_renderer=False))
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

    # Given the directory with the model checkpoints, evaluate the model
    model_files = os.listdir(args.checkpoint_dir)

    # Sort files based on the step number (file formats are [algo]_stepnum_steps.zip
    model_files.sort(key=lambda model_file: int(model_file.split('_')[1]))

    # Discard all model files above the maximum step num
    step_nums = [int(model.split('_')[1]) for model in model_files]
    step_interval = np.diff(step_nums)[0]
    max_index = (args.max_model_step // step_interval) - 1
    if max_index < 0:
        sys.exit('max-model-step is lower than the number of training steps for the first model!')
    elif max_index < len(model_files):
        step_nums = step_nums[:max_index + 1]
        model_files = model_files[:max_index + 1]

    # Also discard any model files that need to be skipped
    # Always keep the first and the last models, so that the resulting graphs have the required start/end points
    if 0 < args.skip_n_models < len(model_files):
        step_nums = step_nums[:-1:args.skip_n_models] + [step_nums[-1]]
        model_files = model_files[:-1:args.skip_n_models] + [model_files[-1]]

    print(f"Step Nums: {step_nums}, Model Files: {model_files}")

    # Loop over the models and obtain rewards for n_eval_episodes trials
    save_dict = dict(
        algorithm=args.algorithm,
        environment=args.environment,
        deterministic_actions=not args.stochastic_actions,
        deterministic_terminations=not args.stochastic_terminations,
        seed=args.seed,
        step_nums=np.array([]),
        rewards=np.empty((0, args.n_eval_episodes), dtype=np.float32),
        max_task_potentials=np.empty((0, args.n_eval_episodes), dtype=np.float32),
        checkpoint_dir=args.checkpoint_dir,     # For posterity
    )

    print(f"Num evaluations: 0/{len(step_nums)}")
    for idx, (step_num, model_file) in enumerate(zip(step_nums, model_files)):
        # Create the model by restoring the checkpoint
        model = algo_cls.load(os.path.join(args.checkpoint_dir, model_file), vec_env, args.device)

        # Evaluate the model
        episode_rewards, _, max_task_potentials = evaluate_policy(
            model=model,
            env=eval_vec_env,
            reward_func=env_config.reward_func,
            task_potential_func=env_config.task_potential_func,
            n_eval_episodes=args.n_eval_episodes,
            deterministic_actions=not args.stochastic_actions,
            deterministic_terminations=not args.stochastic_terminations,
            render=False,
            return_episode_rewards=True,
            return_max_task_potentials=True
        )

        # Store the episode rewards and step num
        save_dict['step_nums'] = np.append(save_dict['step_nums'], step_num)
        save_dict['rewards'] = np.vstack([save_dict['rewards'], episode_rewards])
        save_dict['max_task_potentials'] = np.vstack([save_dict['max_task_potentials'], max_task_potentials])

        print(f"\rNum evaluations: {idx + 1}/{len(step_nums)}")

    # Save the results
    np.savez(file=f'{args.filename}.npz', **save_dict)
