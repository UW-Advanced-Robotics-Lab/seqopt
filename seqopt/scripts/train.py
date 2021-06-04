import argparse
import os

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

import robosuite as suite
from robosuite.wrappers import GymWrapper
from seqopt.seqppo import SequencePPO
from seqopt.seqsac import SequenceSAC

from .config_map import CONFIG_MAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', type=str.lower, choices=['ppo', 'sac'], help="One of 'ppo' or 'sac'")
    parser.add_argument('environment', type=str.lower, choices=CONFIG_MAP.keys(), help=f"One of {CONFIG_MAP.keys()}")
    parser.add_argument('--continue-training', type=str, default='', help='Path to model zip file')
    parser.add_argument('--eval-log-name', type=str, default='', help='Log name')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    eval_log_name = args.eval_log_name if args.eval_log_name != '' else None

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

    # Create a checkpoint callback if the eval log name is specified
    if eval_log_name is not None:
        eval_log_path = os.path.join(algo_config.log_dir, eval_log_name)
        tensorboard_log_path = os.path.join(algo_config.log_dir, 'tensorboard')
        print(f'Logging models to to {eval_log_path}')
        checkpoint_cb = CheckpointCallback(save_freq=algo_config.save_freq,
                                           save_path=os.path.join(eval_log_path, 'checkpoints'),
                                           name_prefix=f'{args.algorithm}',
                                           verbose=verbose)
    else:
        print("No eval_log_name specified...Disabling tensorboard logging and checkpoint saving of model!")
        eval_log_path = None
        tensorboard_log_path = None
        checkpoint_cb = None

    # Continue training from a previous model
    if args.continue_training != '':
        # Both algorithms have the same function signature for loading
        algo = algo_cls.load(args.continue_training, vec_env, algo_config.device)
    else:
        if args.algorithm == 'ppo':
            algo = algo_cls(
                env=vec_env,
                n_steps=algo_config.train_freq,
                batch_size=algo_config.batch_size,
                n_epochs=algo_config.n_epochs,
                gamma=algo_config.gamma,
                tensorboard_log=tensorboard_log_path,
                verbose=args.verbose,
                seed=env_config.seed,
                device=algo_config.device
            )
        elif args.algorithm == 'sac':
            algo = algo_cls(
                env=vec_env,
                buffer_size=algo_config.buffer_size,
                learning_starts=algo_config.learning_starts,
                batch_size=algo_config.batch_size,
                gamma=algo_config.gamma,
                train_freq=algo_config.train_freq,
                gradient_steps=algo_config.gradient_steps,
                n_episodes_rollout=algo_config.n_episodes_rollout,
                tensorboard_log=tensorboard_log_path,
                verbose=args.verbose,
                seed=env_config.seed,
                device=algo_config.device
            )
        else:
            raise ValueError(f"Invalid algorithm: '{args.algorithm}'")

        # Add the options (only if training a fresh model, otherwise it loads parameters from before)
        for option_id in range(env_config.n_options):
            algo.add_option(actor_params=env_config.actor_params[option_id],
                            critic_params=env_config.critic_params[option_id],
                            terminator_params=env_config.terminator_params[option_id],
                            exploration_params=env_config.exploration_params[option_id])

    # Lastly, we make a call to learn()
    if args.algorithm == 'ppo':
        algo.learn(
            total_timesteps=algo_config.total_steps,
            reward_func=env_config.reward_func,
            callback=checkpoint_cb,
            log_interval=algo_config.log_interval,
            eval_env=eval_vec_env,
            eval_freq=algo_config.eval_freq,
            n_eval_episodes=algo_config.n_eval_episodes,
            eval_log_path=eval_log_path,
            reset_num_timesteps=bool(args.continue_training == '')
        )
    elif args.algorithm == 'sac':
        algo.learn(
            total_timesteps=algo_config.total_steps,
            reward_func=env_config.reward_func,
            callback=checkpoint_cb,
            log_interval=algo_config.log_interval,
            eval_env=eval_vec_env,
            eval_freq=algo_config.eval_freq,
            n_eval_episodes=algo_config.n_eval_episodes,
            eval_log_path=eval_log_path,
            reset_num_timesteps=bool(args.continue_training == ''),
            demo_path=algo_config.demo_file_path,
            demo_learning_schedule=algo_config.demo_learning_schedule
        )
    else:
        raise ValueError(f"Invalid algorithm: '{args.algorithm}'")
