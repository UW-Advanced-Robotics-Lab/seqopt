import argparse

from dm_control import viewer
import dmc2gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
import torch as th

from seqopt import environments
from seqopt.algorithms import SequenceSAC


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

    # Create the 'policy' function for the dm_control viewer
    active_option = 0
    def play_policy(time_step):
        global args, active_option

        # Extract and flatten current observation
        obs = np.concatenate([ob.flatten() for ob in time_step.observation.values()])
        obs = np.expand_dims(obs, axis=0)
        obs_tensor = th.as_tensor(obs, device=model.device, dtype=th.float32)

        # Check if we should move to the next option based on the current state (does not apply for the first step
        # in the episode)
        if not time_step.first():
            with th.no_grad():
                terminate, _ = model.sample_termination(active_option,
                                                        obs_tensor,
                                                        deterministic=not args.stochastic_terminations)
                if terminate:
                    active_option = (active_option + 1) % model.num_options

        # Get action from policy
        with th.no_grad():
            action = model.sample_action(active_option,
                                         obs_tensor,
                                         deterministic=not args.stochastic_actions)

        action = np.squeeze(action.cpu().numpy())
        clipped_actions = np.clip(action, vec_env.action_space.low, vec_env.action_space.high)
        return clipped_actions

    viewer.launch(eval_vec_env.envs[0].env.env._env, play_policy)
