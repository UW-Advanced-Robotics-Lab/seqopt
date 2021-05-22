import argparse
import glob
import os
from typing import Optional

from dm_control.viewer import user_input, application
import dmc2gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
import torch as th

from seqopt import environments
from seqopt.algorithms import SequenceSAC


def get_latest_run_id(log_path: Optional[str] = None, log_name: str = "") -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(f"{log_path}/{log_name}_[0-9]*"):
        file_name = path.split(os.sep)[-1]
        ext = os.path.splitext(file_name.split("_")[-1])[0]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def save_demonstration(demonstration_dict, demo_file):
    np.savez(
        demo_file,
        **demonstration_dict
    )

def yes_or_no(question: str, max_tries: int = 3) -> bool:
    num_tries = 0
    yes_no = input(question + "(y/n)").lower().strip()[0]
    if yes_no == 'y':
        return True
    elif yes_no == 'n':
        return False
    else:
        print('Invalid Response!')
        num_tries += 1
        if num_tries >= max_tries:
            return False
        else:
            return yes_or_no(question)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Path to model .zip file of trained agent')
    parser.add_argument('--stochastic-actions', action='store_true', help='Use stochastic policy for actions')
    parser.add_argument('--stochastic-terminations', action='store_true', help='Use stochastic policy for terminations')
    parser.add_argument('--max-episode-steps', type=int, default=500, help='Max time horizon for episode')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generators')
    parser.add_argument('--gpu', action='store_true', help='Run models on GPU')
    parser.add_argument('--save-path', type=str, default='.', help='Save path for demonstration files')
    parser.add_argument('--save-prefix', type=str, default='demo', help='Prefix for saved demonstration files')
    parser.add_argument('--save-on-success', action='store_true', help='Save only successful demos')
    args = parser.parse_args()

    # Use the dmc2gym library to create a dm_control environment in OpenAI gym format
    # The library registers the environment with gym in the process of making it
    # We discard this environment, since this is a hack to get the environment registered with gym
    env_name = 'manipulator'
    task_name = 'bring_ball'
    task_kwargs = dict(random=args.seed)
    env = dmc2gym.make(domain_name=env_name,
                       task_name=task_name,
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

    # Make another copy of the environment for simulation (we don't need the stable baselines environment for this)
    eval_vec_env = make_vec_env(env_id=env_id,
                                n_envs=1,
                                seed=args.seed)

    # Initialize the SequenceSAC algorithm with the trained agent parameters
    model = SequenceSAC.load(path=args.model,
                             env=vec_env,
                             device='cuda' if args.gpu else 'cpu')

    saved, demonstration_states = None, None
    demonstration_file, demo_id = '', 0
    subtask_id = 0
    user_triggered_save = False

    # Create a alias for the dm_control env which is wrapped inside the stable baselines vec env
    dm_env = eval_vec_env.envs[0].env.env._env

    def new_demo_cb():
        # Prior to resetting/restarting the environment, get the random state of its generator
        # so that we can reinitialize environments to the same state at a later time, if required
        random_state = dm_env.task.random.get_state()

        if 'app' in globals():
            globals()['app']._restart_runtime()
            globals()['app']._pause_subject.value = True

        globals()['saved'] = False
        globals()['subtask_id'] = 0
        globals()['user_triggered_save'] = False

        # Initialize a dictionary to store relevant states
        globals()['demonstration_states'] = dict(
            env=globals()['env_name'],
            task=globals()['task_name'],
            task_kwargs=globals()['task_kwargs'],
            random_state=random_state,
            duration=0.0,
            max_reward=-np.inf,
            # Variables that we want to store (other than explicit observations from the environment)
            steps=0,
            subtask_id=[],  # Can be one of 0 (reaching), 1 (grasping) and 2 (placing)
            actions=[],
        )

        # Add observations from the environment (we can also store observations as is, but storing each element
        # of the observation separately allows us to easily deal with changes in the observation space)
        globals()['demonstration_states'].update(
            **{obs_name: [] for obs_name in globals()['dm_env'].observation_spec().keys()}
        )

        globals()['demo_id'] = get_latest_run_id(globals()['args'].save_path, globals()['args'].save_prefix) + 1
        globals()['demonstration_file'] =\
            os.path.join(globals()['args'].save_path,
                         globals()['args'].save_prefix + '_' + str(globals()['demo_id']))

    def trigger_save():
        globals()['user_triggered_save'] = True

    # Call the new demo callback manually for the first demonstration, to initialize appropriate variables
    new_demo_cb()

    # Create the 'policy' function for the dm_control viewer
    def play_policy(time_step):
        global args, subtask_id, demonstration_states, demonstration_file, saved

        # Extract and flatten current observation
        obs = np.concatenate([ob.flatten() for ob in time_step.observation.values()])
        obs = np.expand_dims(obs, axis=0)
        obs_tensor = th.as_tensor(obs, device=model.device, dtype=th.float32)

        # Check if we should move to the next option based on the current state (does not apply for the first step
        # in the episode)
        if not time_step.first():
            with th.no_grad():
                terminate, _ = model.sample_termination(subtask_id,
                                                        obs_tensor,
                                                        deterministic=not args.stochastic_terminations)
                if terminate:
                    subtask_id = (subtask_id + 1) % model.num_options

        # Get action from policy
        with th.no_grad():
            action = model.sample_action(subtask_id,
                                         obs_tensor,
                                         deterministic=not args.stochastic_actions)

        action = np.squeeze(action.cpu().numpy())
        clipped_actions = np.clip(action, vec_env.action_space.low, vec_env.action_space.high)

        # Save demonstration values
        if not saved:
            demonstration_states['duration'] += dm_env.control_timestep()
            demonstration_states['steps'] = dm_env._step_count
            # Add all observation elements
            for obs_name in dm_env.observation_spec().keys():
                demonstration_states[obs_name].append(time_step.observation[obs_name])
            demonstration_states['subtask_id'].append(subtask_id)

            # Determine if demonstration should be saved
            if time_step.reward == 1.0 and args.save_on_success:
                # The task has been successfully accomplished
                print(f"Task successfully accomplished! Saving demonstration to {demonstration_file}")
                save_demonstration(demonstration_states, demonstration_file)
                saved = True
            elif user_triggered_save:
                print(f"User Triggered Save! Saving demonstration to {demonstration_file}")
                print(f"Num obs: {len(demonstration_states['arm_pos'])}, NUm actions: {len(demonstration_states['actions'])}")
                save_demonstration(demonstration_states, demonstration_file)
                saved = True
            elif time_step.last():
                # Ask user if they want to save this demonstration, because this is the last timestep
                if yes_or_no("Save demonstration?"):
                    print(f"Num obs: {len(demonstration_states['joints_pos'])}, NUm actions: {len(demonstration_states['actions'])}")
                    save_demonstration(demonstration_states, demonstration_file)
                    saved = True

            # Store actions (except for the last step, since we can't observe the outcome)
            if not time_step.last() and not saved:
                demonstration_states['actions'].append(action)

        return clipped_actions

    # Instead of the viewer.launch() command, we manually invoke the application, in order to bind custom keyboard
    # callbacks
    app = application.Application(title='Collect Demonstrations', width=1024, height=768)
    # Bind to the environment restart key. We will reinitialize our buffers for demonstrations
    app._input_map.bind(new_demo_cb, application._RESTART)
    app._input_map.bind(trigger_save, user_input.KEY_M)
    app.launch(environment_loader=dm_env, policy=play_policy)