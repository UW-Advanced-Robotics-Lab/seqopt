import os
import numpy as np
from typing import Callable

from dm_control import suite

import btopt.environments


def load_demonstrations(demo_path: str,
                        discount_factor: float,
                        reward_func: Callable,
                        validation_split: float = 0.0):
    assert os.path.isdir(demo_path), 'Please provide a valid path to a directory with demonstration files!'

    # Gather all files names for demonstrations
    demonstration_files = [os.path.join(demo_path, file) for file in os.listdir(demo_path)
                           if os.path.isfile(os.path.join(demo_path, file))]

    # NOTE: It is assumed all files are demonstration files in the folder, and that they have come from the same
    # environment
    # Extract the domain and task from one of the demonstration files
    demo_example = np.load(demonstration_files[0], allow_pickle=True)
    domain_name = demo_example['env'].tolist()
    task_name = demo_example['task'].tolist()
    task_kwargs = demo_example['task_kwargs'].tolist()

    env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs)

    # Now we gather all steps from all demos to create a dataset of observations (inputs) and subtasks (outputs)
    observations = []
    subtasks = []
    actions = []
    returns = []
    for demo_file in demonstration_files:
        demo = np.load(demo_file, allow_pickle=True)
        demo_observations = np.column_stack([np.reshape(demo[obs_name], (demo[obs_name].shape[0], -1))
                                             for obs_name in env.observation_spec().keys()])
        demo_subtasks = demo['subtask_id']
        demo_actions = demo['joint_torques']

        # Calculate the rewards obtained at each timestep, and compute the value of each state based on the rewards
        demo_rewards = reward_func(last_obs=demo_observations[:-1],
                                   obs=demo_observations[1:],
                                   action=demo_actions,
                                   option_id=demo_subtasks[:-1])
        demo_returns = np.zeros(len(demo_rewards))
        for step in reversed(range(len(demo_rewards))):
            if step == len(demo_rewards) - 1:
                demo_returns[step] = demo_rewards[step]
            else:
                demo_returns[step] = demo_rewards[step] + discount_factor * demo_returns[step + 1]
        # print(f"Demo rewards: {demo_rewards}")
        # print(f"Demo returns: {demo_returns}")
        # print(f"Demo length: {len(demo_rewards)}")
        # raise ValueError()

        # total_return = np.sum(demo_rewards)
        # cumulative_rewards = np.concatenate([np.array([0]), np.cumsum(demo_rewards)[:-1]])
        # demo_returns = total_return - cumulative_rewards

        # Ignore last state and its corresponding subtask for each demo (since there is no action corresponding
        # to the last observation)
        observations.append(demo_observations[:-1])
        subtasks.append(demo_subtasks[:-1])
        actions.append(demo_actions)
        returns.append(demo_returns)

    # Stack observations and concatenate subtasks for all demos
    observations = np.vstack(observations)
    subtasks = np.concatenate(subtasks)
    actions = np.vstack(actions)
    returns = np.concatenate(returns)

    # Shuffle the data
    num_total_samples = observations.shape[0]
    shuffle_idxs = np.random.permutation(num_total_samples)
    observations = observations[shuffle_idxs]
    subtasks = subtasks[shuffle_idxs]
    actions = actions[shuffle_idxs]
    returns = returns[shuffle_idxs]

    # Perform a split of the data if validation data is required
    num_validation_samples = np.ceil(validation_split * num_total_samples).astype(np.int32)

    # Print statistics
    print(f'Total Demonstration Samples: {num_total_samples}, '
          f'Total Validation Samples: {num_validation_samples}')

    if num_validation_samples > 0:
        random_idxs = np.random.permutation(num_total_samples)
        train_observations, train_subtasks, train_actions, train_returns =\
            observations[random_idxs[num_validation_samples:]],\
            subtasks[random_idxs[num_validation_samples:]],\
            actions[random_idxs[num_validation_samples:]],\
            returns[random_idxs[num_validation_samples:]]
        validate_observations, validate_subtasks, validate_actions, validate_returns =\
            observations[random_idxs[:num_validation_samples]],\
            subtasks[random_idxs[:num_validation_samples]],\
            actions[random_idxs[:num_validation_samples]],\
            returns[random_idxs[:num_validation_samples]]
        return (train_observations, train_subtasks, train_actions, train_returns),\
               (validate_observations, validate_subtasks, validate_actions, validate_returns)
    else:
        return (observations, subtasks, actions, returns), (None, None, None, None)
