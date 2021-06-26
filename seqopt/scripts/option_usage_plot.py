import argparse

import numpy as np
from stable_baselines3.common.env_util import make_vec_env

import robosuite as suite
from robosuite.wrappers import GymWrapper
from seqopt.fsm.evaluation import evaluate_policy
from seqopt.seqppo import SequencePPO
from seqopt.seqsac import SequenceSAC

from .config_map import CONFIG_MAP

import pandas as pd

import matplotlib
from matplotlib.colors import ColorConverter, ListedColormap
from matplotlib.patches import Polygon
import matplotlib.collections as mcoll
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', type=str.lower, choices=['ppo', 'sac'], help="One of 'ppo' or 'sac'")
    parser.add_argument('environment', type=str.lower, choices=CONFIG_MAP.keys(), help=f"One of {CONFIG_MAP.keys()}")
    parser.add_argument('model', type=str, help='Path to model .zip file')
    parser.add_argument('--stochastic-actions', action='store_true', help='Use stochastic policy for actions')
    parser.add_argument('--stochastic-terminations', action='store_true', help='Use stochastic policy for terminations')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generators')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering of model execution')
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
        if args.no_render:
            env_config.env.update(dict(has_renderer=False))
        else:
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
    _, _, ep_options, ep_potentials = evaluate_policy(
        model=model,
        env=eval_vec_env,
        reward_func=env_config.reward_func,
        task_potential_func=env_config.task_potential_func,
        n_eval_episodes=1,
        deterministic_actions=not args.stochastic_actions,
        deterministic_terminations=not args.stochastic_terminations,
        render=not args.no_render,
        return_episode_rewards=False,
        return_options_used=True,
        return_task_potentials=True
    )
    ep_options = np.asarray(ep_options[0], dtype=np.int)
    #
    # # Print statistics
    # mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    # min_reward, max_reward = np.min(episode_rewards), np.max(episode_rewards)
    # mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

    # DUMMY DATA
    # ep_options = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2], dtype=np.int)

    # We have to create an x-y grid for a 'heatmap'
    # The x values will range from 0 to the length of the episode in integer intervals
    # The y values will only have a range of 0 to 1
    x, y = np.meshgrid(np.arange(len(ep_options)), np.arange(2), sparse=False, indexing='ij')

    # Initialize the plot(s)
    fig, axes = plt.subplots(nrows=2, ncols=1, squeeze=False, sharex=True)
    ax = axes.T[0][0]

    # Draw the heatmap
    kwargs = dict(
        edgecolors='white',
        linewidth=0
    )
    colors = np.expand_dims(ep_options, 0).repeat(2, 0).T

    # Plot the heatmap
    cmap_name = 'tab20c'
    # The cmap scale can range from 0.0 - 1.0. This is used to restrict the range of colors
    # we utilize over the full range of the chosen colormap
    cmap_scale = 9.0 / 20

    ax.pcolormesh(x, y, colors, vmin=0, vmax=(env_config.n_options - 1) / cmap_scale, cmap=cmap_name, **kwargs)

    # Ensure graph limits are equal to the data limits
    ax.set(xlim=(0, x.shape[0] - 1), ylim=(0, x.shape[1] - 1))

    # Set x and y dimensions to be equal (in real graph units)
    # ax.set_aspect(100)

    # Remove any y-ticks
    ax.set_yticks([])

    # Remove borders of graph
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)

    # Label the axes
    # ax.set_xlabel('Time (steps)')
    ax.set_ylabel('Option', rotation='horizontal', ha='right', va='center', fontsize=16)

    # Plot the legend
    # Create handles for each option in the legend
    handles = []
    # Get the used colormap
    cmap = plt.get_cmap(cmap_name)
    for idx in range(env_config.n_options):
        handle = mpatches.Patch(facecolor=cmap(cmap_scale * idx / (env_config.n_options - 1)),
                                label=env_config.option_names[idx],
                                linestyle=None,
                                edgecolor=None)
        handles.append(handle)

    # Now plot the task potential evolution w.r.t time in the second subplot
    ep_task_potentials = np.asarray(ep_potentials[0], dtype=np.float)
    ax = axes.T[0][1]

    # Create a line using a segment to connect each adjacent point
    num_steps = len(ep_task_potentials[1:])
    segments = [[(x1, y1), (x2, y2)] for x1, y1, x2, y2 in zip(np.arange(num_steps - 1),
                                                               ep_task_potentials[:-1],
                                                               np.arange(1, num_steps),
                                                               ep_task_potentials[1:])]
    colors = [cmap(cmap_scale * opt / (env_config.n_options - 1)) for opt in ep_options]
    lc = mcoll.LineCollection(segments, colors=colors, linewidths=2)
    ax.add_collection(lc)
    ax.autoscale()
    # ax.plot(np.arange(len(ep_task_potentials[1:])), ep_task_potentials[1:])

    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    # Label the axes
    ax.set_xlabel('Time (steps)', fontsize=16)
    ax.set_ylabel(r'$\Phi_{task}$', fontsize=16)

    # Adjust the subplots, making space for a legend showing the labels for each option
    fig.subplots_adjust(left=0.07, right=0.8)

    # Add the legend
    box = axes.T[0][0].get_position()
    pad, width = 0.025, 0.1
    cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
    for side in ('top', 'right', 'left', 'bottom'):
        cax.spines[side].set_visible(False)
    cax.set_xticks([])
    cax.set_yticks([])
    cax.legend(handles=handles, title='Option', fancybox=True, loc='center right')

    # Plot Option Usage
    fig.show()
    plt.show()

