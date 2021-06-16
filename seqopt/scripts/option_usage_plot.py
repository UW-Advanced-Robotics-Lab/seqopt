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
    _, _, ep_options = evaluate_policy(
        model=model,
        env=eval_vec_env,
        reward_func=env_config.reward_func,
        n_eval_episodes=1,
        deterministic_actions=not args.stochastic_actions,
        deterministic_terminations=not args.stochastic_terminations,
        render=not args.no_render,
        return_episode_rewards=False,
        return_options_used=True
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
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
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
    ax.set_aspect(100)

    # Remove any y-ticks
    ax.set_yticks([])

    # Remove borders of graph
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)

    # Label the axes
    ax.set_xlabel('Time (steps)')
    ax.set_ylabel('Option', rotation='horizontal', ha='right', va='center')

    # Plot the legend
    # Create handles for each option in the legend
    handles = []
    labels = ['Reach', 'Grasp', 'Place']
    # Get the used colormap
    cmap = plt.get_cmap(cmap_name)
    for idx in range(env_config.n_options):
        handle = mpatches.Patch(facecolor=cmap(cmap_scale * idx / (env_config.n_options - 1)),
                                label=labels[idx],
                                linestyle=None,
                                edgecolor=None)
        handles.append(handle)

    # Code obtained from https://stackoverflow.com/questions/42994338/creating-figure-with-exact-size-and-no-padding-and-legend-outside-the-axes/43001737#43001737
    # to set appropriate bounds for figure and legend
    padpoints = 3
    direction = 'h'
    otrans = ax.figure.transFigure
    # Define the legend
    t = ax.legend(handles=handles, title='Option', fancybox=True,
                  bbox_to_anchor=(1,0.5), loc='center right', bbox_transform=otrans)
    plt.tight_layout(pad=0)
    ax.figure.canvas.draw()
    plt.tight_layout(pad=0)
    ppar = [0, -padpoints / 72.] if direction == "v" else [-padpoints / 72., 0]
    trans2 = matplotlib.transforms.ScaledTranslation(ppar[0], ppar[1], fig.dpi_scale_trans) + \
             ax.figure.transFigure.inverted()
    tbox = t.get_window_extent().transformed(trans2)
    bbox = ax.get_position()
    if direction == "v":
        ax.set_position([bbox.x0, bbox.y0, bbox.width, tbox.y0 - bbox.y0])
    else:
        ax.set_position([bbox.x0, bbox.y0, tbox.x0 - bbox.x0, bbox.height])

    # Plot Option Usage
    fig.show()
    plt.show()

