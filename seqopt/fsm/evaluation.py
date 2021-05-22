from typing import Callable, List, Optional, Tuple, Union

from dmc2gym.wrappers import DMCWrapper
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th

from stable_baselines3.common.vec_env import VecEnv

from seqopt.utils.state_utils import obs_to_box_space


def evaluate_policy(
    model: "seqopt.algorithms.SequenceSAC",
    env: Union[gym.Env, VecEnv],
    reward_func: Callable,
    n_eval_episodes: int = 10,
    deterministic_actions: bool = True,
    deterministic_terminations: bool = True,
    render: bool = False,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.
    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic_actions: Whether to use deterministic or stochastic actions
    :param deterministic_terminations: Whether to use deterministic or stochastic terminations
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of reward per episode
        will be returned instead of the mean.
    :return: Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    # The rendering is a bit different depending on if the environment is a DeepMind Control Suite environment
    # that has been wrapped using the dmc2gym package, or if its a native gym environment
    is_dm_env = isinstance(env.envs[0].env.env, DMCWrapper)

    if render and is_dm_env:
        plt.ion()
        plt.show()
        image = plt.imshow(np.zeros((480, 640), dtype=np.uint8), animated=True)

    # We won't actually change the internal state machine of the finite state/option machine of the model
    # Instead we will track the change in options ourselves and use the predict() method of the state/option machine
    # to obtain the actions to execute
    active_option = 0

    episode_rewards, episode_lengths = [], []
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = obs_to_box_space(env.observation_space, env.reset())
        dones, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not dones:
            with th.no_grad():
                obs_tensor = th.as_tensor(obs, device=model.device)
                action = model.sample_action(active_option,
                                             obs_tensor,
                                             deterministic=deterministic_actions)
            clipped_actions = np.clip(action.cpu().numpy(), env.action_space.low, env.action_space.high)
            if len(clipped_actions.shape) == 1:
                clipped_actions = np.expand_dims(clipped_actions, 0)
            new_obs, _, dones, infos = env.step(clipped_actions)
            new_obs = obs_to_box_space(env.observation_space, new_obs)
            curr_obs = np.vstack([new_obs[idx] if not done \
                else obs_to_box_space(env.observation_space,
                 infos[idx]['terminal_observation'],
                 use_batch_dim=False,
                 ordered_keys=list(env.observation_space.spaces.keys())
                 if isinstance(env.observation_space, gym.spaces.Dict) else None).astype(np.float32) \
            for idx, done in enumerate(dones)])

            reward = reward_func(obs,
                                 curr_obs,
                                 clipped_actions,
                                 np.array([[active_option]])).squeeze()
            # with th.no_grad():
            #     q_vals = []
            #     obs_tensor = th.as_tensor(obs)
            #     for opt in range(model.num_options):
            #         critic_obs = model.mask(obs_tensor, model.critic_obs_masks[opt])
            #         if model.actors[opt] is not None:
            #             actor_obs = model.mask(obs_tensor, model.actor_obs_masks[opt])
            #             critic_act = model.actors[opt](actor_obs)
            #         else:
            #             critic_act = None
            #         q_val = min(model.critics[opt](critic_obs, critic_act))
            #         q_vals.append(q_val)
            #     print(f"Q-values - Option 0: {q_vals[0]}, "
            #           f"Option 1: {q_vals[1]}, "
            #           f"Option 2: {q_vals[2]}")
            obs = new_obs
            episode_reward += reward

            # Check if we should terminate
            with th.no_grad():
                new_obs_tensor = th.as_tensor(new_obs, device=model.device)
                terminate, _ = model.sample_termination(active_option,
                                                        new_obs_tensor,
                                                        deterministic=deterministic_terminations)

            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                # Relevant information
                print(f'Episode {i + 1}/{n_eval_episodes} - '
                      # f'Step {episode_length}/{env.envs[0].env._max_episode_steps} - '
                      f'Option {active_option} - '
                      f'Immediate Reward: {reward} - '
                      f'Reward {episode_reward}',
                      end='\n')

                # Rendering
                if is_dm_env:
                    frame = env.envs[0].env.physics.render(height=480, width=640, camera_id=0)
                    image.set_data(frame)
                    plt.pause(0.001)
                else:
                    env.envs[0].env.render()

            if terminate:
                active_option = (active_option + 1) % model.num_options

            if np.any(dones):
                # This resets the execution state of the finite state/option machine back to the default state
                active_option = 0

        if render:
            # Print new line after end of each episode when rendering
            print('')

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold,\
            "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths

    return mean_reward, std_reward
