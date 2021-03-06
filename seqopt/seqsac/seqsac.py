from collections import deque
import io
import pathlib
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import gym
import numpy as np
from stable_baselines3.common import logger, utils
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, EvalCallback
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.save_util import (
    load_from_zip_file,
    save_to_zip_file
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import (
    get_device,
    get_schedule_fn,
    polyak_update,
    set_random_seed,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecTransposeImage,
    is_wrapped,
)
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.sac.policies import Actor
import torch as th
from torch.nn import functional as F

from seqopt.common.policies import ContinuousCritic, TerminatorPolicy
from seqopt.common.state_counter import StateCounter
from seqopt.common.types import RolloutReturn, Schedule
from seqopt.fsm.buffers import OptionsReplayBuffer
from seqopt.fsm.callbacks import OptionsEvalCallback
from seqopt.seqsac.params import (
    SACActorParams,
    SACCriticParams,
    SACTerminatorParams,
    SACExplorationParams
)
from seqopt.utils.demonstration_utils import load_demonstrations
from seqopt.utils.state_utils import gym_subspace_box, obs_to_box_space


class SequenceSAC(object):

    def __init__(self,
                 env: GymEnv,
                 # Training parameters
                 buffer_size: int = int(1e6),
                 learning_starts: int = 100,
                 batch_size: int = 256,
                 gamma: float = 0.99,
                 train_freq: int = 1,
                 gradient_steps: int = 1,
                 n_episodes_rollout: int = -1,
                 use_chained_model: bool = True,
                 # Extra parameters
                 tensorboard_log: Optional[str] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = 'auto'):
        self.env = env

        # Ensure the vectorized environment has only one environment contained within it
        # Since we cannot run a behavior tree in 'batch' mode, we cannot simultaneously work with multiple environments
        assert env.num_envs == 1,\
            'We do not current support Vectorized Environments with more than 1 environment!'
        # Currently, we only support box spaces for gym environments
        assert isinstance(self.env.observation_space, gym.spaces.Box) or \
               isinstance(self.env.observation_space, gym.spaces.Dict),\
            'Only observation spaces of type gym.spaces.Box or gym.spaces.Dict are allowed!'
        assert isinstance(self.env.action_space, gym.spaces.Box),\
            'Only action spaces of type gym.spaces.Box are allowed!'

        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size  # We will try to adhere to this batch size when we can
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.n_episodes_rollout = n_episodes_rollout
        self.use_chained_model = use_chained_model
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.seed = seed
        self.device = get_device(device)
        self.replay_buffer = None
        self.demo_replay_buffer = None

        # Initialize required variables
        self.default_actions = []
        self.actors = []
        self.critics = []
        self.critic_targets = []
        self.taus = []
        self.target_update_intervals = []
        self.terminators = []
        self.actor_ent_coefs = []
        self.terminator_ent_coefs = []
        self.state_visit_counters = []
        self.exploration_reward_scales = []
        self.exploration_reward_funcs = []

        # Observation and Action masks for neural networks
        self.actor_obs_masks = []
        self.actor_act_masks = []
        self.critic_obs_masks = []
        self.terminator_obs_masks = []

        # Parameters for options
        # NOTE: We only need to store these to re-create the model via saving/loading
        self._params = []

        self.num_options = None
        self._active_option = None
        self._last_obs = None
        self._last_dones = None

        # Additional variables
        self._demo_learning_schedule = None
        self.eval_env = None
        self.num_timesteps = 0
        self._total_timesteps = 0
        self.num_option_timesteps = []
        self.num_option_delta_timesteps = []
        self._episode_num = 0
        self.start_time = None
        self.ep_info_buffer = None
        self.ep_success_buffer = None

    def add_option(self,
                   actor_params: SACActorParams,
                   critic_params: SACCriticParams,
                   terminator_params: Optional[SACTerminatorParams] = None,
                   exploration_params: Optional[SACExplorationParams] = None) -> None:
        # Store the parameters for posterity
        self._params.append(
            dict(
                actor_params=actor_params,
                critic_params=critic_params,
                terminator_params=terminator_params,
                exploration_params=exploration_params
            )
        )

        empty_action_space = False
        # Determine if action space is empty
        if actor_params.action_mask is not None and actor_params.action_mask.size == 0:
            empty_action_space = True

        self.default_actions.append(actor_params.default_action)

        # Initialize actor
        if not empty_action_space:
            actor_obs_space = gym_subspace_box(self.env.observation_space, actor_params.observation_mask)
            features_extractor = FlattenExtractor(actor_obs_space)
            actor = Actor(observation_space=actor_obs_space,
                          action_space=gym_subspace_box(self.env.action_space, actor_params.action_mask),
                          net_arch=actor_params.net_arch if actor_params.net_arch is not None else [256, 256],
                          features_extractor=features_extractor,
                          features_dim=features_extractor.features_dim,
                          activation_fn=actor_params.activation_fn,
                          use_sde=False).to(self.device)
            actor.optimizer = actor_params.optimizer_class(actor.parameters(),
                                                           lr=actor_params.lr_schedule,
                                                           **actor_params.optimizer_kwargs)
        else:
            actor = None

        self.actors.append(actor)
        self.actor_ent_coefs.append(th.as_tensor(actor_params.ent_coef, device=self.device))
        self.actor_obs_masks.append(actor_params.observation_mask)
        self.actor_act_masks.append(actor_params.action_mask)

        # Initialize critic and critic target networks
        # Note: If the action space is empty, the critic basically evaluates the value function instead of the
        #       action-value function
        critic_obs_space = gym_subspace_box(self.env.observation_space, critic_params.observation_mask)
        critic_act_space = gym_subspace_box(self.env.action_space, actor_params.action_mask) \
            if not empty_action_space else None
        features_extractor = FlattenExtractor(critic_obs_space)
        critic = ContinuousCritic(observation_space=critic_obs_space,
                                  action_space=critic_act_space,
                                  net_arch=critic_params.net_arch if critic_params.net_arch is not None else [256, 256],
                                  features_extractor=features_extractor,
                                  features_dim=features_extractor.features_dim,
                                  activation_fn=critic_params.activation_fn,
                                  output_activation_fn=critic_params.output_activation_fn,
                                  n_critics=critic_params.n_critics).to(self.device)
        critic.optimizer = critic_params.optimizer_class(critic.parameters(),
                                                         lr=critic_params.lr_schedule,
                                                         **critic_params.optimizer_kwargs)

        critic_target =\
            ContinuousCritic(observation_space=critic_obs_space,
                             action_space=critic_act_space,
                             net_arch=critic_params.net_arch if critic_params.net_arch is not None else [256, 256],
                             features_extractor=features_extractor,
                             features_dim=features_extractor.features_dim,
                             activation_fn=critic_params.activation_fn,
                             n_critics=critic_params.n_critics).to(self.device)
        critic_target.load_state_dict(critic.state_dict())

        self.critics.append(critic)
        self.critic_targets.append(critic_target)
        self.critic_obs_masks.append(critic_params.observation_mask)
        self.taus.append(critic_params.tau)
        self.target_update_intervals.append(critic_params.target_update_interval)

        # Initialize terminator network
        if terminator_params is not None:
            terminator_obs_space = gym_subspace_box(self.env.observation_space, terminator_params.observation_mask)
            terminator = TerminatorPolicy(observation_space=terminator_obs_space,
                                          lr_schedule=get_schedule_fn(terminator_params.lr_schedule),
                                          net_arch=terminator_params.net_arch,
                                          activation_fn=terminator_params.activation_fn,
                                          optimizer_class=terminator_params.optimizer_class,
                                          optimizer_kwargs=terminator_params.optimizer_kwargs,
                                          use_boltzmann=terminator_params.use_boltzmann).to(self.device)

            self.terminators.append(terminator)
            self.terminator_ent_coefs.append(th.as_tensor(terminator_params.ent_coef, device=self.device))
            self.terminator_obs_masks.append(terminator_params.observation_mask)
        else:
            self.terminators.append(None)
            self.terminator_ent_coefs.append(th.tensor(0., device=self.device))
            self.terminator_obs_masks.append(None)

        # Create a state counter for exploration rewards (if needed)
        if exploration_params is not None:
            state_visit_counter = StateCounter(feature_extractor=exploration_params.features_extractor,
                                               feature_boundaries=exploration_params.feature_boundaries,
                                               device=self.device)
            self.state_visit_counters.append(state_visit_counter)
            self.exploration_reward_scales.append(exploration_params.scale)
            self.exploration_reward_funcs.append(exploration_params.reward_func)
        else:
            self.state_visit_counters.append(None)
            self.exploration_reward_scales.append(None)
            self.exploration_reward_funcs.append(None)

    def collect_rollouts(self,
                         env: VecEnv,
                         reward_func: Callable,
                         callback: BaseCallback,
                         replay_buffer: Optional[OptionsReplayBuffer] = None,
                         n_episodes: int = 1,
                         n_steps: int = -1,
                         learning_starts: int = 0,
                         log_interval: Optional[int] = None,
                         quiet: bool = False) -> RolloutReturn:
        episode_rewards, total_timesteps = [], []
        option_timesteps = [0 for _ in range(self.num_options)]
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "SequenceSAC only supports single environment"

        callback.on_rollout_start()
        continue_training = True

        episode_reward = 0.0
        episode_timesteps = 0
        while total_steps < n_steps or total_episodes < n_episodes:
            active_option = self._active_option

            with th.no_grad():
                # Convert numpy observation to tensor, and add the batch dimension
                obs_tensor = th.unsqueeze(th.as_tensor(self._last_obs, device=self.device), dim=0)
                action = self.sample_action(active_option,
                                            obs_tensor,
                                            deterministic=False,
                                            random=(self.num_timesteps < learning_starts))
            # Format action
            action = action.cpu().numpy()
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            action = np.expand_dims(action, axis=0)

            # Perform action
            new_obs, _, done, info = env.step(action)
            new_obs = obs_to_box_space(self.env.observation_space, new_obs)

            # Remove the batch dimension from outputs of the Vectorized Environment, since we only have 1 env
            new_obs, done, info = new_obs.squeeze(), done.squeeze(), info[0]

            # Note: The new observation doesn't make sense in the context of the last observation when
            #       the episode ends. If the done flag is set, we set the new observation to the terminal observation
            #       that is stored in the infos
            curr_obs = new_obs if not done else obs_to_box_space(self.env.observation_space,
                                                                 info['terminal_observation'],
                                                                 use_batch_dim=False,
                                                                 ordered_keys=list(self.env.observation_space.spaces.keys())
                                                                 if isinstance(self.env.observation_space, gym.spaces.Dict) else None).astype(np.float32)

            # Calculate the reward using our custom reward function
            # Since our reward function works with batched interactions, expand the dimensions of each element
            reward = reward_func(np.expand_dims(self._last_obs, 0),
                                 np.expand_dims(curr_obs, 0),
                                 np.expand_dims(action, 0),
                                 np.array([[active_option]])).squeeze()

            # Assign intrinsic, exploration reward if required
            curr_obs_tensor = th.unsqueeze(th.as_tensor(curr_obs, device=self.device), dim=0)
            if self.state_visit_counters[active_option] is not None and \
                self.exploration_reward_funcs[active_option] is not None:
                # Get the current count
                count = self.state_visit_counters[active_option].get_counts(curr_obs_tensor).cpu().numpy()
                intrinsic_reward = self.exploration_reward_scales[active_option] * \
                                   self.exploration_reward_funcs[active_option](curr_obs, count)
                if not quiet:
                    # Forward pass through the state counter to register a count for the new state
                    self.state_visit_counters[active_option](curr_obs_tensor)
            else:
                intrinsic_reward = np.zeros_like(reward)


            # Store data in replay buffer
            if replay_buffer is not None:
                replay_buffer.add(active_option, self._last_obs, curr_obs, action, reward + intrinsic_reward, done)

            self._last_obs = new_obs

            # Decide whether to terminate option or continue with it
            terminate, _ = self.sample_termination(active_option,
                                                   curr_obs_tensor,
                                                   deterministic=False,
                                                   random=(self.num_timesteps < learning_starts))
            if terminate:
                self._active_option = self.get_next_option(active_option, curr_obs_tensor, deterministic=False)

            if not quiet:
                self.num_timesteps += 1
                self.num_option_timesteps[active_option] += 1
                self.num_option_delta_timesteps[active_option] += 1
            option_timesteps[active_option] += 1
            episode_timesteps += 1
            total_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                option_timesteps = [0 for _ in range(self.num_options)]
                return RolloutReturn(option_timesteps, 0.0, total_steps, total_episodes, continue_training=False)

            episode_reward += reward

            self._current_progress_remaining = 1.0 - float(self.num_timesteps) / float(self._total_timesteps)

            if done:
                total_episodes += 1
                if not quiet:
                    self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)
                # Reset active option to first option
                self._active_option = 0
                # Reset values for next episode
                episode_reward = 0.0
                episode_timesteps = 0

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()
                    self.num_option_delta_timesteps = [0 for _ in range(self.num_options)]

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(option_timesteps, mean_reward, total_steps, total_episodes, continue_training)

    def _dump_logs(self):
        fps = int(self.num_timesteps / (time.time() - self.start_time))
        logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        logger.record("time/fps", fps)
        logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
        logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")

        # Log the number of steps taken for each option since the last log time and total over training time
        for option_id in range(self.num_options):
            logger.record(f"train/option_{option_id}/n_steps", self.num_option_timesteps[option_id])
            logger.record(f"train/option_{option_id}/delta_n_steps", self.num_option_delta_timesteps[option_id])

        # TODO(somesh): Log the mean/std/min/max of rewards seen across all episodes since the last log time

        # Pass the number of timesteps for tensorboard
        logger.dump(step=self.num_timesteps)

    @staticmethod
    def mask(input: th.Tensor, mask: Optional[np.ndarray]) -> th.Tensor:
        if mask is None:
            return input.clone()
        else:
            return input.clone()[..., mask]

    @staticmethod
    def unmask(default_value: th.Tensor, input: th.Tensor, mask: Optional[np.ndarray]) -> th.Tensor:
        if mask is None:
            return input.clone()
        else:
            output = default_value.clone()
            output[..., mask] = input.clone()
            return output

    def _get_default_actions(self, option_id: int, observations: th.Tensor):
        default_action = self.default_actions[option_id]
        if callable(default_action):
            observations_ = observations.squeeze()
            default_actions = default_action(observations_)
            return default_actions.to(self.device)
        else:
            return default_action.to(self.device)

    def action_log_prob(self,
                        option_id: int,
                        observation: th.Tensor,
                        masked_actions: bool = True) -> Tuple[Optional[th.Tensor], th.Tensor]:
        actor = self.actors[option_id]
        default_action = self._get_default_actions(option_id, observation)
        # If we have no control over any output values with this option, return the default action for this option
        if actor is None:
            if masked_actions:
                action = None
            else:
                action = default_action.clone()
            log_prob = th.zeros(observation.shape[0], device=self.device)
        else:
            action_mask = self.actor_act_masks[option_id]
            observation_mask = self.actor_obs_masks[option_id]
            observation_ = self.mask(observation, observation_mask)
            action, log_prob = actor.action_log_prob(observation_)
            if not masked_actions:
                action = self.unmask(default_action, action, action_mask)

        return action, log_prob

    def sample_action(self,
                      option_id: int,
                      observation: th.Tensor,
                      deterministic: bool = False,
                      random: bool = False) -> th.Tensor:
        actor = self.actors[option_id]
        default_action = self._get_default_actions(option_id, observation)
        # If we have no control over any output values with this option, return the default action for this option
        if actor is None:
            return default_action.clone()
        else:
            action_mask = self.actor_act_masks[option_id]
            if random:
                if action_mask is None:
                    action = th.as_tensor(self.env.action_space.sample(), device=self.device)
                else:
                    action = default_action.clone()
                    action[..., action_mask] = th.as_tensor(self.env.action_space.sample()[..., action_mask],
                                                            device=self.device)
            else:
                observation_mask = self.actor_obs_masks[option_id]
                observation_ = self.mask(observation, observation_mask)
                action = actor.forward(observation_, deterministic=deterministic).squeeze()
                action = self.unmask(default_action, action, action_mask)
            return action

    def sample_termination(self,
                           option_id: int,
                           observation: th.Tensor,
                           deterministic: bool = False,
                           random: bool = False) -> Tuple[th.Tensor, th.Tensor]:
        if self.terminators[option_id] is None:
            return th.tensor(False, device=self.device), th.tensor(0., device=self.device)

        if random:
            termination_prob = th.as_tensor(np.random.uniform(0, 1), device=self.device)
            terminate = th.gt(termination_prob, np.random.uniform())
        else:
            observation_mask = self.terminator_obs_masks[option_id]
            observation_ = self.mask(observation, observation_mask)
            terminate, termination_prob, _ =\
                self.terminators[option_id].forward(observation_, deterministic=deterministic)

        return terminate, termination_prob

    def train(self, option_id: int, gradient_steps: int, batch_size: int = 64) -> None:
        critic = self.critics[option_id]
        critic_target = self.critic_targets[option_id]

        loss_dicts = []
        for gradient_step in range(gradient_steps):
            # Sample replay data (collected from RL)
            replay_data = self.replay_buffer.sample(option_id=option_id, batch_size=batch_size)
            if replay_data is None:
                print(f"No replay data for option {option_id}. This was unexpected, please check code!")
                break

            # Sample demo data if available
            if self.demo_replay_buffer is not None:
                demo_replay_data = self.demo_replay_buffer.sample(option_id=option_id, batch_size=batch_size)
            else:
                demo_replay_data = None

            loss_dict = self._optimize(option_id=option_id,
                                       replay_data=replay_data,
                                       demo_replay_data=demo_replay_data)
            loss_dicts.append(loss_dict)

            # Update target networks
            if gradient_step % self.target_update_intervals[option_id] == 0:
                polyak_update(critic.parameters(), critic_target.parameters(), self.taus[option_id])

        # Log info
        # NOTE: It should be safe to assume all loss dicts have the same keys
        for loss_type in loss_dicts[0].keys():
            mean_loss = np.mean([loss_dict[loss_type] for loss_dict in loss_dicts])
            logger.record(f"train/option_{option_id}/{loss_type}", mean_loss)

        if self.demo_replay_buffer is not None:
            logger.record(f"train/lam", self._demo_learning_schedule(self._current_progress_remaining))

    def get_option_values(self,
                          obs: th.Tensor,
                          option_list: Optional[List] = None,
                          use_target_networks: bool = False) -> th.Tensor:
        # If no option list is provided, we will get the values for all options
        if option_list is None:
            option_list = [opt for opt in range(self.num_options)]

        # Store the critics or critic target networks used to compute values, in a list
        if use_target_networks:
            critic_nets = [self.critic_targets[opt] for opt in option_list]
        else:
            critic_nets = [self.critics[opt] for opt in option_list]

        all_option_vals = []    # Store Q(s,o) values
        for option_id, critic_net in zip(option_list, critic_nets):
            # Convert the observation into the form expected by the critic network(s)
            critic_obs = self.mask(obs, self.critic_obs_masks[option_id])

            # Sample an action from the associated option and get the action log prob
            actions_pi, log_prob = self.action_log_prob(option_id, obs)
            actor = self.actors[option_id]
            actor_ent_coef = self.actor_ent_coefs[option_id]

            # Convert any action(s) into the form expected by the critic network(s)
            if actor is None:
                critic_acts = None
            else:
                critic_acts = self.mask(actions_pi, self.actor_act_masks[option_id])

            # Compute Q(s,a,o) from the critic network(s) (we take the min value for stability)
            q_values = th.cat(critic_net(critic_obs, critic_acts), dim=1)
            min_q_values, _ = th.min(q_values, dim=1, keepdim=True)

            # Calculate Q(s,o) by accounting for the action entropy i.e. Q(s,o) = Q(s,a,o) - ent_coef * log pi(a)
            option_vals = min_q_values - actor_ent_coef * log_prob.reshape(-1, 1)
            all_option_vals.append(option_vals)

        # We stack the option values for each option horizontally i.e. Num obs x Num options tensor size
        return th.hstack(all_option_vals)

    def get_softmax_vals(self, option_vals: th.Tensor) -> th.Tensor:
        return F.softmax(option_vals, dim=-1)

    def get_next_option(self,
                        prev_option: int,
                        obs: th.Tensor,
                        deterministic: bool = False):
        # Note that only a single observation may be passed in
        if self.use_chained_model:
            return (prev_option + 1) % self.num_options
        else:
            with th.no_grad():
                option_vals = self.get_option_values(obs, use_target_networks=False).squeeze()

            # If the next option(s) are to be picked stochastically, choose the next option(s) by sampling
            # from a softmax over the option values
            if not deterministic:
                # Compute softmax values
                softmax_vals = self.get_softmax_vals(option_vals)
                # print(f"Option values: {option_vals}, Softmax vals: {softmax_vals}")
                # Sample the index based on the softmax values
                return np.random.choice(np.arange(self.num_options),
                                        p=softmax_vals.cpu().numpy())
            # Otherwise, just choose the option(s) with the highest option value
            else:
                return np.argmax(option_vals.cpu().numpy(), axis=-1)

    def _optimize(self,
                  option_id: int,
                  replay_data: ReplayBufferSamples,
                  demo_replay_data: Optional[ReplayBufferSamples]) -> Dict:
        next_option_id = (option_id + 1) % self.num_options
        actor = self.actors[option_id]
        terminator = self.terminators[option_id]
        critic, next_critic = self.critics[option_id], self.critics[next_option_id]
        critic_target, next_critic_target = self.critic_targets[option_id], self.critic_targets[next_option_id]
        actor_ent_coef, next_actor_ent_coef = self.actor_ent_coefs[option_id], self.actor_ent_coefs[next_option_id]
        # Using a termination entropy using the "option-critic architecture" is ill-advised because it will
        # incentivize the agent to always keep the entropy at 0.5, since it can train the policy over options to
        # re-pick the same option after terminating if needed
        terminator_ent_coef = self.terminator_ent_coefs[option_id] if self.use_chained_model else 0.0

        # Create dictionary for storing loss data
        loss_dict = dict()

        if demo_replay_data is not None:
            num_demo_samples = demo_replay_data.observations.shape[0]
            # Merge both replay data so that we can pass them through the neural networks together
            replay_data = OptionsReplayBuffer.merge_samples(replay_data, demo_replay_data)
            lam = self._demo_learning_schedule(self._current_progress_remaining)
        else:
            num_demo_samples = 0
            lam = 0.0

        # Actions by the current actor for the sampled states
        actions_pi, log_prob = self.action_log_prob(option_id, replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        # Break down the action/observation for the critics if action/observation masks need to be applied
        critic_obs = self.mask(replay_data.observations, self.critic_obs_masks[option_id])
        if actor is None:
            critic_acts = None
        else:
            critic_acts = self.mask(replay_data.actions, self.actor_act_masks[option_id])

        # Compute termination probability for the current option in the next states
        _, termination_prob_ = self.sample_termination(option_id, replay_data.next_observations)
        if terminator is not None:
            termination_ent_ = -terminator_ent_coef * ((1.0 - termination_prob_) * th.log(1.0 - termination_prob_) +
                                                       termination_prob_ * th.log(termination_prob_))
        else:
            termination_ent_ = th.zeros_like(termination_prob_, device=self.device)

        with th.no_grad():
            # Clone the termination probability and termination entropy for the current option (calculated above)
            termination_prob = termination_prob_.clone().detach()

            # We need to calculate Q(s',o) -and- either Q(s',o') or V(s') [former if using chained model, latter if not]
            # These values need to be obtained using the target networks for the purpose of computing targets for
            # training the critics, and from the primary networks for the purpose of training the termination conditions
            if self.use_chained_model:
                target_option_vals = self.get_option_values(replay_data.next_observations,
                                                            option_list=[option_id, next_option_id],
                                                            use_target_networks=True)

                curr_opt_q = target_option_vals[..., 0].unsqueeze(dim=-1)
                next_opt_q = target_option_vals[..., 1].unsqueeze(dim=-1)

                # To be used later for terminator losses
                continue_vals = curr_opt_q.clone()
                terminate_vals = next_opt_q.clone()

                target_q_values = replay_data.rewards.unsqueeze(dim=-1) + \
                                  (1 - replay_data.dones.unsqueeze(dim=-1)) * self.gamma * \
                                  ((1 - termination_prob) * curr_opt_q + termination_prob * next_opt_q +
                                   termination_ent_)
            else:
                # Fetch all option values Q(s,o)
                target_option_vals = self.get_option_values(replay_data.next_observations,
                                                            use_target_networks=True)
                # Get probabilities of all options
                target_option_probs = self.get_softmax_vals(target_option_vals)

                curr_opt_q = target_option_vals[..., option_id].unsqueeze(dim=-1)
                target_vf = th.einsum('ij,ij->i', target_option_vals, target_option_probs).unsqueeze(dim=-1)

                # To be used later for terminator losses
                continue_vals = curr_opt_q.clone()
                terminate_vals = target_vf.clone()

                target_q_values = replay_data.rewards.unsqueeze(dim=-1) + \
                                  (1 - replay_data.dones.unsqueeze(dim=-1)) * self.gamma * \
                                  ((1 - termination_prob) * curr_opt_q + termination_prob * target_vf +
                                   termination_ent_)

        # Get current Q(s,a,o) estimates from the critic of the current option
        # using action from the replay buffer
        current_q_values = critic(critic_obs, critic_acts)

        # Compute critic loss
        if num_demo_samples == 0:
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            loss_dict['critic_loss'] = critic_loss.item()
        else:
            rl_critic_loss = 0.5 * sum([F.mse_loss(current_q[:-num_demo_samples], target_q_values[:-num_demo_samples])
                                        for current_q in current_q_values])
            demo_critic_loss = 0.5 * sum([F.mse_loss(current_q[num_demo_samples:], target_q_values[num_demo_samples:])
                                        for current_q in current_q_values])
            critic_loss = rl_critic_loss + lam * demo_critic_loss
            loss_dict['rl_critic_loss'] = rl_critic_loss.item()
            loss_dict['demo_critic_loss'] = demo_critic_loss.item()
            loss_dict['critic_loss'] = critic_loss.item()

        # Optimize the critic
        critic.optimizer.zero_grad()
        critic_loss.backward()
        th.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        critic.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Mean over all critic networks
        q_values_pi = th.cat(critic(critic_obs, actions_pi), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        if actor is not None:
            # The actor losses is equal to (-)Q(s,o) i.e. we want to maximize Q(s,o) = Q(s,a,o) - ent_coef * log pi(a)
            sample_actor_losses = actor_ent_coef * log_prob - min_qf_pi
            if num_demo_samples == 0:
                actor_loss = sample_actor_losses.mean()
                loss_dict['actor_loss'] = actor_loss.item()
            else:
                rl_actor_loss = sample_actor_losses[:-num_demo_samples].mean()
                demo_actor_loss = sample_actor_losses[num_demo_samples:].mean()
                actor_loss = rl_actor_loss + lam * demo_actor_loss
                loss_dict['rl_actor_loss'] = rl_actor_loss.item()
                loss_dict['demo_actor_loss'] = demo_actor_loss.item()
                loss_dict['actor_loss'] = actor_loss.item()

            # Optimize the actor
            # print(f"Actor loss: {actor_loss}, Log Prob: {log_prob}, Min qf pi: {min_qf_pi}")
            actor.optimizer.zero_grad()
            actor_loss.backward()
            th.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor.optimizer.step()

        # Compute terminator loss (if terminators is not None)
        # We don't need to backpropagate through the value estimates since it doesn't explicitly depend on the
        # termination policy
        if terminator is not None:
            sample_terminator_losses =\
                 -((1.0 - termination_prob_) * continue_vals +
                   termination_prob_ * terminate_vals +
                   termination_ent_)
            if num_demo_samples == 0:
                terminator_loss = sample_terminator_losses.mean()
                loss_dict['terminator_loss'] = terminator_loss.item()
            else:
                rl_terminator_loss = sample_terminator_losses[:-num_demo_samples].mean()
                demo_terminator_loss = sample_terminator_losses[num_demo_samples:].mean()
                terminator_loss = rl_terminator_loss + lam * demo_terminator_loss
                loss_dict['rl_terminator_loss'] = rl_terminator_loss.item()
                loss_dict['demo_terminator_loss'] = demo_terminator_loss.item()
                loss_dict['terminator_loss'] = terminator_loss.item()

            # Optimizer terminator
            terminator.optimizer.zero_grad()
            terminator_loss.backward()
            th.nn.utils.clip_grad_norm_(terminator.parameters(), 0.1)
            terminator.optimizer.step()

        return loss_dict

    def learn(self,
              total_timesteps: int,
              reward_func: Callable,
              task_potential_func: Callable,
              callback: MaybeCallback = None,
              log_interval: int = 1,
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "SequenceSAC",
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True,
              demo_path: Optional[str] = None,
              demo_learning_schedule: Schedule = 0.0
              ) -> "SequenceSAC":
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, reward_func, task_potential_func, callback, eval_freq, n_eval_episodes,
            eval_log_path, reset_num_timesteps, tb_log_name, demo_path, demo_learning_schedule
        )

        callback.on_training_start(locals(), globals())
        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(self.env,
                                            reward_func=reward_func,
                                            callback=callback,
                                            replay_buffer=self.replay_buffer,
                                            n_steps=self.train_freq,
                                            n_episodes=self.n_episodes_rollout,
                                            learning_starts=self.learning_starts,
                                            log_interval=log_interval)
            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                for option_id in range(self.num_options):
                    # Do as many gradient steps as the steps taken under the particular option in the rollout(s)
                    gradient_steps = rollout.option_timesteps[option_id]
                    if gradient_steps > 0:
                        self.train(option_id=option_id, batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)
        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
        self.env.action_space.seed(seed)
        if self.env is not None:
            self.env.seed(seed)
        if self.eval_env is not None:
            self.eval_env.seed(seed)

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        reward_func: Callable,
        task_potential_func: Callable,
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        demo_path: Optional[str] = None,
        demo_learning_schedule: Schedule = 0.0
    ) -> Tuple[int, BaseCallback]:
        """
        Based on:
        https://github.com/DLR-RM/stable-baselines3/blob/723b341c61d168e1460399592d5cebd4c6ef3cc8/stable_baselines3/common/base_class.py#L328

        Initialize different variables needed for training.
        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """
        self._active_option = 0
        self.num_options = len(self.actors)
        self.num_option_timesteps = [0 for _ in range(self.num_options)]
        self.num_option_delta_timesteps = [0 for _ in range(self.num_options)]

        # Ensure that is more than one option exists, terminator networks are defined for all options
        if self.num_options > 1:
            assert all([terminator_net is not None for terminator_net in self.terminators])

        # Initialize the rollout buffer
        self.replay_buffer = OptionsReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            n_options=self.num_options,
            device=self.device,
        )

        # Also initialize the 'Expert' replay buffer from demonstrations if we have any
        if demo_path is not None:
            self._demo_learning_schedule = demo_learning_schedule
            (options, obs, next_obs, acts, rews), _ = load_demonstrations(demo_path, reward_func)
            num_samples = obs.shape[0]
            self.demo_replay_buffer = OptionsReplayBuffer(
                buffer_size=num_samples,
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                n_options=self.num_options,
                device=self.device
            )

            # Add the samples to the replay buffer one-by-one (there are more efficient ways to do this
            # but since we only need to do this once, we can live with it for now)
            for idx in range(num_samples):
                self.demo_replay_buffer.add(options[idx],
                                            obs[idx],
                                            next_obs[idx],
                                            acts[idx],
                                            rews[idx],
                                            np.array([0]))

        self.start_time = time.time()
        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        self._last_obs = obs_to_box_space(self.env.observation_space, self.env.reset()).squeeze()
        if reset_num_timesteps or self._last_obs is None:
            self._last_dones = np.zeros((self.env.num_envs,), dtype=np.bool)

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        # This only happens if we are continuing training a previous model
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps

            # Since we don't store replay buffers from previous model trainings
            # We should try to populate the replay buffer to either the maximum or the no. of training steps
            # observed thus far, using the agent's current policy
            populate_buffer_samples = min(self.buffer_size, self.num_timesteps)

            # Create dummy callback
            class DummyCallback(BaseCallback):
                def _on_step(self) -> bool:
                    return True
            dummy_cb = DummyCallback()
            dummy_cb.init_callback(model=self)

            self.collect_rollouts(self.env,
                                  reward_func=reward_func,
                                  callback=dummy_cb,
                                  replay_buffer=self.replay_buffer,
                                  n_steps=populate_buffer_samples,
                                  n_episodes=-1,
                                  learning_starts=0,
                                  log_interval=None,
                                  quiet=True)

        self._total_timesteps = total_timesteps

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        eval_env = self._get_eval_env(eval_env)

        # Configure logger's outputs
        utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_eval_callback(reward_func,
                                            task_potential_func,
                                            callback,
                                            eval_env,
                                            eval_freq,
                                            n_eval_episodes,
                                            log_path)

        # Set the random seed
        self.set_random_seed(self.seed)

        return total_timesteps, callback

    def get_env(self) -> Optional[VecEnv]:
        return self.env

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: GymEnv,
        device: Union[th.device, str] = "auto",
        **kwargs,
    ) -> "SequenceSAC":
        data, params, pytorch_variables = load_from_zip_file(path, device=device)

        # Create the model using only the required parameters
        model = cls(
            env=env,
            device=device
        )

        # Update the rest of the parameters of the model
        model.__dict__.update(data['cls'])

        # We have to reinitialize the model with the correct parameters
        model_params = model._params.copy()

        # This needs to be reset since we are going to populate the params again through calling add_option()
        model._params = []

        # Add the options
        assert model.num_options == len(model_params), "Inconsistent number of options and parameters!"
        for option_id in range(model.num_options):
            model.add_option(**model_params[option_id])

        # Restore the state dicts
        for option_id in range(model.num_options):
            # Load actor state dicts
            if params['actors'][option_id] is not None:
                actor_dict = params['actors'][option_id]
                model.actors[option_id].load_state_dict(actor_dict['params'])
                model.actors[option_id].optimizer.load_state_dict(actor_dict['optimizer'])

            # Load critic/critic-target state dicts
            critic_dict = params['critics'][option_id]
            model.critics[option_id].load_state_dict(critic_dict['params'])
            model.critic_targets[option_id].load_state_dict(critic_dict['target_params'])
            model.critics[option_id].optimizer.load_state_dict(critic_dict['optimizer'])

            # Load terminator state dicts
            if params['terminators'][option_id] is not None:
                terminator_dict = params['terminators'][option_id]
                model.terminators[option_id].load_state_dict(terminator_dict['params'])
                model.terminators[option_id].optimizer.load_state_dict(terminator_dict['optimizer'])

            # Load state counts
            state_counter_dict = params['state_counters'][option_id]
            if params['state_counters'][option_id] is not None:
                model.state_visit_counters[option_id].counts = state_counter_dict['counts']

        return model

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.
        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "device",
            "env",
            "eval_env",
            "replay_buffer",
            "default_actions",
            "actors",
            "critics",
            "critic_targets",
            "taus",
            "target_update_intervals",
            "terminators",
            "actor_ent_coefs",
            "terminator_ent_coefs",
            "state_visit_counters",
            "exploration_reward_scales",
            "exploration_reward_funcs",
            "actor_obs_masks",
            "actor_act_masks",
            "critic_obs_masks",
            "terminator_obs_masks"
        ]

    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            exclude: Optional[Iterable[str]] = None,
            include: Optional[Iterable[str]] = None
    ) -> None:
        data = dict()

        # Save class variables
        # Copy parameter list so we don't mutate the original dict
        data['cls'] = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data['cls'].pop(param_name, None)

        # Save the state_dicts() of all policies, critics, terminators, state counters and optimizers
        state_dicts = dict(actors=[], critics=[], terminators=[], state_counters=[])
        for option_id in range(self.num_options):
            # Store actor network and optimizer state dicts
            if self.actors[option_id] is not None:
                state_dicts['actors'].append(
                    dict(
                        params=self.actors[option_id].state_dict(),
                        optimizer=self.actors[option_id].optimizer.state_dict()
                    )
                )
            else:
                state_dicts['actors'].append(None)

            # Store critic/critic-target networks and optimizer state dicts
            state_dicts['critics'].append(
                dict(
                    params=self.critics[option_id].state_dict(),
                    target_params=self.critic_targets[option_id].state_dict(),
                    optimizer=self.critics[option_id].optimizer.state_dict()
                )
            )

            # Store terminator network and optimizer state dicts
            if self.terminators[option_id] is not None:
                state_dicts['terminators'].append(
                    dict(
                        params=self.terminators[option_id].state_dict(),
                        optimizer=self.terminators[option_id].optimizer.state_dict()
                    )
                )
            else:
                state_dicts['terminators'].append(None)

            # Store values of state-visit counters
            if self.state_visit_counters[option_id] is not None:
                state_dicts['state_counters'].append(
                    dict(
                        counts=self.state_visit_counters[option_id].get_counts()
                    )
                )
            else:
                state_dicts['state_counters'].append(None)

        save_to_zip_file(path, data=data, params=state_dicts, pytorch_variables=None, verbose=self.verbose)

    def _init_eval_callback(
        self,
        reward_func: Callable,
        task_potential_func: Callable,
        callback: MaybeCallback,
        eval_env: Optional[VecEnv] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
    ) -> BaseCallback:
        """
        Based on:
        https://github.com/DLR-RM/stable-baselines3/blob/723b341c61d168e1460399592d5cebd4c6ef3cc8/stable_baselines3/common/base_class.py#L290

        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations; if None, do not evaluate.
        :param n_eval_episodes: How many episodes to play per evaluation
        :param n_eval_episodes: Number of episodes to rollout during evaluation.
        :param log_path: Path to a folder where the evaluations will be saved
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Create eval callback in charge of the evaluation
        eval_callback = None
        if eval_env is not None:
            eval_callback = OptionsEvalCallback(
                eval_env,
                reward_func=reward_func,
                task_potential_func=task_potential_func,
                best_model_save_path=log_path,
                deterministic_actions=True,
                deterministic_transitions=True,
                log_path=log_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
            )
            callback = CallbackList([callback, eval_callback])

        callback.init_callback(self)
        return callback

    def _get_eval_env(self, eval_env: Optional[GymEnv]) -> Optional[GymEnv]:
        """
        Return the environment that will be used for evaluation.
        :param eval_env:)
        :return:
        """
        if eval_env is None:
            eval_env = self.eval_env

        # if eval_env is not None:
        #     eval_env = self._wrap_env(eval_env, self.verbose)
        #     assert eval_env.num_envs == 1
        return eval_env

    @staticmethod
    def _wrap_env(env: GymEnv, verbose: int = 0) -> VecEnv:
        if not isinstance(env, VecEnv):
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])

        if is_image_space(env.observation_space) and not is_wrapped(env, VecTransposeImage):
            if verbose >= 1:
                print("Wrapping the env in a VecTransposeImage.")
            env = VecTransposeImage(env)

        # check if wrapper for dict support is needed when using HER
        if isinstance(env.observation_space, gym.spaces.dict.Dict):
            env = ObsDictWrapper(env)

        return env
