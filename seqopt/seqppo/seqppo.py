from collections import deque
import io
import pathlib
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import gym
import numpy as np
from stable_baselines3.common import logger, utils
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, EvalCallback
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.save_util import (
    load_from_zip_file,
    save_to_zip_file
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import (
    get_device,
    get_schedule_fn,
    safe_mean,
    set_random_seed,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecTransposeImage,
    is_wrapped,
)
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
import torch as th
from torch.nn import functional

from seqopt.common.types import RolloutBufferSamples
from seqopt.fsm.buffers import OptionsRolloutBuffer
from seqopt.fsm.callbacks import OptionsEvalCallback
from seqopt.common.policies import Actor, ContinuousCritic, TerminatorPolicy
from seqopt.common.state_counter import StateCounter
from seqopt.seqppo.params import (
    PPOActorParams,
    PPOCriticParams,
    PPOTerminatorParams,
    PPOExplorationParams
)
from seqopt.utils.state_utils import gym_subspace_box, obs_to_box_space


class SequencePPO(object):

    def __init__(self,
                 env: GymEnv,
                 # Training parameters
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 # Discount parameters
                 gamma: float = 0.98,
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
        assert isinstance(self.env.observation_space, gym.spaces.Box),\
            'Only observation spaces of type gym.spaces.Box are allowed!'
        assert isinstance(self.env.action_space, gym.spaces.Box),\
            'Only action spaces of type gym.spaces.Box are allowed!'

        self.n_steps = n_steps
        self.batch_size = batch_size  # We will try to adhere to this batch size when we can
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.seed = seed
        self.device = get_device(device)
        self.rollout_buffer = None

        # Initialize required variables
        # Initialize required variables
        self.default_actions = []
        self.actors = []
        self.critics = []
        self.terminators = []
        self.actor_ent_coefs = []
        self.actor_clip_ranges = []
        self.actor_target_kl_divs = []
        self.terminator_ent_coefs = []
        self.terminator_target_kl_divs = []
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

        # Additional variables
        self.num_options = None
        self._active_option = None
        self._last_obs = None
        self._last_dones = None
        self.eval_env = None
        self.num_timesteps = 0
        self._total_timesteps = 0
        self.num_option_samples = None
        self._episode_num = 0
        self.start_time = None
        self.ep_info_buffer = None
        self.ep_success_buffer = None

    def add_option(self,
                   actor_params: PPOActorParams,
                   critic_params: PPOCriticParams,
                   terminator_params: PPOTerminatorParams,
                   exploration_params: Optional[PPOExplorationParams] = None) -> None:
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
            actor = Actor(observation_space=gym_subspace_box(self.env.observation_space, actor_params.observation_mask),
                          action_space=gym_subspace_box(self.env.action_space, actor_params.action_mask),
                          lr_schedule=get_schedule_fn(actor_params.lr_schedule),
                          net_arch=actor_params.net_arch if actor_params.net_arch is not None else [256, 256],
                          activation_fn=actor_params.activation_fn).to(self.device)
        else:
            actor = None

        self.actors.append(actor)
        self.actor_ent_coefs.append(th.as_tensor(actor_params.ent_coef, device=self.device))
        self.actor_clip_ranges.append(actor_params.clip_range)
        self.actor_target_kl_divs.append(actor_params.target_kl)
        self.actor_obs_masks.append(actor_params.observation_mask)
        self.actor_act_masks.append(actor_params.action_mask)

        # Initialize critic (this represents the option-value function, and only accepts state as input)
        # We can use the ContinuousCritic from SAC without allowing any actions to be used as input
        critic_obs_space = gym_subspace_box(self.env.observation_space, critic_params.observation_mask)
        features_extractor = FlattenExtractor(critic_obs_space)
        critic = ContinuousCritic(observation_space=critic_obs_space,
                                  action_space=None,
                                  net_arch=critic_params.net_arch if critic_params.net_arch is not None else [256, 256],
                                  features_extractor=features_extractor,
                                  features_dim=features_extractor.features_dim,
                                  activation_fn=critic_params.activation_fn,
                                  output_activation_fn=critic_params.output_activation_fn,
                                  n_critics=1).to(self.device)
        critic.optimizer = critic_params.optimizer_class(critic.parameters(),
                                                         lr=critic_params.lr_schedule,
                                                         **critic_params.optimizer_kwargs)

        self.critics.append(critic)
        self.critic_obs_masks.append(critic_params.observation_mask)

        # Initialize terminator network
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
        self.terminator_target_kl_divs.append(terminator_params.target_kl)
        self.terminator_obs_masks.append(terminator_params.observation_mask)

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
                         rollout_buffer: OptionsRolloutBuffer,
                         n_rollout_steps: int) -> bool:
        assert self._last_obs is not None, "No previous observation was provided!"
        assert n_rollout_steps > 0, "Number of rollout steps must be more than 0!"

        n_steps = 0

        # Reset the rollout buffer
        rollout_buffer.reset()

        callback.on_rollout_start()

        # Note the option value of the current option in the initial state
        with th.no_grad():
            obs_tensor = th.unsqueeze(th.as_tensor(self._last_obs, device=self.device), dim=0)
            first_option_value = self.get_value(self._active_option, obs_tensor)

        # Populate the rollout buffer
        while n_steps < n_rollout_steps:
            active_option = self._active_option

            with th.no_grad():
                # Convert numpy observation to tensor, and add the batch dimension
                obs_tensor = th.unsqueeze(th.as_tensor(self._last_obs, device=self.device), dim=0)
                action = self.sample_action(active_option,
                                            obs_tensor,
                                            deterministic=False,
                                            random=False)
                # Note the log probability of the option
                action_log_prob, _ = self.get_action_distribution(option_id=active_option,
                                                                  observation=obs_tensor,
                                                                  action=action)

            # Format action
            action = action.cpu().numpy()
            clipped_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            clipped_action = np.expand_dims(clipped_action, axis=0)

            # Perform action
            new_obs, _, done, info = env.step(clipped_action)
            new_obs = obs_to_box_space(self.env.observation_space, new_obs)

            # Remove the batch dimension from outputs of the Vectorized Environment, since we only have 1 env
            new_obs, done, info = new_obs.squeeze(), done.squeeze(), info[0]

            # Note: The new observation doesn't make sense in the context of the last observation when
            #       the episode ends. If the done flag is set, we set the new observation to the terminal observation
            #       that is stored in the infos
            curr_obs = new_obs if not done else obs_to_box_space(
                self.env.observation_space,
                info['terminal_observation'],
                use_batch_dim=False,
                ordered_keys=list(self.env.observation_space.spaces.keys())
                if isinstance(self.env.observation_space, gym.spaces.Dict) else None
            ).astype(np.float32)

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
                # Forward pass through the state counter to register a count for the new state
                self.state_visit_counters[active_option](curr_obs_tensor)
            else:
                intrinsic_reward = np.zeros_like(reward)

            # Decide whether to terminate option or continue with it
            with th.no_grad():
                terminate, termination_prob = \
                    self.sample_termination(active_option,
                                            curr_obs_tensor,
                                            deterministic=False,
                                            random=False)

            # Regardless of staying or continuing with the option, determine the option values
            # of the current option and its neighbour, which will be used later to train the termination
            # condition(s)
            with th.no_grad():
                option_value = self.get_value(active_option, curr_obs_tensor)
                next_option_value = self.get_value((active_option + 1) % self.num_options, curr_obs_tensor)

            if terminate:
                self._active_option = (self._active_option + 1) % self.num_options

            # Add data to rollout buffer
            rollout_buffer.add(
                option_id=active_option,
                obs=self._last_obs,
                next_obs=curr_obs,
                action=action,
                reward=reward + intrinsic_reward,
                done=done,
                action_log_prob=action_log_prob,
                termination_prob=termination_prob,
                option_value=option_value,
                next_option_value=next_option_value
            )

            self._last_obs = new_obs
            self._last_dones = done

            self.num_timesteps += env.num_envs
            n_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

        # Compute advantage and returns for rollout buffer over all collected samples
        rollout_buffer.compute_returns_and_advantage(first_option_value=first_option_value)

        callback.on_rollout_end()

        return True

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

    def get_value(self,
                  option_id: int,
                  observation: th.Tensor) -> th.Tensor:
        return self.critics[option_id](self.mask(observation, self.critic_obs_masks[option_id]))[0]

    def get_action_distribution(self,
                                option_id: int,
                                observation: th.Tensor,
                                action: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actor = self.actors[option_id]
        if actor is None:
            return th.Tensor([0.0]).to(self.device), th.Tensor([0.0]).to(self.device)
        else:
            log_prob, entropy = actor.evaluate_actions(obs=self.mask(observation, self.actor_obs_masks[option_id]),
                                                       actions=self.mask(action, self.actor_act_masks[option_id]))
            return log_prob, entropy

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
                action, _ = actor(observation_, deterministic=deterministic)
                action = self.unmask(default_action, action, action_mask)
            return action

    def sample_termination(self,
                           option_id: int,
                           observation: th.Tensor,
                           deterministic: bool = False,
                           random: bool = False) -> Tuple[th.Tensor, th.Tensor]:
        if random:
            termination_prob = th.as_tensor(np.random.uniform(0, 1), device=self.device)
            terminate = th.gt(termination_prob, np.random.uniform())
        else:
            observation_mask = self.terminator_obs_masks[option_id]
            observation_ = self.mask(observation, observation_mask)
            terminate, termination_prob, _ =\
                self.terminators[option_id](observation_, deterministic=deterministic)

        return terminate, termination_prob

    def train(self, batch_size: int = 64) -> None:
        loss_dicts = [[] for _ in range(self.num_options)]
        actor_kl_divs = [[] for _ in range(self.num_options)]
        terminator_kl_divs = [[] for _ in range(self.num_options)]
        continue_training_actor = [True for _ in range(self.num_options)]
        continue_training_critic = [True for _ in range(self.num_options)]
        continue_training_terminator = [True for _ in range(self.num_options)]
        for epoch in range(self.n_epochs):
            for option_id in range(self.num_options):
                # Check if we are actively trying to train any of the actor/critic/terminator for this option
                if not continue_training_actor[option_id] and \
                   not continue_training_critic[option_id] and \
                   not continue_training_terminator[option_id]:
                    break

                # Sample data to train actor, critic and terminator
                for rollout_data in self.rollout_buffer.get(option_id=option_id,
                                                            max_batch_size=batch_size):
                    # Determine whether to continue training the actor, critic and/or terminator
                    # for this option
                    if continue_training_actor[option_id]:
                        if self.actors[option_id] is None:
                            continue_training_actor[option_id] = False
                        elif len(actor_kl_divs[option_id]) > 0:
                            mean_actor_kl = np.mean(actor_kl_divs[option_id])
                            if mean_actor_kl > 1.5 * self.actor_target_kl_divs[option_id]:
                                print(f"Early stopping at step {epoch} for actor of option {option_id} due to"
                                      f"reaching max kl: {mean_actor_kl:.2f}")
                                continue_training_actor[option_id] = False

                    if continue_training_terminator[option_id]:
                        if len(terminator_kl_divs[option_id]) > 0:
                            mean_terminator_kl = np.mean(terminator_kl_divs[option_id])
                            if mean_terminator_kl > 1.5 * self.terminator_target_kl_divs[option_id]:
                                print(f"Early stopping at step {epoch} for terminator of option {option_id} due to"
                                      f"reaching max kl: {mean_terminator_kl:.2f}")
                                continue_training_terminator[option_id] = False

                    # For the critic, we will train it for at least n/2 epochs, and continue
                    # training it only if we are additionally training the actor or terminator
                    continue_training_critic[option_id] = continue_training_actor[option_id]
                    # continue_training_critic[option_id] = continue_training_actor[option_id] or \
                    #                                       continue_training_terminator[option_id] or \
                    #                                       epoch < self.n_epochs // 2

                    actor_kl, termination_kl, loss_dict =\
                        self._optimize(option_id=option_id,
                                       rollout_data=rollout_data,
                                       train_actor=continue_training_actor[option_id],
                                       train_critic=continue_training_critic[option_id],
                                       train_terminator=continue_training_terminator[option_id])

                    # Store information
                    if actor_kl is not None:
                        actor_kl_divs[option_id].append(actor_kl)
                    if termination_kl is not None:
                        terminator_kl_divs[option_id].append(termination_kl)
                    loss_dicts[option_id].append(loss_dict)

                # Increment the total number of samples seen for each option
                self.num_option_samples[option_id] += self.rollout_buffer.get_num_samples(option_id)

        for option_id in range(self.num_options):
            # Calculate the means of each key in the loss dicts and record it to the logger
            # Note that each loss dict may not have the same keys (since we may only train certain
            # networks in each call to optimize())
            # Also the first loss dict for any option should represent the full set of keys that can
            # We only have data for logging if one or more samples are available
            num_new_samples = self.rollout_buffer.get_num_samples(option_id)
            if num_new_samples > 0:
                for loss_type in loss_dicts[option_id][0].keys():
                    mean_loss = np.mean([loss_dict.get(loss_type) for loss_dict in loss_dicts[option_id]
                                         if loss_dict.get(loss_type)])
                    logger.record(f"train/option_{option_id}/{loss_type}", mean_loss)

                if self.actors[option_id] is not None:
                    logger.record(f"train/option_{option_id}/actor_kl_div", np.mean(actor_kl_divs[option_id]))
                logger.record(f"train/option_{option_id}/terminator_kl_div", np.mean(terminator_kl_divs[option_id][-1]))
            logger.record(f"train/option_{option_id}/new_samples", num_new_samples)
            logger.record(f"train/option_{option_id}/total_samples", self.num_option_samples[option_id])

    def _optimize(self,
                  option_id: int,
                  rollout_data: RolloutBufferSamples,
                  train_actor: bool = True,
                  train_critic: bool = True,
                  train_terminator: bool = True) -> Tuple[Optional[np.float32],
                                                          Optional[np.float32],
                                                          Dict]:
        actor = self.actors[option_id]
        critic = self.critics[option_id]
        terminator = self.terminators[option_id]
        actor_ent_coef = self.actor_ent_coefs[option_id]
        terminator_ent_coef = self.terminator_ent_coefs[option_id]

        # Create dictionary for storing loss data
        loss_dict = dict()

        # OPTIMIZE ACTOR
        # --------------
        if train_actor:
            # NOTE: log_prob and entropy can be None if there is no actor for this option. However,
            #       we shouldn't encounter this scenario due to checks in the function(s) that calls _optimize()
            log_prob, entropy = self.get_action_distribution(option_id=option_id,
                                                             observation=rollout_data.observations,
                                                             action=rollout_data.actions)

            # ratio between old and new policy, should be one at the first iteration
            ratio = th.exp(log_prob - rollout_data.old_log_prob)

            # clipped surrogate loss (i.e. the 'policy gradient' loss)
            clip_range = self.actor_clip_ranges[option_id]
            policy_loss_1 = rollout_data.advantages * ratio
            policy_loss_2 = rollout_data.advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            actor_loss = policy_loss + actor_ent_coef * entropy_loss

            # Log data
            loss_dict['actor_loss'] = actor_loss.item()
            loss_dict['policy_loss'] = policy_loss.item()
            loss_dict['actor_entropy_loss'] = entropy_loss.item()

            # Optimize
            actor.optimizer.zero_grad()
            actor_loss.backward()
            th.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor.optimizer.step()

            # Calculate Actor KL-Divergence
            actor_kl_div = th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy()
        else:
            actor_kl_div = None

        # OPTIMIZE CRITIC
        # ----------------
        if train_critic:
            values = self.get_value(option_id, rollout_data.observations).squeeze()
            value_loss = functional.mse_loss(rollout_data.returns, values)
            if rollout_data.returns.size() != values.size():
                print(f"Returns: {rollout_data.returns}, Values: {values}")

            # Log data
            loss_dict['value_loss'] = value_loss.item()

            # Optimize
            critic.optimizer.zero_grad()
            value_loss.backward()
            th.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic.optimizer.step()

        # OPTIMIZE TERMINATOR
        # --------------------
        # Note that the 'observations' used for the termination policies are the *next* observations
        # with respect to the action policies, since the termination condition is only invoked after
        # completing the action and transitioning to the next state
        if train_terminator:
            _, termination_prob = self.sample_termination(option_id=option_id,
                                                          observation=rollout_data.next_observations)

            # Calculate the loss for the termination policy
            termination_loss = \
                th.mean(termination_prob * (rollout_data.old_option_values - rollout_data.old_next_option_values))
            entropy_loss = th.mean(termination_prob * th.log(termination_prob) +
                                   (1.0 - termination_prob) * th.log(1.0 - termination_prob))
            terminator_loss = termination_loss + terminator_ent_coef * entropy_loss

            # Log data
            loss_dict['terminator_loss'] = terminator_loss.item()
            loss_dict['termination_loss'] = termination_loss.item()
            loss_dict['terminator_entropy_loss'] = entropy_loss.item()

            # Optimize
            terminator.optimizer.zero_grad()
            terminator_loss.backward()
            th.nn.utils.clip_grad_norm_(terminator.parameters(), 0.5)
            terminator.optimizer.step()

            # Calculate Terminator KL-Divergence
            terminator_kl_div = \
                rollout_data.old_termination_prob * \
                (th.log(rollout_data.old_termination_prob) -
                 th.log(termination_prob)) + \
                (1.0 - rollout_data.old_termination_prob) * \
                (th.log(1.0 - rollout_data.old_termination_prob) -
                 th.log(1.0 - termination_prob))
            terminator_kl_div = th.mean(terminator_kl_div).detach().cpu().numpy()
        else:
            terminator_kl_div = None

        return actor_kl_div, terminator_kl_div, loss_dict

    def learn(self,
              total_timesteps: int,
              reward_func: Callable,
              callback: MaybeCallback = None,
              log_interval: int = 1,
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "SequencePPO",
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True
              ) -> "SequencePPO":
        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, reward_func, callback, eval_freq, n_eval_episodes,
            eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())
        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env,
                                                      reward_func=reward_func,
                                                      callback=callback,
                                                      rollout_buffer=self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)
            if not continue_training:
                break

            iteration += 1

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)

            self.train(batch_size=self.batch_size)

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
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
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

        # Initialize the rollout buffer
        self.rollout_buffer = OptionsRolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            n_options=self.num_options,
            gamma=self.gamma,
            device=self.device
        )

        # Create buffers to track total number of learning samples seen for each action and termination policy
        self.num_option_samples = np.zeros(self.num_options, dtype=np.uint32)

        self.start_time = time.time()
        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        self._last_obs = obs_to_box_space(self.env.observation_space, self.env.reset()).squeeze()
        if reset_num_timesteps or self._last_obs is None:
            self._last_dones = np.zeros((self.env.num_envs,), dtype=np.bool)

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        eval_env = self._get_eval_env(eval_env)

        # Configure logger's outputs
        utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_eval_callback(reward_func, callback, eval_env, eval_freq, n_eval_episodes, log_path)

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
    ) -> "SequencePPO":
        data, params, pytorch_variables = load_from_zip_file(path, device=device)

        # Create the model using only the required parameters
        model = cls(
            env=env,
            device=device
        )

        # Update the rest of the parameters of the model
        model.__dict__.update(data['cls'])

        # Add the options
        assert model.num_options == len(model._params), "Inconsistent number of options and parameters!"
        for option_id in range(model.num_options):
            model.add_option(**model._params[option_id])

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
            model.critics[option_id].optimizer.load_state_dict(critic_dict['optimizer'])

            # Load terminator state dicts
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
            "rollout_buffer",
            "default_actions",
            "actors",
            "critics",
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
            "actor_clip_ranges",
            "actor_target_kl_divs",
            "terminator_target_kl_divs"
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
                    optimizer=self.critics[option_id].optimizer.state_dict()
                )
            )

            # Store terminator network and optimizer state dicts
            state_dicts['terminators'].append(
                dict(
                    params=self.terminators[option_id].state_dict(),
                    optimizer=self.terminators[option_id].optimizer.state_dict()
                )
            )

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

        if eval_env is not None:
            eval_env = self._wrap_env(eval_env, self.verbose)
            assert eval_env.num_envs == 1
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
