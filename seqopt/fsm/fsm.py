from typing import Callable, List, Optional, Tuple, Union

import gym
import numpy as np
import torch as th
from stable_baselines3.common.utils import get_device

from .option import Option
from btopt.common.types import OptionPolicyParams, TerminationPolicyParams


class FiniteOptionMachine(object):

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Box,
                 device: Union[str, th.device],
                 use_count_exploration: bool = False
                 ):
        self._active_option = None
        self._finalised = False
        self.action_space = action_space
        self.observation_space = observation_space
        self.device = get_device(device)
        self.use_count_exploration = use_count_exploration
        self.options = []

        # This should ideally be passed in
        self.stop_actions = [th.Tensor([0., 0., 0., 0., -1.]).float().to(device),
                             th.Tensor([0., 0., 0., 0., 1.]).float().to(device),
                             th.Tensor([0., 0., 0., 0., 0.5]).float().to(device)]

    def add_option(self,
                   option_policy_params: OptionPolicyParams = OptionPolicyParams(),
                   termination_policy_params: TerminationPolicyParams = TerminationPolicyParams(),
                   exploration_features_extractor: Optional[Callable] = None,
                   exploration_features_boundaries: Optional[List[th.Tensor]] = None,
                   **kwargs) -> Option:
        assert not self._finalised, "Cannot add more options after calling finalise()!"

        # If count based exploration is enabled, ensure that the user passed in a feature extractor
        # and a list of feature boundaries for each feature
        if self.use_count_exploration:
            if exploration_features_extractor is None or exploration_features_boundaries is None:
                raise ValueError('Count-based exploration is enabled...Ensure both a features '
                                 'extractor and features boundaries are specified for each option!')

        option = Option(observation_space=self.observation_space,
                        action_space=self.action_space,
                        device=self.device,
                        option_policy_params=option_policy_params,
                        termination_policy_params=termination_policy_params,
                        exploration_features_extractor=exploration_features_extractor,
                        exploration_features_boundaries=exploration_features_boundaries,
                        **kwargs)
        self.options.append(option)

        return option

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor,
                                               th.Tensor, th.Tensor, th.Tensor, np.ndarray]:
        # Get action, option-value of active option and the log probability of the action
        action, option_value, action_log_prob = self.options[self.active_option].action_policy.forward(obs)

        # Unmask the action to get the full action (the action_policy by default only returns the actions that
        # it is in-charge of outputting)
        action = self.options[self.active_option].action_policy.unmask_action(action,
                                                                              self.stop_actions[self.active_option])

        # Calculate termination probability of the current option in the current state
        _, termination_prob, _, trainable = self.options[self.active_option].termination_policy.forward(obs)

        # We also require the value of the current state w.r.t the next option in the chain
        next_option = (self.active_option + 1) % self.num_options
        _, next_option_value = self.options[next_option].action_policy.predict(obs)

        # If using count-based exploration, calculate the deliberation margin
        eta = self.options[self.active_option].eta
        lam = self.options[self.active_option].lam
        if self.use_count_exploration:
            deliberation_margin = th.Tensor([0.]).to(self.device)
            # deliberation_margin = 2.0 * eta * (lam / np.sqrt(self.options[self.active_option].get_state_counts(obs)) -
            #                                    (1 - lam) / np.sqrt(self.options[next_option].get_state_counts(obs)))
        else:
            deliberation_margin = th.Tensor([0.]).to(self.device)

        return action, action_log_prob, termination_prob, option_value, next_option_value, deliberation_margin,\
               trainable

    def calculate_intrinsic_reward(self,
                                   obs: th.Tensor,
                                   option: int):
        if self.use_count_exploration:
            next_option = (option + 1) % self.num_options
            lam = self.options[option].lam
            intrinsic_rew = lam / np.sqrt(self.options[option].get_state_counts(obs).cpu().numpy()) - \
                            (1.0 - lam) / np.sqrt(self.options[next_option].get_state_counts(obs).cpu().numpy())
            intrinsic_rew *= self.options[option].eta
        else:
            intrinsic_rew = np.zeros(shape=(obs.shape[0], 1), dtype=np.float32)

        return intrinsic_rew

    def forward_terminations(self, obs:th.Tensor):
        # Fetch option-values of current and next option in chain
        next_option = (self.active_option + 1) % self.num_options
        _, option_value = self.options[self.active_option].action_policy.predict(obs)
        _, next_option_value = self.options[next_option].action_policy.predict(obs)

        # Calculate termination probability of current option in current state (and whether we should terminate)
        terminate, termination_prob, _, _ = self.options[self._active_option].termination_policy.forward(obs)
        if terminate:
            # Reset any stateful components for the terminated option
            self.options[self.active_option].reset()
            self.active_option = next_option

        return termination_prob, option_value, next_option_value

    # def forward(self,
    #             obs: th.Tensor
    #             ) -> Tuple[int, int, th.Tensor, th.Tensor, th.Tensor, th.Tensor,
    #                        th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    #     # Store the initial option (this is the option that was executed in the last timestep)
    #     initial_option = self._active_option
    #
    #     # Get the state counts for the initial option, and the option after/next to the initial option
    #     # The state count refers to the number of times the action policy of a particular option
    #     # has seen that state (states are discretized)
    #     if self.use_count_exploration:
    #         initial_option_state_counts = self.options[initial_option].get_state_counts(obs)
    #         next_option_state_counts = self.options[(initial_option + 1) % self.num_options].get_state_counts(obs)
    #     else:
    #         # Just send back zeros if not using count based exploration
    #         initial_option_state_counts = th.Tensor([0]).to(self.device)
    #         next_option_state_counts = th.Tensor([0]).to(self.device)
    #
    #     # Pass the observation to the termination policy of the current option
    #     # We get back a binary signal determining if we should terminate that option, along with
    #     # the probability of termination and the log of that probability
    #     terminate, initial_beta_prob, log_beta_prob =\
    #         self.options[self._active_option].termination_policy.forward(obs)
    #
    #     # If the option is terminated, generate the action from the action policy of the next option. Otherwise,
    #     # generate the action from the action policy of the same option
    #     if terminate:
    #         self._active_option = (self._active_option + 1) % self.num_options
    #
    #         # If the option was terminated, we re-calculate the beta probability to corresponding to
    #         # the termination probability of the activated option for the given state
    #         beta_prob, _, _ = self.evaluate_terminations(obs, option_id=self._active_option)
    #
    #         # Reset any stateful components for the terminated option
    #         self.options[initial_option].reset()
    #     else:
    #         beta_prob = initial_beta_prob.clone()
    #
    #     # Despite not executing the initial option, we calculate the value of that option
    #     # for the given state anyways
    #     _, initial_opt_value =\
    #         self.options[initial_option].action_policy.predict(obs)
    #
    #     action, value, log_prob =\
    #         self.options[self._active_option].action_policy.forward(obs)
    #
    #     # Unmask the action to get the full action (the action_policy by default only returns the actions that
    #     # it is in-charge of outputting)
    #     action = self.options[self._active_option].action_policy.unmask_action(action, self.stop_actions[self._active_option])
    #
    #     # We also evaluate the value of the next option's policy at the current state
    #     # This will be used in termination policy updates
    #     _, next_opt_value =\
    #         self.options[(self._active_option + 1) % self.num_options].action_policy.predict(obs)
    #
    #     return initial_option, self._active_option, initial_beta_prob, beta_prob,\
    #         action, \
    #         initial_opt_value, value, next_opt_value, log_prob,\
    #         initial_option_state_counts, next_option_state_counts

    def predict(self,
                obs: np.ndarray,
                active_option: int,
                deterministic_action: bool = True,
                deterministic_termination: bool = True) -> Tuple[int, th.Tensor, th.Tensor, th.Tensor]:
        """
        Works similar to the forward() but bypassing any gradient computations and observation normalization updates.
        Furthermore, it doesn't change the internal state of the finite option machine.
        """
        # Disable gradient computation
        with th.no_grad():
            # Convert observation to FloatTensor
            obs = th.as_tensor(obs.astype(np.float32)).to(self.device)

            # Generate an action based on the active option
            action, value = self.options[active_option].action_policy.predict(obs, deterministic=deterministic_action)

            # Unmask the action to get the full action (the action_policy by default only returns the actions that
            # it is in-charge of outputting)
            action = self.options[active_option].action_policy.unmask_action(action, self.stop_actions[active_option])

            # Predict if we should terminate the active option for the next step
            terminate, beta = \
                self.options[active_option].termination_policy.predict(obs, deterministic=deterministic_termination)

            next_active_option = (active_option + 1) % self.num_options if terminate else active_option
        return next_active_option, beta, action, value

    def reset_noise(self, n_envs: int = 1, option_id: Optional[int] = None):
        if option_id is None:
            for option_id in range(self.num_options):
                if self.options[option_id].action_policy.params.use_sde:
                    self.options[option_id].action_policy.policy.reset_noise(n_envs)
        else:
            if self.options[option_id].action_policy.params.use_sde:
                self.options[option_id].action_policy.policy.reset_noise(n_envs)

    def evaluate_actions(self,
                         obs: th.Tensor,
                         acts: th.Tensor,
                         option_id: int) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluates value, log probability and entropy for the given observation, action and option
        """
        values, log_probs, entropy = self.options[option_id].action_policy.evaluate_actions(obs, acts)

        return values, log_probs, entropy

    def evaluate_terminations(self, obs: th.Tensor, option_id: int) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluates termination probability (and its log and entropy) of a specified option for a given observation
        """
        beta_probs, log_beta_probs, entropy = self.options[option_id].termination_policy.evaluate_terminations(obs)

        return beta_probs, log_beta_probs, entropy

    def finalise(self) -> None:
        assert self.num_options > 0, "No options created! Please create one or more options!"
        self._active_option = 0
        self._finalised = True

    def get_action_policy_params(self, option_id: int) -> OptionPolicyParams:
        return self.options[option_id].action_policy.params

    def get_termination_policy_params(self, option_id: int) -> TerminationPolicyParams:
        return self.options[option_id].termination_policy.params

    def optimize_actions(self,
                         option_id: int,
                         value_loss: th.Tensor,
                         policy_loss: Optional[th.Tensor] = None) -> None:
        self.options[option_id].action_policy.optimize(value_loss=value_loss,
                                                       policy_loss=policy_loss)

    def optimize_terminations(self,
                              option_id: int,
                              loss: th.Tensor) -> None:
        self.options[option_id].termination_policy.optimize(loss)

    # Getters/Setters for the options

    @property
    def active_option(self) -> int:
        return self._active_option

    @active_option.setter
    def active_option(self, option_id: int) -> None:
        assert option_id < self.num_options, f"option_id needs to be less than {self.num_options}!"
        self._active_option = option_id

    @property
    def num_options(self) -> int:
        return len(self.options)

    def reset(self) -> None:
        self._active_option = 0

        # Reset any stateful components of all options
        for option in self.options:
            option.reset()

    # TRAINING/TESTING settings

    def deterministic(self, actions: bool = True, terminations: bool = True) -> None:
        for option in self.options:
            option.deterministic(actions=actions, terminations=terminations)

    def train(self, enable: bool = True) -> None:
        for option in self.options:
            option.train(enable=enable)
