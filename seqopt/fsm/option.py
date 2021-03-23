from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Optional, Tuple, Union

import gym
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.utils import get_schedule_fn

from btopt.common.policies import OptionTerminationPolicy, ValueCritic
from btopt.common.state_counter import StateCounter
from btopt.common.types import OptionPolicyParams, TerminationPolicyParams
from btopt.utils.state_utils import gym_subspace_box


class GenericOptionPolicy(ABC):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 device: th.device,
                 observation_mask: Optional[Union[Iterable[Union[int, bool]], slice]] = None,
                 normalize_observations: bool = True,
                 normalization_epsilon: float = 1e-8):
        self.observation_mask = observation_mask
        self.device = device

        # Store the full observation space and the subspace that is actually required by the node
        self.full_observation_space = observation_space
        self.observation_space = gym_subspace_box(observation_space, observation_mask)

        # Create a RunningMeanStd object to track observation statistics for the relevant part of the observation space
        self.normalize_observations = normalize_observations
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape) if normalize_observations else None
        self.normalization_epsilon = normalization_epsilon

        # Mode
        self.training = True
        self.deterministic_outputs = False

    def train(self, enable: bool = True) -> None:
        self.training = enable

    def deterministic(self, enable: bool = True) -> None:
        self.deterministic_outputs = enable

    def normalize_obs(self, obs: th.Tensor, update: bool = False) -> th.Tensor:
        assert self.obs_rms is not None, "Cannot call normalize_obs() without enabling observation normalization!"

        # TODO(somesh): Check if gradients are affected in some way due to the conversion of the observation
        #               to a numpy array for normalization, and subsequent conversion back to a tensor
        # Avoid modifying the observation directly, make a copy of it first
        obs_ = obs.clone().cpu().numpy()

        # Normalize the observation using the statistics collected thus far
        normalized_obs = (obs_ - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.normalization_epsilon)

        # We ensure that we only update the observation statistics with the current observation after we have
        # used the previous statistics to compute the normalized observation
        if update:
            self.obs_rms.update(obs_)

        # Cast to float32 (numpy's default type is float64, while Pytorch uses float32 tensors)
        return th.as_tensor(normalized_obs.astype(np.float32)).to(obs.device)

    @abstractmethod
    def forward(self,
                obs: th.Tensor):
        # Apply the observation mask to the tensor to obtain the relevant observation values
        obs = self.mask_tensor(obs, self.observation_mask)

        # Normalize the observation and update the statistics if in training mode
        if self.normalize_observations:
            obs = self.normalize_obs(obs, update=self.training)

        return obs

    @abstractmethod
    def predict(self, obs: th.Tensor):
        # Apply the observation mask to the tensor to obtain the relevant observation values
        obs = self.mask_tensor(obs, self.observation_mask)

        # Normalize the observation and update the statistics if in training mode
        if self.normalize_observations:
            obs = self.normalize_obs(obs, update=False)

        return obs

    @abstractmethod
    def optimize(self, loss: th.Tensor) -> None:
        raise NotImplementedError('This behavior is not instantiable! Please use a subclass of this behavior!')

    @staticmethod
    def mask_tensor(tensor: th.Tensor,
                    mask: Optional[Union[Iterable[Union[int, bool]], slice]] = None):
        if mask is None:
            return tensor
        else:
            # We apply the mask to the last dimension of the tensor
            return tensor[..., mask]

    @staticmethod
    def unmask_tensor(tensor: th.Tensor,
                      default_full_tensor: th.Tensor,
                      mask: Optional[Union[Iterable[Union[int, bool]], slice]] = None,
                      ):
        if mask is None:
            return tensor
        else:
            output_tensor = default_full_tensor.clone()
            output_tensor[..., mask] = tensor
            return output_tensor


class ActionPolicy(GenericOptionPolicy):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Box,
                 device: th.device,
                 option_policy_params: OptionPolicyParams = OptionPolicyParams(),
                 exploration_features_extractor: Optional[Callable] = None,
                 exploration_features_boundaries: Optional[List[th.FloatTensor]] = None,
                 normalization_epsilon: float = 1e-8):
        super(ActionPolicy, self).__init__(
            observation_space=observation_space,
            device=device,
            observation_mask=option_policy_params.observation_mask,
            normalize_observations=option_policy_params.normalize_observations,
            normalization_epsilon=normalization_epsilon
        )

        # Store the full action space and the subspace that is actually required by the node
        self.full_action_space = action_space

        # There is the possibility that we only want the default action to be executed every time
        # the option is invoked. This corresponds to the case where the action mask is an empty
        # numpy array (note, this is different from the case where the action mask is None, whereby
        # the entire action space is used). In this case, we set the action space to None
        if option_policy_params.action_mask is not None and option_policy_params.action_mask.size == 0:
            self.action_space = None
        else:
            self.action_space = gym_subspace_box(action_space, option_policy_params.action_mask)
        self.params = option_policy_params

        # If both the exploration features extractor and feature boundaries are specified
        # create a state counter to keep track of state visit statistics
        if exploration_features_extractor is not None and exploration_features_boundaries is not None:
            self.state_counter = StateCounter(feature_extractor=exploration_features_extractor,
                                              feature_boundaries=exploration_features_boundaries,
                                              device=self.device).to(self.device)
        else:
            self.state_counter = None

        # Check if we need a special output activation function for the value function
        self.vf_activation_fn = self.params.vf_activation_fn
        self.vf_activation_fn_kwargs = self.params.vf_activation_fn_kwargs

        # If we do not have an action space, we do not require 2 out of the 3 neural networks we usually
        # create i.e. The Actor policy of the Actor-Critic is not required since we have no actions
        # to output, and we don't need to estimate the action-value since we have no actions. Hence,
        # we only need a single neural network that captures the value of the state for this option.
        # This is represented by the Critic of the Actor-Critic
        if self.action_space is not None:
            self.policy = ActorCriticPolicy(observation_space=self.observation_space,
                                            action_space=self.action_space,
                                            lr_schedule=get_schedule_fn(self.params.lr_schedule),
                                            net_arch=self.params.net_arch,
                                            activation_fn=self.params.activation_fn,
                                            use_sde=option_policy_params.use_sde,
                                            log_std_init=option_policy_params.log_std_init,
                                            full_std=option_policy_params.full_std,
                                            sde_net_arch=option_policy_params.sde_net_arch,
                                            squash_output=option_policy_params.squash_output).to(self.device)
        else:
            # Obtain the network architecture of the value network from the policy params if specified
            vf_net_arch = []
            if self.params.net_arch is not None:
                vf_only_layers = []
                for i, layer in enumerate(self.params.net_arch):
                    # Append hidden layers from the 'shared' network
                    if isinstance(layer, int):
                        vf_net_arch.append(layer)
                    elif isinstance(layer, dict):
                        if 'vf' in layer:
                            vf_only_layers.extend(layer['vf'])
                    else:
                        raise ValueError(f'Unexpected net arch format: {self.params.net_arch}')

                vf_net_arch.extend(vf_only_layers)

            if len(vf_net_arch) == 0:
                vf_net_arch = None

            self.value_critic = ValueCritic(observation_space=self.observation_space,
                                            lr_schedule=get_schedule_fn(self.params.lr_schedule),
                                            net_arch=vf_net_arch,
                                            activation_fn=self.params.activation_fn,
                                            features_extractor_class=FlattenExtractor).to(self.device)

    def evaluate_actions(self, obs: th.Tensor, acts: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Get masked + normalized observation (statistics for normalization are not updated)
        obs = super().predict(obs)

        if self.action_space is not None:
            # Get masked action
            acts = self.mask_tensor(acts, self.params.action_mask)

            values, log_probs, entropy = self.policy.evaluate_actions(obs, acts)
        else:
            values = self.value_critic.forward(obs)
            log_probs, entropy = th.Tensor([0.]).to(self.device), th.Tensor([0.]).to(self.device)

        if self.vf_activation_fn is not None:
            values = self.vf_activation_fn(values, **self.vf_activation_fn_kwargs)

        return values, log_probs, entropy

    def reset(self):
        pass

    def get_state_counts(self, obs: th.Tensor):
        assert self.state_counter is not None, 'Invalid operation! Can only call get_state_counts() ' \
                                               'if count-based exploration is enabled!'
        return self.state_counter.get_counts(obs)

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Prior to masking and/or normalizing the observation, we pass the raw observation
        # through the state counter if count-based exploration is enabled
        if self.state_counter is not None:
            self.state_counter(obs)

        # Get the masked + normalized observation
        obs = super().forward(obs)

        if self.action_space is not None:
            actions, values, log_prob =\
                self.policy.forward(obs, deterministic=self.deterministic_outputs)
        else:
            actions = th.Tensor([]).to(self.device)
            values = self.value_critic.forward(obs)
            log_prob = th.Tensor([0.]).to(self.device)

        if self.vf_activation_fn is not None:
            values = self.vf_activation_fn(values, **self.vf_activation_fn_kwargs)

        return actions, values, log_prob

    def predict(self, obs: th.Tensor, deterministic: bool = True) -> Tuple[th.Tensor, th.Tensor]:
        # Get masked + normalized observation (statistics for normalization are not updated)
        obs = super().predict(obs)

        if self.action_space is not None:
            actions, values, _ = self.policy.forward(obs, deterministic=deterministic)
        else:
            actions, values = th.Tensor([]).to(self.device), self.value_critic.forward(obs)

        if self.vf_activation_fn is not None:
            values = self.vf_activation_fn(values, **self.vf_activation_fn_kwargs)

        return actions, values

    def optimize(self,
                 value_loss: Optional[th.Tensor] = None,
                 policy_loss: Optional[th.Tensor] = None,
                 optimizer: Optional[th.optim.Optimizer] = None) -> None:
        if value_loss is None and policy_loss is None:
            return

        if self.action_space is not None:
            optimizer = optimizer if optimizer is not None else self.policy.optimizer

            # Add the policy gradient and value losses since the ActorCritic policy
            # encapsulates both the actor and value critic
            losses = [policy_loss, value_loss]
            valid_losses = list(filter(lambda loss: loss is not None, losses))
            loss = th.stack(valid_losses, dim=0).sum(dim=0).sum(dim=0)

            # Zero the gradients for the actor-critic optimizer
            optimizer.zero_grad()

            # Calculate the gradients for the loss
            loss.backward()

            # Clip the gradients if required
            th.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                        self.params.max_grad_norm)

            # Update the parameters
            optimizer.step()
        # If there is no action space, we can only have a value loss
        elif value_loss is not None:
            optimizer = optimizer if optimizer is not None else self.value_critic.optimizer
            # In the case that we do not generate any actions, we only optimize the value critic
            optimizer.zero_grad()
            value_loss.backward()
            th.nn.utils.clip_grad_norm_(self.value_critic.parameters(),
                                        self.params.max_grad_norm)
            optimizer.step()

    def unmask_action(self, action: th.Tensor, default_full_action: th.Tensor):
        return self.unmask_tensor(action, default_full_action, self.params.action_mask)

    # Pre-training (Behaviour Cloning) Helper Methods
    @property
    def trainable_parameters(self):
        if self.action_space is not None:
            return self.policy.parameters()
        else:
            return self.value_critic.parameters()

    def pretrain(self,
                 observations: np.ndarray,
                 actions: np.ndarray,
                 value_targets: np.ndarray,
                 train_value: bool = True,
                 train_policy: bool = True,
                 optimizer: Optional[th.optim.Optimizer] = None):
        value_loss, policy_loss = self.evaluate_loss(observations,
                                                     actions,
                                                     value_targets,
                                                     evaluate_value_loss=train_value,
                                                     evaluate_policy_loss=train_policy)
        self.optimize(value_loss=value_loss,
                      policy_loss=policy_loss,
                      optimizer=optimizer)

    def evaluate_loss(self,
                      observations: np.ndarray,
                      actions: np.ndarray,
                      value_targets: np.ndarray,
                      evaluate_value_loss: bool = True,
                      evaluate_policy_loss: bool = True,
                      ):
        observations = th.from_numpy(observations).float().to(self.device)

        # Predict actions and values using the neural networks
        pred_actions, pred_values = self.predict(observations, deterministic=True)

        # Calculate Value Loss
        if evaluate_value_loss:
            # Add batch dimension to value targets
            value_targets = th.from_numpy(value_targets).float().to(self.device)
            value_targets = th.unsqueeze(value_targets, -1)
            value_loss = th.nn.functional.mse_loss(pred_values, value_targets)
        else:
            value_loss = None

        # Calculate Policy Loss
        if self.action_space is not None and evaluate_policy_loss:
            if self.params.action_mask is not None:
                actions = actions[..., self.params.action_mask]

            # Convert the arrays to tensors
            actions_ = th.from_numpy(actions).float().to(self.device)

            # Perform supervised learning (using MSE loss) on this data
            policy_loss = th.nn.functional.mse_loss(pred_actions, actions_)
        else:
            policy_loss = None

        return value_loss, policy_loss


class TerminationPolicy(GenericOptionPolicy):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 device: th.device,
                 termination_policy_params: TerminationPolicyParams = TerminationPolicyParams(),
                 normalization_epsilon: float = 1e-8):
        super(TerminationPolicy, self).__init__(
            observation_space=observation_space,
            device=device,
            observation_mask=termination_policy_params.observation_mask,
            normalize_observations=termination_policy_params.normalize_observations,
            normalization_epsilon=normalization_epsilon
        )

        self.params = termination_policy_params
        # 'Expert' defined termination condition
        self.termination_condition = termination_policy_params.termination_condition

        self.policy = OptionTerminationPolicy(
            observation_space=self.observation_space,
            lr_schedule=get_schedule_fn(self.params.lr_schedule),
            net_arch=self.params.net_arch,
            activation_fn=self.params.activation_fn,
            use_boltzmann=self.params.use_boltzmann
        ).to(self.device)

    def evaluate_terminations(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        full_obs = obs.clone()
        # Get masked + normalized observation (statistics for normalization are not updated)
        obs = super().predict(obs)

        if self.termination_condition:
            non_expert_idxs, beta_probs = self.termination_condition(full_obs)
            beta_probs = th.clip(beta_probs, 1e-6, 1.0 - 1e-6)
            log_beta_probs = th.log(beta_probs)
            entropy = -(beta_probs * log_beta_probs + (1 - beta_probs) * th.log(1 - beta_probs))

            if len(non_expert_idxs) > 0:
                beta_probs[non_expert_idxs], log_beta_probs[non_expert_idxs], entropy[non_expert_idxs] = \
                    self.policy.evaluate(obs[non_expert_idxs])
        else:
            beta_probs, log_beta_probs, entropy = self.policy.evaluate(obs)

        return beta_probs, log_beta_probs, entropy

    def reset(self):
        pass

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, np.ndarray]:
        full_obs = obs.clone()
        # Get the masked + normalized observation
        obs = super().forward(obs)

        # Check if we should obtain the termination probabilities from the expert defined condition(s)
        if self.termination_condition:
            trainable_terminations = np.zeros(obs.shape[0], dtype=np.bool)
            non_expert_idxs, beta = self.termination_condition(full_obs)
            beta = th.clip(beta, 1e-6, 1.0 - 1e-6)
            log_prob = th.log(beta)
            if self.deterministic_outputs:
                terminate = th.gt(beta, 0.5)
            else:
                terminate = th.gt(beta, np.random.uniform())

            if len(non_expert_idxs) > 0:
                trainable_terminations[non_expert_idxs] = True
                terminate[non_expert_idxs], beta[non_expert_idxs], log_prob[non_expert_idxs] = \
                    self.policy.forward(obs[non_expert_idxs], deterministic=self.deterministic_outputs)
        else:
            trainable_terminations = np.ones(obs.shape[0], dtype=np.bool)
            terminate, beta, log_prob = self.policy.forward(obs, deterministic=self.deterministic_outputs)

        return terminate, beta, log_prob, trainable_terminations

    def predict(self, obs: th.Tensor, deterministic: bool = True) -> Tuple[th.Tensor, th.Tensor]:
        full_obs = obs.clone()
        # Get masked + normalized observation (statistics for normalization are not updated)
        obs = super().predict(obs)

        if self.termination_condition:
            non_expert_idxs, beta = self.termination_condition(full_obs)
            beta = th.clip(beta, 1e-6, 1.0 - 1e-6)
            if deterministic:
                terminate = th.gt(beta, 0.5)
            else:
                terminate = th.gt(beta, np.random.uniform())

            if len(non_expert_idxs) > 0:
                terminate[non_expert_idxs], beta[non_expert_idxs], _ = \
                    self.policy.forward(obs[non_expert_idxs], deterministic=deterministic)
        else:
            terminate, beta, _ = self.policy.forward(obs, deterministic=deterministic)

        return terminate, beta

    def optimize(self, loss: th.Tensor, optimizer: Optional[th.optim.Optimizer] = None) -> None:
        optimizer = optimizer if optimizer is not None else self.policy.optimizer

        # Zero the gradients for the optimizer
        optimizer.zero_grad()

        # Calculate the gradients for the loss
        loss.backward()

        # Clip the gradients if required
        th.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                    self.params.max_grad_norm)

        # Update the parameters
        optimizer.step()

    # Pre-training (Behaviour Cloning) Helper Methods
    @property
    def trainable_parameters(self):
        return self.policy.parameters()

    def pretrain(self, observations: np.ndarray,
                 terminations: np.ndarray,
                 weights: Optional[np.ndarray] = None,
                 optimizer: Optional[th.optim.Optimizer] = None):
        loss = self.evaluate_loss(observations, terminations, weights)
        self.optimize(loss, optimizer)

    def evaluate_loss(self, observations: np.ndarray, terminations: np.ndarray, weights: Optional[np.ndarray] = None):
        # First, extract only the observations that are used by the termination policy
        observations_ = observations.copy()
        if self.observation_mask is not None:
            observations_ = observations[..., self.observation_mask]

        # Convert the arrays to tensors
        observations_ = th.from_numpy(observations_).float().to(self.device)
        terminations_ = th.from_numpy(terminations).to(self.device)

        # Perform supervised learning (using MSE loss) on this data
        _, beta_prob, _ = self.policy.forward(observations_, deterministic=True)
        if weights is None:
            # Regular mse loss
            loss = th.nn.functional.mse_loss(beta_prob, terminations_.float())
        else:
            # Weighted mse loss
            weights_ = th.from_numpy(weights).to(self.device)
            loss = th.sum(weights_ * (beta_prob - terminations_.float()) ** 2)

        return loss

class Option(object):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Box,
                 device: th.device,
                 option_policy_params: OptionPolicyParams = OptionPolicyParams(),
                 termination_policy_params: TerminationPolicyParams = TerminationPolicyParams(),
                 exploration_features_extractor: Optional[Callable] = None,
                 exploration_features_boundaries: Optional[List[th.FloatTensor]] = None,
                 explore_eta: float = 1.0,
                 explore_lambda: float = 0.6,
                 normalization_eps: float = 1e-8):
        assert 0 <= explore_lambda <= 1, "Lambda for count-based explorations must be between 0 and 1 (inclusive)!"
        self._explore_eta = explore_eta
        self._explore_lambda = explore_lambda

        self._termination_policy = TerminationPolicy(observation_space=observation_space,
                                                     device=device,
                                                     termination_policy_params=termination_policy_params,
                                                     normalization_epsilon=normalization_eps)
        self._action_policy = ActionPolicy(observation_space=observation_space,
                                           action_space=action_space,
                                           device=device,
                                           option_policy_params=option_policy_params,
                                           exploration_features_extractor=exploration_features_extractor,
                                           exploration_features_boundaries=exploration_features_boundaries,
                                           normalization_epsilon=normalization_eps)

    @property
    def action_policy(self) -> ActionPolicy:
        return self._action_policy

    @property
    def eta(self) -> float:
        return self._explore_eta

    @property
    def lam(self) -> float:
        return self._explore_lambda

    @property
    def termination_policy(self) -> TerminationPolicy:
        return self._termination_policy

    def deterministic(self, actions: bool = True, terminations: bool = True) -> None:
        self.action_policy.deterministic(actions)
        self.termination_policy.deterministic(terminations)

    def get_state_counts(self, obs):
        return self.action_policy.get_state_counts(obs)

    def train(self, enable: bool = True) -> None:
        self.action_policy.train(enable)
        self.termination_policy.train(enable)

    def reset(self):
        # Call resets on the termination and action policies
        self.action_policy.reset()
        self.termination_policy.reset()
