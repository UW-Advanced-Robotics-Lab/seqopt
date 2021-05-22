from collections import deque
from typing import Generator, List, Optional, Union

import gym
import numpy as np
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import get_device
import torch as th

from seqopt.common.types import RolloutBufferSamples


class OptionsRolloutBuffer:
    """
    Based on:
    https://github.com/DLR-RM/stable-baselines3/blob/723b341c61d168e1460399592d5cebd4c6ef3cc8/stable_baselines3/common/buffers.py#L259

    This class collects experiences seen by on-policy rollouts over a set of options, additionally classifying the
    experiences into separate buffers in order to allow updating the different policies existing in our options
    framework with relevant experiences.

    A major distinction from the RolloutBuffer from Stable Baselines 3 is that with the options framework, we need
    to keep track of different experiences with regards to different options. Along the same line, the experiences
    used for updating the termination condition of an option and the actual policy of that option can and will
    generally consist of different experiences (the latter will always be a subset of the former).
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        n_options: int,
        gamma: float = 0.99,
        device: Union[th.device, str] = "cpu",
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_options = n_options
        self.gamma = gamma
        self.device = get_device(device)

        self.pos = 0
        self.full = False

        # Buffers for sample data
        self.observations, self.next_observations, self.actions = None, None, None
        self.rewards, self.dones = None, None
        self.action_log_probs, self.termination_probs = None, None
        self.option_values, self.next_option_values = None, None
        self.advantages, self.returns = None, None
        self.active_options = None

        self.option_idxs = None
        self.generator_ready = False
        self.reset()

    def reset(self):
        observation_dim = gym.spaces.flatdim(self.observation_space)
        action_dim = gym.spaces.flatdim(self.action_space)

        self.observations = np.zeros((self.buffer_size, observation_dim), dtype=self.observation_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, observation_dim), dtype=self.observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, action_dim), dtype=self.action_space.dtype)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.option_values = np.zeros(self.buffer_size, dtype=np.float32)
        self.action_log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.termination_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.option_values = np.zeros(self.buffer_size, dtype=np.float32)
        self.next_option_values = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.active_options = np.zeros(self.buffer_size, dtype=np.int)

        # Keep track of which indexes correspond to which options
        self.option_idxs = [[] for _ in range(self.n_options)]

        self.pos = 0
        self.full = False

    def add(self,
            option_id: int,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            action_log_prob: th.Tensor,
            termination_prob: th.Tensor,
            option_value: th.Tensor,
            next_option_value: th.Tensor):
        # Add the new experience
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.action_log_probs[self.pos] = action_log_prob.clone().cpu().numpy().item()
        self.termination_probs[self.pos] = termination_prob.clone().cpu().numpy().item()
        self.option_values[self.pos] = option_value.clone().cpu().numpy().item()
        self.next_option_values[self.pos] = next_option_value.clone().cpu().numpy().item()
        self.active_options[self.pos] = option_id

        # Each sample provides us with exactly 1 learning sample for updating the action policy of the given option
        # and 1 learning sample for updating the termination policy of the given option
        # For the buffers corresponding to the each option, add the current step as a reference for important values
        self.option_idxs[option_id].append(self.pos)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, first_option_value: th.Tensor) -> None:
        """
        Compute targets for option values, Q(s,o) and Q(s,a,o), based on the rewards seen over the collected
        trajectories
        """
        # Unlike the original PPO algorithm, we do not use Generalized Advantage Estimation (GAE) for calculating
        # advantages and/or returns. Instead, we will base the returns on 1-step TD targets and the advantages will
        # be computed over any consecutive steps taken using that option
        discounted_target = 0.0
        for step in reversed(range(self.buffer_size)):
            # For the last sample, we don't know (or care) if the option terminated or not, so we just assume it did
            # This ensures that we bootstrap on the next options values in calculating the advantages/returns
            if step == self.buffer_size - 1:
                next_option_changed = True
            else:
                next_option_changed = (self.active_options[step] != self.active_options[step + 1])

            # Calculate: (1 - beta(s')) * Q(s',o) + beta(s') * Q(s',o')
            next_state_value = (1.0 - self.termination_probs[step]) * self.option_values[step] + \
                               self.termination_probs[step] * self.next_option_values[step]

            # Determine if the episode ends here
            # NOTE: We don't use the 'done' flags from the environment, since for the environments that we are
            # concerned with, there are actually no 'absorbing' states
            next_non_terminal = 1.0

            # Update the discounted target for Q(s,a,o)
            # This is a variable step target that keeps aggregating over consecutive steps
            # experienced by any given option
            if next_option_changed:
                discounted_target = next_state_value
            discounted_target = self.rewards[step] + next_non_terminal * self.gamma * discounted_target

            # Check the value of Q(s,o) [NOTE: For each sample, we store Q(s',o) and Q(s',o') which corresponds
            # to the next observation and not the original observation under which the action was taken]
            if step == 0:
                val = first_option_value.clone().cpu().numpy().flatten()
            else:
                if self.active_options[step - 1] == self.active_options[step]:
                    val = self.option_values[step - 1]
                else:
                    val = self.next_option_values[step - 1]

            # Now the advantage estimate for action policies, A(s,a,o), is given by Q(s,a,o) - Q(s,o)
            self.advantages[step] = discounted_target - val

            # For training the critic, we directly use a 1-step return
            self.returns[step] = self.rewards[step] + next_non_terminal * self.gamma * next_state_value

    def get_num_samples(self, option_id: int):
        return len(self.option_idxs[option_id])

    def get(self,
            option_id: int,
            max_batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""

        # Get the no. of samples available for updating the policy of this option
        num_samples = self.get_num_samples(option_id)

        indices = np.random.permutation(num_samples)

        # Return everything, don't create minibatches if no batch size given or the batch size specified
        # is more than the number of samples available
        if max_batch_size is None or max_batch_size > num_samples:
            batch_size = num_samples
        else:
            batch_size = max_batch_size

        start_idx = 0
        while start_idx < num_samples:
            # Since we store the relevant indexes of the primary buffer in our option specific
            # list, we extract the global indexes corresponding to the random indices generated
            global_inds = [self.option_idxs[option_id][i]
                           for i in indices[start_idx: start_idx + batch_size]]
            yield self._get_samples(global_inds)
            start_idx += batch_size

    # Define function to form the RolloutBufferPolicySamples object
    def _get_samples(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.next_observations[batch_inds],
            self.actions[batch_inds],
            self.action_log_probs[batch_inds],
            self.termination_probs[batch_inds],
            self.option_values[batch_inds],
            self.next_option_values[batch_inds],
            self.advantages[batch_inds],
            self.returns[batch_inds]
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)


class OptionsReplayBuffer:
    def __init__(self,
                 buffer_size: int,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Box,
                 n_options: int,
                 device: Union[th.device, str] = "cpu"):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_options = n_options
        self.device = get_device(device)

        self.pos = 0
        self.full = False

        observation_dim = gym.spaces.flatdim(self.observation_space)
        action_dim = gym.spaces.flatdim(self.action_space)

        self.observations = np.zeros((self.buffer_size, observation_dim), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, observation_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.options = np.zeros(self.buffer_size, dtype=np.int)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.option_sample_idxs = [deque() for _ in range(self.n_options)]

    def add(self,
            option: int,
            obs: np.ndarray,
            next_obs: np.ndarray,
            act: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray) -> None:

        if self.full:
            # Pop index from queue associated with the current option
            self.option_sample_idxs[self.options[self.pos]].pop()

        self.options[self.pos] = option
        self.option_sample_idxs[option].appendleft(self.pos)
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(act).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, option_id: int, batch_size: int) -> Optional[ReplayBufferSamples]:
        num_samples = len(self.option_sample_idxs[option_id])

        if num_samples == 0:
            return None
        else:
            batch_inds = [self.option_sample_idxs[option_id][pos]
                          for pos in np.random.randint(0, num_samples, size=batch_size)]
            return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds) -> ReplayBufferSamples:
        data = (
            self.observations[batch_inds, :],
            self.actions[batch_inds, :],
            self.next_observations[batch_inds, :],
            self.dones[batch_inds],
            self.rewards[batch_inds],
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def merge_samples(samples_1: ReplayBufferSamples, samples_2: ReplayBufferSamples):
        return ReplayBufferSamples(*tuple([th.cat([v1, v2]) for v1, v2 in zip(samples_1, samples_2)]))

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)
