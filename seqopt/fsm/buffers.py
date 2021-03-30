from collections import deque
from typing import Generator, List, Optional, Union

import gym
import numpy as np
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import get_device
import torch as th


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
