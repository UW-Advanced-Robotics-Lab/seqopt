
"""A Module that uses discretized binning to count visited states."""
from typing import Callable, List, Optional

import torch
from torch import nn
import time


class StateCounter(nn.Module):
    def __init__(self, feature_extractor: Callable, feature_boundaries: List[torch.FloatTensor], device: torch.device):
        """
        Args:
            feature_extractor: The hash function should output N (batch) x K (#features) tensors
            feature_boundaries: List (size = #features) of tensors for boundaries of each feature (variable tensor len)
        """
        super(StateCounter, self).__init__()

        # Set the hash function and the boundaries for each feature
        self.feature_extractor = feature_extractor
        self.feature_boundaries = feature_boundaries

        # Calculate the number of bins required for each feature dimension
        grid_dims = tuple(map(lambda boundaries: len(boundaries) + 1, feature_boundaries))

        # Ensure that the feature boundary tensors are moved to the correct device
        self.device = device
        for i in range(len(feature_boundaries)):
            self.feature_boundaries[i] = self.feature_boundaries[i].to(self.device)

        # Register a buffer to accumulate counts
        self.register_buffer('counts', tensor=torch.ones(grid_dims).long(), persistent=True)

    def _get_features(self, states):
        # Clone the tensor to avoid modifying it directly
        states_ = states.clone().detach()

        # Pass the states through the feature extractor function
        # s = time.time()
        features = self.feature_extractor(states_)
        # s2 = time.time()
        # print(f'elapsed: {s2-s}')

        # Ensure the correct number of features are extracted by the hash function
        assert features.shape[1] == len(self.feature_boundaries), 'The number of features specified are not correct!'

        return features

    def _get_bins(self, features):
        # Calculate the bin indexes for each feature
        bin_idxs = torch.stack([torch.bucketize(features[..., feature_idx], boundaries)
                                for feature_idx, boundaries in enumerate(self.feature_boundaries)],
                               dim=1).long()

        return tuple(bin_idxs.transpose(0, 1))

    def forward(self, states, accumulator='counts'):
        # Pass the features from the states
        features = self._get_features(states)

        # Get the required bins
        idxs = self._get_bins(features)

        # Accumulate counts
        acc = self._buffers.get(accumulator)
        acc.index_put_(idxs, torch.ones(1).long().to(self.device), accumulate=True)

    def get_counts(self, states: Optional[torch.Tensor] = None, accumulator='counts'):
        if states is not None:
            # Get features from states
            features = self._get_features(states)

            # Get bin indices
            idxs = self._get_bins(features)

            return self._buffers.get(accumulator)[idxs]
        else:
            return self._buffers.get(accumulator)
