from typing import Iterable, Optional, Type, Union

from gym import spaces
import numpy as np


def gym_subspace_box(gym_space: Union[spaces.Box, spaces.Dict],
                     idxs: Optional[Union[Iterable[Union[int, bool]], slice]] = None):
    if isinstance(gym_space, spaces.Dict):
        gym_space = spaces.utils.flatten_space(gym_space)
        assert isinstance(gym_space, spaces.Box), "Spaces of type spaces.Dict cannot be transformed to spaces.Box"

    # We only deal with one dimensional Box spaces
    assert len(gym_space.shape) == 1, 'Only 1-dimensional gym.spaces.Box spaces are supported!'

    if idxs is None:
        return gym_space
    else:
        # Create a new gym space with appropriate size and low/high limits
        # based on the original gym space passed in, and the relevant indexes of that space
        gym_subspace = spaces.Box(low=gym_space.low[idxs],
                                  high=gym_space.high[idxs],
                                  shape=(gym_space.low[idxs].shape[0],))

        return gym_subspace


def obs_to_box_space(obs_space: Union[spaces.Box, spaces.Dict],
                     obs: Type[Union[spaces.Box, spaces.Dict]],
                     use_batch_dim=True,
                     ordered_keys=None):
    if isinstance(obs_space, spaces.Box):
        return obs
    elif isinstance(obs_space, spaces.Dict):
        obs_dims = 2 if use_batch_dim else 1
        if ordered_keys is None:
            new_obs = np.hstack([v if len(v.shape) == obs_dims else np.expand_dims(v, -1) for v in obs.values()])
        else:
            new_obs = np.hstack([obs[k] if len(obs[k].shape) == obs_dims else np.expand_dims(obs[k], -1)
                                 for k in ordered_keys])

        return new_obs
    else:
        assert False, f"Unsupported Observation Space: {type(obs_space)}"
