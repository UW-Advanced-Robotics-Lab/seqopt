from numbers import Number
from typing import Any, Dict, Iterable, Optional, Union

from gym import spaces
import numpy as np
from stable_baselines3.common.vec_env import VecEnv


# def reduce_state(full_state: Dict[str, Union[Number, Iterable[Number]]],
#                  observation_dict: Optional[Dict[str, Optional[slice, Iterable]]] = None) -> Dict[str, Any]:
#     """
#     Extracts required states from a (single-level i.e. not nested) dictionary containing all states
#
#     Args:
#         full_state(dict): (Single-level) Dictionary containing keys (strings) denoting an observation and value(s)
#                           associated with each observation
#         observation_dict(dict, None): (Single-level) Dictionary containing keys (strings) denoting observations to
#                                       extract and optional values denoting specific indexes of observations to extract.
#                                       If set to None, the full state is returned
#     Returns:
#         (Single-level) Dictionary containing the specified subset of states
#     """
#     # Extract the required observations
#     state = dict()
#     if observation_dict is not None:
#         # `self.observations` is a dict that contains the contains the keys of the observations that we want to use
#         # and optional values that specify slices or indexes of the particular observations that we want to use
#         for obs_key, obs_idxs in observation_dict.items():
#             if isinstance(obs_idxs, (slice, Iterable)):
#                 state[obs_key] = full_state[obs_key][obs_idxs]
#             elif obs_idxs is None:
#                 state[obs_key] = full_state[obs_key]
#             else:
#                 raise ValueError('Invalid index type {} for observation {}'.format(type(obs_idxs), obs_key))
#     else:
#         state.update(**full_state)
#
#     return state
#
#
# def flatten_dict(dictionary: Dict[Any, Union[Number, Iterable[Number]]],
#                  dtype: np.dtype = np.float) -> np.array:
#     """
#     Concatenates all values of a (single-level) dictionary into a single 1D numpy array.
#
#     Args:
#         dictionary(dict): (Single-level) Dictionary containing key-value pairs
#         dtype(np.dtype): Type of the output numpy array
#
#     Returns:
#         All values of the (single-level) dictionary concatenated together in a 1D numpy array
#     """
#     values = np.empty(shape=(0, 0))
#
#     for value in dictionary.values():
#         values = np.append([values, value])
#
#     return values.astype(dtype=dtype)


def gym_subspace_box(gym_space: spaces.Box,
                     idxs: Optional[Union[Iterable[Union[int, bool]], slice]] = None):
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
