from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Type, Union

import numpy as np
import torch as th


# Type alias
Schedule = Callable[[float], float]


class RolloutReturn(NamedTuple):
    option_timesteps: List[int]
    episode_reward: float
    episode_timesteps: int
    n_episodes: int
    continue_training: bool


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    next_observations: th.Tensor
    actions: th.Tensor
    old_log_prob: th.Tensor
    old_termination_prob: th.Tensor
    old_option_values: th.Tensor
    old_next_option_values: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


def update(obj: Any, kv: Dict):
    for k, v in kv.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


# This class represents the parameters for the actors/critic/terminators/state_counters common to all algorithms
@dataclass
class ActorParams:
    default_action: Union[Callable[[th.Tensor], th.Tensor], th.Tensor]
    observation_mask: Optional[Union[Iterable[Union[int, bool]], slice]] = None
    action_mask: Optional[np.ndarray] = None
    lr_schedule: Union[float, Schedule] = 1e-3
    net_arch: Optional[List[int]] = None
    ent_coef: float = 1e-4
    activation_fn: Type[th.nn.Module] = th.nn.ReLU
    max_grad_norm: float = 0.5
    optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam
    optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())

    def update(self, kv):
        update(self, kv)


@dataclass
class CriticParams:
    observation_mask: Optional[Union[Iterable[Union[int, bool]], slice]] = None
    lr_schedule: Union[float, Schedule] = 3e-4
    net_arch: Optional[List[int]] = None
    activation_fn: Type[th.nn.Module] = th.nn.ReLU
    output_activation_fn: Optional[Union[th.nn.Module, Type[th.nn.Module]]] = None
    max_grad_norm: float = 0.5
    optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam
    optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())

    def update(self, kv):
        update(self, kv)


@dataclass
class TerminatorParams:
    observation_mask: Optional[Union[Iterable[Union[int, bool]], slice]] = None
    lr_schedule: Union[float, Schedule] = 5e-8
    net_arch: Optional[List[int]] = None
    use_boltzmann: bool = True     # Use Boltzmann (Softmax) distribution to compute probabilities for termination
    ent_coef: float = 5e-2
    activation_fn: Type[th.nn.Module] = th.nn.ReLU
    max_grad_norm: float = 0.5
    optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam
    optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())

    def update(self, kv):
        update(self, kv)


@dataclass
class ExplorationParams:
    # State-Counting features and corresponding boundaries
    features_extractor: Callable
    feature_boundaries: List[th.Tensor]
    reward_func: Optional[Callable[[th.Tensor, int], float]] = None
    # Hyperparameters
    scale: float = 1.0

    def update(self, kv):
        update(self, kv)
