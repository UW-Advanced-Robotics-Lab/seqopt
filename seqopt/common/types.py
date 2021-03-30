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


@dataclass
class ActorParams:
    default_action: np.ndarray
    observation_mask: Optional[Union[Iterable[Union[int, bool]], slice]] = None
    action_mask: Optional[np.ndarray] = None
    lr_schedule: Union[float, Schedule] = 3e-4
    net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None
    ent_coef: float = 1e-3
    activation_fn: Type[th.nn.Module] = th.nn.ReLU
    max_grad_norm: float = 0.5
    optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam
    optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())


@dataclass
class CriticParams:
    observation_mask: Optional[Union[Iterable[Union[int, bool]], slice]] = None
    n_critics: int = 2
    lr_schedule: Union[float, Schedule] = 3e-4
    net_arch: Optional[List[int]] = None
    tau: float = 0.005
    target_update_interval: int = 1
    activation_fn: Type[th.nn.Module] = th.nn.ReLU
    max_grad_norm: float = 0.5
    optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam
    optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())


@dataclass
class TerminatorParams:
    observation_mask: Optional[Union[Iterable[Union[int, bool]], slice]] = None
    lr_schedule: Union[float, Schedule] = 3e-6
    net_arch: Optional[List[int]] = None
    use_boltzmann: bool = True     # Use Boltzmann (Softmax) distribution to compute probabilities for termination
    ent_coef: float = 0.0
    activation_fn: Type[th.nn.Module] = th.nn.ReLU
    target_kl: Optional[float] = 5e-4
    max_grad_norm: float = 0.5
    optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam
    optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())


@dataclass
class ExplorationParams:
    # State-Counting features and corresponding boundaries
    features_extractor: Callable
    feature_boundaries: List[th.Tensor]
    # Hyperparameters
    scale: float = 1.0
