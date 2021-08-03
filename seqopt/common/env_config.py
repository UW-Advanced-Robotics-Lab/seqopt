from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from stable_baselines3.common.utils import get_schedule_fn

from seqopt.common.types import (
    ActorParams,
    CriticParams,
    TerminatorParams,
    ExplorationParams,
    Schedule
)


@dataclass
class EnvConfig:
    # Environment information
    env: Union[str, dict]   # The dict representation is reserved for robosuite environments
    seed: int
    obs_dict: Dict[str, np.array]
    reward_func: Callable
    task_potential_func: Callable
    # Option information
    n_options: int
    option_names: List[str]
    actor_params: List[ActorParams]
    critic_params: List[CriticParams]
    terminator_params: List[Optional[TerminatorParams]]
    exploration_params: List[Optional[ExplorationParams]]

    def validate(self):
        assert len(self.actor_params) == self.n_options
        assert len(self.critic_params) == self.n_options
        assert len(self.terminator_params) == self.n_options
        assert len(self.exploration_params) == self.n_options


@dataclass
class AlgorithmConfig:
    total_steps: int
    gamma: float = 0.99
    train_freq: int = 1
    batch_size: int = 256
    n_eval_episodes: int = 3
    log_interval: int = 1
    save_freq: int = 1000
    eval_freq: int = 1000
    log_dir: str = ''
    device: str = 'cpu'


@dataclass
class PPOConfig(AlgorithmConfig):
    n_epochs: int = 10


@dataclass
class SACConfig(AlgorithmConfig):
    buffer_size: int = int(1e6)
    gradient_steps: int = 1
    n_episodes_rollout: int = -1
    learning_starts: int = 100
    demo_file_path: Optional[str] = None
    demo_learning_schedule: Schedule = get_schedule_fn(0.0)
