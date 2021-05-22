from dataclasses import dataclass

from seqopt.common.types import ActorParams, CriticParams, TerminatorParams, ExplorationParams


# We can add additional data members for the option parameters and/or override default values for parameters
# defined in the parent classes


@dataclass
class PPOActorParams(ActorParams):
    clip_range: float = 0.2
    target_kl: float = 5e-2


@dataclass
class PPOCriticParams(CriticParams):
    pass


@dataclass
class PPOTerminatorParams(TerminatorParams):
    target_kl: float = 5e-4


@dataclass
class PPOExplorationParams(ExplorationParams):
    pass
