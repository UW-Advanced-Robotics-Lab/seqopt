from dataclasses import dataclass

from seqopt.common.types import ActorParams, CriticParams, TerminatorParams, ExplorationParams


# We can add additional data members for the option parameters and/or override default values for parameters
# defined in the parent classes


@dataclass
class SACActorParams(ActorParams):
    pass


@dataclass
class SACCriticParams(CriticParams):
    n_critics: int = 2
    tau: float = 0.005
    target_update_interval: int = 1


@dataclass
class SACTerminatorParams(TerminatorParams):
    pass


@dataclass
class SACExplorationParams(ExplorationParams):
    pass
