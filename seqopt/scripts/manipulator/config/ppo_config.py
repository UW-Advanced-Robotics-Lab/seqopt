import torch as th

from seqopt.common.env_config import PPOConfig
from seqopt.seqppo.params import (
    PPOActorParams,
    PPOCriticParams,
    PPOTerminatorParams,
    PPOExplorationParams
)

from .common import get_env_config


ENV_CONFIG = get_env_config()

# Update any parameters for the options
assert ENV_CONFIG.n_options == 3, "Unexpected number of options!"
ENV_CONFIG.validate()

# This is a hack to cast the parameters to the derived class
for option_id in range(ENV_CONFIG.n_options):
    ENV_CONFIG.actor_params[option_id].__class__ = PPOActorParams
    ENV_CONFIG.critic_params[option_id].__class__ = PPOCriticParams
    ENV_CONFIG.terminator_params[option_id].__class__ = PPOTerminatorParams
    if ENV_CONFIG.exploration_params[option_id] is not None:
        ENV_CONFIG.exploration_params[option_id].__class__ = PPOExplorationParams

# Update actor parameters
ENV_CONFIG.actor_params[0].update(
    dict(
        lr_schedule=3e-4,
        ent_coef=1e-3,
        target_kl=5e-2
    )
)

ENV_CONFIG.actor_params[1].update(
    dict(
        lr_schedule=3e-4,
        ent_coef=1e-3,
        target_kl=5e-2
    )
)

ENV_CONFIG.actor_params[2].update(
    dict(
        lr_schedule=3e-4,
        ent_coef=1e-3,
        target_kl=5e-2
    )
)

# Update critic parameters
ENV_CONFIG.critic_params[0].update(
    dict(
        lr_schedule=3e-4,
        # output_activation_fn=th.nn.LeakyReLU(negative_slope=1e-1)
    )
)

ENV_CONFIG.critic_params[1].update(
    dict(
        lr_schedule=3e-4,
        # output_activation_fn=th.nn.LeakyReLU(negative_slope=1e-1)
    )
)

ENV_CONFIG.critic_params[2].update(
    dict(
        lr_schedule=3e-4,
        # output_activation_fn=th.nn.LeakyReLU(negative_slope=1e-1)
    )
)

# Update terminator parameters
ENV_CONFIG.terminator_params[0].update(
    dict(
        lr_schedule=1e-5,
        ent_coef=5e-1,
        target_kl=5e-4
    )
)

ENV_CONFIG.terminator_params[1].update(
    dict(
        lr_schedule=1e-5,
        ent_coef=5e-1,
        target_kl=5e-4
    )
)

ENV_CONFIG.terminator_params[2].update(
    dict(
        lr_schedule=1e-5,
        ent_coef=0.0,
        target_kl=5e-4
    )
)

# Update exploration parameters
if ENV_CONFIG.exploration_params[1] is not None:
    ENV_CONFIG.exploration_params[1].update(
        dict(
            scale=5.0
        )
    )

# Define all SAC parameters
ALGORITHM_CONFIG = PPOConfig(
    total_steps=int(5e7),
    gamma=0.98,
    batch_size=256,
    train_freq=10240,
    save_freq=50000,
    eval_freq=20000,
    n_eval_episodes=3,
    log_interval=1,
    n_epochs=10,
    log_dir='experiments/manipulator/seqppo'
)
