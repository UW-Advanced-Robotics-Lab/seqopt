from stable_baselines3.common.utils import get_schedule_fn

from seqopt.common.env_config import SACConfig
from seqopt.seqsac.params import (
    SACActorParams,
    SACCriticParams,
    SACExplorationParams
)

from .benchmark_common import get_env_config


ENV_CONFIG = get_env_config()

# Update any parameters for the options
assert ENV_CONFIG.n_options == 1, "Unexpected number of options!"
ENV_CONFIG.validate()

# This is a hack to cast the parameters to the derived class
for option_id in range(ENV_CONFIG.n_options):
    ENV_CONFIG.actor_params[option_id].__class__ = SACActorParams
    ENV_CONFIG.critic_params[option_id].__class__ = SACCriticParams
    assert ENV_CONFIG.terminator_params[option_id] is None
    if ENV_CONFIG.exploration_params[option_id] is not None:
        ENV_CONFIG.exploration_params[option_id].__class__ = SACExplorationParams

# Update actor parameters
ENV_CONFIG.actor_params[0].update(
    dict(
        lr_schedule=3e-4,
        ent_coef=1e-3
    )
)
# Update critic parameters
ENV_CONFIG.critic_params[0].update(
    dict(
        lr_schedule=3e-5,
    )
)

# Update exploration parameters
if ENV_CONFIG.exploration_params[0] is not None:
    ENV_CONFIG.exploration_params[0].update(
        dict(
            scale=10.0
        )
    )


# Define schedule for demo learning
def demo_schedule(progress_remaining):
    return max(2 * progress_remaining - 1, 0)

# Define all SAC parameters
ALGORITHM_CONFIG = SACConfig(
    total_steps=int(3e7),
    buffer_size=int(3e5),
    gamma=0.95,
    batch_size=256,
    n_eval_episodes=1,
    log_interval=1,
    train_freq=1,
    save_freq=9000,
    eval_freq=9000,
    gradient_steps=1,
    n_episodes_rollout=-1,
    learning_starts=100,
    demo_file_path=None,
    demo_learning_schedule=get_schedule_fn(demo_schedule),
    log_dir='experiments/door/sac_benchmark'
)
