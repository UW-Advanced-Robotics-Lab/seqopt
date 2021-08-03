from .door.config.ppo_config import ENV_CONFIG as DOOR_PPO_ENV_CONFIG
from .door.config.ppo_config import ALGORITHM_CONFIG as DOOR_PPO_ALGO_CONFIG
from .door.config.sac_config import ENV_CONFIG as DOOR_SAC_ENV_CONFIG
from .door.config.sac_config import ALGORITHM_CONFIG as DOOR_SAC_ALGO_CONFIG

from .door.config.benchmark_sac_config import ENV_CONFIG as DOOR_BENCHMARK_SAC_ENV_CONFIG
from .door.config.benchmark_sac_config import ALGORITHM_CONFIG as DOOR_BENCHMARK_SAC_ALGO_CONFIG

from .manipulator.config.ppo_config import ENV_CONFIG as MANIPULATOR_PPO_ENV_CONFIG
from .manipulator.config.ppo_config import ALGORITHM_CONFIG as MANIPULATOR_PPO_ALGO_CONFIG
from .manipulator.config.sac_config import ENV_CONFIG as MANIPULATOR_SAC_ENV_CONFIG
from .manipulator.config.sac_config import ALGORITHM_CONFIG as MANIPULATOR_SAC_ALGO_CONFIG

from .manipulator.config.benchmark_sac_config import ENV_CONFIG as MANIPULATOR_BENCHMARK_SAC_ENV_CONFIG
from .manipulator.config.benchmark_sac_config import ALGORITHM_CONFIG as MANIPULATOR_BENCHMARK_SAC_ALGO_CONFIG


CONFIG_MAP = dict(
    door=dict(ppo=(DOOR_PPO_ENV_CONFIG, DOOR_PPO_ALGO_CONFIG),
              sac=(DOOR_SAC_ENV_CONFIG, DOOR_SAC_ALGO_CONFIG)),
    door_benchmark=dict(sac=(DOOR_BENCHMARK_SAC_ENV_CONFIG, DOOR_BENCHMARK_SAC_ALGO_CONFIG)),
    manipulator=dict(ppo=(MANIPULATOR_PPO_ENV_CONFIG, MANIPULATOR_PPO_ALGO_CONFIG),
                     sac=(MANIPULATOR_SAC_ENV_CONFIG, MANIPULATOR_SAC_ALGO_CONFIG)),
    manipulator_benchmark=dict(sac=(MANIPULATOR_BENCHMARK_SAC_ENV_CONFIG, MANIPULATOR_BENCHMARK_SAC_ALGO_CONFIG)),
)
