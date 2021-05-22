from .door.config.ppo_config import ENV_CONFIG as DOOR_PPO_ENV_CONFIG
from .door.config.ppo_config import ALGORITHM_CONFIG as DOOR_PPO_ALGO_CONFIG
from .door.config.sac_config import ENV_CONFIG as DOOR_SAC_ENV_CONFIG
from .door.config.sac_config import ALGORITHM_CONFIG as DOOR_SAC_ALGO_CONFIG

from .manipulator.config.ppo_config import ENV_CONFIG as MANIPULATOR_PPO_ENV_CONFIG
from .manipulator.config.ppo_config import ALGORITHM_CONFIG as MANIPULATOR_PPO_ALGO_CONFIG
from .manipulator.config.sac_config import ENV_CONFIG as MANIPULATOR_SAC_ENV_CONFIG
from .manipulator.config.sac_config import ALGORITHM_CONFIG as MANIPULATOR_SAC_ALGO_CONFIG

from .kitchen.config.ppo_config import ENV_CONFIG as KITCHEN_PPO_ENV_CONFIG
from .kitchen.config.ppo_config import ALGORITHM_CONFIG as KITCHEN_PPO_ALGO_CONFIG
from .kitchen.config.sac_config import ENV_CONFIG as KITCHEN_SAC_ENV_CONFIG
from .kitchen.config.sac_config import ALGORITHM_CONFIG as KITCHEN_SAC_ALGO_CONFIG


CONFIG_MAP = dict(
    door=dict(ppo=(DOOR_PPO_ENV_CONFIG, DOOR_PPO_ALGO_CONFIG),
              sac=(DOOR_SAC_ENV_CONFIG, DOOR_SAC_ALGO_CONFIG)),
    manipulator=dict(ppo=(MANIPULATOR_PPO_ENV_CONFIG, MANIPULATOR_PPO_ALGO_CONFIG),
                     sac=(MANIPULATOR_SAC_ENV_CONFIG, MANIPULATOR_SAC_ALGO_CONFIG)),
    kitchen=dict(ppo=(KITCHEN_PPO_ENV_CONFIG, KITCHEN_PPO_ALGO_CONFIG),
                 sac=(KITCHEN_SAC_ENV_CONFIG, KITCHEN_SAC_ALGO_CONFIG))
)
