import inspect

from dm_control import suite
from gym.envs.registration import registry, register, make, spec

from seqopt.environments import manipulator

# Export any custom dm_control environments
suite._DOMAINS.update({name: module for name, module in locals().items()
                       if inspect.ismodule(module) and hasattr(module, 'SUITE')})

# Export any custom gym environments
for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    register(
        id='FetchModifiedPickAndPlace{}-v1'.format(suffix),
        entry_point='seqopt.environments.robotics:FetchPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

