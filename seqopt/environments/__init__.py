import inspect

from dm_control import suite

from seqopt.environments import manipulator


suite._DOMAINS.update({name: module for name, module in locals().items()
                       if inspect.ismodule(module) and hasattr(module, 'SUITE')})
