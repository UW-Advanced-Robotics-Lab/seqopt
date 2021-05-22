# Activate the MuJoCo license manually here, because of license registration clashes between
# robosuite and dm_control

import os

from dm_control.mujoco.wrapper import util
from dm_control.mujoco.wrapper.mjbindings import mjlib
import dm_control.mujoco.wrapper.core as dm_core


license_path = util.get_mjkey_path()
result = mjlib.mj_activate(util.to_binary_string(license_path))

# Disable robosuite re-activation of license
os.environ['MUJOCO_PY_SKIP_ACTIVATE'] = '1'

# Disable dm_control re-activation of license
dm_core._REGISTERED = True

