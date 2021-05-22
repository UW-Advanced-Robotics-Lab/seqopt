import numpy as np
from robosuite.models.robots.robot_model import RobotModel
from robosuite.utils.mjcf_utils import xml_path_completion


class WAM(RobotModel):
    """
    WAM is a 7dof dexterous robot arm created by Barrett

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
        bottom_offset (3-array): (x,y,z) offset desired from initial coordinates
    """

    def __init__(self, idn=0, bottom_offset=(0, 0, -0.913)):
        super().__init__(xml_path_completion("robots/wam/robot.xml"), idn=idn, bottom_offset=bottom_offset)

    @property
    def dof(self):
        return 7

    @property
    def gripper(self):
        return "WAMBarrettGripperDexterous"

    @property
    def default_controller_config(self):
        return "default_wam"

    @property
    def init_qpos(self):
        return np.array([0., 1.05, 0., 2.0, 0., 0., 1.57])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length/2, 0, 0)
        }

    @property
    def arm_type(self):
        return "single"

    @property
    def _joints(self):
        return ["wam_joint_1", "wam_joint_2", "wam_joint_3", "wam_joint_4",
                "wam_joint_5", "wam_joint_6", "wam_joint_7"]

    @property
    def _eef_name(self):
        return "right_hand"

    @property
    def _robot_base(self):
        return "base"

    @property
    def _actuators(self):
        return {
            "pos": [],  # No position actuators for sawyer
            "vel": [],  # No velocity actuators for sawyer
            "torq": ["torq_j1", "torq_j2", "torq_j3",
                     "torq_j4", "torq_j5", "torq_j6", "torq_j7"]
        }

    @property
    def _contact_geoms(self):
        return ["s_collision", "ah1_collision", "ah2_collision", "f_collision",
                "ws1_collision", "ws2_collision"]

    @property
    def _root(self):
        return 'base'

    @property
    def _links(self):
        return ["wam_link_1", "wam_link_2", "wam_link_3", "wam_link_4",
                "wam_link_5", "wam_link_6", "wam_link_7"]
