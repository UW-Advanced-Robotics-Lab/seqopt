from collections import OrderedDict
import numpy as np
from scipy.spatial.transform import Rotation
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from robosuite.environments.robot_env import RobotEnv
from robosuite.robots import SingleArm

from robosuite.models.arenas import TableArena
from robosuite.models.objects import DoorObject
from robosuite.models.tasks import ManipulationTask, UniformRandomSampler

from robosuite.models.grippers import GripperModel


class Door(RobotEnv):
    """
    This class corresponds to the door opening task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        gripper_visualizations (bool or list of bool): True if using gripper visualization.
            Useful for teleoperation. Should either be single bool if gripper visualization is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_latch (bool): if True, uses a spring-loaded handle and latch to "lock" the door closed initially
            Otherwise, door is instantiated with a fixed handle

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler instance): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        use_indicator_object (bool): if True, sets up an indicator object that
            is useful for debugging.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        controller_configs=None,
        gripper_types="default",
        gripper_visualizations=False,
        initialization_noise="default",
        use_latch=True,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        # First, verify that only one robot is being inputted
        self._check_robot_configuration(robots)

        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = (0.8, 0.3, 0.05)
        self.table_offset = (-0.2, -0.35, 0.8)

        # reward configuration
        self.use_latch = use_latch
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[0.07, 0.09],
                y_range=[-0.01, 0.01],
                ensure_object_boundary_in_range=False,
                rotation=(-np.pi / 2. - 0.25, -np.pi / 2.),
                rotation_axis='z',
            )

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            gripper_visualizations=gripper_visualizations,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the door is opened

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.25], proportional to the distance between door handle and robot arm
            - Rotating: in [0, 0.25], proportional to angle rotated by door handled
              - Note that this component is only relevant if the environment is using the locked door version

        Note that a successfully completed task (door opened) will return 1.0 irregardless of whether the environment
        is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # else, we consider only the case if we're using shaped rewards
        elif self.reward_shaping:
            # Add reaching component
            dist = np.linalg.norm(self._gripper_to_handle)
            reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
            reward += reaching_reward
            # Add rotating component if we're using a locked door
            if self.use_latch:
                handle_qpos = self.sim.data.qpos[self.handle_qpos_addr]
                reward += np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25)

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Verify the correct robot has been loaded
        assert isinstance(self.robots[0], SingleArm), \
            "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        door = DoorObject(
            name="Door",
            friction=0.0,
            damping=0.1,
            lock=self.use_latch,
            joints=[],  # ensures that door object does not have a free joint
        )
        self.mujoco_objects = OrderedDict([("Door", door)])
        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena, 
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.mujoco_objects, 
            visual_objects=None, 
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        # Additional object references from this env
        self.object_body_ids = dict()
        self.object_body_ids["door"] = self.sim.model.body_name2id("door")
        self.object_body_ids["frame"] = self.sim.model.body_name2id("frame")
        self.object_body_ids["latch"] = self.sim.model.body_name2id("latch")
        self.door_handle_site_id = self.sim.model.site_name2id("door_handle")
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr("door_hinge")
        if self.use_latch:
            self.handle_qpos_addr = self.sim.model.get_joint_qpos_addr("latch_joint")

        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.robots[0].gripper.important_geoms["left_finger"]
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.robots[0].gripper.important_geoms["right_finger"]
        ]

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for the Door object
            door_pos, door_quat = self.model.place_objects()
            door_body_id = self.sim.model.body_name2id("Door")
            self.sim.model.body_pos[door_body_id] = door_pos[0]
            self.sim.model.body_quat[door_body_id] = door_quat[0]

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:

            `'robot-state'`: contains robot-centric information.

            `'object-state'`: requires @self.use_object_obs to be True. Contains object-centric information.

            `'image'`: requires @self.use_camera_obs to be True. Contains a rendered frame from the simulation.

            `'depth'`: requires @self.use_camera_obs and @self.camera_depth to be True.
            Contains a rendered depth map from the simulation

        Returns:
            OrderedDict: Observations from the environment
        """
        di = super()._get_observation()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix
            pr = self.robots[0].robot_model.naming_prefix

            eef_pos = self._eef_xpos
            door_pos = np.array(self.sim.data.body_xpos[self.object_body_ids["door"]])
            handle_pos = self._handle_xpos
            hinge_qpos = np.array([self.sim.data.qpos[self.hinge_qpos_addr]])

            di["door_pos"] = door_pos
            di["handle_pos"] = handle_pos
            di[pr + "door_to_eef_pos"] = door_pos - eef_pos
            di[pr + "handle_to_eef_pos"] = handle_pos - eef_pos
            di["hinge_qpos"] = hinge_qpos

            # Some custom observations (for reward and exploration calculations only)
            di['reach_dist'] = np.array([np.linalg.norm(di[pr + "handle_to_eef_pos"])])
            di['gripper_euler_angles'] = Rotation.from_quat(di[pr + 'eef_quat']).as_euler('xyz')

            # gripper_geom_groups = [['gripper0_thumb_proximal_collision',
            #                         'gripper0_thumb_distal_collision',
            #                         'gripper0_thumb_pad_collision'],
            #                        ['gripper0_index_proximal_collision',
            #                         'gripper0_index_distal_collision',
            #                         'gripper0_index_pad_collision'],
            #                        ['gripper0_pinky_proximal_collision',
            #                         'gripper0_pinky_distal_collision',
            #                         'gripper0_pinky_pad_collision']]
            #
            # grasped = np.array([np.float32(self._check_grasp(gripper_geom_groups, 'handle'))])
            # self._check_handle_grasp()
            # di['grasped'] = grasped

            # Get the average angle of the finger joints (only the first part of all fingers)
            joint_names = ['gripper0_joint_thumb', 'gripper0_joint_index', 'gripper0_joint_pinky']
            joint_ids = [self.sim.model.joint_name2id(jnt) for jnt in joint_names]
            fingertip_dist = np.mean(self.sim.data.qpos[joint_ids])
            di['fingertip_dist'] = np.array([fingertip_dist])

            # Check if handle is grasped
            grasped = np.array([np.float32((fingertip_dist < 0.5) and self._check_handle_in_grasp())])
            di['grasped'] = grasped

            di['object-state'] = np.concatenate([
                di["door_pos"],
                di["handle_pos"],
                di[pr + "door_to_eef_pos"],
                di[pr + "handle_to_eef_pos"],
                di["hinge_qpos"],
                # Our custom observations
                di['grasped'],
                di['fingertip_dist'],
                di['reach_dist'],
                di['gripper_euler_angles']
            ])

            # Also append handle qpos if we're using a locked door version with rotatable handle
            if self.use_latch:
                handle_qpos = np.array([self.sim.data.qpos[self.handle_qpos_addr]])
                di["handle_qpos"] = handle_qpos
                di['object-state'] = np.concatenate([di["object-state"], di["handle_qpos"]])

        return di

    def _check_handle_in_grasp(self):
        # To ensure that the handle is grasped, we check if the line extending from the end effector site
        # to the palm, intersects at least two of the four planar faces of the handle (not taking into account
        # the outward facing face of the handle). This requires some geometry for the intersection of lines
        # with planes, and bounds checking to ensure that the intersection actually happens since the size of
        # all entities is finite. An additional check it to ensure that one or more fingers is actually
        # making contact with the handle

        handle_id = self.sim.model.geom_name2id('handle')

        # Get the position (center) of the handle and the orientation of the handle in world coordinates
        handle_xpos = self.sim.data.geom_xpos[handle_id]
        handle_xmat = self.sim.data.geom_xmat[handle_id].reshape(3, 3)
        handle_size = self.sim.model.geom_size[handle_id]

        # Get the corners of the cuboid representing the handle (there should be 8 values)
        assert len(handle_size) == 3, 'Expected a box shape for handle! Got something else!'
        corners = []
        for dy in [-handle_size[1], handle_size[1]]:
            for dz in [-handle_size[2], handle_size[2]]:
                for dx in [-handle_size[0], handle_size[0]]:
                    rel_pos = np.dot(handle_xmat, np.array([dx, dy, dz]))
                    pos = handle_xpos + rel_pos
                    corners.append(pos)

        # Segregate the corners into points for the 4 faces
        faces = np.asarray([
            [corners[0], corners[1], corners[3], corners[2]],  # Front face
            [corners[2], corners[3], corners[7], corners[6]],  # Top face
            [corners[4], corners[5], corners[7], corners[6]],  # Back face
            [corners[0], corners[1], corners[5], corners[4]]   # Bottom face
        ])

        # Get the end-effector site and the point on the palm that line along the line extending to the palm
        # from the end effector
        eef_site_id = self.sim.model.site_name2id('gripper0_grip_site')
        end_effector_pos = self.sim.data.site_xpos[eef_site_id]
        palm_site_id = self.sim.model.site_name2id('gripper0_ft_frame')
        palm_pos = self.sim.data.site_xpos[palm_site_id]
        line = np.vstack([end_effector_pos, palm_pos])

        # Check if at least 2 faces are intersected by the line
        n_faces_intersected = 0
        for face_id in range(len(faces)):
            if self._intersect(line, faces[face_id]):
                n_faces_intersected += 1
            if n_faces_intersected >= 2:
                # print(f"Handle inside end-effector grasp")
                return True

        return False

    def _intersect(self, line_pts, plane_pts):
        line_vector = line_pts[1] - line_pts[0]

        # Assume plane pts are provides in a CCW direction (doesn't matter, since we correct for it anyway)
        normal_vector = np.cross(plane_pts[1] - plane_pts[0], plane_pts[2] - plane_pts[1])

        # Calculate dot product of line vector and normal vector of plane
        line_plane_dotp = np.dot(line_vector, normal_vector)
        # We want to ensure that the normal vector points in the same half-space as the line vector
        if line_plane_dotp < 0:
            normal_vector *= -1

        # They intersect (at least for infinitely long objects) as long as the line vector and normal vector
        # are not perpendicular. We use a tolerance value instead of zero
        if abs(np.dot(line_vector / np.linalg.norm(line_vector), normal_vector / np.linalg.norm(normal_vector))) > 1e-3:
            # Calculate point of intersection
            line_frac = np.dot((plane_pts[0] - line_pts[0]), normal_vector) / line_plane_dotp
            if 0 <= line_frac <= 1:
                # Calculate the intersection point on the plane
                intersect_pt = line_pts[0] + line_frac * line_vector

                # Check if the intersection point lies within the points of the plane
                # We want to ignore the normal dimension of all points
                # Basically, we want to go from a 3D -> 2D Coordinate system
                # We set plane_pts[0] as the origin
                origin = plane_pts[0].copy()
                plane_pts = plane_pts - origin
                intersect_pt = intersect_pt - origin
                # Define new basis vectors (any points on the plane should provide an axis perpendicular to
                # the normal vector)
                x_axis = plane_pts[1] / np.linalg.norm(plane_pts[1])
                y_axis = np.cross(normal_vector, x_axis)
                y_axis /= np.linalg.norm(y_axis)

                # Project all points into this 2D space
                new_plane_pts = [np.array([np.dot(plane_pt, x_axis), np.dot(plane_pt, y_axis)])
                                 for plane_pt in plane_pts]
                new_intersection_pt = np.array([np.dot(intersect_pt, x_axis), np.dot(intersect_pt, y_axis)])

                # Use the shapely API to determine if the point lies within the polygon formed by the plane in 2D space
                polygon = Polygon(new_plane_pts)
                point = Point(new_intersection_pt)

                return polygon.contains(point)

            else:
                # The line intersects the plane at some point, but not within the bounds of both of the
                # provided line points
                return False
        else:
            return False

    def _check_grasp(self, gripper, object_geoms):
        """
        Checks whether the specified gripper as defined by @gripper is grasping the specified object in the environment.
        By default, this will return True if at least one geom in both the "left_fingerpad" and "right_fingerpad" geom
        groups are in contact with any geom specified by @object_geoms. Custom gripper geom groups can be
        specified with @gripper as well.
        Args:
            gripper (GripperModel or str or list of str or list of list of str): If a MujocoModel, this is specific
            gripper to check for grasping (as defined by "left_fingerpad" and "right_fingerpad" geom groups). Otherwise,
                this sets custom gripper geom groups which together define a grasp. This can be a string
                (one group of single gripper geom), a list of string (multiple groups of single gripper geoms) or a
                list of list of string (multiple groups of multiple gripper geoms). At least one geom from each group
                must be in contact with any geom in @object_geoms for this method to return True.
            object_geoms (str or list of str or MujocoModel): If a MujocoModel is inputted, will check for any
                collisions with the model's contact_geoms. Otherwise, this should be specific geom name(s) composing
                the object to check for contact.
        Returns:
            bool: True if the gripper is grasping the given object
        """
        # Convert object, gripper geoms into standardized form
        o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            # Parse each element in the gripper_geoms list accordingly
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        # Search for collisions between each gripper geom group and the object geoms group
        for g_group in g_geoms:
            if not self.check_contact(g_group, o_geoms):
                return False
        return True

    def check_contact(self, geoms_1, geoms_2=None):
        """
        Finds contact between two geom groups.
        Args:
            geoms_1 (str or list of str or MujocoModel): an individual geom name or list of geom names or a model. If
                a MujocoModel is specified, the geoms checked will be its contact_geoms
            geoms_2 (str or list of str or MujocoModel or None): another individual geom name or list of geom names.
                If a MujocoModel is specified, the geoms checked will be its contact_geoms. If None, will check
                any collision with @geoms_1 to any other geom in the environment
        Returns:
            bool: True if any geom in @geoms_1 is in contact with any geom in @geoms_2.
        """
        # Check if either geoms_1 or geoms_2 is a string, convert to list if so
        if type(geoms_1) is str:
            geoms_1 = [geoms_1]

        if type(geoms_2) is str:
            geoms_2 = [geoms_2]

        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            # check contact geom in geoms
            c1_in_g1 = self.sim.model.geom_id2name(contact.geom1) in geoms_1
            c2_in_g2 = self.sim.model.geom_id2name(contact.geom2) in geoms_2 if geoms_2 is not None else True
            # check contact geom in geoms (flipped)
            c2_in_g1 = self.sim.model.geom_id2name(contact.geom2) in geoms_1
            c1_in_g2 = self.sim.model.geom_id2name(contact.geom1) in geoms_2 if geoms_2 is not None else True
            if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
                return True
        return False

    def _check_success(self):
        """
        Check if door has been opened.

        Returns:
            bool: True if door has been opened
        """
        hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        return hinge_qpos > 0.3

    def _visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to door handle
        if self.robots[0].gripper_visualization:
            # get distance to door handle
            dist = np.sum(
                np.square(
                    self._handle_xpos
                    - self.sim.data.get_site_xpos(self.robots[0].gripper.visualization_sites["grip_site"])
                )
            )

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.robots[0].eef_site_id] = rgba

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable

        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        if type(robots) is list:
            assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"

    @property
    def _eef_xpos(self):
        """
        Grabs End Effector position

        Returns:
            np.array: End effector(x,y,z)
        """
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])

    @property
    def _handle_xpos(self):
        """
        Grabs the position of the door handle handle.

        Returns:
            np.array: Door handle (x,y,z)
        """
        return self.sim.data.geom_xpos[self.sim.model.geom_name2id('handle')]
        # return self.sim.data.site_xpos[self.door_handle_site_id]

    @property
    def _gripper_to_handle(self):
        """
        Calculates distance from the gripper to the door handle.

        Returns:
            np.array: (x,y,z) distance between handle and eef
        """
        return self._handle_xpos - self._eef_xpos
