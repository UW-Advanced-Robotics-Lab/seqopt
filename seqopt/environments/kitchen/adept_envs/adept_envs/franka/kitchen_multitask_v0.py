""" Kitchen environment for long horizon manipulation """
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from .. import robot_env
from ..utils.configurable import configurable
from gym import spaces
from dm_control.mujoco import engine

@configurable(pickleable=True)
class KitchenV0(robot_env.RobotEnv):

    CALIBRATION_PATHS = {
        'default':
        os.path.join(os.path.dirname(__file__), 'robot/franka_config.xml')
    }
    # Converted to velocity actuation
    ROBOTS = {'robot': 'seqopt.environments.kitchen.adept_envs.adept_envs.franka.robot.franka_robot:Robot_VelAct'}
    MODEl = os.path.join(
        os.path.dirname(__file__),
        '../franka/assets/franka_kitchen_jntpos_act_ab.xml')
    N_DOF_ROBOT = 9
    N_DOF_OBJECT = 21

    def __init__(self, robot_params={}, frame_skip=40):
        self.goal_concat = True
        self.obs_dict = {}
        self.robot_noise_ratio = 0.0  # 10% as per robot_config specs
        self.goal = np.zeros((30,))
        self.ids = None

        super().__init__(
            self.MODEl,
            robot=self.make_robot(
                n_jnt=self.N_DOF_ROBOT,  #root+robot_jnts
                n_obj=self.N_DOF_OBJECT,
                **robot_params),
            frame_skip=frame_skip,
            camera_settings=dict(
                distance=4.5,
                azimuth=-66,
                elevation=-65,
            ),
        )
        self.init_qpos = self.sim.model.key_qpos[0].copy()

        # For the microwave kettle slide hinge
        self.init_qpos = np.array([ 1.48388023e-01, -1.76848573e+00,  1.84390296e+00, -2.47685760e+00,
                                    2.60252026e-01,  7.12533105e-01,  1.59515394e+00,  4.79267505e-02,
                                    3.71350919e-02, -2.66279850e-04, -5.18043486e-05,  3.12877220e-05,
                                   -4.51199853e-05, -3.90842156e-06, -4.22629655e-05,  6.28065475e-05,
                                    4.04984708e-05,  4.62730939e-04, -2.26906415e-04, -4.65501369e-04,
                                   -6.44129196e-03, -1.77048263e-03,  1.08009684e-03, -2.69397440e-01,
                                    3.50383255e-01,  1.61944683e+00,  1.00618764e+00,  4.06395120e-03,
                                   -6.62095997e-03, -2.68278933e-04])

        self.init_qvel = self.sim.model.key_qvel[0].copy()

        self.act_mid = np.zeros(self.N_DOF_ROBOT)
        self.act_amp = 2.0 * np.ones(self.N_DOF_ROBOT)

        act_lower = -1*np.ones((self.N_DOF_ROBOT,))
        act_upper =  1*np.ones((self.N_DOF_ROBOT,))
        self.action_space = spaces.Box(act_lower, act_upper)

        obs_upper = 8. * np.ones(self.obs_dim)
        obs_lower = -obs_upper
        self.observation_space = spaces.Box(obs_lower, obs_upper)

    def setup(self):
        # Store ids of important entities
        self.ids = dict()

        # Add geoms
        self.ids['geoms'] = dict()
        self.ids['geoms'].update({
            'slide_handle': self.sim.model.name2id('slide_handle', 'geom'),
            'left_finger_col': self.sim.model.name2id('panda0_leftfinger_col', 'geom'),
            'right_finger_col': self.sim.model.name2id('panda0_rightfinger_col', 'geom')
        })

        # Add sites
        self.ids['sites'] = dict()
        self.ids['sites'].update({
            'end_effector': self.sim.model.name2id('end_effector', 'site')
        })

        # Add joints
        self.ids['joints'] = dict()
        self.ids['joints'].update({
            'slide': self.sim.model.name2id('slidedoor_joint', 'joint'),
            'left_finger': self.sim.model.name2id('panda0_finger_joint1', 'joint'),
            'right_finger': self.sim.model.name2id('panda0_finger_joint2', 'joint')
        })

    def _get_reward_n_score(self, obs_dict):
        raise NotImplementedError()

    def step(self, a, b=None):
        a = np.clip(a, -1.0, 1.0)

        if not self.initializing:
            a = self.act_mid + a * self.act_amp  # mean center and scale
        else:
            self.goal = self._get_task_goal()  # update goal if init

        self.robot.step(
            self, a, step_duration=self.skip * self.model.opt.timestep)

        # observations
        obs = self._get_obs()

        #rewards
        reward_dict, score = self._get_reward_n_score(self.obs_dict)

        # termination
        done = False

        # finalize step
        env_info = {
            'time': self.obs_dict['t'],
            'obs_dict': self.obs_dict,
            'rewards': reward_dict,
            'score': score,
            'images': np.asarray(self.render(mode='rgb_array'))
        }
        # self.render()
        return obs, reward_dict['r_total'], done, env_info

    def _get_obs(self):
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
            self, robot_noise_ratio=self.robot_noise_ratio)

        # (Somesh) Custom observations
        handle_pos = self.sim.data.geom_xpos[self.ids['geoms']['slide_handle']]
        fingertip_dist = self.sim.data.qpos[self.ids['joints']['left_finger']] +\
                         self.sim.data.qpos[self.ids['joints']['right_finger']]
        grip_pos = self.sim.data.site_xpos[self.ids['sites']['end_effector']]
        reach_dist = np.linalg.norm(handle_pos - grip_pos)
        slide_dist = np.abs(self.sim.model.jnt_range[self.ids['joints']['slide']][1] -
                            self.sim.data.qpos[self.ids['joints']['slide']])
        # Determine if the handle is grasped
        # Conditions that need to be met:
        #   1. Both left and right fingers are in contact with the handle
        #   2. x position of handle should be (somewhere) in between x positions of the contact locations (to ensure
        #      the fingers grab on either side of the handle)
        left_finger_contact_pos = None
        right_finger_contact_pos = None
        for contact in self.sim.data.contact:
            # If we have already determined contact locations for both fingers, stop searching
            if left_finger_contact_pos is not None and right_finger_contact_pos is not None:
                break

            # Check if the door handle is involved in the contact
            if self.ids['geoms']['slide_handle'] in [contact.geom1, contact.geom2]:
                # Check if the left finger was involved in the contact
                if self.ids['geoms']['left_finger_col'] in [contact.geom1, contact.geom2]:
                    left_finger_contact_pos = contact.pos.copy()

                # Check if the right finger was involved in the contact
                if self.ids['geoms']['right_finger_col'] in [contact.geom1, contact.geom2]:
                    right_finger_contact_pos = contact.pos.copy()

        if left_finger_contact_pos is not None and right_finger_contact_pos is not None:
            # print(f"Contact Positions - Left Finger: {left_finger_contact_pos}, "
            #       f"Right Finger: {right_finger_contact_pos} \n"
            #       f"Handle Position - {self.sim.data.geom_xpos[self.ids['geoms']['slide_handle']]}")
            # Get the mean x value of the contacts
            contact_mean_x = .5 * (left_finger_contact_pos[0] + right_finger_contact_pos[0])

            # Get the x value of the center of the door handle
            handle_x = self.sim.data.geom_xpos[self.ids['geoms']['slide_handle']][0]

            # If the mean x value is within half the radius of the center of the handle, we consider it gripped
            if np.abs(handle_x - contact_mean_x) <= 0.011:
                grasped = np.float32(1.0)
                # print("Grasped handle!")
            else:
                grasped = np.float32(0.0)
        else:
            grasped = np.float32(0.0)

        self.obs_dict = {}
        self.obs_dict['t'] = t
        self.obs_dict['qp'] = qp
        self.obs_dict['qv'] = qv
        self.obs_dict['obj_qp'] = obj_qp
        self.obs_dict['obj_qv'] = obj_qv
        self.obs_dict['goal'] = self.goal
        self.obs_dict['handle_pos'] = handle_pos
        self.obs_dict['fingertip_dist'] = fingertip_dist
        self.obs_dict['grasped'] = grasped
        self.obs_dict['slide_dist'] = slide_dist
        self.obs_dict['reach_dist'] = reach_dist

        if self.goal_concat:
            print(f"Obs dict: {self.obs_dict}")
            return np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp'], self.obs_dict['goal'],
                                   self.obs_dict['handle_pos'],
                                   np.expand_dims(self.obs_dict['fingertip_dist'], 0),
                                   np.expand_dims(self.obs_dict['grasped'], 0),
                                   np.expand_dims(self.obs_dict['slide_dist'], 0),
                                   np.expand_dims(self.obs_dict['reach_dist'], 0)])

    def reset_model(self):
        reset_pos = self.init_qpos[:].copy()
        reset_vel = self.init_qvel[:].copy()
        self.robot.reset(self, reset_pos, reset_vel)
        self.sim.forward()
        self.goal = self._get_task_goal()  #sample a new goal on reset
        return self._get_obs()

    def evaluate_success(self, paths):
        # score
        mean_score_per_rollout = np.zeros(shape=len(paths))
        for idx, path in enumerate(paths):
            mean_score_per_rollout[idx] = np.mean(path['env_infos']['score'])
        mean_score = np.mean(mean_score_per_rollout)

        # success percentage
        num_success = 0
        num_paths = len(paths)
        for path in paths:
            num_success += bool(path['env_infos']['rewards']['bonus'][-1])
        success_percentage = num_success * 100.0 / num_paths

        # fuse results
        return np.sign(mean_score) * (
            1e6 * round(success_percentage, 2) + abs(mean_score))

    def close_env(self):
        self.robot.close()

    def set_goal(self, goal):
        self.goal = goal

    def _get_task_goal(self):
        return self.goal

    # Only include goal
    @property
    def goal_space(self):
        len_obs = self.observation_space.low.shape[0]
        env_lim = np.abs(self.observation_space.low[0])
        return spaces.Box(low=-env_lim, high=env_lim, shape=(len_obs//2,))

    def convert_to_active_observation(self, observation):
        return observation

class KitchenTaskRelaxV1(KitchenV0):
    """Kitchen environment with proper camera and goal setup"""

    def __init__(self):
        super(KitchenTaskRelaxV1, self).__init__()

    def _get_reward_n_score(self, obs_dict):
        reward_dict = {}
        reward_dict['true_reward'] = 0.
        reward_dict['bonus'] = 0.
        reward_dict['r_total'] = 0.
        score = 0.
        return reward_dict, score

    def render(self, mode='human'):
        if mode =='rgb_array':
            camera = engine.MovableCamera(self.sim, 1920, 2560)
            camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
            img = camera.render()
            return img
        else:
            super(KitchenTaskRelaxV1, self).render()
