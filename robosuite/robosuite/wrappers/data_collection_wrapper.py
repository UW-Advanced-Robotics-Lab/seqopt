"""
This file implements a wrapper for saving simulation states to disk.
This data collection wrapper is useful for collecting demonstrations.
"""

import glfw
import os
import time
import numpy as np

from robosuite.wrappers import Wrapper


class DataCollectionWrapper(Wrapper):
    def __init__(self, env, directory, collect_freq=1, flush_freq=100,
                 keypress_for_start=False, keypress_for_save=False):
        """
        Initializes the data collection wrapper.

        Args:
            env (MujocoEnv): The environment to monitor.
            directory (str): Where to store collected data.
            collect_freq (int): How often to save simulation state, in terms of environment steps.
            flush_freq (int): How frequently to dump data to disk, in terms of environment steps.
        """
        super().__init__(env)

        # the base directory for all logging
        self.directory = directory

        # in-memory cache for simulation states and action info
        self.states = []
        self.action_infos = []  # stores information about actions taken

        # how often to save simulation state, in terms of environment steps
        self.collect_freq = collect_freq
        self._collecting_data = False

        # how frequently to dump data to disk, in terms of environment steps
        self.flush_freq = flush_freq

        # Start data collection based on keypress
        # This allows user freedom to adjust environment prior to collecting data for episode
        self.keypress_for_start = keypress_for_start

        # Save trajectory/episode once user decides it's good to save
        # This allows for discarding episodes if the user is unhappy with the demonstration
        # Note that this renders the flush_freq invalid, as we only create files/directories once user
        # decides that they want to save this episode
        self.keypress_for_save = keypress_for_save

        # Enable key bindings
        env.viewer.add_keyup_callback("any", self.on_release)

        if not os.path.exists(directory):
            print("DataCollectionWrapper: making new directory at {}".format(directory))
            os.makedirs(directory)

        # store logging directory for current episode
        self.ep_directory = None

        # remember whether any environment interaction has occurred
        self.has_interaction = False

    def _start_new_episode(self):
        """
        Bookkeeping to do at the start of each new episode.
        """

        # flush any data left over from the previous episode if any interactions have happened
        # only do this if the saving is not explicitly triggered by a keypress
        if self.has_interaction and not self.keypress_for_save:
            self._flush()

        self._collecting_data = True

        # timesteps in current episode
        self.t = 0
        self.has_interaction = False

    def _on_first_interaction(self):
        """
        Bookkeeping for first timestep of episode.
        This function is necessary to make sure that logging only happens after the first
        step call to the simulation, instead of on the reset (people tend to call
        reset more than is necessary in code).

        Raises:
            AssertionError: [Episode path already exists]
        """

        self.has_interaction = True

        # create a directory with a timestamp
        t1, t2 = str(time.time()).split(".")
        self.ep_directory = os.path.join(self.directory, "ep_{}_{}".format(t1, t2))
        assert not os.path.exists(self.ep_directory)
        print("DataCollectionWrapper: making folder at {}".format(self.ep_directory))
        os.makedirs(self.ep_directory)

        # save the model xml
        xml_path = os.path.join(self.ep_directory, "model.xml")
        self.env.model.save_model(xml_path)

    def _flush(self):
        """
        Method to flush internal state to disk.
        """
        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, "state_{}_{}.npz".format(t1, t2))
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__
        np.savez(
            state_path,
            states=np.array(self.states),
            action_infos=self.action_infos,
            env=env_name,
        )
        self.states = []
        self.action_infos = []

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate data collection

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        ret = super().reset()
        self._collecting_data = False
        if not self.keypress_for_start:
            self._start_new_episode()
        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ret = super().step(action)

        if self._collecting_data:
            self.t += 1

            # on the first time step, make directories for logging
            # only do this if we don't save based on user keypress
            if not self.has_interaction and not self.keypress_for_save:
                self._on_first_interaction()

            # collect the current simulation state if necessary
            if self.t % self.collect_freq == 0:
                state = self.env.sim.get_state().flatten()
                self.states.append(state)

                info = {}
                info["actions"] = np.array(action)
                self.action_infos.append(info)

            # flush collected data to disk if necessary
            if self.t % self.flush_freq == 0 and not self.keypress_for_save:
                self._flush()

        return ret

    def close(self):
        """
        Override close method in order to flush left over data
        """
        self._start_new_episode()
        self.env.close()

    def on_release(self, window, key, scancode, action, mods):
        """
        Key handler for key releases.

        Args:
            window: [NOT USED]
            key (int): keycode corresponding to the key that was pressed
            scancode: [NOT USED]
            action: [NOT USED]
            mods: [NOT USED]
        """
        # user-commanded trajectory saving
        if key == glfw.KEY_S:
            print('Saving episode...')
            # This is kind of a hack
            # The _on_first_iteration() function has the sole purpose of creating the directory
            # structure for the episode and storing the robot model xml
            self._on_first_interaction()
            # Save all the collected states and actions
            self._flush()
        # user-command collection start
        elif key == glfw.KEY_C:
            self._start_new_episode()
            print('Started data collection for new episode...')
