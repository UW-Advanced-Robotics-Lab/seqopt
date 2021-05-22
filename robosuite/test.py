import cProfile

import robosuite as suite
from robosuite.wrappers import GymWrapper
import time


if __name__ == "__main__":

    RENDER = True

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            "Door",
            robots="Jaco",                   # use WAM robot
            use_camera_obs=False,           # do not use pixel observations
            has_offscreen_renderer=False,   # not needed since not using pixel obs
            has_renderer=RENDER,              # make sure we can render to the screen
            reward_shaping=True,            # use dense rewards
            control_freq=20,                # control should happen fast enough so that simulation looks smooth
        )
    )

    def sim_time_test():
        sim_start_time = time.time()
        # Run for 1000 steps
        observation = env.reset()
        for t in range(1000):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print(f"Episode finished after {t + 1} timesteps")
                break
            if RENDER:
                env.render()
        return time.time() - sim_start_time

    cProfile.run("sim_time_test()")
    print(f"Simulation time for {t + 1} steps: {sim_time_test()} s")

