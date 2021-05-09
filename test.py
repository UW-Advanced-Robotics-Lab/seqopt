import cProfile
import seqopt.environments
import gym
import time

env = gym.make('kitchen_relax-v1')
_ = env.reset()

def tic_toc(env, steps=1000):
    s = time.time()
    for _ in range(steps):
        act = env.action_space.sample()
        _ = env.step(act)
    return time.time() - s

print(f"Total time: {tic_toc(env)}")
#cProfile.run("env.step(env.action_space.sample())")
