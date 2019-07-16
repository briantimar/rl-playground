import numpy as np
import gym

env = gym.make('CartPole-v0')
env.reset()

action = 1
for __ in range(100):
    obs, reward, done, info = env.step(action)
    env.render()