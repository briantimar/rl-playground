import gym
import numpy as np

#cart-pole environment
env = gym.make('CartPole-v0')
env.reset()
initact=0
act=initact
for _ in range(1000):
    env.render()
    # env.step(env.action_space.sample())
    # obs = position, velocity, angle, angular velocity
    obs, reward, done, info = env.step(act)
    act = obs[2] >0
env.close()