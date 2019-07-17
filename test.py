import gym
import numpy as np

#cart-pole environment
env = gym.make('CartPole-v0')
env.reset()

rewards = []
observations = []
for _ in range(1000):
    env.render()
    # obs = position, velocity, angle, angular velocity
    obs, reward, done, info = env.step(env.action_space.sample())
    rewards.append(reward)
    observations.append(obs)
    
env.close()