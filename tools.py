import numpy as np
import gym
import torch

def do_episode(policy, env, max_timesteps):
    """ Run one episode.
        policy: a stochastic policy model. Given current states as inputs, outputs logits which define probabilities of various actions.
        env: openai gym environment.
        max_timesteps: max number of timesteps the environment is allowed to run (ie one episode)

        Returns: states, actions, rewards,  log_probs
        """
    #sample initial state from the environment
    obs = torch.tensor(env.reset(),dtype=torch.float32)
    state_trajectory = [obs]
    rewards = []
    action_trajectory = []
    log_probs = []

    for t in range(max_timesteps):
        #sample action from policy
        action, log_prob = policy.sample_action_with_log_prob(obs)
        action_trajectory.append(action)
        log_probs.append(log_prob)
        #update the environment 
        obs, reward, done, __ = env.step(action.numpy())
        obs = torch.tensor(obs,dtype=torch.float32)
        state_trajectory.append(obs)
        rewards.append(reward)
        
    state_trajectory = torch.stack(state_trajectory)
    action_trajectory = torch.stack(action_trajectory)
    rewards = torch.tensor(rewards)
    log_probs = torch.stack(log_probs)

    return state_trajectory, action_trajectory, rewards, log_probs

def render_trajectory(env, actions):
    """Render environment under list of (numpy) actions"""
    __ = env.reset()
    env.render()
    for a in actions:
        env.step(a)
        env.render()
    return env

if __name__ == '__main__':
    from models import MLP
    policy = MLP([4,10,2])
    env = gym.make('CartPole-v0')
    max_timesteps=100
    states, actions, rewards, lps = do_episode(policy, env, max_timesteps)
    render_trajectory(env, actions.numpy())