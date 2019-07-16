import numpy as np
import gym

def do_episode(policy, env, max_timesteps):
    """ Run one episode.
        policy: a stochastic policy model. Given current states as inputs, outputs logits which define probabilities of various actions.
        env: openai gym environment.
        max_timesteps: max number of timesteps the environment is allowed to run (ie one episode)

        Returns: states, actions, rewards,  log_probs
        """
    #sample initial state from the environment
    obs, reward, done, __ = env.reset()
    state_trajectory = [obs]
    rewards = []
    action_trajectory = []
    log_probs = []

    for t in range(max_timesteps):
        #sample action from policy
        action, log_prob = policy.sample_action_with_logprob(obs)
        action_trajectory.append(action)
        log_probs.append(log_prob)
        #update the environment 
        obs, reward, done, __ = env.step(action)
        state_trajectory.append(obs)
        rewards.append(reward)
        

    return state_trajectory, action_trajectory, rewards, log_probs


