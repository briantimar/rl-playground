import numpy as np
import gym
import torch

def do_episode(policy, env, max_timesteps, stop_on_done=True, render=False):
    """ Run one episode.
        policy: a stochastic policy model. Given current states as inputs, outputs logits which define probabilities of various actions.
        env: openai gym environment.
        max_timesteps: max number of timesteps the environment is allowed to run (ie one episode)
        stop_on_done: bool, whether to stop when the environment is 'done'.
        render: bool, whether to render the environment.
        Returns: states, actions, rewards,  log_probs
        """
    #sample initial state from the environment
    obs = torch.tensor(env.reset(),dtype=torch.float32)
    state_trajectory = [obs]
    rewards = []
    action_trajectory = []
    log_probs = []

    for t in range(max_timesteps):
        if render:
            env.render()
        #sample action from policy
        action, log_prob = policy.sample_action_with_log_prob(obs)
        action_trajectory.append(action)
        log_probs.append(log_prob)
        #update the environment 
        obs, reward, done, __ = env.step(action.numpy())
        obs = torch.tensor(obs,dtype=torch.float32)
        state_trajectory.append(obs)
        rewards.append(reward)

        if done and stop_on_done:
            break
        
    state_trajectory = torch.stack(state_trajectory)
    action_trajectory = torch.stack(action_trajectory)
    rewards = torch.tensor(rewards)
    log_probs = torch.stack(log_probs)

    return state_trajectory, action_trajectory, rewards, log_probs

def compute_rewards_to_go(rewards, discount=1.0):
    """ input: (N, T) tensor of reward values received at each timestep.
        output: (N, T) tensor of rewards to go Q, where 
            Q[i] = sum_j=i^T gamma^j r[j]
            and gamma is the time-discounting factor.
        """
    Q = torch.zeros_like(rewards)
    running_sum = torch.zeros(rewards.size(0))
    for i in reversed(range(Q.size(1))):
        running_sum += rewards[:,i] * discount**i
        Q[:,i] = running_sum
    return Q   


def effective_cost_function(log_probs, rewards, states, baseline=None):
    """ Computes a scalar torch tensor whose gradient is an estimator of the expected-return cost function
        log_probs: (N,T) tensor of log-probabilities
        rewards: (N,T) tensor of rewards
        states: (N, T) tensor of states immediately prior to rewards
        baseline: if not None, a function of the state which is subtracted from the reward-to-go. Should return
        tensor of the same shape as states.
        """
    reward_to_go = compute_rewards_to_go(rewards)
    if baseline is not None:
        reward_to_go = reward_to_go - baseline(states)
    return - (reward_to_go * log_probs).mean()
    
def do_vpg_training(policy, env, max_episode_timesteps, 
                    optimizer, batch_size, num_batches, verbose=True):
    """ Run vanilla policy-grad training on the given policy network and environment.
        policy: torch model for the stochastic policy
        env: openai gym environment
        max_episode_timesteps: max number of timesteps to run an episode
        optimizer: torch optimizer for policy parameters.
        batch_size: how many episodes to batch together when performing policy updates
        num_batches: how many batch updates to perform before halting training."""
    
    avg_returns = []
    try:
        for ib in range(num_batches):
            batch_rewards = []
            batch_log_probs = []
            batch_states = []
            for i in range(batch_size):
                # run a single episode
                states, actions, rewards, log_probs = do_episode(policy, env, 
                                                max_timesteps=max_episode_timesteps, stop_on_done=False)
                batch_rewards.append(rewards)
                batch_log_probs.append(log_probs)
                batch_states.append(states[:-1])
    
            batch_rewards = torch.stack(batch_rewards)
            batch_log_probs = torch.stack(batch_log_probs)
            batch_states = torch.stack(batch_states)

            loss = effective_cost_function(batch_log_probs, batch_rewards, batch_states,  
                                                            baseline=None)
            
            avg_return = batch_rewards.mean(dim=0).sum().numpy()
            if verbose:
                print("Avg return for batch {0}: {1:.3f}".format(ib, avg_return))
            avg_returns.append(avg_return)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    except KeyboardInterrupt:
        print("Halting training early!")

    return avg_returns

def get_sample_trajectory(policy, env, halt_on_done=False, max_timesteps=200):
    """ Sample an action trajectory from a particular policy model. """

    obs = env.reset()
    done = False
    actions = []
    rewards = []
    t = 0
    while (not halt_on_done or not done) and t < max_timesteps:
        obs = torch.tensor(obs, dtype=torch.float32)
        action, __ = policy.sample_action_with_log_prob(obs)
        action = action.numpy()
        actions.append(action)
        obs, reward, done, __ = env.step(action)
        rewards.append(reward)
        t += 1
    return actions, rewards


if __name__ == '__main__':
    from models import MLP
    policy = MLP([4,10,2])
    env = gym.make('CartPole-v0')
    max_timesteps=200
    batch_size = 100
    num_batches = 20
    optimizer = torch.optim.Adam(policy.parameters(),lr=.1)

    avg_returns = do_vpg_training(policy, env, max_timesteps, 
                                optimizer=optimizer,batch_size=batch_size, num_batches=num_batches)
    traj, rewards = get_sample_trajectory(policy, env)
    render_trajectory(env, traj)