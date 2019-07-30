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
    """ input: (T,) tensor of reward values received at each timestep.
        output: (T,) tensor of rewards to go Q, where 
            Q[i] = sum_j=i^T gamma^(j-i) r[j]
            and gamma is the time-discounting factor.
        """
    Q = torch.zeros_like(rewards)
    Q.copy_(rewards)
    T = len(Q)
    running_sum = 0
    for i in reversed(range(T)):
        Q[i] = Q[i] + discount * running_sum
        running_sum = Q[i]
    return Q   

def effective_cost_function(log_probs, rewards_to_go, states, 
                                        value_model = None,
                                        external_baseline=None,
                                        baseline=None,
                                                        ):
    """ Computes a scalar torch tensor whose gradient is an estimator of the expected-return cost function
        log_probs: (T,) tensor of log-probabilities
        rewards_to_go: (T,) tensor of future rewards from each action in trajectory
        states: (T,d) tensor of states immediately prior to rewards
        baseline: if not None, string specifying the type of baseline to apply
            if 'value_model': value_model should take (T,d) tensor of states, return (T,) tensor of values
            if 'external': external_baseline should hold (T,) tensor of baseline values to subtract.
        returns: (J_baseline, J_no_baseline), a tuple of scalar tensors holding the baselined and non-baselined
        cost functions respectively.
        """
    J_no_baseline = - (rewards_to_go * log_probs).sum()

    if baseline is not None:
        if (baseline not in ['external', 'value_model']):
            raise NotImplementedError

        elif baseline == 'value_model':
            if value_model is None:
                raise ValueError("Please supply value model")
            baselinefn = lambda s: value_model(s)

        elif baseline == 'external':
            if external_baseline is None:
                raise ValueError("Please supply external baseline")
            baselinefn = lambda s: external_baseline

        rewards_to_go = rewards_to_go - baselinefn(states)
    J_baseline = - (rewards_to_go * log_probs).sum()
    return J_baseline, J_no_baseline
    
def do_vpg_training(policy, env, max_episode_timesteps, 
                    optimizer, batch_size, num_batches, 
                        baseline=None, value_modelstepper=None, discount=1.0, verbose=True):
    """ Run vanilla policy-grad training on the given policy network and environment.

        Parameters:
        policy: torch model for the stochastic policy
        env: openai gym environment
        max_episode_timesteps: max number of timesteps to run an episode
        optimizer: torch optimizer for policy parameters.
        batch_size: how many episodes to batch together when performing policy updates
        num_batches: how many batch updates to perform before halting training."""
    
    BASELINE_TYPES = ['running_average_Q', 'value_model']

    if baseline is not None and baseline not in BASELINE_TYPES:
        raise ValueError("Allowed baseline types: ", BASELINE_TYPES)
    if baseline == 'value_model' and value_modelstepper is None:
        raise ValueError("Please supply a value ModelStepper")

    avg_returns = []
    try:
        # running average of the reward-to-go
        running_average_Q = 0

        for ib in range(num_batches):
            #collect cost function components with baseline...
            batch_costfn = []
            # ... and without
            batch_costfn_nb = []

            #also keep track of state trajectories and corresponding Q-values
            batch_states = []
            batch_Q = []
            # and rewards, to see model progress
            batch_returns = []
            for i in range(batch_size):
                # run a single episode
                states, actions, rewards, log_probs = do_episode(policy, env, 
                                                max_timesteps=max_episode_timesteps, stop_on_done=True)
                #Q-values for the episode
                rewards_to_go = compute_rewards_to_go(rewards, discount=discount)
                
                #obtain episode's contribution to the gradient
                if baseline=='running_average_Q':
                    baseline_type = 'external'
                    external_baseline = running_average_Q * torch.ones(len(rewards_to_go))
                    value_model = None
                    running_average_Q = .9 * running_average_Q + .1 * rewards_to_go.mean().numpy()

                elif baseline == 'value_model':
                    baseline_type=baseline
                    external_baseline = None
                    value_model = value_modelstepper.model
                
                elif baseline is None:
                    baseline_type = baseline
                    external_baseline = None
                    value_model = None

                J, J_nb = effective_cost_function(log_probs, rewards_to_go, states, 
                                                        baseline=baseline_type,
                                                        external_baseline=external_baseline, 
                                                        value_model=value_model)

                batch_costfn.append(J)
                batch_costfn_nb.append(J_nb)    
                batch_states.append(states[:-1])
                batch_Q.append(rewards_to_go)
                batch_returns.append(rewards.sum())

            #after a few trajectories have been collected, update policy parameters, and value model if 
            # appropriate
            if baseline == 'value_model':
                #update value function model
                states_all = torch.cat(batch_states)            
                Q_all = torch.cat(batch_Q)
                value_modelstepper.step(states_all, Q_all.view(-1,1))

            loss = torch.stack(batch_costfn).mean()
            loss_nb = torch.stack(batch_costfn_nb).mean()
            batch_returns = torch.stack(batch_returns)
            avg_return = batch_returns.mean()
            avg_returns.append(avg_return)

            if verbose:
                print("Avg return for batch {0}: {1:.3f}".format(ib, avg_return))
                if baseline == 'value_model':
                    print("Value model loss fn: {0:.3f}".format(value_modelstepper.losses[-1]))
            
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

