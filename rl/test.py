import torch
import numpy as np
import unittest


class testTools(unittest.TestCase):

    def test_do_episode(self):
        from .tools import do_episode       
        from .models import MLP 
        import gym
        
        env = gym.make('CartPole-v0')
        max_timesteps=2
        
        policy = MLP([4,5,2], activation=torch.relu, output_critic=False)
        states, actions, rewards, logprobs, critics = do_episode(policy, env, max_timesteps=max_timesteps,
                                                                    stop_on_done=False)
        self.assertEqual(tuple(states.shape), (max_timesteps+1, 4))
        for s in [actions, rewards, logprobs]:
            self.assertEqual(tuple(s.shape), (max_timesteps,))

        policy = MLP([4,5,3], activation=torch.relu, output_critic=True)
        states, actions, rewards, logprobs, critics = do_episode(policy, env, max_timesteps=max_timesteps,
                                                                    stop_on_done=False)
        self.assertEqual(tuple(critics.shape), (max_timesteps,))

class testVPG(unittest.TestCase):

    def test_compute_reward_to_go(self):
        from .tools import compute_rewards_to_go
        rewards = torch.tensor([1, 0, 1, 1], 
                                dtype=torch.float)
        Qtarget = torch.tensor([3, 2, 2, 1], 
                                dtype=torch.float)
        Q = compute_rewards_to_go(rewards)
        gamma = .5
        Q_discount = compute_rewards_to_go(rewards, discount=gamma)
        Qtarget_discount = torch.tensor([1 + .5*.5*1.5, .5 * 1.5, 1.5, 1], 
                                dtype=torch.float)
        self.assertAlmostEqual( (Q - Qtarget).abs().sum().numpy(), 0)
        self.assertAlmostEqual( (Q_discount - Qtarget_discount).abs().sum().numpy(), 0 )

    def test_effective_cost_function(self):
        from .tools import effective_cost_function
        lps = torch.ones(2) * np.log(.5)
        rewards_to_go = torch.tensor([2.,1.])
        states = torch.ones(2)
        external_baseline = torch.tensor([1.,1.])
        J, J_nb = effective_cost_function(lps, rewards_to_go,states,baseline='external',
                                            external_baseline=external_baseline)
        self.assertAlmostEqual(J.detach().numpy(), np.log(2))
        self.assertAlmostEqual(J_nb.detach().numpy(), 3 * np.log(2))


class testModels(unittest.TestCase):

    def test_policy(self):
        from .models import MLP
        pol1 = MLP([1, 10, 2], output_critic=False)
        x = torch.ones(5,1)
        action, logprobs = pol1.sample_action_with_log_prob(x)
        self.assertEqual(tuple(action.shape), (5,))
        self.assertEqual(tuple(logprobs.shape), (5,))

        pol2 = MLP([1, 10, 3], output_critic=True)
        action, logprobs, critic = pol2.sample_action_with_log_prob(x)
        for output in [action, logprobs, critic]:
            self.assertEqual(tuple(output.shape), (5,))

        x2 = torch.ones(1)
        action, logprobs = pol1.sample_action_with_log_prob(x2)
        self.assertEqual(tuple(action.shape), ())
        action, logprobs, critic = pol2.sample_action_with_log_prob(x2)
        self.assertEqual(tuple(action.shape), ())

    def test_model_stepper(self):
        from .models import ModelStepper, MLP
        from .models import HyperParams
        from torch.nn import MSELoss
        hp = HyperParams(lr=.1, layer_sizes=[2, 1], epochs=50)
        x = torch.ones((1, 2))
        y = torch.ones((1, 1))

        ms = ModelStepper(MLP, lossfn=MSELoss(), hyperparams=hp,
                                optimizer=torch.optim.SGD)
        
        for ep in range(hp.epochs):
            ms.step(x, y)
        yout = ms.eval(x)
        self.assertAlmostEqual(yout.detach().item(), y.item(), places=4)
        
    def test_policy_family(self):
        from .models import MLP, PolicyFamily
        pol1 = MLP([1, 4, 2])
        pol2 = MLP([1, 5, 2])
        pf = PolicyFamily([pol1, pol2])
        action, __ = pf.sample_action_with_log_prob(torch.ones(1))
        a1, __ = pol1.sample_action_with_log_prob(torch.ones(1))
        self.assertEqual(action.shape, a1.shape)

if __name__ == '__main__':
    unittest.main()