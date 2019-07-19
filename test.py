import torch
import numpy as np
import unittest

class testVPG(unittest.TestCase):

    def test_compute_reward_to_go(self):
        from tools import compute_rewards_to_go
        rewards = torch.tensor([[1, 0, 1, 1], 
                                [0, 0, 1, 1]],dtype=torch.float)
        Qtarget = torch.tensor([[3, 2, 2, 1], 
                                [2, 2, 2, 1]],dtype=torch.float)
        Q = compute_rewards_to_go(rewards)
        self.assertAlmostEqual( (Q - Qtarget).abs().sum().numpy(), 0)


if __name__ == '__main__':
    unittest.main()