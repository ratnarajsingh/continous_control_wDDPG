"""
Using a well known noise function directly, source referenced
For each action, it generates a random number within a sigma distance centered around mu. This small change is added to the actions in DDPG class
"""

# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------


import numpy as np
import numpy.random as nr

class OUNoise:
    """docstring for OUNoise"""
    def __init__(self,n_actions,seed,mu=0, theta=0.15, sigma=0.3):
        self.action_dim = n_actions
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state