#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self, true_probs):
        self.true_probs = true_probs
        self.num_arms = len(true_probs)

    def pull_arm(self, arm):
        return np.random.binomial(1, self.true_probs[arm])

def thompson_sampling(num_arms, num_rounds, true_probs):
    bandit = BernoulliBandit(true_probs)
    num_wins = np.zeros(num_arms)
    num_trials = np.zeros(num_arms)
    rewards = []

    for _ in range(num_rounds):
        sampled_theta = [np.random.beta(num_wins[i] + 1, num_trials[i] - num_wins[i] + 1) for i in range(num_arms)]
        chosen_arm = np.argmax(sampled_theta)
        reward = bandit.pull_arm(chosen_arm)
        
        num_wins[chosen_arm] += reward
        num_trials[chosen_arm] += 1
        rewards.append(reward)

    return rewards

def main():
    np.random.seed(42)

    # Number of arms in the bandit
    num_arms = 3

    # True probabilities of success for each arm
    true_probs = [0.3, 0.5, 0.7]

    # Number of rounds to simulate
    num_rounds = 1000

    # Run Thompson Sampling algorithm
    rewards = thompson_sampling(num_arms, num_rounds, true_probs)

   


# In[ ]:




