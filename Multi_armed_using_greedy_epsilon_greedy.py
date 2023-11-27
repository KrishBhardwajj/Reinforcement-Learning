#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.true_means = np.random.normal(0, 1, num_arms)
        self.action_values = np.zeros(num_arms)
        self.action_counts = np.zeros(num_arms)

    def pull_arm(self, arm):
        reward = np.random.normal(self.true_means[arm], 1)
        self.action_values[arm] = (
            (self.action_values[arm] * self.action_counts[arm] + reward) /
            (self.action_counts[arm] + 1)
        )
        self.action_counts[arm] += 1
        return reward

def explore_bandit(bandit, num_steps):
    rewards = []
    for _ in range(num_steps):
        arm = np.random.randint(bandit.num_arms)
        reward = bandit.pull_arm(arm)
        rewards.append(reward)
    return rewards

def greedy_bandit(bandit, num_steps):
    rewards = []
    for _ in range(num_steps):
        arm = np.argmax(bandit.action_values)
        reward = bandit.pull_arm(arm)
        rewards.append(reward)
    return rewards

def epsilon_greedy_bandit(bandit, num_steps, epsilon):
    rewards = []
    for _ in range(num_steps):
        if np.random.rand() < epsilon:
            arm = np.random.randint(bandit.num_arms)
        else:
            arm = np.argmax(bandit.action_values)
        reward = bandit.pull_arm(arm)
        rewards.append(reward)
    return rewards

# Example usage:
num_arms = 3
num_steps = 1000
epsilon = 0.1

bandit = Bandit(num_arms)

explore_rewards = explore_bandit(bandit, num_steps)
greedy_rewards = greedy_bandit(bandit, num_steps)
epsilon_greedy_rewards = epsilon_greedy_bandit(bandit, num_steps, epsilon)

# Plot results
plt.plot(np.cumsum(explore_rewards), label='Exploration')
plt.plot(np.cumsum(greedy_rewards), label='Greedy')
plt.plot(np.cumsum(epsilon_greedy_rewards), label='Epsilon-Greedy')
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Total Reward')
plt.title('Multi-Armed Bandit Strategies')
plt.show()


# In[ ]:




