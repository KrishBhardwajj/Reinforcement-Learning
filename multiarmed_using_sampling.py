#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

class ThompsonSamplingBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.true_rewards = np.random.normal(0, 1, num_arms)  # True reward distribution for each arm
        self.alpha = np.ones(num_arms)  # Initialize success counts for each arm
        self.beta = np.ones(num_arms)  # Initialize failure counts for each arm

    def choose_action(self):
        # Sample a random reward for each arm from the Beta distribution
        sampled_rewards = np.random.beta(self.alpha, self.beta)
        # Choose the arm with the highest sampled reward
        action = np.argmax(sampled_rewards)
        return action

    def update_values(self, action, reward):
        # Update success or failure counts based on the observed reward
        if reward == 1:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1

def run_thompson_sampling_bandit(num_arms, num_steps):
    bandit = ThompsonSamplingBandit(num_arms)
    rewards = []

    for step in range(num_steps):
        action = bandit.choose_action()
        reward = np.random.binomial(1, bandit.true_rewards[action])  # Simulate pulling the chosen arm
        bandit.update_values(action, reward)
        rewards.append(reward)

    return rewards

# Example usage
num_arms = 5
num_steps = 1000

rewards = run_thompson_sampling_bandit(num_arms, num_steps)

# You can then analyze the rewards to see how well the algorithm performed
average_reward = np.mean(rewards)
print(f"Average reward: {average_reward}")

# Plot cumulative rewards over time
cumulative_rewards = np.cumsum(rewards)
plt.plot(range(1, num_steps + 1), cumulative_rewards / np.arange(1, num_steps + 1))
plt.xlabel('Steps')
plt.ylabel('Average Cumulative Reward')
plt.title('Thompson Sampling Bandit')
plt.show()


# In[ ]:




