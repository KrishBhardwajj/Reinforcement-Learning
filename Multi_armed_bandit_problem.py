#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

class MultiArmedBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.true_rewards = np.random.normal(0, 1, num_arms)  # True reward distribution for each arm
        self.estimated_values = np.zeros(num_arms)  # Estimated values of each arm
        self.action_counts = np.zeros(num_arms)  # Number of times each arm has been pulled

    def choose_action(self, epsilon):
        if np.random.rand() < epsilon:
            # Explore: choose a random arm
            action = np.random.randint(self.num_arms)
        else:
            # Exploit: choose the arm with the highest estimated value
            action = np.argmax(self.estimated_values)
        return action

    def update_values(self, action, reward):
        # Update estimated value for the chosen arm using incremental formula
        self.action_counts[action] += 1
        self.estimated_values[action] += (reward - self.estimated_values[action]) / self.action_counts[action]

def run_bandit(num_arms, num_steps, epsilon):
    bandit = MultiArmedBandit(num_arms)
    rewards = []

    for step in range(num_steps):
        action = bandit.choose_action(epsilon)
        reward = np.random.normal(bandit.true_rewards[action], 1)  # Simulate pulling the chosen arm
        bandit.update_values(action, reward)
        rewards.append(reward)

    return rewards

# Example usage
num_arms = 5
num_steps = 1000
epsilon = 0.1

rewards = run_bandit(num_arms, num_steps, epsilon)

# You can then analyze the rewards to see how well the algorithm performed
average_reward = np.mean(rewards)
print(f"Average reward: {average_reward}")


# In[ ]:




