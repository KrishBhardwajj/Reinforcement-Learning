#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class QLearning:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        # Initialize Q-table with zeros
        self.Q = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        # Epsilon-greedy policy for action selection
        if np.random.rand() < self.exploration_prob:
            # Explore: choose a random action
            action = np.random.choice(self.num_actions)
        else:
            # Exploit: choose the action with the highest Q-value
            action = np.argmax(self.Q[state, :])
        return action

    def update_q_value(self, state, action, reward, next_state):
        # Q-value update using the Q-learning update rule
        self.Q[state, action] = (1 - self.learning_rate) * self.Q[state, action] + \
                               self.learning_rate * (reward + self.discount_factor * np.max(self.Q[next_state, :]))

def run_q_learning(num_states, num_actions, num_episodes):
    q_learning_agent = QLearning(num_states, num_actions)

    for episode in range(num_episodes):
        # Reset the environment to the initial state
        state = 0  # Assuming the initial state is 0

        while state != num_states - 1:  # Continue until reaching the terminal state
            # Choose an action using the epsilon-greedy policy
            action = q_learning_agent.choose_action(state)

            # Simulate the chosen action and observe the next state and reward
            next_state = state + 1  # Moving to the right
            reward = 0 if next_state != num_states - 1 else 1  # Reward 1 upon reaching the terminal state

            # Update the Q-value based on the observed transition
            q_learning_agent.update_q_value(state, action, reward, next_state)

            # Move to the next state
            state = next_state

    return q_learning_agent.Q

# Example usage
num_states = 5
num_actions = 2
num_episodes = 1000

final_q_table = run_q_learning(num_states, num_actions, num_episodes)
print("Final Q-table:")
print(final_q_table)


# In[ ]:




