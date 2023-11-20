#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class GridWorld:
    def __init__(self, size):
        self.size = size
        self.states = [(i, j) for i in range(size) for j in range(size)]
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.num_actions = len(self.actions)

    def is_valid_state(self, state):
        return state in self.states

    def is_terminal_state(self, state):
        return state == (0, 0) or state == (self.size - 1, self.size - 1)

    def take_action(self, state, action):
        i, j = state
        if action == "UP":
            i = max(i - 1, 0)
        elif action == "DOWN":
            i = min(i + 1, self.size - 1)
        elif action == "LEFT":
            j = max(j - 1, 0)
        elif action == "RIGHT":
            j = min(j + 1, self.size - 1)
        return i, j

def policy_evaluation(grid_world, policy, gamma, theta):
    size = grid_world.size
    V = np.zeros((size, size))

    while True:
        delta = 0
        for i in range(size):
            for j in range(size):
                if not grid_world.is_terminal_state((i, j)):
                    old_v = V[i, j]
                    action = policy[i, j]
                    next_i, next_j = grid_world.take_action((i, j), action)
                    reward = -1  # Constant reward for each step
                    V[i, j] = reward + gamma * V[next_i, next_j]
                    delta = max(delta, np.abs(old_v - V[i, j]))

        if delta < theta:
            break

    return V

def policy_improvement(grid_world, V, gamma):
    size = grid_world.size
    policy = np.zeros((size, size), dtype=np.int)

    for i in range(size):
        for j in range(size):
            if not grid_world.is_terminal_state((i, j)):
                action_values = []
                for action in grid_world.actions:
                    next_i, next_j = grid_world.take_action((i, j), action)
                    reward = -1  # Constant reward for each step
                    action_values.append(reward + gamma * V[next_i, next_j])
                best_action = np.argmax(action_values)
                policy[i, j] = best_action

    return policy

def policy_iteration(grid_world, gamma, theta):
    size = grid_world.size
    policy = np.random.randint(0, grid_world.num_actions, size=(size, size))
    policy_stable = False

    while not policy_stable:
        V = policy_evaluation(grid_world, policy, gamma, theta)
        new_policy = policy_improvement(grid_world, V, gamma)
        policy_stable = np.array_equal(policy, new_policy)
        policy = new_policy

    return policy

# Example usage:
grid_size = 4
gamma = 0.9
theta = 0.001

grid_world = GridWorld(grid_size)
optimal_policy = policy_iteration(grid_world, gamma, theta)

print("Optimal Policy:")
print(optimal_policy)


# In[ ]:




