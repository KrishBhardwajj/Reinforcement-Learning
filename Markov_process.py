#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class MarkovProcess:
    def __init__(self, transition_matrix, initial_state):
        """
        Initialize the Markov process.

        Parameters:
        - transition_matrix: a 2D NumPy array representing the transition probabilities.
        - initial_state: the initial state of the process.
        """
        self.transition_matrix = transition_matrix
        self.current_state = initial_state

    def step(self):
        """
        Move to the next state based on the transition probabilities.
        """
        probabilities = self.transition_matrix[self.current_state, :]
        new_state = np.random.choice(len(probabilities), p=probabilities)
        self.current_state = new_state

    def simulate(self, num_steps):
        """
        Simulate the Markov process for a specified number of steps.

        Parameters:
        - num_steps: the number of steps to simulate.
        """
        trajectory = [self.current_state]
        for _ in range(num_steps - 1):
            self.step()
            trajectory.append(self.current_state)
        return trajectory

# Example usage
transition_matrix = np.array([[0.7, 0.3], [0.2, 0.8]])  # Example transition matrix for a 2-state process
initial_state = 0
num_steps = 10

markov_process = MarkovProcess(transition_matrix, initial_state)
trajectory = markov_process.simulate(num_steps)

print("Simulated trajectory:", trajectory)


# In[ ]:




