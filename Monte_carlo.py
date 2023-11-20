#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def get_card_value():
    """Get a random card value between 1 and 11."""
    return np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])

def get_initial_state():
    """Get the initial state of the Blackjack game (player's sum, dealer's face-up card, and whether the player has a usable ace)."""
    return (get_card_value(), get_card_value(), np.random.choice([True, False]))

def get_action():
    """Get a random action: 0 for 'stick' and 1 for 'hit'."""
    return np.random.choice([0, 1])

def play_game(policy):
    """
    Play one episode of the Blackjack game.

    Parameters:
    - policy: A function that takes the current state as input and returns the action to take.

    Returns:
    - List of tuples (state, action, reward).
    """
    states_actions_rewards = []
    player_sum, dealer_card, usable_ace = get_initial_state()

    # Player's turn
    while player_sum < 12:  # Always 'hit' if player's sum is less than 12
        action = 1
        next_card = get_card_value()
        player_sum += next_card
        if next_card == 1:
            usable_ace = True

    while action == 1 and player_sum < 21:
        action = policy(player_sum, dealer_card, usable_ace)
        if action == 1:
            next_card = get_card_value()
            player_sum += next_card
            if next_card == 1:
                usable_ace = True

    # Dealer's turn
    dealer_sum = dealer_card
    while dealer_sum < 17:
        next_card = get_card_value()
        dealer_sum += next_card
        if next_card == 1 and dealer_sum <= 11:
            dealer_sum += 10  # Convert Ace to 11

    # Determine the winner and calculate rewards
    if player_sum > 21 or (dealer_sum <= 21 and dealer_sum >= player_sum):
        reward = -1
    elif player_sum == dealer_sum:
        reward = 0
    else:
        reward = 1

    states_actions_rewards.append(((player_sum, dealer_card, usable_ace), action, reward))
    return states_actions_rewards

def monte_carlo_policy(player_sum, dealer_card, usable_ace):
    """
    A simple policy for the Monte Carlo method: 'stick' if player's sum is 20 or 21, otherwise 'hit'.
    """
    return 0 if player_sum >= 20 else 1

def update_q_values(q_values, returns, states_actions_rewards):
    """Update Q-values using the Monte Carlo return."""
    for state, action, reward in states_actions_rewards:
        if state not in returns:
            returns[state] = []
        returns[state].append(reward)
        q_values[state][action] = np.mean(returns[state])
    return q_values

def monte_carlo_simulation(num_episodes):
    """Perform Monte Carlo simulation to estimate Q-values."""
    q_values = {}
    returns = {}

    for _ in range(num_episodes):
        states_actions_rewards = play_game(monte_carlo_policy)
        q_values = update_q_values(q_values, returns, states_actions_rewards)

    return q_values

# Example usage:
num_episodes = 100000
estimated_q_values = monte_carlo_simulation(num_episodes)

# Display the estimated Q-values for some states
print("Estimated Q-values:")
for state, values in estimated_q_values.items():
    print(f"State: {state}, Q-values: {values}")


# In[ ]:




