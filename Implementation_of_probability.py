#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.stats
import statistics

# Example 1: Probability Distribution and Cumulative Distribution Function using scipy.stats

# Define a normal distribution with mean=0 and standard deviation=1
normal_distribution = scipy.stats.norm(0, 1)

# Calculate the probability density function (PDF) at a specific point (e.g., x=1)
pdf_at_1 = normal_distribution.pdf(1)
print(f"PDF at x=1: {pdf_at_1}")

# Calculate the cumulative distribution function (CDF) at a specific point (e.g., x=1)
cdf_at_1 = normal_distribution.cdf(1)
print(f"CDF at x=1: {cdf_at_1}")

# Example 2: Generating Random Samples from a Distribution using scipy.stats

# Generate 5 random samples from the normal distribution
random_samples = normal_distribution.rvs(size=5)
print(f"Random Samples: {random_samples}")

# Example 3: Basic Probability Calculations using statistics module

# Calculate the mean of a list of values
data = [1, 2, 3, 4, 5]
mean_value = statistics.mean(data)
print(f"Mean: {mean_value}")

# Calculate the standard deviation of a list of values
std_deviation = statistics.stdev(data)
print(f"Standard Deviation: {std_deviation}")

# Calculate the probability of an event in a discrete sample space
sample_space = [1, 2, 3, 4, 5, 6]
event = [1, 3, 5]
probability_event = len(event) / len(sample_space)
print(f"Probability of the event: {probability_event}")


# In[ ]:




