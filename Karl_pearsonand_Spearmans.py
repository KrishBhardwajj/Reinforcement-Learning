#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy.stats import pearsonr, spearmanr

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 4, 6]

# Karl Pearson's correlation coefficient
pearson_corr, _ = pearsonr(x, y)
print(f"Karl Pearson's correlation coefficient: {pearson_corr}")

# Spearman's rank correlation coefficient
spearman_corr, _ = spearmanr(x, y)
print(f"Spearman's rank correlation coefficient: {spearman_corr}")

pearson_corr, pearson_p_value = pearsonr(x, y)
spearman_corr, spearman_p_value = spearmanr(x, y)
print(f"Karl Pearson's correlation coefficient: {pearson_corr}, p-value: {pearson_p_value}")
print(f"Spearman's rank correlation coefficient: {spearman_corr}, p-value: {spearman_p_value}")



# In[ ]:




