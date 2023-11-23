#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
import random
l1 = [random.randint(x,100) for x in range(0,100)]
l2 = [random.randint(x,100) for x in range(0,100)]
plt.scatter(l1,l2)


# In[2]:


l3 = [x for x in range(0,12)]
l4 = [x+12 for x in range(0,12)]

l5 = [abs(x-12) for x in range(0,12)]


# In[3]:


plt.subplot(1,2,1)
plt.scatter(l3,l4)
plt.subplot(1,2,2)
plt.scatter(l3,l5)


# In[4]:


import numpy as np

arr1 = np.array(l1)
arr2 = np.array(l2)

rel = np.corrcoef(arr1,arr2)

print(rel)
plt.plot(rel)


# In[5]:


arr1 = np.array(np.random.randint((1000), size=(500)))
arr2 = np.array(np.random.choice((arr1*57), size=(500)))
arr3 = np.array(np.random.choice((-(arr1*75)), size=(500)))
arr4 = np.array(np.random.randint((1000), size=(500)))
plt.figure(figsize=(10,7))
plt.subplot(1,2,1)
plt.scatter(arr1,arr2)
plt.subplot(1,2,2)
plt.scatter(arr1,arr3)


# In[ ]:




