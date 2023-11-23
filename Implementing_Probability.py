#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,4,5])
print(np.corrcoef(x,y))
plt.scatter(x,y)


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3,4,5])
y=np.array([0,0,0,0,0])
print(np.corrcoef(x,y))
plt.scatter(x,y)


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3,4,5])
y=np.array([-1,-2,-3,-4,-5])
print(np.corrcoef(x,y))
plt.scatter(x,y)


# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data={"states":["Ohio","Ohio","Ohio","Nevada","Nevada","Nevada"],
     "year":[2000,2001,2002,2001,2002,2003],
     "pop":[1.5,1.7,3.6,2.4,2.9,3.2],
     "sex":["M","M","F","F","F","M"],
     "income":[2000,8000,1000,9000,6000,2000],
     "age":[35,98,65,51,34,11]}
df=pd.DataFrame(data)
print(df)


# In[10]:


pd.plotting.scatter_matrix(df, figsize=(10, 10))
plt.show()





# In[ ]:




