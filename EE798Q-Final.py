#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


originaldata=pd.read_csv("https://raw.githubusercontent.com/jetharam171/EE798Q/main/Open%20pit%20blasting%2001-02-2023%20000000%20To%2001-05-2023%20235959.csv")


# In[24]:


originaldata.head()


# In[25]:


originaldata.tail()


# In[26]:


originaldata.shape


# In[27]:


originaldata.info()


# In[28]:


originaldata.dtypes


# In[29]:


originaldata.columns


# In[30]:


originaldata.isnull().sum()


# ### After interpolation for blank space

# In[31]:


df=pd.read_csv("https://raw.githubusercontent.com/jetharam171/EE798Q/main/Open%20pit%20blasting%20without%20any%20data%20loss.csv")


# In[32]:


df.isnull().sum()


# In[33]:


df.describe()


# In[34]:


sns.boxplot( y="Singrauli, Surya Kiran Bhawan Dudhichua  PM10 (µg/m3)",data=df)


# In[35]:


sns.boxplot( y="Singrauli, Surya Kiran Bhawan Dudhichua  PM2.5 (µg/m3)",data=df)


# In[36]:


sns.boxplot( y="Singrauli, Surya Kiran Bhawan Dudhichua  NO (µg/m3)",data=df)


# In[37]:


sns.boxplot( y="Singrauli, Surya Kiran Bhawan Dudhichua  NO2 (µg/m3)",data=df)


# In[38]:


sns.boxplot( y="Singrauli, Surya Kiran Bhawan Dudhichua  NOX (ppb)",data=df)


# In[39]:


sns.boxplot( y="Singrauli, Surya Kiran Bhawan Dudhichua  CO (mg/m3)",data=df)


# In[40]:


sns.boxplot( y="Singrauli, Surya Kiran Bhawan Dudhichua  SO2 (µg/m3)",data=df)


# In[41]:


sns.boxplot( y="Singrauli, Surya Kiran Bhawan Dudhichua  NH3 (µg/m3)",data=df)


# In[42]:


sns.boxplot( y="Singrauli, Surya Kiran Bhawan Dudhichua  Ozone (µg/m3)",data=df)


# In[43]:


sns.boxplot( y="Singrauli, Surya Kiran Bhawan Dudhichua  Benzene (µg/m3)",data=df)


# In[44]:


df.describe()


# In[45]:


df.hist(figsize=(10,20))


# In[46]:


sns.pairplot(df,diag_kind='kde')


# In[47]:


sns.pairplot(df)


# In[48]:


corr=df.corr()
corr


# In[49]:


sns.heatmap(corr,annot=True)


# In[50]:


df.dtypes

