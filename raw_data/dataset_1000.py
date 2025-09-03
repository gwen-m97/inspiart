#!/usr/bin/env python
# coding: utf-8

# In[2]:


import dataset_prep


# In[3]:


df = dataset_prep.load_data()


# In[4]:


df.head


# In[5]:


df_filtered = dataset_prep.filter_dataframe(df)
df_merged = dataset_prep.merging_metadata(df_filtered)


# In[9]:


df_sample = dataset_prep.data_sampling(df_merged, sample_size=1000, number_styles=10)


# In[ ]:




