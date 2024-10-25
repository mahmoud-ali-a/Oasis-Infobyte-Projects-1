#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd


# In[9]:


df = pd.read_csv('retail_sales_dataset.csv')


# In[11]:


print(df.head())


# In[10]:


print(df.isnull().sum())


# In[12]:


df.dropna(inplace=True)


# In[13]:


print(df.describe())


# In[13]:


print(df.mode())


# In[14]:


median = df['Age'].median()


# In[15]:


std_dev = df['Age'].std()


# In[16]:


print(f'Median: {median}, Standard Deviation: {std_dev}')


# In[17]:


df['Date'] = pd.to_datetime(df['Date'])


# In[23]:


Amount_by_date = df.groupby('Date')['Total Amount'].sum()


# In[24]:


import matplotlib.pyplot as plt


# In[25]:


plt.figure(figsize=(10,6))


# In[29]:


Amount_by_date.plot()
plt.title('Amount_by_date')
plt.ylabel('Amount_by_date')
plt.show()


# In[30]:


# Top 10 best-selling products


# In[39]:


print(df.columns)


# In[40]:


# Top 10 best-selling products
top_products = df.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False).head(10)
print(top_products)


# In[41]:


customer_data = df.groupby('Customer ID')['Total Amount'].sum().sort_values(ascending=False)
print(customer_data)


# In[42]:


import seaborn as sns


# In[43]:


# Bar chart for top products


# In[44]:


plt.figure(figsize=(10,6))
sns.barplot(x=top_products.index, y=top_products.values)
plt.title('Top 10 Best-Selling Products')
plt.xticks(rotation=45)
plt.show()


# In[45]:


# Heatmap for correlations between features


# In[46]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# In[ ]:




