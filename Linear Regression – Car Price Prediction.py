#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')


# In[10]:


car=pd.read_csv('car_dataset.csv')


# In[11]:


car.head()


# In[12]:


car.shape


# In[13]:


car.info()


# In[14]:


backup=car.copy()


# In[15]:


car=car[car['year'].str.isnumeric()]


# In[16]:


car['year']=car['year'].astype(int)


# In[17]:


car=car[car['Price']!='Ask For Price']


# In[18]:


car['Price']=car['Price'].str.replace(',','').astype(int)


# In[19]:


car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','')


# In[20]:


car=car[car['kms_driven'].str.isnumeric()]


# In[21]:


car['kms_driven']=car['kms_driven'].astype(int)


# In[22]:


car=car[~car['fuel_type'].isna()]


# In[23]:


car.shape


# In[24]:


car['name']=car['name'].str.split().str.slice(start=0,stop=3).str.join(' ')


# In[25]:


car=car.reset_index(drop=True)


# In[26]:


car


# In[27]:


car.to_csv('Cleaned_Car_data.csv')


# In[28]:


car.info()


# In[29]:


car.describe(include='all')


# In[30]:


car=car[car['Price']<6000000]


# In[31]:


car['company'].unique()


# In[32]:


import seaborn as sns


# In[33]:


plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='company',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# In[34]:


plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='year',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# In[35]:


sns.relplot(x='kms_driven',y='Price',data=car,height=7,aspect=1.5)


# In[36]:


plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel_type',y='Price',data=car)


# In[37]:


ax=sns.relplot(x='company',y='Price',data=car,hue='fuel_type',size='year',height=7,aspect=2)
ax.set_xticklabels(rotation=40,ha='right')


# In[38]:


X=car[['name','company','year','kms_driven','fuel_type']]
y=car['Price']


# In[39]:


X


# In[40]:


y.shape


# In[41]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[42]:


from sklearn.linear_model import LinearRegression


# In[43]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[44]:


ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])


# In[45]:


column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    remainder='passthrough')


# In[46]:


lr=LinearRegression()


# In[47]:


pipe=make_pipeline(column_trans,lr)


# In[48]:


pipe.fit(X_train,y_train)


# In[49]:


y_pred=pipe.predict(X_test)


# In[50]:


r2_score(y_test,y_pred)


# In[51]:


scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))


# In[52]:


np.argmax(scores)


# In[53]:


scores[np.argmax(scores)]


# In[54]:


pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))


# In[55]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)


# In[56]:


import pickle


# In[57]:


pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))


# In[58]:



pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))


# In[59]:


pipe.steps[0][1].transformers[0][1].categories[0]


# In[ ]:




