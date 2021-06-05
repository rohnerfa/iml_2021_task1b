#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#import the data
data = pd.read_csv('train.csv',index_col='Id')

#extract X and y
y = data['y'].to_numpy()
X = data.drop(['y'],axis=1).to_numpy()

#transform X and y
X_transformed = np.concatenate((X,np.square(X),np.exp(X),np.cos(X),np.ones((y.size,1))),axis=1)


# In[2]:


a = np.logspace(-3,3,100)
model = linear_model.RidgeCV(alphas=a, fit_intercept=False, scoring='neg_mean_squared_error',cv=5).fit(X_transformed,y)
print(f'The RMSE is: {mean_squared_error(y,model.predict(X_transformed),squared=False)}.')


# In[3]:


coef = model.coef_
print(f'The coefficients for this model are:\n{coef}')


# In[4]:


#write output
submission = pd.Series(coef)
submission.to_csv('submission_final.csv', index=False, header=False)

