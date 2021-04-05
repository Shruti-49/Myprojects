#!/usr/bin/env python
# coding: utf-8

# Data Science and Business Analytics

# #GRIPMAR21

# Author : Shruti Chittora

# Task 1 : Prediction using Supervised ML

# A simple linear regression task as it involves just 2 variables i.e hours and score

# Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Getting the data

# In[4]:


url='http://bit.ly/w-data'
data=pd.read_csv(url)
print('Done!')


# In[5]:


data.head(5)


# In[6]:


data.dtypes


# In[7]:


data.shape


# In[8]:


data.describe()


# Data Visualization

# In[18]:


data.plot(x='Hours',y='Scores',style='o')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours vs Scores')
plt.show()


# From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# Preparing the data
# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[34]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values
print(x)
print(y)


# splitting data into training and test set
# We will do this by using Scikit-Learn's built-in train_test_split() method:

# In[21]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# 
# Training the ML Algorithm
# for this we requires a linear regression Algorithm

# In[22]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)


# In[23]:


# Plotting for the test data
plt.scatter(X_train,y_train)
plt.title('Training set')
plt.plot(X_train,reg.predict(X_train))
plt.xlabel('Hour Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[24]:


#acuracy of training
reg.score(X_train,y_train)


# In[25]:


# Plotting the regression line
line = reg.coef_*X+reg.intercept_
# Plotting for the test data
plt.scatter(X_train,y_train)
plt.title('Training set')
plt.plot(X_train,reg.predict(X_train))
plt.xlabel('Hour Studied')
plt.ylabel('Percentage Score')
plt.show()


# Making Predictions
# Now that we have trained our algorithm, it's time to make some predictions.

# In[26]:



print(X_test) # Testing data - In Hours
y_pred = reg.predict(X_test) # Predicting the scores


# 
# Task is to check what will be score if a student studies for 9.25 hours per day?

# In[27]:


hours=9.25
own_pred=reg.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format((own_pred)[0]))


# Evaluating the model
# 
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

# In[28]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


# In[29]:


# Visualizing the training set
plt.scatter(X_train,y_train)
plt.title('Training set')
plt.plot(X_train,reg.predict(X_train))
plt.xlabel('Hour Studied')
plt.ylabel('Percentage Score')
plt.show()


# Hence Task1 is completed!

# In[ ]:




