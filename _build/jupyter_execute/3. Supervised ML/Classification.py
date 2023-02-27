#!/usr/bin/env python
# coding: utf-8

# # Classification
# 
# ## What is Classification
# <div style="text-align:justify">
# Classification consists in taking a vector of inputs X to produce a vector of outputs Y that will associate each input x to a class. The training data set is composed of the vector X of n observations (that can include b parameters), and Y the vector the associated true values. The algorithm will try to minimize a loss function in order to create a model that can predict the class of each x with a certain level of precision. During the training phase, the true output is already defined and is given to the algorithm so it can learn how to differenciate those classes. Once the model is trained, it is supposed to be able to correctly predict those output for new values.
# <br><br>
# An example of classification could be to predict the added value of a new client : small, average, important. The input could include the type of client (individual or company), the location of the store in which they made their first purchase, the amount of their first purchase, the day of the week... The model will transform the input vector in a value corresponding to one of the classes. During the training phase, you should take data that already exists. Here is a simple example of a classification model : 
# </div>
# 
# #### 1. Training the model 
# <center>
# <img src="pictures/classificationTraining.png">
# </center>
# 
# #### 2. Predict new values 
# <center>
# <img src="./pictures/classificationPredictions.png">
# </center>
# 
# ## How does a classification model work? 
# 
# <div style="text-align:justify">
# Once the data is correclty collected, cleaned and formated, you can proceed to develop and train a classification model. There are several algorithm that will help you to do this task. However, it is important to understand how they work under the hood, and the right way to use them.
# </div>
# 
# ### The loss function
# 
# <div style="text-align:justify">
# As specified in the graphs above, one of the main part of the training process consist in comparing the predicted values to the true output. <br>
# For a classification problem, the loss can be defined as : 
# </div>
# 
# $$
# 
# \mathcal{l}(Y,f(X)) = \sum_{i=1}^{n} 1_{Y_i \ne f(X_i)}
# 
# $$
# 
# ### What are the mains challenges 
# 
# <div style="text-align:justify">
# As classification greatly depends on the quality of the input, therefore the first challenge we encounter is to feed the model with the proper data and the right amount of information. Indeed, giving too much information to the model can lead to over-fitting, especially if the information is not independ. It will also take more time to train the model and to get precise results. On the other hand, by not giving enough data to the model, you won't be able to train it correctly and it will never reach a correct level of precision. 
# </div>

# ## How to implement classification with python :
# 
# <div style="text-align:justify">
# In this example, I choose to do imports as we go. This is supposed to help you understand in which situation and part of the process the different packages are useful. However, in reality, a good practice is to import all the packages in the first cell of the notebook. 
# </div>

# ### 1. Load the data 
# 
# <div style="text-align:justify">
# Pandas and numpy are the main packages used for loading/creating data sets. However, in a real case scenario, you might need a specific package to access a data base (boto3, sql...) or to do data transformation.
# <br><br>
# </div>
# 
# <div style="text-align:justify">
# Then, download the data set : https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset?resource=download) and unzip the file and load it in python with pandas. As the data sets are from an external source, we will check that there are correctly splited.
# </div>

# In[1]:


import pandas as pd 
import numpy as np 


# In[2]:


train = pd.read_csv("../datasets/loan/train.csv")
test = pd.read_csv("../datasets/loan/test.csv")

print(list(train.columns))
print(list(test.columns),"\n")

S1 = set(train["Loan_ID"])
S2 = set(test["Loan_ID"])

if S1.intersection(S2) != set():
    print("Error, there are common elements in train and test data sets : ", S1.intersection(S2))
else:
    print("Datasets are correctly splited")


# <div style="text-align:justify">
# Now that the data has been loaded and that we checked that the two datasets are different, we can do data exploration.
# </div>

# ### 2. Clean and visualize the data 
# 
# <div style="text-align:justify">
# We will import two libraries for data visualisation : seaborn and matplotlib are very common library used for data visualization. Other common are ggplot, altair ...
# </div>

# In[3]:


import matplotlib.pyplot as plt 
import seaborn as sns


# <div style="text-align:justify">
# First, let's look at the type of data. We will focuse on the train dataset. Some useful functions are head, info, isnull and describe. We see that Loan_ID is not a relevant field for our problem, therefre we can drop it immediatly. 
# </div>

# In[4]:


train.head()


# In[5]:


train.drop('Loan_ID',axis=1,inplace=True)
train.describe()


# In[6]:


train.isnull().sum()


# <div style="text-align:justify">
# As we can see, there are many null values. Keeping them like this would negatively affect the performance of the model, therefore we have to find a way to remove those values. One way to deal with such values is to simply remove the data entry. However, this reduces the size of the data set on which we will train the model. As a consequence, it is often better to find a way to replace null values by other values that are likely to be close to the truth. This can be done by inserting the most common value or the mode, the mean, a zero value... <br>
# Here we will treat each cases differently : 
# </div>

# - Drop NA
# 
# <div style="text-align:justify">
# For LoanAmount and Credit_History we can either replace the missing values by their means, or drop the entries with missing values. As both of those fields have an important standart deviations in comparison to their mean, it could create a significant noise to replace them by their mean (cf. describe table). Therefore, we will drop the lines with either a missing loan amount or loan amount term. We can do that with the dropna function by specifying the columns in which it should look for NA values. 
# </div>

# In[7]:


train = train.dropna(subset=['LoanAmount', 'Credit_History'])
train[['LoanAmount', 'Credit_History']].isnull().sum()


# - Fill with the mean
# 
# <div style="text-align:justify">
# For Loan_Amount_Term we can replace the missing values by the mean of this duration.
# </div>

# In[8]:


Mean_Amout_Term = int(train['Loan_Amount_Term'].mean())
print("Missing values will be replaced by : ",Mean_Amout_Term)
train['Loan_Amount_Term'].fillna(Mean_Amout_Term ,inplace=True)


# - Fill with the mode
# 
# <div style="text-align:justify">
# For other fields we can replace the missing values by their mode. This would make sense as their mean is close to this value. 
# </div>

# In[9]:


print("The mode of Gender history is : ",train['Self_Employed'].mode()[0]) 
train['Married'].fillna(train['Married'].mode()[0],inplace=True)
train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)


# Let's recount the missing values now : 

# In[10]:


train.isnull().sum()


# In[11]:


train.info()


# In[12]:


train.nunique()


# <div style="text-align:justify">
# Those results are making sense, except for dependents that is an "object" (see train.info()) when it is supposed to be an integer. As we can see below, values above 3 are regrouped in a unique field "+3" that we can replace by "3" as in order to have only integer. 
# </div>

# In[13]:


train['Dependents'].value_counts()


# In[14]:


train['Dependents'].replace('3+',3,inplace=True)
train['Dependents']=train['Dependents'].astype('int')
train['Dependents'].value_counts()


# <div style="text-align:justify">
# Now that the trained dataset is cleaned, we can use data visualisation in order to get a better understanding of our working material. We will look for outsiders values, correlations...
# </div>

# ##### 1. Outsiders

# In[15]:


sns.catplot(data=train,kind='box')
plt.xticks(rotation=45)
plt.grid()
plt.show()


# ##### 2. Correlation

# In[16]:


plt.figure(figsize=(15,5))
sns.heatmap(train.corr(),annot=True)
plt.show()


# ##### 3. Discrete values 

# In[17]:


Discrete_Values = ['Loan_Status','Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

fig, axs = plt.subplots(figsize=(25,6),ncols=3,nrows=3)
plt.subplots_adjust(hspace=0.5)

for column, ax in zip(Discrete_Values, axs.ravel()):
    sns.countplot(x=train[column],hue=train['Loan_Status'],ax=ax)

plt.show()


# ##### 4. Continous values 

# In[18]:


Continous_Values = ['ApplicantIncome','CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome']

fig, axs = plt.subplots(figsize=(25,6),ncols=5,nrows=1)
plt.subplots_adjust(hspace=0.5)

for column, ax in zip(Continous_Values, axs.ravel()):
    sns.kdeplot(x=train[column],hue=train['Loan_Status'],ax=ax,fill=True)
plt.show()

fig, axs = plt.subplots(figsize=(25,6),ncols=5,nrows=1)
plt.subplots_adjust(hspace=0.5)
for column, ax in zip(Continous_Values, axs.ravel()):
    sns.kdeplot(x=train[column],hue=train['Gender'],ax=ax,fill=True)
plt.show()


# ### 3. Create the model

# ### 4. Train the model

# ### 5. Make predictions 
