#!/usr/bin/env python
# coding: utf-8

# # Data Manipulation

# ### 1. Load the data 
# 
# Pandas and numpy are the main packages used for loading/creating data sets. However, in a real case scenario, you might need a specific package to access a data base (boto3, sql...) or to do data transformation. Then, download the data set : https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset?resource=download) and unzip the file and load it in python with pandas. As the data sets are from an external source, we will check that there are correctly splited.

# ```python
# import pandas as pd 
# import numpy as np 
# 
# train = pd.read_csv("../datasets/loan/train.csv")
# test = pd.read_csv("../datasets/loan/test.csv")
# 
# print(list(train.columns),"\n")
# print(list(test.columns),"\n")
# 
# S1 = set(train["Loan_ID"])
# S2 = set(test["Loan_ID"])
# 
# if S1.intersection(S2) != set():
#     print("Error, there are common elements in train and test data sets : ", S1.intersection(S2))
# else:
#     print("Datasets are correctly splited")
# ```

# In[1]:


import pandas as pd 
import numpy as np 

train = pd.read_csv("../datasets/loan/train.csv")
test = pd.read_csv("../datasets/loan/test.csv")

print(list(train.columns),"\n")
print(list(test.columns),"\n")

S1 = set(train["Loan_ID"])
S2 = set(test["Loan_ID"])

if S1.intersection(S2) != set():
    print("Error, there are common elements in train and test data sets : ", S1.intersection(S2))
else:
    print("Datasets are correctly splited")


# We see that the train dataset has one column more than the test dataset, it is the target feature, what we want to predict. Now that the data has been loaded and that we checked that the two datasets are different, we can do data exploration.
# 

# ### 2. Clean and visualize the data 
# 
# We will import two libraries for data visualisation : seaborn and matplotlib are very common library used for data visualization. Other common librairies are ggplot, altair ...
# 
# First, let's look at the type of data. We will focuse on the train dataset. Some useful functions are head, info, isnull and describe. We see that Loan_ID is not a relevant field for our problem, therefre we can drop it immediatly. 

# In[2]:


import matplotlib.pyplot as plt 
import seaborn as sns

train.head()


# In[3]:


train.drop('Loan_ID',axis=1,inplace=True)
print(train.isnull().sum())
train.describe()


# <div style="text-align:justify">
# As we can see, there are many null values. Keeping them like this would negatively affect the performance of the model, therefore we have to find a way to remove those values. One way to deal with such values is to simply remove the data entry. However, this reduces the size of the data set on which we will train the model. As a consequence, it is often better to find a way to replace null values by other values that are likely to be close to the truth. This can be done by inserting the most common value or the mode, the mean, a zero value... <br>
# Here we will treat each cases differently : 
# </div>

# - Drop NA
# 
# <div style="text-align:justify">
# For LoanAmount and Credit_History we can either replace the missing values by their means, or drop the entries with missing values. As both of those fields have an important standart deviations in comparison to their mean, it could create a significant noise to replace them by their mean (cf. describe table). Therefore, we will drop the lines with either a missing loan amount or loan amount term. We can do that with the dropna function by specifying the columns in which it should look for NA values. 
# </div>

# In[4]:


train = train.dropna(subset=['LoanAmount', 'Credit_History'])
train[['LoanAmount', 'Credit_History']].isnull().sum()


# - Fill with the mean
# 
# <div style="text-align:justify">
# For Loan_Amount_Term we can replace the missing values by the mean of this duration.
# </div>

# In[5]:


Mean_Amout_Term = int(train['Loan_Amount_Term'].mean())
print("Missing values will be replaced by : ",Mean_Amout_Term)
train['Loan_Amount_Term'].fillna(Mean_Amout_Term ,inplace=True)


# - Fill with the mode
# 
# <div style="text-align:justify">
# For other fields we can replace the missing values by their mode. This would make sense as their mean is close to this value. 
# </div>

# In[6]:


print("The mode of Gender history is : ",train['Self_Employed'].mode()[0]) 
train['Married'].fillna(train['Married'].mode()[0],inplace=True)
train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)


# We can recount the values with ```train.isnull().sum()```and see that we don't have null values anymore. We can also look a the types of the column, and the number of unique values. The results are making sense, except for dependents that is an "object" (see train.info()) when it is supposed to be an integer. As we can see below, values above 3 are regrouped in a unique field "+3" that we can replace by "3" as in order to have only integer. Now that the trained dataset is cleaned, we can use data visualisation in order to get a better understanding of our working material. We will look for outsiders values, correlations...

# In[7]:


train['Dependents'].value_counts()

train['Dependents'].replace('3+',3,inplace=True)
train['Dependents']=train['Dependents'].astype('int')
train['Dependents'].value_counts()


# ##### 1. Outsiders

# In[8]:


sns.catplot(data=train,kind='box')
plt.xticks(rotation=45)
plt.grid()
plt.show()


# ##### 2. Correlation

# In[9]:


plt.figure(figsize=(15,5))
sns.heatmap(train[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]].corr(),annot=True)
plt.show()


# ##### 3. Discrete values 

# In[10]:


Discrete_Values = ['Loan_Status','Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

fig, axs = plt.subplots(figsize=(25,6),ncols=3,nrows=3)
plt.subplots_adjust(hspace=0.5)

for column, ax in zip(Discrete_Values, axs.ravel()):
    sns.countplot(x=train[column],hue=train['Loan_Status'],ax=ax)

plt.show()


# ##### 4. Continous values 

# In[11]:


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


# ## Tips and Tricks 

# In[12]:


# Loop with a modulo

for i in range(10):
    print(i%3, i//3)


# In[13]:


for i in range(10):
    print(i//5)


# In[14]:


from enum import Enum
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# In[15]:


memory = [] #A list of tuples containing the client ID, the store location and the purchase 
memory.append((1,"Paris",25))
memory.append((13,"Paris",23))
memory.append((17,"Marseille",25))
print(memory)

IDs, Stores, Purchases = zip(*memory)
print(IDs, '\n', Stores, '\n', Purchases)


# In[ ]:




