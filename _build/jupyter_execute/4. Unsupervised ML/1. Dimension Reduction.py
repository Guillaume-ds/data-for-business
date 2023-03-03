#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Machine Learning and Dimension Reduction 

# <style>body {text-align: justify}</style>
# 
# ## What is unsupervised machine learning ? 
# 
# Unsupervised machine learning consists in working with unlabeled data, often in order to create clusters or groups of observations sharing similar features. For instance, creating a customer segmentation is an unsupervised learning task. It differs from a supervised learning task such as classification in the sense that the segments are not given, i.e. the algorithm has to <em>create</em> the segments in order to classify the customers. Contrary to supervised learning, the dataset is only composed of observations, and does not have target features :
# 
# - Dataset in the case of *supervised* machine learning : $ \mathcal{D} = \{(X_1,Y_1), ... ,(X_n,Y_n) \} \in \mathcal{X}^n, \mathcal{Y}^n $
# 
# - Dataset in the case of *unsupervised* machine learning : $ \mathcal{D} = \{X_1, ... ,X_n \} \in \mathcal{X}^n $
# 
# Letting the machine create the segments will allow the user to discover hidden patterns and insights that are often quite complex to detect, especially with a large number of features. For example, it can be relatively easy to find similar behaviour for customers that are the same age, but the task gets harder as we increase the number of features, and comparing the behaviour of customers with respect to their age, location, average purchases, day of the week ... can be way harder. Unsupervised machine learning alogrithms aim at finding a solution for such tasks. 

# ## Dimension reduction
# 
# The dimension of $ \mathcal{X} $ often makes it hard to work with. Therefore, one the first task to take care of in unsupervised machine learning is to reduce the dimension of the space under study. This can be done with a map $ \phi $ from $ \mathcal{X} $ to a new space $ \mathcal{X}' $ of smaller dimension, i.e. a function that will transform the elements of $ \mathcal{X} $ into elements of $ \mathcal{X}' $. It is important in this process to take a close look at the reconstruction error of the application $ \tilde{\phi} $ : $ \mathcal{X}' \rightarrow \mathcal{X} $, so that $ \tilde{\phi}(\phi(\mathcal{X})) $ is equal to $ \mathcal{X} $. The relationship between observations must also be preserved : $(\phi(X_i),\phi(X_j))$ should have a similar relationship as $ (X_i,X_j)$.
# 
# The formalised motivation behind the reduction of the dimension of the space is the <strong>hight dimensional geometry curse - Folks theorem</strong>, which states that <em>when $d$ is large enough, all points are almost equidistant</em>. 
# 
# - If $X_1,...,X_n$ in the hypercube of dimension $d$ such that their coordinates are i.i.d then : 
# 
# $$
# \frac{min \vert\vert X_i - X_j \vert\vert_p}{max \vert\vert X_i - X_j \vert\vert_p} = 1 + \mathcal{O}_p(\sqrt{\frac{log(n)}{d}})
# $$
# 
# In reality, creating such a map is difficult and reducing the dimension of the space can be a challenge. For instance, we can consider a dataset about customers with many information that we will resize in order to obtain a smaller dataset by grouping some similar features together : 
# 
# <center>
# <figure>
# <img src="./pictures/resized_dataset.png" width="500" height="250">
# </figure>
# </center>
# 
# However, reducing the size of the space leads to a loss of information. Indeed, each dimension corresponds to a feature, thus a space of dimension $k \le d$ will include $d-k$ less features.  Therefore, it is necessary to carefully select the dimensions to be removed in order to preserve the principal components.
