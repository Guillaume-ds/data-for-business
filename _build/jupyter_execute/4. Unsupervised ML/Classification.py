#!/usr/bin/env python
# coding: utf-8

# <style>body {text-align: justify}</style>
# <div style="text-align:center;">
# <h1>Dimension Reduction and Clustering</h1>
# </div>

# 
# Unsupervised machine learning consist in working with unlabeled data in order to create clusters or groups of observations sharing similar features. Contrary to supervised learning, the dataset only stores observations :
# 
# $$
# 
# \mathcal{D} = \{X_1, ... ,X_n \} \in \mathcal{X}^n
# 
# $$
# 
# However, the dimension of $ \mathcal{X} $ often makes it hard to work with. Therefore, one the first task to do in unsupervised machine learning is to reduce the dimension of the space. This can be done with a map $ \phi $ from $ \mathcal{X} $ to a new space $ \mathcal{X}' $ of smaller dimension. 
# 

# 
