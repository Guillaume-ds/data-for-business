#!/usr/bin/env python
# coding: utf-8

# # Dimension Reduction and Clustering

# <style>body {text-align: justify}</style>
# 
# ## What is unsupervised machine learning ? 
# 
# Unsupervised machine learning consists in working with unlabeled data, often in order to create clusters or groups of observations sharing similar features. For instance, creating a customer segmentation is an unsupervised learning task. It differs from a supervised learning task such as classification in the sense that the segments are not given, i.e. the algorithm has to <em>create</em> the segments in order to classify the customers. Contrary to supervised learning, the dataset is only composed of observations, and does not have target features :
# 
# - Dataset in the case of supervised machine learning : $ \mathcal{D} = \{(X_1,Y_1), ... ,(X_n,Y_n) \} \in \mathcal{X}^n, \mathcal{Y}^n $
# 
# - Dataset in the case of unsupervised machine learning : $ \mathcal{D} = \{X_1, ... ,X_n \} \in \mathcal{X}^n $
# 
# Letting the machine create the segments will allow the user to discover hidden patterns and insights that are often quite complex to detect, especially with a large number of features. For example, it can be relatively easy to find similar behaviour for customers that are the same age, but the task gets harder as we increase the number of feature, and comparing the behaviour of customers with respect to their age, location, average purchases, day of the week ... can be way harder. Unsupervised machine learning alogrithm aim at finding a solution for such tasks. 

# ## Dimension reduction
# 
# The dimension of $ \mathcal{X} $ often makes it hard to work with. Therefore, one the first task to take care of in unsupervised machine learning is to reduce the dimension of the space under study. This can be done with a map $ \phi $ from $ \mathcal{X} $ to a new space $ \mathcal{X}' $ of smaller dimension, i.e. a function that will transform the elements of $ \mathcal{X} $ into elements of $ \mathcal{X}' $. It is important in this process to take a close look at the reconstruction error of the application $ \tilde{\phi} $ from $ \mathcal{X}' $ to $ \mathcal{X} $ : $ \tilde{\phi}(\phi(\mathcal{X})) $ should be equal to $ \mathcal{X} $. The relationship between observations must also be preserved : $(\phi(X_i),\phi(X_j))$ should have a similar relationship as $ (X_i,X_j)$.
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
# However, reducing the size of the space leads to a loss of information. Indeed, each dimension corresponds to a feature, thus a space of dimension $k \le d$ will include $d-k$ less features.  Therefore, it is necessary to carefully select the dimensions to be removed in ordeer to preserve the principal components.

# ## Principal Component Analysis
# 
# As the name suggests, the objective of principal component analysis is to select the most important features in order to create a subspace of smaller dimension that retains the most information, it aims at decreasing the dimension of the dataset from $d$ to $k$ with $ d \ge k$, while capturing the essence of the original data. We consider the dispertion of the data with respect to a feature as the importance of this feature, i.e the more dispersed the data is with respect to a feature, the more useful this feature will be in order to create clusters.
# 
# ### First example in 2 dimensions
# <center>
#     <figure>
#         <img src="./pictures/PCA_rotation.png">
#         <figcaption>Fig.1 - PCA example from wikipedia with the transformation to apply</figcaption>
#     </figure>
# </center>
# 
# On the example above, we can clearly see that the feature represented on the X-axis is the principal component, whereas the one on the Y-axis is less important. Indeed, the variation is greater on the X-axis as it is on the Y-axis. The objective will be to rotate the data as shown in red in order to increase the variance with respect to the X axis, and decrease the variance with respect to the Y axis. A little bit of mathematical formalization is necessary in order to clearly understand how one can compute the importance and the dispersion of each feature. 
# 
# 1. **Find the center of the dataset**. We can see on the graph above that the origin of the 2 vectors is at the center of the dataset. This point can be found by computing the mean of each feature, here we see that the mean of the data on the X axis is 1, and that the mean on the Y axis is 3. Therefore, the center of the dataset is the point (1,3). Then when move the data set so that its center is on the origin. This operation does not change the dataset, we only "move" the axis so that the origin is on the mean. 
# 2. **Find the direction of the main vector**. The data above has a nice shape that makes quite obvious the direction in which the main vector should be. However, it is still necessary to compute the precise direction. This can be done by fitting a line through the dataset, the same way than for a linear regression problem, either by minimizing the square distances between the line and the points ($d_i$) or by maximizing the square distances between the projection of the point on the line ($d*_i$) and the origin. The equivalent holds because of the pytagorian theorem : since : $b^2_i = d_i^2 + d_i^{*2}$, then : $min(\sum_{i=1}^d d_i^2) \equiv max(\sum_{i=1}^d d^{*2}_i)$. When can have this intuition by looking at the line : as the line gets closer to the data point $i$, $d_i$ decreases and $d*_i$ increases as it gets closer to its hypothenuse. 
# <center>
#     <figure>
#         <img src="./pictures/fit_line.png" width="450" height="300">
#         <figcaption>Fig.2 - Line fitting through the dataset</figcaption>
#     </figure>
# </center>
# 
# Then, we can find the best fitting line by solving one of the two optimisation problem above. Here, the fitting line as a slope of 1/3, i.e. for each variation of 3 on the X axis, there is only a variation of 1 on the Y axis. This means that the feature X is 3 times more important than Y for the PC 1, and thus the PC1 can be seen as the result of a linear combination such as : 3.X + Y. Indeed, in principal component analysis, we replace the features by a linear combination of the features. Therefore, a data point will not depend on $X$ and $Y$, but on two linear combination of both. 
# 
# Now that we have the the direction of the fitting line, we need to find the unit vector, i.e. the vector that corresponds to moving of one unit on the fitting line. Note that this unit vector corresponds to the eingenvector. We start from the vector above $(3,1)$ that we will scale to a length of 1 by using the pytagorian theorem. Obviously, the lienar relation between X and Y remains, as : $3 \times \frac{1}{\sqrt{10}} = \frac{3}{\sqrt{10}}$. This means that for the principal component 1, that we see on the figure below, the feature X is 3 times as important as the feature Y.
# 
# $$
# length_{vectorA}^2 = 3^2 + 1^2 \iff length_{vectorA} = \sqrt{10}
# $$
# 
# $$
# PC1\ unit\ vector = (\frac{3}{\sqrt{10}},\frac{1}{\sqrt{10}})
# $$
# 
# <center>
#     <figure>
#         <img src="./pictures/unit_vector.png" width="450" height="300">
#         <figcaption>Fig.3 - Computing the unit vector</figcaption>
#     </figure>
# </center>
# 
# Because there are only 2 dimensions in this example, the PC2 is the line perpendicular to PC1 that goes through the origin. Indeed, in order to classify to importance of each PC, they have to be independent, so PC2 has to be orthogonal to PC1, which, in a 2D space, is a perpendicular line. Otherwise, a part of a change in PC1 could be explained as a change in PC2,wich can easely be represented graphically. The slope of the PC2 line is -3, wich means that PC2 is a linear combination of the form : -X+3Y. Then again we scale the vector : 
# 
# $$
# PC2\ unit\ vector = (\frac{-1}{\sqrt{10}},\frac{3}{\sqrt{10}})
# $$
# 
# Once we have the two unit vectors and therefore the two PC, we can project the data set onto this new space. Then we rotate the new space and we obtain a new representation of the dataset that extends the most important feature. We can see that the relation between the points remains the same as on fig.2, however the variance increased on the X axis and decreased on the Y axis. 
# 
# <div style="display:flex;justify-content:space-around">
#     <img src="./pictures/PCA_projection.png" width="450" height="300">
#     <img src="./pictures/PCA_newspace.png" width="450" height="300" style="transform:rotate(-4deg);">
# </div>
# 
# We can see that the points are now represented as a function of PC1 and PC2, both of which are linear combinations of X and Y. Finally, we can compute the importance of each feature in the total variation of the PCs. First, we compute the sum of square distances of the projections of the points relatively to the origin divided by (n-1). For PC1 it is the sum of square distances between each green cross and the origin devided by 9 (there are 10 data points), and for PC2 we do the same with the blue crosses. Let's say for the sake of the example that the variation for PC1 is equal to 15, and that it is equal to 5 for PC2. Then the total variation is equal to 20, and PC1 accounts for 15/20 = 75% of the variation, and PC2 accounts for 25% of the variation. 
# 
# ### Generalisation in more dimensions
# In a real life scenario where there are many more features and dimensions, it is impossible to proceed by representing the dataset in the entire space. Indeed, even the simple example above of a customer dataset is in 5 dimensions, and thus impossible to represent graphically. In order to classify the importance of each component, it is necessary to adopt a mathematical approach and to compute the covariance matrix. The covariance matrix, called $\Sigma$, is a matrix where the element $(i,j)$ corresponds to the covariance of the features $X_i,X_j$. Note : the covariance of $X_i,X_i$ is equal to the variance of $X_i$, therefore the diagonal of the covariance matrix is composed of the variance of each feature.
# 
# $$
# cov(X_i,X_j) = E[(X_i - \mu_i)(X_j - \mu_j)] 
# $$
# 
# <div style="text-align:center">
# 
# with $\mu_i$, $\mu_j$ the expected values of the $i^{th}$ and $j^{th}$ features.
# 
# </div>
# 
# $$
# \Sigma = E[(\boldsymbol{X}−E[\boldsymbol{X}])(\boldsymbol{X}−E[\boldsymbol{X}])^T]
# $$
# 
# <div style="text-align:center">
# 
# with $\boldsymbol{X}$ the matrix of observations.
# 
# </div>
# 
# In order to do a PCA, we start by computing the covariance matrix, and then we compute the corresponding eingenvectors and eigenvalues. Because the features are supposed to be linearly independent, we should find the same number of eingenvectors and eingenvalues as features. We classify the eigenvalues in an increasing order and select the k-biggest (the data scientist gets to choose k!). Finally we project the dataset onto the newly formed space of size k. 
# 
# <center>
#     <figure>
#         <img src="./pictures/PCA_summary.png" width="600" height="300">
#         <figcaption>Fig.4 - Summary</figcaption>
#     </figure>
# </center>
# 
# In conclusion, we start by looking at the dataset in order to compute the covariance matrix. Then we compute the eigenvectors and eigenvalues. Finally, we can project the data onto a smaller space. Here, the space is in 4 dimensions, which can be represented with the help of a color chart.

# ## Clustering
# 
# There are a variety of clustering methods, which are very difficult to compare. The image below from scikit learn documentation shows 
# <center>
# <figure>
# <img src="./pictures/plot_cluster_comparison.png">
# <figcaption>Fig.1 - Clustering method from scikit learn</figcaption>
# </figure>
# </center>

# 
