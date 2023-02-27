#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks
# 
# ### Convolutional Layers 
# 
# <div style="text-align: justify">
# A convolutional neural network is a neural network that uses convolution instead of matrix product. Convolution consists in applying the same transformation to different parts of the image, that will reflect the dependence between pixels and thus will detect the main features of the image. With convolution, the inputs are not connected to every neurons, but instead they are linked to their neighbours. This is a sparse connectivity, as opposed to the dense connectivity of a traditionnal neural network, wich are easier to optimize. 
# <br><br>
# The formal definition of the Convolution operator : 
# </div>
# 
# $$
# \begin{align}
# O(i,j) 
# & = (I*K)(i,j) \\ 
# & = \sum_k \sum_I I(i + k , j + I) K (k,I)
# \end{align}
# \\ 
# \text{Where I is the input, and K the kernel to be learned}
# $$
# 
# There are four hyper parameters that control the size of the ouput volume  :
# 1. **Size of the Kernel** (usually 3x3 or 5x5)
# 2. **Depth of the output volume** 
# 3. **Stride** that correspond to the shift in pixel 
# 4. **Zero Padding**
# 
# We do not specify the depth of the kernel as it is usually the same as the input layer. Also, the kernel operates on the all depth of the input volume. 
# 
# We obtain the equality : 
# 
# $$
# O = \lfloor \frac{I + 2P - K}{S} \rfloor + 1
# $$ 
# 
# With : 
# - O the output size (height/width)
# - I the input size (height/width)
# - P the padding 
# - K the size of the filter (height/width)
# - S the stride 
# 
# ### Polling layers 
# 
# <div style="text-align: justify">
# The pooling layer operates independently on every depth slice of the input and resizes it, therefore the depth of the output of a pooling layer is always equal to the depth of its input. It uses a given function, usually the max function. 
# </div>

# #### LeNet
# 
# 

# In[ ]:




