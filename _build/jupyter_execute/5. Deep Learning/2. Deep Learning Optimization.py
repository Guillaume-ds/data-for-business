#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Problems and Optimization

# <div style="text-align:justify">
# Optimization is a crucial step in deep learning, that concerns every problem with various degrees of difficulty. Depending on the task and the data set, optimization will be more or less challenging. Therefore it is important to understand how one can optimize the main algorithms in order to choose the right one for a given problem. 
# </div>
# 
# #### Main challenges : 
# - Convexity of the function (a non-convex function may have many local minima)
# - Set of the optimization function
# - Dimension of the vector of parameters
# - Cost of gradient computing
# - Irregularity of a twice differentiable function :
#     - L-smooth if : $ \forall w \in R^d, eig[f''(w)]\le L $
#     - $\mu$-strongly convex if : $\forall w \in R^d, eig[f''(w)]\ge \mu $
# 
# 
# <div style="text-align:justify">
# Those are some of the main challenges we encouter while doing optimization. Therefore, it is important to understand precisely how we can tackle such issues. 
# </div>

# ## Gradient Descent Procedure
# 
# We aim at minimizing a function $ f : R^d \rightarrow R$
# 
# <div style="display:flex;justify-content: space-around;">
# <img src="pictures/minimum.png">
# <img src="pictures/minimum2D.png">
# </div>

# In order to solve this problem, we have ton compute the gradient of the function, which is the vector of partial derivatives : 
# 
# $$
# \nabla f(\left.x_{1}, x_{2}, \ldots, x_{n}\right)=\left[\begin{array}{c}
# \dfrac{\partial f}{\partial x_1}(\left.x_{1}, x_{2}, \ldots, x_{n}\right)\\
# \dfrac{\partial f}{\partial x_2}(\left.x_{1}, x_{2}, \ldots, x_{n}\right) \\
# \vdots \\
# \dfrac{\partial f}{\partial x_n}(\left.x_{1}, x_{2}, \ldots, x_{n}\right) 
# \end{array}\right]
# $$
# 
# We call *critical points* the points such that $ \nabla f(w^*) = 0 $ 
# 
# *Remark* critical points are not always extrema !
# 
# If this function is twice differenciable, we can compute the hessian matrix given by : 
# 
# $$
# \nabla^2 f(\left.x_{1}, x_{2}, \ldots, x_{n}\right)=
# \begin{bmatrix}
#     \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \dots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
#     \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \dots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
#     \vdots & \vdots & \ddots & \vdots \\
#     \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \dots & \frac{\partial^2 f}{\partial x_n^2}
# \end{bmatrix}
# $$
# 
# ### Convex Function 

# 
