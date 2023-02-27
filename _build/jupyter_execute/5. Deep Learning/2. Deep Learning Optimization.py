#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Problems and Optimization

# <div style="text-align:justify">
# Optimization is a crucial step in deep learning, that concerns every problem with various degrees of difficulty. Depending on the task and the data set, optimization will be more or less challenging. Therefore it is important to understand how one can optimize the main algorithms in order to choose the right one for a given problem. 
# </div>
# 
# ### Main challenges : 
# - Convexity of the function (a non-convex function may have many local minima)
# - Set of the optimization function
# - Dimension of the vector of parameters
# - Cost of gradient computing
# - Irregularity of a twice differentiable function :
#     - $L$-smooth if : $ \forall w \in R^d, eig[f''(w)]\le L $
#     - $\mu$-strongly convex if : $\forall w \in R^d, eig[f''(w)]\ge \mu $
# 
# 
# <div style="text-align:justify">
# Those are some of the main challenges we encouter while doing optimization. Therefore, it is important to understand precisely how we can tackle such issues. 
# </div>
# 
# #### Convex functions 
# 
# Convex functions are easier to study and help to solve optimization problems.
# 
# $ f : R^d \rightarrow R$ is a convex function if : 
# 
# $$
# f(\lambda x + (1-\lambda)y)\le \lambda f(x) + (1-\lambda)f(y) \hspace{0.4cm }\forall (x,y)\in R^d,\lambda \in [0,1]
# $$
# 
# If f is differentiable, it is convex if : 
# 
# $$
# f(x) \ge f(y) + \langle \nabla f(y),x-y \rangle \hspace{0.4cm }\forall (x,y)\in R^d
# $$
# 
# Also, if f is twice differentiable, it is convex if : 
# 
# $$
# h^T \nabla^2f(x)h \ge 0 \hspace{0.4cm }\forall h\in R^d, x \in R
# $$
# 
# *Remark: Those conditions are all necessary and sufficient if the function is twice differentiable, i.e. verifying one implies that the others are verified.*
# 
# Therefore, for a twice differentiable function we can compute $ f(w) $, but also $ \nabla f(w) $ to learn in wich direction f is increasing and $ \nabla^2 f(w) $ to see the curvature of the function. Computing the two derivatives helps to solve optimization problems faster. There are two types of algorithm that use this idea : 
# 1. **First-order algorithm** : Use $ f(w) $ and $ \nabla f(w) $ for differentiable and convex functions.
# 1. **Second-order algorithm** : Use $ f(w) $, $ \nabla f(w) $ and $ \nabla^2 f(w) $. It is very efficient when computing the Hessian Matrix is not to time consumming. 
# 
# #### Strongly Convex functions 
# 
# $ f : R^d \rightarrow R$ is a $\mu$-strongly convex function if the following function is also convex : 
# 
# $$
# w \rightarrow f(w) - \frac{\mu}{2}\vert\vert w \vert\vert^2_2
# $$
# 
# Equivalently : 
# 
# $$
# f(w) \ge f(w') + \langle\nabla f(w'),w-w' \rangle + \frac{\mu}{2}\vert\vert w - w' \vert\vert^2
# $$
# 
# Graphically, a smooth-convex function is above the tangent and the tangent + $\mu$ x quadratic function : 
# 
# <center>
# <img src="pictures/stronglyConvex.png">
# </center>
# 
# 
# #### L-Smooth functions 
# 
# A function f is said to be $L$-smooth if f is twice differentiable and if :
# 
# $$
# \vert\vert \nabla f(x) - \nabla f(y) \vert\vert \le \mathcal{L} \vert\vert x - y \vert\vert \hspace{0.4cm }\forall (x,y)\in R^d
# $$
# 
# Equivalently : 
# 
# $$
# f(w) \le f(w') + \langle\nabla f(w'),w-w' \rangle + \frac{\mathcal{L}}{2}\vert\vert w - w' \vert\vert^2
# $$
# 
# Graphically, a smooth-convex function is above the tangent and below the tangent + quadratic function : 
# 
# <center>
# <img src="pictures/Lsmooth.png">
# </center>

# ## Gradient Descent Procedure
# 
# We aim at minimizing a function $ f : R^d \rightarrow R$. However, local extremum are making it difficult to find the overall extrema. 
# 
# Here are two exemples, in 1d and 2d:
# 
# <div style="display:flex;justify-content: space-around;">
# <img src="pictures/minimum.png">
# <img src="pictures/minimum2D.png">
# <br>
# </div>
# 
# 
# 
# We search the vector of weights : $ \tilde{w} = (w_1, ... , w_n, b)$
# 
# We do an iterative procedure in order to find the correct $\tilde{w}$
# $$ 
# \tilde{w}_{t+1} \leftarrow \tilde{w}_t - \eta \nabla_w \mathcal{L}(\tilde{w}_t) 
# $$

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
# We call *critical points* or *local extremum* the points such that $ \nabla f(w^*) = 0 $. 
# - If $w^*$ is a local minimum, then $ \nabla f(w^*)=0$ and $\nabla^2f(w^*)$ is positive semi-definite.
# - If $ \nabla f(w^*)=0$ and $\nabla^2f(w^*)$ then $w^*$ is a local optimum.
# 
# *Remark: critical points are not always extrema !*
# 
# There are two main types of algorithm for a gradient descent procedure : Gradient Descent (GD) and Stochastic Gradient Descent (SGD). They are useful for convex and smooth functions, but there results vary depending on the type of function. 
# 
# <center>
# <img src="pictures/gradientEfficiency.png">
# </center>
# 
# ##### Tyler expansion around a point :
# 
# $$ 
# f(w) = f(w^{(0)}) + \langle \nabla f(w^{(0)}),w-w^{(0)} \rangle + O(\vert\vert w - w^{(0)} \vert\vert)
# $$
# 
# ##### Convergence of Gradient Descent :
# 
# Let $ f : R^d \rightarrow R$ be a $L$-smooth convex function, with $w*$ the minimum of $f$ on $R^d$.
# 
# Then, with a step size $\eta \le \frac{1}{L}$ : 
# 
# $$
# f(w^{k}) - f(w^*) \le \frac{\vert\vert w^{(0)} - w^* \vert\vert^2_2 }{2\eta k}
# $$
# 
# In particular for : $\eta = \frac{1}{L}$ :
# 
# $$
# f(w^{k}) - f(w^*) \le L \frac{\vert\vert w^{(0)} - w^* \vert\vert^2_2 }{2}
# $$
# 
# iterations are sufficient to get an $\epsilon$-approximation of the minimal value of $f$
# 
# And we naturaly choose to do the gradient descent as follow : 
# 
# $$
#  
# w^{(k+1)} = w^{(k)} - \frac{1}{L} \nabla f(w^{(k)})
# 
# $$
# 
# Let $ f : R^d \rightarrow R$ be a $L$-smooth, $\mu$-strongly convex function, with $w*$ the minimum of $f$ on $R^d$.
# 
# Then, with a step size $\eta \le \frac{1}{L}$ : 
# 
# $$
# f(w^{k}) - f(w^*) \le \frac{L}{2} (1- \eta \mu)^k \vert\vert w^{(0)} - w^* \vert\vert^2_2
# $$
# 
# ### Stochastic Gradient Descent 
# 
# TBD
# 
# 

# ## Convegence rates 
# 
# Number of iterations for a precision $\epsilon$ :
# - GD = $ k \ge \frac{1}{\epsilon} $
# - SGD = $ k \ge \frac{1}{\epsilon^2} $
# 
# Complexity per iteration :  
# - GD = $nd$
# - SGD = $d$
# 
# Total complexity for a precision $\epsilon$ :
# - GD = $ \frac{nd}{\epsilon} $
# - SGD = $ \frac{d}{\epsilon^2} $

# ## Gradient methods with reduced variance 
# 
# ### SAG & SAGA
# 
# $$
# 
# w_k = w_{k-1} - \frac{\eta}{n} \sum^n_{i=1}{g_k(i)}
# 
# \hspace{0.5cm} \text{with} \hspace{0.2cm}
# 
# {g_k(i)} = \left\{ \begin{array}{ll}
# \nabla f_i (w_{k-1}) & \text{if} i = i_k \\ 
# {g_{k-1}(i)} & \text{otherwise}
# \end{array} \right. 
# 
# $$
# 
# ### SVRG
# 
# ### Newton's method for smooth functions 
# 
# $$
# f(w) \approx f(w') + \nabla f(w')^T(w-w') + \frac{1}{2} (w - w')^T \nabla^2 f(w')(w-w')
# $$
# 
# With a step : 
# 
# $$
# d_k
#  = - [\nabla^2 f(w_k)]^{-1} \nabla f(w_k) 
# $$
# 
# ## Momentum methods
# 
# ### Polyak's momentum algorithm
# 
# Do until convergence : 
# 1. $ v^{(k)} = \beta (w^{(k)} - w^{(k+1)}) - \eta_k \nabla f(w^{(k)}) $
# 2. $ w^{(k+1)} = w^{(k)} + v^{(k)} $
# 
# Polyak's algorithm can fail and start repeating cycles 
# 
# ### Nesterov's momentum algorithm : improved Polyak's momentum algorithm
# 
# Do until convergence : 
# 1. $ v^{(k)} = w^{(k)} - \eta_k \nabla f(w^{(k)}) $
# 2. $ w^{(k+1)} = v^{(k+1)} + \beta_{k+1}(v^{(k+1)} - v^{(k)}) $

# 
