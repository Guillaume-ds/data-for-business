#!/usr/bin/env python
# coding: utf-8

# # Financial Data

# ## 1 - Basic Financial contracts and their prices
# 
# Futur cash flow denoted $A_T$ : 
# - $T$ is the maturity, a time horizon in the futur
# - The value $A_T$ is unknown today
# 
# The spot price $ price_t(A_T,T)$ : 
# - It is the price to pay at time $t$ in order to receive $A_T$ at $T$ 
# 
# The forward price $ F_t(A_T,T)$ : 
# - It is the price determined at time $t$ but to pay at $T$ in order to receive $A_T$  
# 
# At time $t$, we enter into a forward contract by agreeing on the value of $ F_t(A_T,T)$. However, there are no cash flow at that time : the price to pay to enter a forward is 0.
# 
# *A long position in a forward is to receive $A_T$ and pay $F$* 
# 
# *A short position in a forward is to receive $F$ in exchange for $A_T$*
# 
# We assume that there ie a risk-free rate r such that N eur invested at t = N(1 + r (T-t)) at T
# 
# So for any cash flox $A_T$ we have : 
# 
# $$
# F_t(A_T,T) = price_t(A_T,T)(1+r(T-t))
# $$
# 
# Value at time = $ F_t(A_T,T) - price_t(A_T,T)(1+r(T-t))$
# 

# ### Options 
# 
# An option is a financial contract whose value at maturity T depends on the value of S 
# The option payoff = $g(S_T)$ 
# 

# In[ ]:




