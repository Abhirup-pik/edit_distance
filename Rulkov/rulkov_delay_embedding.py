#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 21:21:08 2019

@author: banerjee
"""

import numpy as np 
import matplotlib.pyplot as plt 
from recurrence import*
from detrend import*

def RulkovMap(alpha,beta,sigma,x,y):
    X=(alpha/(1+x**2))+y
    Y=y-sigma*x-beta
    return X,Y

# Map dependent parameters 

#alpha=4.1    # alpha's range 4.1 - 4.7

alpha=4.5
beta=0.0009
sigma=0.0011
iterates=10000

# Initial Condition
xi=0.1
yi=0.3
x=[xi]
y=[yi]

for _ in range(0,iterates):
    xi,yi=RulkovMap(alpha,beta,sigma,xi,yi)
    x.append(xi)
    y.append(yi)
    
# Plot the time series
    

X=np.array(x[8000:])
Y=np.array(y[8000:])
X_detrend=detrend_time_series(X,4)
Y_detrend=detrend_time_series(Y,4)


plt.subplot(2,2,1)   
plt.plot(X,'r',linewidth=0.8)
plt.xlabel("iterations")
plt.ylabel("x")
plt.subplot(2,2,2)
plt.plot(Y,'b',linewidth=0.8)
plt.xlabel("iterations")
plt.ylabel("y")

#
#''' Recurrence plot using delay-embedding method '''
#tau=mi(Y,10)
#m=fnn(Y,3,15)

m=1
tau=1
Rp_x=rp(X,m,tau,0.08, norm="euclidean", threshold_by="frr", normed=False)
Rp_y=rp(Y,m,tau,0.08, norm="euclidean", threshold_by="frr", normed=False)

plt.subplot(2,2,3)
plt.imshow(Rp_x,cmap='binary',origin='lower')
plt.subplot(2,2,4)
plt.imshow(Rp_y,cmap='binary',origin='lower')
plt.tight_layout()







