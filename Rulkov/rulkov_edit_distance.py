#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:49:38 2019

@author: banerjee
"""

import numpy as np 
import matplotlib.pyplot as plt 
from recurrence import rp
from detrend import detrend_time_series
from DistanceMatrix import dm

NT=8000 # transit time
N= 2000 # system size 
DP= 200 # delete point
NN=N-DP # No of points after deletion 

''' to make our time series irregular , we can also delete some random points '''
# Map parameters 

alpha=4.5
beta=0.0009
sigma=0.0011
iterates=10000

# Initial Condition
x=np.zeros(NT+1)
y=np.zeros(NT+1)
x[0]=0.1
y[0]=0.3

for i in range(1,NT+1):
    x[i]=alpha/(1+x[i-1]**2)+y[i-1]
    y[i]=y[i-1]-sigma*x[i-1]-beta

X=np.zeros(N)
Y=np.zeros(N)
T=np.zeros(N)
X[0],Y[0],T[0]=x[-1],y[-1],1



for j in range(1,N):
    X[j]=alpha/(1+X[j-1]**2)+Y[j-1]
    Y[j]=Y[j-1]-sigma*X[j-1]-beta
    T[j]=j+1

# detrending the time series 
    
X_detrend=detrend_time_series(X,4)
Y_detrend=detrend_time_series(Y,4)


''' to find the events, here we are using one constraints '''

X_evnt=X[X>0]
T_evnt=T[X>0]

#plt.subplot(2,1,1)   
#plt.plot(T,X)
#plt.plot(T_evnt,X_evnt,'r *')
#plt.xlabel("iterations")
#plt.ylabel("x")
#plt.subplot(2,1,2)
#plt.plot(Y,'b',linewidth=0.8)
#plt.xlabel("iterations")
#plt.ylabel("y")
#plt.tight_layout()



'''deleting random point '''

#XY=np.zeros((N,2))
#XY[:,0]=T
#XY[:,1]=X
#rand=np.random.choice(N,DP,replace=False)
#rand.sort()
#
#for i in range(len(rand)):
#     XY[rand[i],:]=0
#XA=XY[np.nonzero(XY)].reshape(NN,2)
#
#T_evnt=XA[:,0]
#X_evnt=XA[:,1]





'''Constructing the distance matrix using edit-distance method'''


DM_x=dm(T,T_evnt,X_evnt,5)
eps_x=np.percentile(DM_x,16)

j_x=np.where(DM_x<=eps_x)

R_Px=np.zeros(DM_x.shape,float)
R_Px[j_x]=1

# Delay embedding method
m=1
tau=1
R_Py=rp(Y_detrend,m,tau,0.08, norm="euclidean", threshold_by="frr", normed=False)


plt.figure(1)
plt.imshow(R_Px,cmap='binary',origin='lower')



plt.figure(2)
plt.imshow(R_Py,cmap='binary',origin='lower')


























