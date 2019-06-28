#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:28:31 2019

@author: Abhirup banerjee
"""

import numpy as np


def dm(t,T_hat,A,n):    
    """
      Args  t : original time 
            T_hat : Event time 
            A : event  values 
            n : window size 
     Returns  
            dis_mat= distance matrix 
    """
    wins=np.arange(min(t), max(t), n * np.mean(np.diff(t))) # creating the windows 
   
    # Metric distance parameter
   
    p0=float(len(A)/T_hat[-1]-T_hat[0]) # parameter for time
    p1=1.0 / (np.mean(np.abs(np.diff(A)))) # parameter for amplitude
    p2=1.0 # adding or deleting parameter

    
    M=int(len(wins))
    dis_mat=np.zeros((M,M))
    for i in range(M-1):
        i1=np.where((T_hat >= wins[i]) & (T_hat < wins[i+1]))[0] # findig the consequetive windows
        tli=T_hat[i1] - T_hat[i1[0]]
        ali=A[i1]
        
        for j in range(i+1,M-1):
            
            i2 = np.where((T_hat[:] >= wins[j]) & (T_hat < wins[j + 1]))[0]
            tlj=T_hat[i2]-T_hat[i2[0]]
            alj=A[i2]
    
            dis_mat[i,j]=cost(tli,ali,tlj,alj,p0,p1,p2,i1,i2)
    
    dis_mat=dis_mat.T+dis_mat
    #dis_mat=np.delete(dis_mat,(-1),axis=0)
    #dis_mat=np.delete(dis_mat,(-1),axis=1)
    return dis_mat

def cost(tli,ali,tlj,alj,p0,p1,p2,t1,t2):
        '''
        Args: tli, ali,tlj,alj - are the values of time and amplitude of consequative windows , p0,p1,p2 are the cost parameter 
        t1,t2 provide the length of two windows 
        '''
        
        nspi = len(t1)
        nspj = len(t2)
    
        G = np.zeros([nspi + 1, nspj + 1])
    
        G[:, 0] = np.arange(nspi + 1)
        G[0, :] = np.arange(nspj + 1)
    
        if nspi and nspj > 0:
            for i in range(1, nspi + 1):
                for j in range(1, nspj + 1):
                    try:
                        G[i, j] = min(G[i - 1, j] + p2, \
                           G[i, j - 1] + p2, \
                           G[i - 1, j - 1] + p1 * abs(alj[i - 1] - ali[j - 1]) + p0 * abs(tli[i - 1] - tlj[j - 1]))
                    except:
                        G[i, j] = min(G[i - 1, j] + p2, G[i, j - 1] + p2)
            d = G[-1, -1]
        else:
            d = abs(nspi - nspj) * p2
        return d