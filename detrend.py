#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:59:10 2019

@author: banerjee
"""
import numpy as np
from sklearn.linear_model import LinearRegression
def detrend_time_series(data, window_size):
    """ making small window ; pyunicorn
    """
    #  Get length of data array
    n = data.shape[0]
    #  Initialize a local copy of data array
    detrended_data = np.empty(n)

    #  Detrend data
    for j in range(n):
        #  Get distance of sample from boundaries of time series
        dist = min(j, n - 1 - j)

        if window_size / 2 > dist:
            half_size = int(dist)
        else:
            half_size = int(window_size / 2)

        detrended_data[j] = data[j] - data[j - half_size:j + half_size + 1].mean()

    return detrended_data


def linear_regression_detrend(data):
    X = [i for i in range(0, len(data))]
    X = np.reshape(X, (len(data), 1))
    model = LinearRegression()
    model.fit(X, data)
    # calculate trend
    trend = model.predict(X)
    detrended = [data[i]-trend[i] for i in range(0, len(data))]
    return detrended

def diff_detrend(data):
    ''' detrend by differencing '''
    return np.diff(data)