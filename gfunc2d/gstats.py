#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:19:27 2018

@author: christian
"""

#import h5py
import numpy as np
#import scipy


#test_output = '/Users/christian/stellar_ages/Gaia_benchmark/test/output.h5'


def smooth_gfunc2d(g):
    kernel = np.array([0.25, 0.5, 0.25])
    func = lambda x: np.convolve(x, kernel, mode='same')
    g1 = np.apply_along_axis(func, 0, g)
    g2 = np.apply_along_axis(func, 1, g1)
    
    return g2

def norm_gfunc(g, method='maxone'):
    if method == 'maxone':
        gnorm = g / np.amax(g)
#    elif method == 'other_method':
#        gnorm = ...
    else:
        raise ValueError('Invalid normalization method')

    return gnorm