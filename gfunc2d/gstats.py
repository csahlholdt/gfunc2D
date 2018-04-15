#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:19:27 2018

@author: christian
"""

import h5py
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton


def smooth_gfunc2d(g):
    kernel = np.array([0.25, 0.5, 0.25])
    func = lambda x: np.convolve(x, kernel, mode='same')
    g1 = np.apply_along_axis(func, 0, g)
    g2 = np.apply_along_axis(func, 1, g1)
    
    return g2


def norm_gfunc(g, method='maxone'):
    if np.amax(g) == 0:
        return g

    if method == 'maxone':
        gnorm = g / np.amax(g)
#    elif method == 'other_method':
#        gnorm = ...
    else:
        raise ValueError('Unknown normalization method')

    return gnorm


def gfunc_age(g, norm=True, norm_method='maxone'):
    g_age = np.sum(g, axis=1)
    if norm:
        g_age = norm_gfunc(g_age, norm_method)

    return g_age


def gfunc_age_mode(g_age, age_grid):
    ind = np.argmax(g_age)
    age_mode = age_grid[ind]

    return age_mode


def conf_glim(conf_level):
    assert conf_level > 0 and conf_level < 1

    zero_func = lambda x: 2*norm.cdf(np.sqrt(-2*np.log(x))) - 1 - conf_level
    glim = newton(zero_func, 0.6)

    return glim


def gfunc_age_conf(g_age, age_grid, conf_level=0.68):
    glim = conf_glim(conf_level)

    ages_lim = age_grid[g_age > glim]
    age_low, age_high = ages_lim[0], ages_lim[-1]

    if age_low == age_grid[0]:
        age_low = None
    if age_high == age_grid[-1]:
        age_high = None
    age_conf = (age_low, age_high)

    return age_conf


def age_mode_and_conf(g_age, age_grid, conf_levels=[0.68, 0.90]):
    n = len(conf_levels)
    age_arr = np.zeros(1+2*n)

    age_arr[n] = gfunc_age_mode(g_age, age_grid)
    for i in range(1, n+1):
        try:
            age_arr[n-i:n+i+1:2*i] = gfunc_age_conf(g_age, age_grid, conf_level=conf_levels[i-1])
        except:
            age_arr[:] = None
            break

    return age_arr


def print_age_stats(output_h5, filename):
    with h5py.File(output_h5) as out:
        ages = out['grid/tau'][:]
        gf_group = out['gfuncs']
        star_id = np.array(gf_group)

        n_star = len(star_id)
        age_arr = np.zeros((n_star, 5))
        for i, star in enumerate(star_id):
            g = gf_group[star][:]
            g = smooth_gfunc2d(g)
            g = norm_gfunc(g)
            g_age = gfunc_age(g)
            age_arr[i] = age_mode_and_conf(g_age, ages)

        # Pad identifier strings (for prettier output)
        id_len = max((10, max([len(x) for x in star_id])))
        star_id_pad = [x.ljust(id_len) for x in star_id]

        # Combine identifiers and data in DataFrame and write to txt
        pd_arr = pd.DataFrame(age_arr, index=star_id_pad)
        pd_arr.to_csv(filename, sep='\t', index_label='#ID number',
                      header=['5', '16', 'Mode', '84', '95'],
                      float_format='%2.2f', na_rep='nan')
