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
            age_arr_i = age_arr[i]

            age_arr_i[2] = gfunc_age_mode(g_age, ages)
            age_arr_i[1:4:2] = gfunc_age_conf(g_age, ages)
            age_arr_i[0:5:4] = gfunc_age_conf(g_age, ages, conf_level=0.90)

        # Pad identifier strings (for prettier output)
        id_len = max([len(x) for x in star_id])
        for i, sid in enumerate(star_id):
            star_id[i] = sid + (id_len - len(sid))*' '

        # Combine identifiers and data in DataFrame and write to txt
        pd_arr = pd.DataFrame(age_arr, index=star_id)
        pd_arr.to_csv(filename, sep='\t', index_label='#ID number',
                      header=['5', '16', 'Mode', '84', '95'],
                      float_format='%2.2f', na_rep='nan')
