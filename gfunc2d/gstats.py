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
        gmax = np.amax(g)
        if gmax == 0:
            gnorm = np.ones_like(g)
        else:
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
        id_len = max((10, max([len(x) for x in star_id])))
        star_id_pad = [x.ljust(id_len) for x in star_id]

        # Combine identifiers and data in DataFrame and write to txt
        pd_arr = pd.DataFrame(age_arr, index=star_id_pad)
        pd_arr.to_csv(filename, sep='\t', index_label='#ID number',
                      header=['5', '16', 'Mode', '84', '95'],
                      float_format='%2.2f', na_rep='nan')


def estimate_samd(gfunc_files, case='1D', betas=None, stars=None, dilut=None,
                  figdir=None):
    # Load data
    g2d = []
    tau_grid, feh_grid = None, None
    for gfunc_file in gfunc_files:
        with h5py.File(gfunc_file, 'r') as gfile:
            if tau_grid is not None and feh_grid is not None:
                tau_grid_new = gfile['grid/tau'][:]
                feh_grid_new = gfile['grid/feh'][:]
                if not (np.array_equal(tau_grid, tau_grid_new)\
                        and np.array_equal(feh_grid, feh_grid_new)):
                    raise ValueError('All g-functions must be defined on ' +\
                                     'the same metallicity grid!')
            else:
                tau_grid = gfile['grid/tau'][:]
                feh_grid = gfile['grid/feh'][:]
            for starid in gfile['gfuncs']:
                if stars is None or starid in stars:
                    gfunc = gfile['gfuncs/' + starid][:]
                    gfunc = smooth_gfunc2d(gfunc)
                    gfunc = norm_gfunc(gfunc)
                    g2d.append(gfunc)

    g2d = np.array(g2d)

    # Make grid more coarse (optionally, increases performance)
    if dilut is not None:
        g2d = g2d[:, ::dilut, ::dilut]
        tau_grid = tau_grid[::dilut]
        feh_grid = feh_grid[::dilut]

    # Number of tau/feh-values and number of stars
    l = len(feh_grid)
    m = len(tau_grid)
    n = g2d.shape[0]

    if case == '1D':
        g = np.sum(g2d, axis=2)
        k = m
    elif case == '2D':
        k = m*l
        g = g2d.reshape(n, k, order='F')

    #------------------------------------------------
    # Set up for estimating age distribution phi(1:k)
    #------------------------------------------------

    # weights for integrals over theta
    w = np.ones(k) / k

    # constant prior, normalized
    Phi = np.ones(k)
    Phi /= np.dot(w, Phi)

    # initial guess for phi and lambda
    phi = Phi
    lamda = -1 # this gives r_j = 0 for beta = 0

    # initial beta and step
    if betas is None:
        beta, dbeta, beta_max = 0.01, 0.01, 1.00
    else:
        beta, dbeta, beta_max = betas

    # max number of Newton-Raphson iterations per beta
    max_iter = 10
    min_tol = 1.e-20

    # Gw = G matrix, with each column multiplied by w(j)
    gw = np.zeros((n, k))
    for j in range(k):
        gw[:,j] = g[:,j] * w[j];

    # array to hold Gwu = Gw matrix, each row divided by u(i)
    gwu = np.zeros((n, k))

    finished = False
#    y_max_plot = 5
#    if figdir is None:
#        figdir = './'

    # list to hold beta, L, E
    Q = []
    # list to hold phi (the age/age-metallicity distribution)
    samd = []
#    # set-up for plotting phi
#    if case == '1D':
#        fig, ax = plt.subplots()
#        phi_plot, = ax.plot(tau_grid, phi)
#        ax.set_xlabel('Age [Gyr]')
#        ax.set_ylabel('Relative frequency')
#        ax.set_title('beta = ' + str(round(beta, 4)))
#        ax.axhline(1, c='g', ls='--')
#        ax.set_xlim([0, tau_grid[-1]])
#        ax.set_ylim([0, y_max_plot])
#        ax.grid()
#        fig.savefig(os.path.join(figdir, 'samd1d_0.pdf'))
#    elif case == '2D':
#        fig, ax = plt.subplots()
#        dtau = tau_array[1] - tau_array[0]
#        dfeh = feh_array[1] - feh_array[0]
#        plot_lims = (tau_array[0]-dtau, tau_array[-1]+dtau,
#                     feh_array[0]-dfeh, feh_array[-1]+dfeh)
#        cax = ax.imshow(phi.reshape(m, l), origin='lower', extent=plot_lims,
#                        aspect='auto', interpolation='none')
#        cbar = plt.gcf().colorbar(cax)
#        cbar.set_label('Relative frequency')
#        ax.set_xlabel('Age [Gyr]')
#        ax.set_ylabel('[Fe/H]')
#        ax.set_title('beta = ' + str(round(beta, 4)))
#        ax.grid()
#        ax.set_xlim(plot_lims[:2])
#        ax.set_ylim(plot_lims[2:])
#        fig.savefig(os.path.join(figdir, 'samd2d_0.pdf'))

    beta_i = 1
    while not finished:
        for iterr in range(max_iter):
            u = np.dot(gw, phi)

            for i in range(n):
                gwu[i, :] = gw[i, :] / u[i]

            # residuals
            r = w * (1 + np.log(phi/Phi)) - beta * np.sum(gwu, 0) + lamda * w
            R = np.dot(w, phi) - 1

            # Hessian
            H = np.diag(w / phi) + beta * np.dot(gwu.T, gwu)

            # full matrix
            M = np.zeros((k+1, k+1))
            M[:k, :k] = H
            M[-1, :k] = M[:k, -1] = w
            h = np.append(-r, -R)

            s = np.append(1/np.sqrt(np.diag(H)), 1.)
            S = np.diag(s)
            M1 = np.dot(np.dot(S, M), S)
            h1 = np.dot(S, h)

            con = np.linalg.cond(M1)
            if con is np.inf:
                finished = True
                break

            Delta1 = np.linalg.solve(M1, h1)
            Delta = np.dot(S, Delta1)

            Delta_phi = Delta[:k]
            Delta_lambda = Delta[-1]

            f = 1.
            phi_test = phi + f * Delta_phi
            while min(phi_test) < 0:
                f *= 0.5
                phi_test = phi + f * Delta_phi
            phi = phi_test
            lamda += f * Delta_lambda
            if min(phi) < min_tol or beta >= beta_max:
                finished = True
                break

        # re-normalise to avoid exponential growth of rounding errors
        phi /= np.dot(w, phi)
        if case == '1D':
            samd.append(phi)
        elif case == '2D':
            samd.append(phi.reshape(m, l, order='F'))

        # entropy
        E = np.sum(w * phi * np.log(phi / Phi))

        # total negative log-likelihood
        L = -np.sum(np.log(u))

        # add to list Q
        Q.append([beta, L, E])

        # plot current phi
        # need to reduce y-scale?
#        if np.amax(phi) > y_max_plot*0.95:
#            y_max_plot *= 2
#
#        ax.set_title('beta = ' + str(round(beta, 4)))
#        if case == '1D':
#            ax.set_ylim([0, y_max_plot])
#            phi_plot.set_ydata(phi)
#            fig.savefig(os.path.join(figdir, 'samd1d_' + str(beta_i) + '.pdf'))
#        elif case == '2D':
#            continue

        beta += dbeta
        beta_i += 1

    return samd, Q, tau_grid, feh_grid
