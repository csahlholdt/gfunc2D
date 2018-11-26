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
    '''
    Smooth a 2D G function.

    Parameters
    ----------
    g : array
        2D G function.

    Returns
    -------
    g2 : array
        Smoothed 2D G function.
    '''

    kernel = np.array([0.25, 0.5, 0.25])
    func = lambda x: np.convolve(x, kernel, mode='same')
    g1 = np.apply_along_axis(func, 0, g)
    g2 = np.apply_along_axis(func, 1, g1)

    return g2


def norm_gfunc(g, method='maxone'):
    '''
    Normalise G function.

    Parameters
    ----------
    g : array
        1D/2D G function.

    method : str, optional
        Normalisation method.
        Default is 'maxone' which scales the maximum value to unity.

    Returns
    -------
    gnorm : array
        Normalised G function.
    '''

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
    '''
    Get the 1D age G function from the 2D G function.

    Parameters
    ----------
    g : array
        1D/2D G function.

    norm : bool, optional
        Normalise the G function before returning.
        Default value is True.

    norm_method : str, optional
        Normalisation method to use if `norm=True`.
        Default is 'maxone'.

    Returns
    -------
    g_age : array
        1D age G function.
    '''

    g_age = np.sum(g, axis=1)
    if norm:
        g_age = norm_gfunc(g_age, norm_method)

    return g_age


def gfunc_age_mode(g_age, age_grid):
    '''
    Get the mode of a 1D age G function.

    Parameters
    ----------
    g_age : array
        1D age G function.

    age_grid: array
        The age values on which `g_age` is defined.
        Must be same length as g_age.

    Returns
    -------
    age_mode : array
        Mode of the 1D age G function.
    '''

    ind = np.argmax(g_age)
    age_mode = age_grid[ind]

    return age_mode


def conf_glim(conf_level):
    '''
    Get the limiting value of a 1D age G function corresponding
    to a certain age confidence level.
    This is calculating assuming that the G function can be approximated
    by a Gaussian.

    Parameters
    ----------
    conf_level : float
        Confidence level as a fraction (between 0 and 1).

    Returns
    -------
    glim : float
        Limit on the G function setting the confidence limits.
    '''

    assert conf_level > 0 and conf_level < 1

    zero_func = lambda x: 2*norm.cdf(np.sqrt(-2*np.log(x))) - 1 - conf_level
    glim = newton(zero_func, 0.6)

    return glim


def gfunc_age_conf(g_age, age_grid, conf_level=0.68):
    '''
    Get the confidence interval of the age from a 1D G function.

    Parameters
    ----------
    g_age : array
        1D age G function.

    age_grid : array
        The age values on which `g_age` is defined.
        Must be same length as g_age.

    conf_level : float
        Confidence level as a fraction (between 0 and 1).
        Default value is 0.68 corresponding to 1 sigma for a Gaussian.

    Returns
    -------
    age_conf : tuple
        Lower and upper limit on the confidence interval of the age.
        None is returned if no limit exists (the G function does not fall
        below the critical value given by the confidence level before
        hitting the edge of the age_grid).
    '''

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
    '''
    Function for printing ages and confidence intervals to a text file
    based on an output hdf5 file (containing the 2D G functions).
    The age statistics which are printet are the mode of the G function as
    well as the 68 and 90% confidence intervals (this can be changed in
    the source code).

    Parameters
    ----------
    output_h5 : str
        Path to the output hdf5 file.

    filename : str
        Name of the text file with the age output.

    conf_level : float
        Confidence level as a fraction (between 0 and 1).
        Default value is 0.68 corresponding to 1 sigma for a Gaussian.
    '''

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
                  max_iter=10, min_tol=1.e-20):
    '''
    Function for estimating the sample age metallicity distribution (samd) OR
    simply the sample age distribution (sad).

    This uses a Newton-Raphson minimisation to find the function phi which
    maximises the likelihood L(phi) = sum(L_i(phi)), where
        L_i(phi) = int(G_i(theta)*phi(theta)) ,
    and G_i are the G functions and theta is either the age (in the 1D case)
    or both the age and metallicity (in the 2D) case.

    Parameters
    ----------
    gfunc_files : list
        List of paths to the output hdf5 files containing the 2D G functions.
        If more than one, the output in the different files MUST be defined on
        the same age/metallicity grids.

    case : str
        Determines whether the 2D (samd) or 1D (sad) is calculated.
        '2D' for samd and '1D' for sad. Default is '1D'.

    betas : tuple
        Beta is a regularization parameter which regulates how strongly the
        solution favors a flat (constant) function (0 is most strict, higher
        numbers are less strict).
        betas should be a tuple containing the three floats beta, dbeta, and
        beta_max. beta is the initial value, dbeta is the step, and beta_max is
        the maximum value which, if hit, stops the computation.
        beta (the initial value) should be close to 0 and dbeta not too large
        to allow a gentle convergence towards a sensible solution.
        Default is None in which case the values (0.01, 0.01, 1.00) are used.

    stars : list of str
        List of star identifiers (as used in the gfunc_files) to be included in
        the calculation.
        Default is None in which case all stars are included.

    dilut : int
        Dilution factor. If specified, only every `dilut`th age and metallicity
        grid point is considered. This increases performance by lowering the
        size of the problem.
        Default is None in which case all grid points are considered.

    max_iter : int
        Maximum number of Newton-Raphson iterations per beta.
        Default value is 10

    min_tol : float
        Minimum value that the samd/sad must reach in order to end the
        calculation.

    Returns
    -------
    samd : list
        List of samd/sad with one entry for each value of beta.

    Q : list
        List of same length as `samd`. Each entry is a list giving the values of
        beta, the negative log-likelihood of the solution, and its entropy.

    tau_grid : array
        Age grid on which the input G functions were defined (taken from on of
        the `gfunc_files`).

    feh_grid : array
        Metallicity grid on which the input G functions were defined (taken
        from on of the `gfunc_files`).
    '''
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

    # Define matrix with n g-functions
    # (in 2D case each g-function is flattened first)
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

    # Gw = G matrix, with each column multiplied by w(j)
    gw = np.zeros((n, k))
    for j in range(k):
        gw[:,j] = g[:,j] * w[j];

    # array to hold Gwu = Gw matrix, each row divided by u(i)
    gwu = np.zeros((n, k))

    finished = False

    # list to hold beta, L, E
    Q = []
    # list to hold phi (the age/age-metallicity distribution)
    samd = []

    # Perform Newton-Raphson minimisation
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

        beta += dbeta

    return samd, Q, tau_grid, feh_grid
