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
from scipy.linalg import block_diag


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
        Default is 'maxone' which scales the max value to one.

    Returns
    -------
    gnorm : array
        Normalised G function.
    '''

    gmax = np.amax(g)
    if method == 'maxone':
        if gmax == 0:
            gnorm = np.ones_like(g)
        else:
            gnorm = g / np.amax(g)
    elif method == 'sumone':
        if gmax == 0:
            gnorm = np.ones_like(g) / np.size(g)
        else:
            gnorm = g / np.sum(g)
#    elif method == 'some other method':
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


def gfunc_age_quantile(g_age, age_grid, q):
    '''
    The the q'th quantile of the 1D age G function
    interpolated onto the age grid.
    (q = 0.5 gives the median age)

    Parameters
    ----------
    g_age : array
        1D age G function.

    age_grid: array
        The age values on which `g_age` is defined.
        Must be same length as g_age.

    q : float
        Quantile to compute, which must be between 0 and 1.

    Returns
    -------
    age_q : float
        Age quantile.
    '''

    g_age_cumsum = np.cumsum(g_age)
    g_age_cumsum /= g_age_cumsum[-1]
    age_q_ind = np.argmin((np.abs(g_age_cumsum-q)))
    age_q = age_grid[age_q_ind]

    return age_q


def gfunc_age_mode(g_age, age_grid, use_median=False):
    '''
    Get the mode or mean of a 1D age G function.

    Parameters
    ----------
    g_age : array
        1D age G function.

    age_grid: array
        The age values on which `g_age` is defined.
        Must be same length as g_age.

    use_median : bool, optional
        Use median instead of mode if True.
        Default is False.

    Returns
    -------
    age_mode : float
        Mode (or mean) of the 1D age G function.
    '''

    if use_median:
        age_mode = gfunc_age_quantile(g_age, age_grid, 0.5)
    else:
        ind = np.argmax(g_age)
        age_mode = age_grid[ind]

    return age_mode


def conf_glim(conf_level):
    '''
    Get the limiting value of a 1D age G function corresponding
    to a certain age confidence level.
    This is calculated assuming that the G function can be approximated
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


def gfunc_age_conf(g_age, age_grid, conf_level=0.68, use_median=False):
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

    if use_median:
        age_low = gfunc_age_quantile(g_age, age_grid, 0.5-conf_level/2)
        age_high = gfunc_age_quantile(g_age, age_grid, 0.5+conf_level/2)
    else:
        glim = conf_glim(conf_level)
        ages_lim = age_grid[g_age > glim]
        age_low, age_high = ages_lim[0], ages_lim[-1]

    if age_low == age_grid[0]:
        age_low = None
    if age_high == age_grid[-1]:
        age_high = None
    age_conf = (age_low, age_high)

    return age_conf

  
def age_mode_and_conf(g_age, age_grid, conf_levels=[0.68, 0.90],
                      use_median=False):
    '''
    Get the mode (or mean) and confidence interval for a 1D age G function
    with a number of confidence intervals.

    Parameters
    ----------
    g_age : array
        1D age G function.

    age_grid : array
        The age values on which `g_age` is defined.
        Must be same length as g_age.

    conf_levels : list of float
        Confidence levels as fractions (between 0 and 1).
        Default value is [0.68, 0.90].

    use_median : bool, optional
        Use median instead of mode if True.
        Default is False.

    Returns
    -------
    age_arr : array
        List of length 1+2*len(`conf_levels`). The middle entry is
        the age mode (or mean), and the surrounding values are the
        confidence interval.
        For example with conf_levels=[0.68, 0.90] it returns the list
        [5%, 16%, mode, 84%, 95%].
    '''

    n = len(conf_levels)
    age_arr = np.zeros(1+2*n)

    age_arr[n] = gfunc_age_mode(g_age, age_grid, use_median)
    for i in range(1, n+1):
        try:
            age_arr[n-i:n+i+1:2*i] = gfunc_age_conf(g_age, age_grid,
                                                    conf_level=conf_levels[i-1],
                                                    use_median=use_median)
        except:
            age_arr[:] = None
            break

    return age_arr


def print_age_stats(output_h5, filename, smooth=False, use_median=False):
    '''
    Function for printing ages and confidence intervals to a text file
    based on an output hdf5 file (containing the 2D G functions).
    The age statistics which are printed are the mode of the G function as
    well as the 68 and 90% confidence intervals (this can be changed in
    the source code).

    Parameters
    ----------
    output_h5 : str
        Path to the output hdf5 file.

    filename : str
        Name of the text file with the age output.

    smooth : bool
        If True, smooth the G functions before calculating the ages and
        confidence intervals.
        Note: This only applies to 2D G functions, if the G functions in
        output_h5 are 1D, nothing happens.
        Default value is False.

    use_median : bool, optional
        Use median of G function instead of mode if True.
        Default is False.
    '''

    with h5py.File(output_h5, 'r') as out:
        ages = out['grid/tau'][:]
        gf_group = out['gfuncs']
        if len(gf_group) == 1:
            star_id = np.array([star for star in gf_group])
        else:
            star_id = np.array(gf_group)
            try:
                star_id_sort = np.argsort([int(x) for x in star_id])
                star_id = star_id[star_id_sort]
            except ValueError:
                print('star_id not sorted in '+filename)

        n_star = len(star_id)
        age_arr = np.zeros((n_star, 5))
        for i, star in enumerate(star_id):
            g = gf_group[star][:]
            gdim = g.ndim
            if gdim == 2:
                if smooth:
                    g = smooth_gfunc2d(g)
                g_age = gfunc_age(g)
            else:
                g_age = g
            g_age = norm_gfunc(g_age)
            age_arr[i] = age_mode_and_conf(g_age, ages, use_median=use_median)

        # Pad identifier strings (for prettier output)
        id_len = max((10, max([len(x) for x in star_id])))
        star_id_pad = [x.ljust(id_len) for x in star_id]

        # Combine identifiers and data in DataFrame and write to txt
        pd_arr = pd.DataFrame(age_arr, index=star_id_pad)
        astr = 'aMedian' if use_median else 'aMode'
        pd_arr.to_csv(filename, sep='\t', index_label='#IDnumber',
                      header=['a5', 'a16', astr, 'a84', 'a95'],
                      float_format='%2.2f', na_rep='nan')


def estimate_samd(gfunc_files, case='1D', betas=None, alpha=0, stars=None,
                  grid_slice=None, grid_thin=None, max_iter=10, min_tol=1.e-20):
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

    case : str, optional
        Determines whether the 2D (samd) or 1D (sad) is calculated.
        '2D' for samd and '1D' for sad. Default is '1D'.

    betas : tuple, optional
        Beta is a regularization parameter which regulates how strongly the
        solution favors a flat (constant) function (0 is most strict, higher
        numbers are less strict).
        betas should be a tuple containing the three floats beta, dbeta, and
        beta_max. beta is the initial value, dbeta is the step, and beta_max is
        the maximum value which, if hit, stops the computation.
        beta (the initial value) should be close to 0 and dbeta not too large
        to allow a gentle convergence towards a sensible solution.
        Default is None in which case the values (0.01, 0.01, 1.00) are used.

    alpha : int, optional
        Value of the smoothing parameter. Higher values will favor solutions
        with smaller point-to-point variations (first derivatives).
        Not implemented proberly in the '2D' case.
        Default value is 0.

    stars : list of str, optional
        List of star identifiers (as used in the gfunc_files) to be included in
        the calculation.
        Default is None in which case all stars are included.

    grid_slice : tuple of ints, optional
        Grid slice indices. If specified, it must be a tuple of four integers,
        and only the ages from grid_slice[0] to grid_slice[1] and the
        metallicities from grid_slice[2] to grid_slice[3] are considered.
        This increases performance by decresaing the size of the problem.
        Default value is None in which case all grid points are considered.
        Note that if gfunctions are saved as 1D, only the first two integers
        are used (it is too late to thin in metallicity).

    grid_thin : tuple of ints, optional
        Thinning factor. If specified, it must be a tuple of two integers, and
        only every `grid_thin[0]`th age and every `grid_thin[1]`th metallicity
        grid point is considered. This increases performance by decreasing the
        size of the problem.
        This thinning is performed after slicing with grid_slice.
        Default is None in which case all grid points (in the grid_slice
        selection) are considered.

    max_iter : int, optional
        Maximum number of Newton-Raphson iterations per beta.
        Default value is 10

    min_tol : float, optional
        Minimum value that the samd/sad is allowed to reach.
        Default value is 1e-20.

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
    for i, gfunc_file in enumerate(gfunc_files):
        # Allow for stars to be a list of lists (one for each gfunc-file)
        if stars is not None and isinstance(stars[0], list):
            stars_i = stars[i]
        elif stars is not None:
            stars_i = stars
        with h5py.File(gfunc_file, 'r') as gfile:
            saved_2d = gfile['header/save2d'][()].decode('ascii') == 'True'
            if not saved_2d and case == '2D':
                raise ValueError('Need 2D functions in output for case="2D"')
            if tau_grid is not None and feh_grid is not None:
                tau_grid_new = gfile['grid/tau'][:]
                feh_grid_new = gfile['grid/feh'][:]
                if not (np.array_equal(tau_grid, tau_grid_new)\
                        and np.array_equal(feh_grid, feh_grid_new)):
                    raise ValueError('All g-functions must be defined on ' +\
                                     'the same age/metallicity grid!')
            else:
                tau_grid = gfile['grid/tau'][:]
                feh_grid = gfile['grid/feh'][:]
            for starid in gfile['gfuncs']:
                if stars is None or starid in stars_i:
                    gfunc = gfile['gfuncs/' + starid][:]
                    # Make gfunc more coarse (optionally, increases performance)
                    if grid_slice is not None:
                        if saved_2d:
                            gfunc = gfunc[grid_slice[0]:grid_slice[1],
                                          grid_slice[2]:grid_slice[3]]
                        else:
                            gfunc = gfunc[grid_slice[0]:grid_slice[1]]

                    if grid_thin is not None:
                        if saved_2d:
                            gfunc = gfunc[::grid_thin[0], ::grid_thin[1]]
                        else:
                            gfunc = gfunc[::grid_thin[0]]
                    #gfunc = smooth_gfunc2d(gfunc)
                    gfunc = norm_gfunc(gfunc)
                    g2d.append(gfunc)

    g2d = np.array(g2d)

    # Make grid more coarse (optionally, increases performance)
    if grid_slice is not None:
        tau_grid = tau_grid[grid_slice[0]:grid_slice[1]]
        feh_grid = feh_grid[grid_slice[2]:grid_slice[3]]
    if grid_thin is not None:
        tau_grid = tau_grid[::grid_thin[0]]
        feh_grid = feh_grid[::grid_thin[1]]

    # Number of tau/feh-values and number of stars
    l = len(feh_grid)
    m = len(tau_grid)
    n = g2d.shape[0]

    # Define matrix with n g-functions
    # (in 2D case each g-function is flattened first)
    if case == '1D':
        if saved_2d:
            g = np.sum(g2d, axis=2)
        else:
            g = g2d
        k = m
    elif case == '2D':
        k = m*l
        # reshape with "age-order" (each row is a sequence of functions
        # like in the '1D' case)
        g = g2d.reshape(n, k, order='F')
    # add small number to avoid log(0)
    g += 1e-10

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
    gw = g * w

    # Derivative matrix
    T = np.diag([-1]+(m-2)*[-2]+[-1])
    T += np.diag((m-1)*[1], k=1)
    T += np.diag((m-1)*[1], k=-1)

    if case == '2D':
        T1 = np.diag(np.ones(m)*(-3))
        T1[0][0] = T1[-1][-1] = -2
        T1 += np.diag(np.ones(m-1), k=1)
        T1 += np.diag(np.ones(m-1), k=-1)
        T2 = np.diag(np.ones(m)*(-4))
        T2[0][0] = T2[-1][-1] = -3
        T2 += np.diag(np.ones(m-1), k=1)
        T2 += np.diag(np.ones(m-1), k=-1)
        T_repeat = [T1] + [T2 for i in range(l-2)] + [T1]
        T = block_diag(*T_repeat)
        T += np.diag(np.ones(k-m), k=m)
        T += np.diag(np.ones(k-m), k=-m)

    # Tw = T matrix, with each column multiplied by w(j)
    Tw = T * w

    # list to hold beta, L, E, R
    Q = []
    # list to hold phi (the age/age-metallicity distribution)
    samd = []

    # Perform Newton-Raphson minimisation
    finished = False
    while not finished:
        for iterr in range(max_iter):
            u = np.dot(gw, phi)
            v = np.dot(Tw, phi)

            u[u == 0] = 1 # avoid division by zero
            gwu = gw / u[:, np.newaxis]
            Twv = Tw * v[:, np.newaxis]

            # residuals
            r = w * (1 + np.log(phi/Phi)) - beta * np.sum(gwu, 0) \
                + 2*alpha*beta * np.sum(Twv, 0) + lamda * w
            R = np.dot(w, phi) - 1

            # Hessian
            H = np.diag(w / phi) + beta * np.dot(gwu.T, gwu) \
                + 2*alpha*beta * np.dot(Tw.T, Tw)

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

            phi[phi < min_tol] = min_tol
            if beta >= beta_max:
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
        L = -np.sum(np.log(np.dot(gw, phi)))

        # total regularization term
        R = np.sum(np.dot(Tw, phi)**2)

        # add to list Q
        Q.append([beta, L, E, R])

        beta += dbeta

    return samd, Q, tau_grid, feh_grid
