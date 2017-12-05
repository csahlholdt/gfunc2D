"""
Python version of the MATLAB script gFunc2D.m
    C Sahlholdt, 2017 Oct 27 (translated to Python)
    L Howes, Lund Observatory, 2016 Sep 15 (adapted to YY and plx)
    L Lindegren, 2016 Oct 4 (returning 2D G function)
        - based on gFunc.m by L. Lindegren.
"""

import numpy as np
import h5py
import gridtools
from marg_mu import marginalise_mu


def gfunc2D(isogrid, fitparams, alpha, isodict=None):
    '''
    Returns a 2D array of the normalised G-function as a function of age and
    metallicity, along with the corresponding arrays of ages and metallicities.

    Parameters
    ----------
    isogrid : str
        Name of the isochrone hdf5 grid (including the full path).
    fitparams : dict
        Dictionary consisting of the parameters to be fitted in the format
        fitparams = {'param_name': (value, uncertainty), ...}.
    alpha : float
        Value of [alpha/Fe]. Must exist in the grid.
    isodict : dict, optional
        The isochrone hdf5 grid loaded as a dictionary using
        gridtools.load_as_dict(). Supplying this dictionary is optional but it
        speeds up the code significantly since the data has already been loaded
        into the memory.

    Returns
    -------
    g2D : array of float
        2D array of the normalised G-function as a function of age (rows) and
        metallicity (columns).
    ages : array of float
        Array of ages in the rows of g2D.
    fehs : array of float
        Array of metallicities in the columns of g2D.
    '''
    # Exponent for power-law IMF
    beta = 2.7

    # Prior on the distance modulus
    mu_prior = 10
    # Statistical weight on mu_prior
    mu_prior_w = 0.0

    with h5py.File(isogrid, 'r') as gridfile:
        # Get arrays of alpha, metallicities, and ages
        alphas, fehs, ages = gridtools.get_afa_arrays(gridfile)

        # Check that the chosen [alpha/Fe] is available in the grid
        if alpha not in alphas:
            raise ValueError('[alpha/Fe] = ' + str(alpha) +\
                             ' not found in grid. ' +\
                             'Change alpha to one of the following: '+\
                             str(alphas))

        # Check that the chosen fitparams are accommodated by the grid
        # and add metadata (attribute) used in the fitting proccess.
        fitparams, app_mag = gridtools.prepare_fitparams(gridfile, fitparams)

        # The hdf5 grid is loaded into a python dictionary if the dictionary
        # has not been loaded (and passed to this function) in advance.
        if isodict is None:
            isodict = gridtools.load_as_dict(gridfile, (alpha, alpha))

    # Initialize g-function
    g2D = np.zeros((len(ages), len(fehs)))

    for i_feh, feh in enumerate(fehs):
        for i_age, age in enumerate(ages):
            g2D_i = 0

            # Get the hdf5 path to the desired isochrone and pick out the
            # isochrone
            isopath = gridtools.get_isopath(alpha, feh, age)
            iso_i = isodict[isopath]

            # Get mass array and calculate the change in mass for each
            # entry based on the two surrounding entries
            masses = iso_i['Mini']
            dm = (masses[2:] - masses[:-2]) / 2

            # Pick out the values for which the change in mass is positive
            pdm = dm > 0
            masses = masses[1:-1][pdm]
            dm = dm[pdm]

            # Calculate total X2 (chi2) for all parameters but the
            # distance modulus
            chi2 = np.zeros(len(masses))
            for param in fitparams:
                if param == 'plx':
                    continue

                obs_val, obs_unc, attr = fitparams[param]
                if attr == 'none':
                    if param == 'FeH':
                        iso_val = feh*np.ones(len(masses))
                    elif param == 'logT' or param == 'logL':
                        iso_val = 10**iso_i[param][1:-1][pdm]
                    else:
                        iso_val = iso_i[param][1:-1][pdm]
                    chi2 += ((obs_val - iso_val)/obs_unc)**2
                elif attr == 'color':
                    colors = param.split('-')
                    m1 = iso_i[colors[0]][1:-1][pdm]
                    m2 = iso_i[colors[1]][1:-1][pdm]
                    iso_val =  m1 - m2
                    chi2 += ((obs_val - iso_val)/obs_unc)**2

            # The value of the G-function is calculated based on all models
            # with X2 < 100 (if any)
            low_chi2 = chi2 < 100
            if any(low_chi2):
                # Initial mass function (prior on mass)
                phi_M = masses[low_chi2]**(-beta)

                # X2 and mass change for models with low X2
                chi2 = chi2[low_chi2]
                dm = dm[low_chi2]

                # The array to be summed in order to marginalise over the
                # mass
                marg_m = dm * phi_M * np.exp(-0.5*chi2)

                # At this step the marginalisation over the distance
                # modulus is carried out for each model individually (i.e.
                # for each mass) if an apparent magnitude is in fitparams.
                if app_mag is not None:
                    lik_int_mu = np.ones(len(chi2))
                    obs_mag, obs_unc = fitparams[app_mag][:2]
                    iso_mags = iso_i[app_mag][1:-1][pdm][low_chi2]
                    plx_obs, plx_unc = fitparams['plx'][:2]
                    for i_mass in range(len(chi2)):
                        iso_mag = iso_mags[i_mass]
                        lik_int_mu[i_mass] = marginalise_mu(plx_obs,
                                                            plx_unc,
                                                            obs_mag,
                                                            obs_unc,
                                                            iso_mag,
                                                            mu_prior,
                                                            mu_prior_w)

                    # The marginalisation over mass is carried out to give
                    # the value of the G-function for the current
                    # metallicity and age.
                    g2D_i = np.sum(marg_m * lik_int_mu)
                else:
                    # If no apparent magnitude is given
                    g2D_i = np.sum(marg_m)

            g2D[i_age, i_feh] += g2D_i

    return g2D, ages, fehs
