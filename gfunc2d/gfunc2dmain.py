import os
import sys
import h5py
import numpy as np
from datetime import datetime
import gfunc2d.gridtools as gt
from gfunc2d.marg_mu import marginalise_mu as margm
from gfunc2d.gplot import loglik_save, contour_save, hr_save
from gfunc2d.gstats import print_age_stats


def gfunc2d(isogrid, fitparams, alpha, isodict=None):
    '''
    Python version of the MATLAB script gFunc2D.m
    C Sahlholdt, 2017 Oct 27 (translated to Python)
    L Howes, Lund Observatory, 2016 Sep 15 (adapted to YY and plx)
    L Lindegren, 2016 Oct 4 (returning 2D G function)
        - based on gFunc.m by L. Lindegren.

    Calculates, for a single star, a 2D array of the G-function as a function
    of age and metallicity.

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
        into the memory (very useful when looping this function over several
        stars).

    Returns
    -------
    g2D : array of float
        2D array of the G-function as a function of age (rows) and metallicity
        (columns).

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
        alphas, fehs, ages = gt.get_afa_arrays(gridfile)

        # Check that the chosen [alpha/Fe] is available in the grid
        if alpha not in alphas:
            raise ValueError('[alpha/Fe] = ' + str(alpha) +\
                             ' not found in grid. ' +\
                             'Change alpha to one of the following: '+\
                             str(alphas))

        # Check that the chosen fitparams are accommodated by the grid
        # and add metadata (attribute) used in the fitting proccess.
        fitparams, app_mag = gt.prepare_fitparams(gridfile, fitparams)

        # The hdf5 grid is loaded into a python dictionary if the dictionary
        # has not been loaded (and passed to this function) in advance.
        if isodict is None:
            isodict = gt.load_as_dict(gridfile, (alpha, alpha))

    # Initialize g-function
    g2D = np.zeros((len(ages), len(fehs)))

    for i_feh, feh in enumerate(fehs):
        for i_age, age in enumerate(ages):
            g2D_i = 0

            # Get the hdf5 path to the desired isochrone and pick out the
            # isochrone
            isopath = gt.get_isopath(alpha, feh, age)
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
                        lik_int_mu[i_mass] = margm(plx_obs, plx_unc, obs_mag,
                                                   obs_unc, iso_mag, mu_prior,
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


def gfunc2d_run(inputfile, isogrid, outputdir, inputnames, fitnames,
                alpha=0.0, make_gplots=True, make_hrplots=False,
                output_ages=True):
    '''
    docstring
    '''

    # Get the date and time
    time = datetime.now().isoformat(timespec='minutes')

    # Check if outputfile already exists
    output_h5 = os.path.join(outputdir, 'output.h5')
    if os.path.exists(output_h5):
        raise IOError(output_h5 + ' already exists. Move/remove it and try again.')

    # Create output directories if they do not exist
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        print('Created output directory: ' + outputdir)
    if not os.path.exists(os.path.join(outputdir, 'figures')):
        if make_gplots or (make_hrplots is not False):
            os.makedirs(os.path.join(outputdir, 'figures'))
            print('Created figure directory: ' +\
                  os.path.join(outputdir, 'figures'))

    # Prepare output hdf5 groups and fill in header
    with h5py.File(output_h5) as h5out:
        h5out.create_group('header')
        h5out.create_group('gfuncs')
        h5out.create_group('grid')

        h5out['header'].create_dataset('inputfile', data=np.string_(inputfile))
        h5out['header'].create_dataset('isogrid', data=np.string_(isogrid))
        h5out['header'].create_dataset('fitnames', data=np.string_(fitnames))
        h5out['header'].create_dataset('datetime', data=np.string_(time))
        h5out['header'].create_dataset('alpha', data=alpha)

    # Load stellar data
    data = np.genfromtxt(inputfile, dtype=None, names=inputnames)
    # Get indices of inputnames which should be fitted
    fit_inds = [inputnames.index(x) for x in fitnames]

    # Load isochrones into memory in the form of a python dictionary
    # Also get available parameters in the grid and their units
    print('\nLoading isochrones into memory...', end=''); sys.stdout.flush()
    with h5py.File(isogrid, 'r') as gridfile:
        isodict = gt.load_as_dict(gridfile, alpha_lims=(alpha-0.01, alpha+0.01))
        gridparams, gridunits = gt.get_gridparams(gridfile, return_units=True)
    print(' done!\n')

    # Loop over stars in the input file
    for i, name in enumerate(data['name']):
        # Set stellar name and data
        name = name.decode('ASCII')
        data_i = data[i]

        # Make fitparams dictionary
        fitparams = {inputnames[k]: (data_i[k], data_i[k+1]) for k in fit_inds}

        # Compute G-function
        print('Processing ' + name + '...', end=''); sys.stdout.flush()
        g, tau_array, feh_array = gfunc2d(isogrid, fitparams,
                                          alpha, isodict=isodict)

        # Save G-function
        with h5py.File(output_h5) as h5out:
            h5out['gfuncs'].create_dataset(name, data=g)

        # Save plots of the G-function if make_plots
        if make_gplots:
            loglik_name = os.path.join(outputdir, 'figures', name + '_loglik.pdf')
            contour_name = os.path.join(outputdir, 'figures', name + '_contour.pdf')
            loglik_save(g, tau_array, feh_array, loglik_name)
            contour_save(g, tau_array, feh_array, contour_name)

        if make_hrplots is not False:
            hr_axes = make_hrplots
            try:
                hrx_data_index = inputnames.index(make_hrplots[0])
                hry_data_index = inputnames.index(make_hrplots[1])
                hrx_grid_index = gridparams.index(make_hrplots[0])
                hry_grid_index = gridparams.index(make_hrplots[1])
            except:
                raise ValueError('Both of ' + str(make_hrplots) +\
                                 ' must be in inputnames and in gridparams!')
            try:
                feh_index = inputnames.index('FeH')
                feh_i = data_i[feh_index]
            except:
                feh_i = 0
            hr_vals = (data_i[hrx_data_index], data_i[hry_data_index])
            hr_units = (gridunits[hrx_grid_index], gridunits[hry_grid_index])
            hr_name = os.path.join(outputdir, 'figures', name + '_hr.pdf')
            if hr_units[1] == 'mag':
                plx = data_i[inputnames.index('plx')]
                hr_save(isodict, hr_axes, hr_vals, hr_units, hr_name,
                        par=plx, feh=feh_i)
            else:
                hr_save(isodict, hr_axes, hr_vals, hr_units, hr_name,
                        feh=feh_i)

        # Print progress
        print(' ' + str(round((i+1) / len(data) * 100)) + '%')

    # Save (tau, feh)-grid
    with h5py.File(output_h5) as h5out:
        h5out['grid'].create_dataset('tau', data=tau_array)
        h5out['grid'].create_dataset('feh', data=feh_array)

    # Optionally, write ages to text file
    if output_ages:
        output_ages_file = os.path.join(outputdir, 'ages.txt')
        print_age_stats(output_h5, output_ages_file)
