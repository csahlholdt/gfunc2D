import os
import sys
import h5py
import numpy as np
from datetime import datetime
import gfunc2d.gridtools as gt
from gfunc2d.marg_mu import marginalise_mu as margm
from gfunc2d.marg_mu import marginalise_mu_simple as margm2
from gfunc2d.gplot import loglik_save, contour_save, hr_save
from gfunc2d.gstats import print_age_stats
from gfunc2d.utilities import is_color


def gfunc2d(isogrid, fitparams, alpha, isodict=None, margm_fast=True):
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

    margm_fast : bool, optional
        If fitting to the parallax ('plx' in fitparams), one can choose a fast
        method for the marginalisation over the distance modulus by setting this
        value to True. A slower (but slightly more exact) method is used otherwise.
        Default value is True.

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

    ### Some fixed parameters which could be given as input instead
    # Exponent for power-law IMF
    beta = 2.35

    # Prior on the distance modulus
    mu_prior = 10
    # Statistical weight on mu_prior
    mu_prior_w = 0.0
    ###

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

            # Calculate total chi2 for all parameters but the distance modulus
            # The parallax is skipped explicitly, and any magnitudes are
            # skipped due to their attribute being 'mag' which is not handled.
            chi2 = np.zeros(len(masses))
            for param in fitparams:
                if param == 'plx':
                    continue

                obs_val, obs_unc, attr = fitparams[param]
                if attr == 'none':
                    if param == 'logT' or param == 'logL':
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
            # with chi2 < 100 (if any)
            low_chi2 = chi2 < 100
            if any(low_chi2):
                # Initial mass function (prior on mass)
                phi_M = masses[low_chi2]**(-beta)

                # chi2 and mass change for models with low X2
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
                    # Get data
                    obs_mag, obs_unc = fitparams[app_mag][:2]
                    iso_mags = iso_i[app_mag][1:-1][pdm][low_chi2]
                    plx_obs, plx_unc = fitparams['plx'][:2]
                    # Define 3-sigma interval of distance modulus based on
                    # observed parallax
                    plx_int = [plx_obs-3*plx_unc, plx_obs+3*plx_unc]
                    mu_plx_int = [-5*np.log10(plx_int[1]/100),
                                  -5*np.log10(plx_int[0]/100)]

                    for i_mass in range(len(chi2)):
                        run_marg_mu = False
                        iso_mag = iso_mags[i_mass] # Absolute magnitude
                        # 3-sigma interval of mu from magnitudes
                        mu_mag_int = [(obs_mag-iso_mag)-3*obs_unc,
                                      (obs_mag-iso_mag)+3*obs_unc]

                        # Only run marginalisation if the 3-sigma intervals
                        # of mu based on the magnitude and parallax overlap
                        # (or if the parallax is negative within 3-sigma)
                        if plx_int[1] < 0:
                            run_marg_mu = True
                        elif plx_int[0] < 0 and mu_plx_int[0] < mu_mag_int[1]:
                            run_marg_mu = True
                        elif plx_int[0] > 0:
                            if mu_plx_int[0] <= mu_mag_int[1] and mu_mag_int[0] <= mu_plx_int[1]:
                                run_marg_mu = True

                        if run_marg_mu:
                            if margm_fast:
                                lik_int_mu[i_mass] = margm2(plx_obs, plx_unc, obs_mag,
                                                            obs_unc, iso_mag, mu_prior,
                                                            mu_prior_w)
                            else:
                                lik_int_mu[i_mass] = margm(plx_obs, plx_unc, obs_mag,
                                                           obs_unc, iso_mag, mu_prior,
                                                           mu_prior_w)
                        else:
                            # If the magnitude and parallax imply values of mu
                            # which are too different
                            lik_int_mu[i_mass] = 0

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
                hr_axes=None, output_ages=True, margm_fast=True,
                save2d=True):
    '''
    Run gfunc2d on a list of stars with parameters given in a text file.
    The output G-functions are saved to an HDF5 file.

    Parameters
    ----------
    inputfile : str
        Name of the input file (including the full path) with one line
        of data for each star and columns separated by whitespace.

    isogrid : str
        Name of the isochrone hdf5 grid (including the full path).

    outputdir : str
        Name of the output directory.
        The directory will be created if it does not exist.

    inputnames : list
        List containing the names of the parameters in the columns of
        the input file. For parameters which are to be used in the fit
        the name must match the name of the corresponding parameter in
        the models. E.g. for surface gravity, the name must be 'logg'.

    fitnames : list
        List containing the names of the parameters to be fitted.
        This must be a subset of inputnames.

    alpha : float, optional
        Value of [alpha/Fe]. Must exist in the grid.
        Default value is 0.0.

    make_gplots : bool, optional
        If True, the 2D G-function of each star is plotted and saved
        to the output directory.
        Default value is True.

    make_hrplots : bool, optional
        If True, an HR diagram is plotted for each star and saved
        to the output directory.
        Default value is False.

    hr_axes : tuple, optional
        Tuple giving the parameters to plot on the x and y-axis of
        the HR diagram. E.g. ('logT', 'logg')
        Only used if make_hrplots is True.
        Default value is None.

    output_ages : bool, optional
        If True, individual age estimates are saved to text files in
        the output directory. Two files are saved: one with the mode
        of the G-function as the age estimate and another with the
        median.
        Default value is True.

    margm_fast : bool, optional
        If fitting to the parallax ('plx' in fitparams), one can choose a fast
        method for the marginalisation over the distance modulus by setting this
        value to True. A slower (but slightly more exact) method is used
        otherwise.
        Default value is True.

    save2d : bool, optional
        If True, the 2D G-functions are saved in full in the output file.
        Otherwise, only the 1D age G-functions are saved.
        Default value is True.
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
    with h5py.File(output_h5, 'a') as h5out:
        h5out.create_group('header')
        h5out.create_group('gfuncs')
        h5out.create_group('grid')

        h5out['header'].create_dataset('inputfile', data=np.string_(inputfile))
        h5out['header'].create_dataset('isogrid', data=np.string_(isogrid))
        h5out['header'].create_dataset('fitnames', data=np.string_(fitnames))
        h5out['header'].create_dataset('datetime', data=np.string_(time))
        h5out['header'].create_dataset('alpha', data=alpha)
        h5out['header'].create_dataset('margm_fast', data=np.string_(str(margm_fast)))
        h5out['header'].create_dataset('save2d', data=np.string_(str(save2d)))

    # Load stellar data
    data = np.genfromtxt(inputfile, dtype=None, names=inputnames, encoding=None)

    # Get indices of inputnames which should be fitted
    try:
        fit_inds = [inputnames.index(x) for x in fitnames]
    except:
        raise ValueError('Problem with fitnames. Check that all fitnames are in inputnames.')

    # Load isochrones into memory in the form of a python dictionary
    # Also get available parameters in the grid and their units
    print('\nLoading isochrones into memory...', end=''); sys.stdout.flush()
    with h5py.File(isogrid, 'r') as gridfile:
        isodict = gt.load_as_dict(gridfile, alpha_lims=(alpha-0.01, alpha+0.01))
        gridparams, gridunits = gt.get_gridparams(gridfile, return_units=True)
    print(' done!\n')

    # Make sure that the case of a single star is handled correctly
    sids = np.atleast_1d(data['sid'])
    # Loop over stars in the input file
    for i, name in enumerate(sids):
        if not isinstance(name, str):
            name = str(name)
        # Set stellar data
        if len(sids) == 1:
            data_i = data[()]
        else:
            data_i = data[i]

        # Make fitparams dictionary
        fitparams = {inputnames[k]: (data_i[k], data_i[k+1]) for k in fit_inds}

        # Compute G-function
        print('Processing ' + name + '...', end=''); sys.stdout.flush()
        g, tau_array, feh_array = gfunc2d(isogrid, fitparams,
                                          alpha, isodict=isodict,
                                          margm_fast=margm_fast)

        # Save G-function
        with h5py.File(output_h5, 'r+') as h5out:
            if name in h5out['gfuncs']:
                print(' Star_id already in output, skipping...')
                continue
            if save2d:
                h5out['gfuncs'].create_dataset(name, data=g)
            else:
                g_age = np.sum(g, axis=1)
                h5out['gfuncs'].create_dataset(name, data=g_age)

        # Save plots of the G-function if make_plots
        if make_gplots:
            loglik_name = os.path.join(outputdir, 'figures', name + '_loglik.pdf')
            contour_name = os.path.join(outputdir, 'figures', name + '_contour.pdf')
            loglik_save(g, tau_array, feh_array, loglik_name)
            contour_save(g, tau_array, feh_array, contour_name)

        # Save plots of HR-diagram if make_hrplots
        if make_hrplots and hr_axes is not None:
            # Find index of relevant data in input and in grid
            try:
                hrx_data_index = inputnames.index(hr_axes[0])
                hry_data_index = inputnames.index(hr_axes[1])
                hry_grid_index = gridparams.index(hr_axes[1])
                if is_color(hr_axes[0]):
                    hrx_grid_index = gridparams.index(hr_axes[0].split('-')[0])
                    hrx_grid_index = gridparams.index(hr_axes[0].split('-')[1])
                else:
                    hrx_grid_index = gridparams.index(hr_axes[0])
            except:
                raise ValueError('Both of ' + str(hr_axes) +\
                                 ' must be in inputnames and in gridparams!')

            # Find input metallicity. If no metallicity used, plot feh=0.
            for potential_feh in ['FeHini', 'FeHact']:
                try:
                    feh_index = inputnames.index(potential_feh)
                    feh_i = data_i[feh_index]
                    break
                except:
                    continue
            else:
                feh_i = 0
            hr_vals = (data_i[hrx_data_index], data_i[hry_data_index])
            hr_units = (gridunits[hrx_grid_index], gridunits[hry_grid_index])
            hr_name = os.path.join(outputdir, 'figures', name + '_hr.pdf')
            if hr_units[1] == 'mag':
                plx = data_i[inputnames.index('plx')]
                hr_save(isodict, name, hr_axes, hr_vals, hr_units, hr_name,
                        par=plx, feh=feh_i)
            else:
                hr_save(isodict, name, hr_axes, hr_vals, hr_units, hr_name,
                        feh=feh_i)

        # Print progress
        if len(sids) == 1:
            print(' 100%')
        else:
            print(' ' + str(round((i+1) / len(data) * 100)) + '%')

    # Save (tau, feh)-grid
    with h5py.File(output_h5, 'r+') as h5out:
        h5out['grid'].create_dataset('tau', data=tau_array)
        h5out['grid'].create_dataset('feh', data=feh_array)

    # Optionally, write ages to text file
    if output_ages:
        output_ages_file_mode = os.path.join(outputdir, 'ages_mode.txt')
        output_ages_file_median = os.path.join(outputdir, 'ages_median.txt')
        print_age_stats(output_h5, output_ages_file_mode, use_median=False)
        print_age_stats(output_h5, output_ages_file_median, use_median=True)
