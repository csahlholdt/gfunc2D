import numpy as np
from gfunc2d.utilities import is_color, find_nearest


def get_isopath(alpha, feh=None, age=None):
    '''
    Function to get the grid path corresponding to the isochrone with the given
    values of alpha, feh, and age.

    Parameters
    ----------
    alpha : float
        Value of [alpha/Fe].

    feh : float, optional
        Value of [Fe/H].

    age : float, optional
        Value of the age.

    Returns
    -------
    path : str
        The grid path to the isochrone with the input parameters.
    '''

    path = 'alphaFe=' + format(alpha, '.4f') + '/'
    if feh is not None:
        path += 'FeH=' + format(feh, '.4f') + '/'
        if age is not None:
            path += 'age=' + format(age, '.4f') + '/'

    return path


def get_afa_arrays(gridfile):
    '''
    Function for getting all values of [alpha/Fe], [Fe/H], and age available in
    the grid.
    Note that the function only takes [Fe/H] from the first value of [alpha/Fe]
    and it only takes ages from the first value of [Fe/H]. Therefore, the same
    values of [Fe/H] should be available for all [alpha/Fe] and the same ages
    should be available for all values of [Fe/H].

    Parameters
    ----------
    gridfile : h5py.File object or dict
        Isochrone hdf5.File object or isochrones loaded as dictionary using the
        load_as_dict() function.

    Returns
    -------
    alpha_array : array of float
        Array of all values of [alpha/Fe] available in the grid in ascending
        order.

    feh_array : array of float
        Array of all values of [Fe/H] available in the grid in ascending order.

    age_array : array of float
        Array of all values of age available in the grid in ascending order.
    '''
    # Check if gridfile has been loaded as dictionary
    if isinstance(gridfile, dict):
        alpha_list, feh_list, age_list = [], [], []
        # Go through each key in the dictionary and get arrays of unique values
        # of alpha, feh, age.
        for path in gridfile:
            alpha = float(path.split('/')[0].split('=')[1])
            feh = float(path.split('/')[1].split('=')[1])
            age = float(path.split('/')[2].split('=')[1])

            if alpha not in alpha_list:
                alpha_list.append(alpha)
            if feh not in feh_list:
                feh_list.append(feh)
            if age not in age_list:
                age_list.append(age)

        alpha_array = np.array(alpha_list)
        alpha_array.sort()
        feh_array = np.array(feh_list)
        feh_array.sort()
        age_array = np.array(age_list)
        age_array.sort()
    else:
        # Take first alpha value in the grid and the first [Fe/H] value for that
        # alpha
        alpha = list(gridfile)[0]
        feh = list(gridfile[alpha])[0]

        # Make array of [alpha/Fe] values in the grid and sort
        alpha_list = list(gridfile)
        alpha_array = np.array([float(x.split('=')[1]) for x in alpha_list])
        alpha_array.sort()

        # Make array of [Fe/H] values for the first alpha in the grid and sort
        feh_list = list(gridfile[alpha])
        feh_array = np.array([float(x.split('=')[1]) for x in feh_list])
        feh_array.sort()

        # Make array of age values for the first (alpha, [Fe/H]) in the grid and
        # sort
        age_list = list(gridfile[alpha][feh])
        age_array = np.array([float(x.split('=')[1]) for x in age_list])
        age_array.sort()

    return alpha_array, feh_array, age_array


def get_isochrone(gridfile, alpha, feh, age):
    '''
    Function for extracting a single isochrone with parameters alpha, feh, and
    age from an hdf5 isochrone grid.
    If an isochrone with the given parameters is not found in the grid the
    closest one is chosen.

    Parameters
    ----------
    gridfile : h5py.File object or dict
        Isochrone hdf5.File object or isochrones loaded as dictionary using the
        load_as_dict() function.

    alpha : float
        Value of [alpha/Fe].

    feh : float, optional
        Value of [Fe/H].

    age : float, optional
        Value of the age.

    Returns
    -------
    q : dict
        Dictionary holding the data of the isochrone with parameter names as
        keys and numpy arrays as values.

    closest_afa : tuple
        The actual values of alpha, feh, and age of the returned isochrone.
        Only different from the input values if no isochrone was found with
        those values.
    '''

    isopath = get_isopath(alpha, feh, age)

    # If the path is not found in the grid
    if isopath not in gridfile:
        alphas, fehs, ages = get_afa_arrays(gridfile)
        alpha = find_nearest(alphas, alpha)
        feh = find_nearest(fehs, feh)
        age = find_nearest(ages, age)
        isopath = get_isopath(alpha, feh, age)

    # Save afa which was actually used
    closest_afa = (alpha, feh, age)

    # Read isochrone data and return as dictionary
    group = gridfile[isopath]
    q = {}
    for param in group:
        q[param] = group[param][:]

    return q, closest_afa


def get_gridparams(gridfile, return_units=False):
    '''
    Function to get the names (and optionally units) of all parameters
    available for each isochrone in the grid.
    Note that this assumes (reasonably) that all isochrones have the same
    parameters stored.

    Parameters
    ----------
    gridfile : h5py.File object
        Isochrone hdf5.File object.

    return_units : bool, optional
        If true, return a list of units for the parameters.
        Default is False.

    Returns
    -------
    params : list of str
        Names of the parameters available for each isochrone in the grid.

    units : list of str
        Units of the parameters available for each isochrone in the grid.
        If return_units is False, None is returned instead of the list.
    '''

    # Function to be passed to the gridfile.visit() method
    # It returns the group if 'age=' is in the name.
    fun = lambda x: x if 'age=' in x else None

    # Find an age group
    age_group = gridfile.visit(fun)

    # Return a list of the names of the datasets in the age group. Also return
    # a list of units if return_units is True.
    params = list(gridfile[age_group])
    if not return_units:
        return params, None
    else:
        units = []
        for param in params:
            unit = gridfile[age_group][param].attrs.get('unit').decode('ASCII')
            units.append(unit)
        return params, units


def prepare_fitparams(gridfile, fitparams):
    '''
    Function for checking and preparing the fitparams dictionary before the fit
    is carried out. It checks that the necessary parameters are available in
    the isochrone grid and adds metadata used in the fitting process.

    A ValueError is raised if:
    One (or both) of the magnitudes of a given color index is unavailable in
    the grid.
    An apparent magnitude is given in fitparams without a parallax.
    There is more than one apparent magnitude in fitparams.
    An unknown parameter is found in fitparams (one that is not a color, the
    parallax or one of the parameters in the grid).

    Parameters
    ----------
    gridfile : h5py.File object
        Isochrone hdf5.File object.

    fitparams : dict
        Dictionary with inputparameters for the isochrone fit.
        See gfunc_2D() for a description of the format.

    Returns
    -------
    fitparams_attr : dict
        Updated fitparams dictionary with metadata (attributes) for each
        parameter.

    mag : str
        The apparent magnitude (name of the filter) given in fitparams.
        If no apparent magnitude is fitted, None is returned instead.
    '''
    # Get parameters available in the grid as well as their units
    libparams, libunits = get_gridparams(gridfile, return_units=True)
    libparams += ['FeH']
    libunits += ['dex']
    fitparams_attr = {}
    mag = None

    # For each parameter in fitparams, check that the necessary parameters are
    # available in the grid and add an attribute (a string) for use in the
    # fitting process
    for param in fitparams:
        if is_color(param):
            colors = param.split('-')
            if colors[0] in libparams and colors[1] in libparams:
                fitparams_attr[param] = (*fitparams[param], 'color')
            else:
                raise ValueError('One of the magnitudes of the color index '\
                                  + param + ' was not found in the grid')
        elif param == 'plx':
            fitparams_attr[param] = (*fitparams[param], 'none')
        elif param in libparams:
            if libunits[libparams.index(param)] == 'mag':
                if mag is None:
                    if 'plx' in fitparams:
                        fitparams_attr[param] = (*fitparams[param], 'mag')
                        mag = param
                    else:
                        raise ValueError('The magnitude ' + param +\
                            ' is in fitparams without a parallax ("plx")')
                else:
                    raise ValueError('More than one apparent magnitude ' +\
                                     'in fitparams')
            else:
                fitparams_attr[param] = (*fitparams[param], 'none')
        else:
            raise ValueError('The parameter ' + param +\
                             ' was not found in the grid')

    return fitparams_attr, mag


def load_as_dict(gridfile, alpha_lims=None, feh_lims=None, age_lims=None):
    '''
    Function for loading the isochrones of an hdf5 grid into a python
    dictionary.

    Parameters
    ----------
    gridfile : h5py.File object
        Isochrone hdf5.File object.

    alpha_lims : seq of float, optional
        Lower and upper limits (closed interval) on [alpha/Fe].
        If no limits are given the default is to take all values available in
        the grid.

    feh_lims : seq of float, optional
        Lower and upper limits (closed interval) on [Fe/H].
        If no limits are given the default is to take all values available in
        the grid.

    ages : seq of float, optional
        Lower and upper limit (closed interval) on the age.
        If no limits are given the default is to take all values available in
        the grid.

    Returns
    -------
    isodict : dict
        Dictionary holding the isochrone data. The keys are grid paths to the
        isochrones (as returned by get_isopath()) and the values are dictionaries
        like the ones returned by get_isochrone().
    '''

    # Get arrays of alphas, fehs, ages
    alphas, fehs, ages = get_afa_arrays(gridfile)
    # Pick out the desired parameter ranges if limits are given
    if alpha_lims is not None:
        alphas = alphas[(alphas >= alpha_lims[0]) & (alphas <= alpha_lims[1])]
    if feh_lims is not None:
        fehs = fehs[(fehs >= feh_lims[0]) & (fehs <= feh_lims[1])]
    if age_lims is not None:
        ages = ages[(ages >= age_lims[0]) & (ages <= age_lims[1])]

    # Initialize dictionary
    isodict = {}
    for alpha in alphas:
        for feh in fehs:
            for age in ages:
                # For each alpha, feh, age, the isochrone (as a dictionary
                # returned by get_isochrone()) is added to isodict under the
                # path to that isochrone in the hdf5 grid.
                isopath = get_isopath(alpha, feh, age)
                isodata = get_isochrone(gridfile, alpha, feh, age)
                isodict[isopath] = isodata

    return isodict
