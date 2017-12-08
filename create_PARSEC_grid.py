import numpy as np
import h5py
import os
import re


def get_param_names(datafile):
    '''
    Finds the header of a PARSEC isochrone table and returns a list of
    parameter names excluding the first two which are always Z and the age.

    Parameters
    ----------
    datafile : str
        Name of the isochrone table (including the path).

    Returns
    -------
    param_names : list
        Names of the parameters in the isochrone table.
    '''
    with open(datafile, 'r') as data:
        prev_line = ''
        for line in data:
            if prev_line.startswith('#') and not line.startswith('#'):
                param_names = prev_line.split('\n')[0].split()[3:]
                return param_names
            else:
                prev_line = line
        else:
            raise IOError("Datafile lacking parameter header")


def Z_age_from_data(datafile):
    '''
    Finds all unique pairs of Z and age in the isochrone table and returns them
    along with the number of models for each pair.

    Parameters
    ----------
    datafile : str
        Name of the isochrone table (including the path).

    Returns
    -------
    Z_age : list
        Every unique pair of Z and age.

    counts : list
        The number of models with each pair of Z and age.
    '''
    Z_age, counts = np.unique(np.loadtxt(datafile, usecols=(0,1)), axis=0,
                              return_counts=True)
    return Z_age, counts


def feh_age_from_filename(datafile):
    '''
    Returns the metallicity and age taken from the filename. The metallicity is
    assumed to be the first number in the filename and the age the second. The
    age should be in Gyr.

    Parameters
    ----------
    datafile : str
        Name of the isochrone table (including the path).

    Returns
    -------
    feh : float
        [Fe/H] as given in the filename

    age : float
        Age as given in the filename
    '''
    # Find all substrings in the filename which are numbers
    re_match = re.findall('-?[0-9\.]+', datafile)
    # Take feh and age from the first two matches
    feh, age = re_match[:2]

    return float(feh), float(age)


def add_isotable_to_grid(datafile, gridfile, params, units, solar_Z=0.0152,
                        feh_age_filename=True, append_mags=None):
    '''
    Adds isochrone(s) from a single isochrone table to an isochrone grid file.

    Parameters
    ----------
    datafile : str
        Name of the isochrone table (including the path).

    gridfile : str
        Name of the isochrone grid file (including the path).
        If it does not exist it will be created.

    params : list
        Names of the parameters found in the isochrone table excluding the
        first two which are always Z and the age.

    units : list
        Units for each of the parameters in param.

    solar_Z : float, optional
        The solar value of Z which is used to calculate FeH.
        Default value: 0.0152

    feh_age_filename : bool, optional
        If True, the metallicity and age are taken from the filename. Note that
        in this case the datafile should only contain a single isochrone (of
        the metallicity and age given by the filename). The metallicity is
        assumed to be the first number in the filename and the age the second.
        If False, the metallicities and ages of the isochrones in datafile (can
        be multiple in the same file) are taken from the first two columns of
        the datafile. In this case [FeH] is calculated as log10(Z/solar_Z).
        Default is True.

    append_mags : list, optional
        List of photometric filters for which data should be appended to an
        existing grid. The names in the list must be in params. If given, the
        grid must already exist with all parameters and the photometric data
        are simply appended.
        Default value is None in which case the isochrone data are added as¨
        usual
    '''
    data = np.loadtxt(datafile)

    with h5py.File(gridfile) as Grid:
        if feh_age_filename:
            feh, age = feh_age_from_filename(datafile)

            gridpath = 'alphaFe=0.0000/' +\
                       'FeH=' + format(feh, '.4f') + '/' +\
                       'age=' + format(age, '.4f') + '/'

            if append_mags is None:
                for ind, pname in enumerate(params):
                    gname = gridpath + pname
                    Grid[gname] = data[:, ind+2]
                    Grid[gname].attrs.create('unit', np.string_(units[ind]))
            else:
                for ind, pname in enumerate(params):
                    gname = gridpath + pname
                    if pname == 'Mini':
                        if any(Grid[gname] - data[:, ind+2] > 0.01):
                            raise ValueError('Masses of the isochrone being\
                                              appended do not match the masses\
                                              already stored in the grid')
                    elif pname not in append_mags:
                        continue
                    else:
                        Grid[gname] = data[:, ind+2]
                        Grid[gname].attrs.create('unit', np.string_(units[ind]))
        else:
            Z_age, Z_age_counts = get_Z_age(datafile)

            n = 0
            for i, count in enumerate(Z_age_counts):
                data_i = data[n:n+count]
                Z, age = Z_age[i]
                feh = np.log10(Z/solar_Z)

                gridpath = 'alphaFe=0.0000' + '/' +\
                           'FeH=' + format(feh, '.4f') + '/' +\
                           'age=' + format(age, '.4f') + '/'

                for ind, pname in enumerate(params):
                    gname = gridpath + pname
                    if pname == 'Mini':
                        if any(Grid[gname] - data[:, ind+2] > 0.01):
                            raise ValueError('Masses of the isochrone being\
                                              appended do not match the masses\
                                              already stored in the grid')
                    elif pname not in append_mags:
                        continue
                    else:
                        Grid[gname] = data[:, ind+2]
                        Grid[gname].attrs.create('unit', np.string_(units[ind]))

                n += count


def build_PARSEC(datadir, gridfile, phot_filters, version,
                 feh_age_filename=True, append_mags=False):
    '''
    Function for building a PARSEC isochrone grid file based on multiple
    isochrone tables located in a given directory.
    The isochrone tables should be of the same format and must have the
    extension '.dat'

    Parameters
    ----------
    datadir : str
        Directory of the isochrone tables.

    gridfile : str
        Name of the isochrone grid file (including the path). If it does not
        exist it will be created.

    phot_filters : list
        Names of the photometric filters for which magnitudes are included in
        the isochrone tables.

    version : str, optional
        PARSEC version number. So far only '1.1' and '1.2S' are supported.
        The version number determines the names given to the parameters of the
        isochrones.

    feh_age_filename : bool, optional
        See add_isotable_to_grid().

    append_mags : bool, optional
        If True, the magnitude data (in the phot_filters) are appended to an
        existing grid and all other parameters are not stored.
    '''
    supported_versions = ['1.1', '1.2S']
    print("Building PARSEC isochrone grid from files in directory '"
          + datadir + "'...")

    if version in supported_versions:
        if version == '1.1' or version == '1.2S':
            param_names = ['Mini', 'Mact',
                           'logL', 'logT','logg', 'mbol'] + phot_filters
            param_units = ['solar', 'solar', 'log10(solar)', 'log10(K)',
                           'log10(cm/s2)', 'mag'] + ['mag']*len(phot_filters)
            solar_Z = 0.0152
    else:
        raise ValueError("Unknown version. Supported versions are: " +\
                         str(supported_versions))

    for datafile in os.listdir(datadir):
        if datafile.endswith('.dat'):
            print("Processing '" + datafile + "'...")
            if append_mags:
                add_isotable_to_grid(os.path.join(datadir, datafile), gridfile,
                                     param_names, param_units, solar_Z,
                                     feh_age_filename, append_mags=phot_filters)
            else:
                add_isotable_to_grid(os.path.join(datadir, datafile), gridfile,
                                     param_names, param_units, solar_Z,
                                     feh_age_filename)
    else:
        print("Build done!")
