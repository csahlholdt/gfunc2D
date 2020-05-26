import numpy as np
import h5py
import os
import re


def makeYY(datadir, gridfile):
    # tau values [Gyr]:
    tau = np.arange(0.1, 15.1, 0.1)

    # Number of mass values per feh/alpha/tau combination:
    nm = 140

    # Get list of isochrone data files
    isofiles = [file for file in os.listdir(datadir) if file.endswith('.dat')]

    # Number of lines from the start of one data block to the next
    line_jump = nm + 2

    # Names and units of parameters in the columns.
    # The data is saved under these names in the hdf5 file
    # and the units are added as attributes (metadata).
    params = [('Mini', 'solar'),
              ('logT', 'log10(K)'),
              ('logL', 'log10(solar)'),
              ('logg', 'log10(cm/s2)'),
              ('FeHini', 'dex'),
              ('U', 'mag'),
              ('B', 'mag'),
              ('V', 'mag'),
              ('R', 'mag'),
              ('I', 'mag'),
              ('J', 'mag'),
              ('H', 'mag'),
              ('Ks', 'mag'),
              ('G', 'mag')]

    print("Building YY isochrone grid from files in directory '"
          + datadir + "'...\n")

    with h5py.File(gridfile, 'a') as YYgrid:
        for datafile in isofiles:
            print("Processing '" + datafile + "'...")

            # Get alphaFe and FeH from the name of the datafile
            alpha_data = float(datafile[7:10])
            feh_data = float(datafile[1:6])

            for i, age in enumerate(tau):
                gridpath = 'alphaFe=' + format(alpha_data, '.4f') + '/' +\
                           'FeH=' + format(feh_data, '.4f') + '/' +\
                           'age=' + format(age, '.4f') + '/'

                # Read nm lines from the datafile which contain the data for
                # the relevant feh/alpha/tau combination
                data_i = np.genfromtxt(os.path.join(datadir, datafile),
                                       skip_header=3+i*line_jump, max_rows=nm)

                # Make a new array and fill in the data to be stored e.g. the
                # magnitudes as calculated from the colors given in the
                # datafile.
                data_mags = np.zeros((data_i.shape[0], len(params)))
                V = data_i[:, 4]
                VmI = data_i[:, 8]
                for i in range(4):
                    data_mags[:, i] = data_i[:, i]
                data_mags[:, 4] = feh_data
                data_mags[:, 5] = V + data_i[:, 5] + data_i[:, 6] # U = V+(U-B)+(B-V)
                data_mags[:, 6] = V + data_i[:, 6] # B = V+(B-V)
                data_mags[:, 7] = V # V = V
                data_mags[:, 8] = V - data_i[:, 7] # R = V-(V-R)
                data_mags[:, 9] = V - VmI # I = V-(V-I)
                data_mags[:, 10] = V - data_i[:, 9] # J = V-(V-J)
                data_mags[:, 11] = V - data_i[:, 10] # H = V-(V-H)
                data_mags[:, 12] = V - data_i[:, 11] - 0.044 # Ks = V-(V-K)-0.044
                G = V - 0.0354 - 0.0561*VmI - 0.1767*VmI**2 - 0.0108*VmI**3
                data_mags[:, 13] = G

                # Store the data in the grid file
                for ind, (pname, punit) in enumerate(params):
                    YYgrid[gridpath + pname] = data_mags[:, ind]
                    YYgrid[gridpath + pname].attrs.create('unit', np.string_(punit))

    print("\nBuild done!")


def makePARSEC(datadir, gridfile, phot_filters, version,
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
        existing grid and all other parameters (which may already be in the
        grid) are not stored.
    '''

    supported_versions = ['1.1', '1.2S']
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

    print("Building PARSEC isochrone grid from files in directory '"
          + datadir + "'...")

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
    Adds isochrone(s) from a single isochrone table (text file) to an isochrone
    grid file.

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
        Units for each of the parameters in `params`.

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
        Default value is None in which case the isochrone data are added as
        usual.
    '''

    data = np.loadtxt(datafile)
    with h5py.File(gridfile, 'a') as Grid:
        if feh_age_filename:
            feh, age = feh_age_from_filename(os.path.basename(datafile))

            gridpath = 'alphaFe=0.0000/' +\
                       'FeH=' + format(feh, '.4f') + '/' +\
                       'age=' + format(age, '.4f') + '/'

            add_isochrone_to_grid(data, Grid, gridpath, params, units,
                                  append_mags)
            if append_mags is None:
                Grid[gridpath + 'FeHini'] = feh*np.ones(len(data[:,0]))
                Grid[gridpath + 'FeHini'].attrs.create('unit', np.string_('dex'))
        else:
            Z_age, Z_age_counts = Z_age_from_data(datafile)

            n = 0
            for i, count in enumerate(Z_age_counts):
                data_i = data[n:n+count]
                Z, age = Z_age[i]
                feh = np.log10(Z/solar_Z)

                gridpath = 'alphaFe=0.0000' + '/' +\
                           'FeH=' + format(feh, '.4f') + '/' +\
                           'age=' + format(age, '.4f') + '/'

                add_isochrone_to_grid(data_i, Grid, gridpath, params, units,
                                      append_mags)

                if append_mags is None:
                    Grid[gridpath + 'FeHini'] = feh*np.ones(len(data[:,0]))
                    Grid[gridpath + 'FeHini'].attrs.create('unit', np.string_('dex'))

                n += count


def add_isochrone_to_grid(data_array, h5py_grid, gridpath, params, units,
                          append_mags):
    '''
    Adds a single isochrone stored as a numpy array loaded from a text file in
    an open h5py grid-file.

    Parameters
    ----------
    data_array : array
        Array containing the isochrone data with the metallicity and age in the
        first two columns and the parameters in `param` in the remaining
        columns.

    h5py_grid : h5py.File object
        Isochrone hdf5.File object in which the data will be stored.

    gridpath : str
        Path into the h5py file where the data will be stored. Should
        correspond to the alpha enhancement, metallicity, and age of the
        isochrone being stored.
        E.g. 'alphaFe=0.0000/FeH=0.0000/age=1.0000/'

    params : list
        Names of the parameters found in the `data_array`, excluding the
        first two which are always Z and the age.

    units : list
        Units for each of the parameters in `params`.

    append_mags : list, optional
        List of photometric filters for which data should be appended to an
        existing grid. The names in the list must be in params. If given, the
        grid must already exist with all parameters and the photometric data
        are simply appended.
        Default value is None in which case the isochrone data are added as
        usual.
    '''

    if append_mags is None:
        for ind, pname in enumerate(params):
            gname = gridpath + pname
            h5py_grid[gname] = data_array[:, ind+2]
            h5py_grid[gname].attrs.create('unit', np.string_(units[ind]))
    else:
        for ind, pname in enumerate(params):
            gname = gridpath + pname
            if pname == 'Mini':
                if any(h5py_grid[gname] - data_array[:, ind+2] > 0.01):
                    raise ValueError('Masses of the isochrone being\
                                      appended do not match the masses\
                                      already stored in the grid')
            elif pname not in append_mags:
                continue
            else:
                h5py_grid[gname] = data_array[:, ind+2]
                h5py_grid[gname].attrs.create('unit', np.string_(units[ind]))


def append_asteroseismology(gridfile, solar_T=5777):
    '''
    Append asteroseismic parameters (dnu, numax) based on scaling relations to
    an existing grid.

    Parameters
    ----------
    gridfile : str
        Name of the isochrone grid file (including the path).
        If it does not exist it will be created.

    solar_T : float, optional
        Solar effective temperature (in Kelvin) to use in the scaling relations.
        Default value is 5777.
    '''

    grid = h5py.File(gridfile, 'r+')
    for a in list(grid.keys()):
        for f in list(grid[a].keys()):
            for ag in list(grid[a+'/'+f].keys()):
                path = a+'/'+f+'/'+ag+'/'
                logL = grid[path+'logL'][:]
                logT = grid[path+'logT'][:]
                Mact = grid[path+'Mact'][:]
                R = 10**(logL/2 - 2*logT + 2*np.log10(solar_T))

                # Scaling relations
                dnu = Mact**(1/2)*R**(-3/2)
                numax = Mact*R**(-2)*(10**logT/solar_T)**(-1/2)

                # Add to grid
                grid[path+'dnu'] = dnu
                grid[path+'dnu'].attrs.create('unit', np.string_('muHz (solar)'))
                grid[path+'numax'] = numax
                grid[path+'numax'].attrs.create('unit', np.string_('muHz (solar)'))

    grid.close()


