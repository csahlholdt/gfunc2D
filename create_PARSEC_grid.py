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
    assumed to be the first number in the filename and the age the second.

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
    re_match = re.findall('-?[0-9\.]+', datafile)
    feh, age = re_match[:2]

    return float(feh), float(age)


def add_isotable_to_lib(datafile, libfile, params, units, solar_Z=0.0152,
                        feh_age_filename=True):
    '''
    Adds isochrone(s) from a single isochrone table to an isochrone grid file.

    Parameters
    ----------
    datafile : str
        Name of the isochrone table (including the path).

    libfile : str
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
        If True, the metallicity and age is taken from the filename. Note that
        in this case the datafile should only contain a single isochrone (of
        the metallicity and age given by the filename). The metallicity is
        assumed to be the first number in the filename and the age the second.
        If False, the metallicities and ages of the isochrones in datafile (can
        be multiple in the same file) are taken from the first two columns of
        the datafile. In this case [FeH] is calculated as log10(Z/solar_Z).
        Default is True.
    '''
    data = np.loadtxt(datafile)

    with h5py.File(libfile) as library:
        if feh_age_filename:
            feh, age = feh_age_from_filename(datafile)

            gridpath = 'alphaFe=0.0000' + '/' +\
                       'FeH=' + format(feh, '.4f') + '/' +\
                       'age=' + format(age, '.4f') + '/'

            for ind, pname in enumerate(params):
                library[gridpath + pname] = data[:, ind+2]
                library[gridpath + pname].attrs.create('unit', np.string_(units[ind]))

                ### TEMPORARY ###
                #if pname == 'V':
                #    V = data[:, ind+2]
                #elif pname == 'I':
                #    I = data[:, ind+2]
                #    VmI = V - I
                #    library[gridpath + 'G'] = V - 0.0354 - 0.0561*VmI - 0.1767*VmI**2 - 0.0108*VmI**3
                #    library[gridpath + 'G'].attrs.create('unit', np.string_('mag'))
                ### TEMPORARY ###
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
                    library[gridpath + pname] = data_i[:, ind+2]
                    library[gridpath + pname].attrs.create('unit', np.string_(units[ind]))

                n += count


def build_PARSEC(datadir, libfile, phot_filters, version):
    '''
    Function for building a PARSEC isochrone library file based on multiple
    isochrone tables located in a given directory.
    The isochrone tables should be of the same format and must have the
    extension '.dat'

    Parameters
    ----------
    datadir : str
        Directory of the isochrone tables.

    libfile : str
        Name of the isochrone library file (including the path). If it does not
        exist it will be created.

    phot_filters : list
        Names of the photometric filters for which magnitudes are included in
        the isochrone tables.

    version : str, optional
        PARSEC version number. So far only '1.1' and '1.2S' are supported.
        The version number determines the names given to the parameters of the
        isochrones.
    '''
    supported_versions = ['1.1', '1.2S']
    print("Building PARSEC isochrone library from files in directory '"
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
            add_isotable_to_lib(os.path.join(datadir, datafile), libfile,
                                param_names, param_units, solar_Z=solar_Z)
    else:
        print("Build done!")


datadir = '/home/christian/Downloads/PARSEC_isochrones/test_ubv/'
build_PARSEC(datadir, datadir + 'PARSEC_lib12S.h5',
             ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'Ks'], '1.2S')

#build_PARSEC(datadir, datadir + 'PARSEC_lib12S.h5',
#             ['G', 'G_BP', 'G_RP'], '1.2S')
