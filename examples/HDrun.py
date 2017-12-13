import os
import sys
import h5py
import numpy as np
from datetime import datetime
from gfunc2D.gfunc2D import gfunc2D
from gfunc2D.gridtools import load_as_dict
from gfunc2D.gplot import gplot_loglik, gplot_contour

# Define input file, output directory, and path to an isochrone grid
inputfile = '/Users/christian/code/gfunc2D/examples/HDinput.txt'
outputdir = '/Users/christian/code/gfunc2D/examples/HDoutput/'
isogrid = '/Users/christian/isochrones/grids/YY_grid.h5'
make_plots = True

# Set names of the parameters in the columns of the input file
# NOTE: The names must macth the name of the corresponding parameter in the
# isochrone grid; therefore, the effective temperature is called 'logT'.
inputnames = ('name', 'FeH', 'FeH_unc', 'logT', 'logT_unc',
              'logg', 'logg_unc', 'plx', 'plx_unc', 'B', 'B_unc',
              'V', 'V_unc')

# Set the parameters which are fitted to
# This must be a subset of the above 'inputnames'
fitnames = ('FeH', 'logT', 'logg')

# Value of [alpha/Fe]
alpha = 0.0

########## END OF INPUT ##########
##################################

# Get the date and time
time = datetime.now().isoformat(timespec='minutes')

# Check if outputfile already exists
output_h5 = os.path.join(outputdir, 'output.h5')
if os.path.exists(output_h5):
    raise IOError(output_h5 + ' already exists. Move/remove it and try again.')

# Create output directories if they do not exist
if make_plots and not os.path.exists(os.path.join(outputdir, 'figures')):
    os.makedirs(os.path.join(outputdir, 'figures'))
    print('Created output directory: ' + outputdir)
elif not os.path.exists(outputdir):
    os.makedirs(os.path.join(outputdir, 'figures'))
    print('Created output directory: ' + outputdir)

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
print('\nLoading isochrones into memory...', end=''); sys.stdout.flush()
with h5py.File(isogrid, 'r') as gridfile:
    isodict = load_as_dict(gridfile, alpha_lims=(alpha-0.01, alpha+0.01))
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
    g, tau_array, feh_array = gfunc2D(isogrid, fitparams, alpha, isodict=isodict)

    # Save G-function
    with h5py.File(output_h5) as h5out:
        h5out['gfuncs'].create_dataset(name, data=g)

    # Save plots of the G-function if make_plots
    if make_plots:
        loglik_plot = os.path.join(outputdir, 'figures', name + '_loglik.pdf')
        contour_plot = os.path.join(outputdir, 'figures', name + '_contour.pdf')
        gplot_loglik(g, tau_array, feh_array, savename=loglik_plot, show=False)
        gplot_contour(g, tau_array, feh_array, savename=contour_plot, show=False)

    # Print progress
    print(' ' + str(round((i+1) / len(data) * 100)) + '%')

# Save (tau, feh)-grid
with h5py.File(os.path.join(outputdir, 'output.h5')) as h5out:
    h5out['grid'].create_dataset('tau', data=tau_array)
    h5out['grid'].create_dataset('feh', data=feh_array)
