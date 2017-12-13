import os
import sys
import h5py
import numpy as np
from gfunc2D.gfunc2D import gfunc2D
from gfunc2D.gridtools import load_as_dict
from gfunc2D.gplot import gplot_loglik, gplot_contour

# Define input file, output directory, and path to an isochrone grid
inputfile = 'HDinput.txt'
outputdir = 'HDoutput'
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

# Load stellar data
data = np.genfromtxt(inputfile, dtype=None, names=inputnames)
# Get indices of inputnames which should be fitted
fit_inds = [inputnames.index(x) for x in fitnames]

# Create output directory if it does not exist
if not os.path.exists(outputdir):
    print('Output directory "' + outputdir + '" created\n')
    os.makedirs(outputdir)

# Load isochrones into memory in the form of a python dictionary
print('Loading isochrones into memory...', end=''); sys.stdout.flush()
with h5py.file(isogrid, 'r') as gridfile:
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

    # Save G-function and the (tau, feh)-grid
    outfile = os.path.join(outputdir, name)
    np.savez(outfile, g=g, tau_array=tau_array, feh_array=feh_array)

    # Save plots of the G-function if make_plots
    if make_plots:
        loglik_plot = os.path.join(outputdir, name + '_loglik.pdf')
        contour_plot = os.path.join(outputdir, name + '_contour.pdf')
        gplot_loglik(g, tau_array, feh_array, savename=loglik_plot, show=False)
        gplot_contour(g, tau_array, feh_array, savename=contour_plot, show=False)

    # Print progress
    print(' ' + str(round((i+1) / len(data) * 100)) + '%')
