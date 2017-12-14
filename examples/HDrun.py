from gfunc2d.gfunc2dmain import gfunc2d_run

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

gfunc2d_run(inputfile, isogrid, outputdir, inputnames, fitnames,
            alpha, make_plots)
