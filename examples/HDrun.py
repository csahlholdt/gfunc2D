from gfunc2d.gfunc2dmain import gfunc2d_run

# Define input file, output directory, and path to an isochrone grid
inputfile = '/Users/csahlholdt/code/gfunc2D/examples/HDinput.txt'
outputdir = '/Users/csahlholdt/code/gfunc2D/examples/HDoutput/'
isogrid = '/Users/csahlholdt/isochrones/grids/PARSEC/3.3/PARSEC_label6_interp.h5'

# Choose the figures to be produced
make_gplots = True
make_hrplots = True
hr_axes = ('logT', 'logg')

# Output age summary in text file?
output_ages = True

# Set names of the parameters in the columns of the input file
# NOTE: The names must macth the name of the corresponding parameter in the
# isochrone grid; therefore, the effective temperature is called 'logT'.
inputnames = ('sid', 'FeHini', 'FeHini_unc', 'logT', 'logT_unc',
              'logg', 'logg_unc', 'plx', 'plx_unc', 'V', 'V_unc',
              'Ks', 'Ks_unc')

# Set the parameters which are fitted to
# This must be a subset of the above 'inputnames'
fitnames = ('FeHini', 'logT', 'logg')
#fitnames = ('FeHini', 'logT', 'Ks', 'plx')

# Value of [alpha/Fe]
alpha = 0.0

########## END OF INPUT ##########
##################################

gfunc2d_run(inputfile, isogrid, outputdir, inputnames, fitnames,
            alpha, make_gplots, make_hrplots, hr_axes)
