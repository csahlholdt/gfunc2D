import os
import sys
import h5py
import numpy as np
from gfunc2D.gfunc2D import gfunc2D
from gfunc2D.gridtools import load_as_dict

inputfile = 'HDinput.txt'
outputdir = 'HDoutput'
isogrid = '/Users/christian/isochrones/grids/YY_lib.h5'

inputnames = ('name', 'FeH', 'FeH_unc', 'logT', 'logT_unc',
			  'logg', 'logg_unc', 'plx', 'plx_unc', 'B', 'B_unc',
			  'V', 'V_unc')

fitnames = ('FeH', 'logT', 'logg')

alpha = 0.0


########## END OF INPUT ##########
##################################

data = np.genfromtxt(inputfile, dtype=None, names=inputnames)
fit_inds = [inputnames.index(x) for x in fitnames]

if not os.path.exists(outputdir):
	print('Output directory "' + outputdir + '" created\n')
	os.makedirs(outputdir)

print('Loading isochrones into memory...', end=''); sys.stdout.flush()
with h5py.File(isogrid, 'r') as gridfile:
	isodict = load_as_dict(gridfile, alpha_lims=(alpha-0.01, alpha+0.01))
print(' done!\n')

for i, name in enumerate(data['name']):
	name = name.decode('ASCII')
	data_i = data[i]

	fitparams = {inputnames[k]: (data_i[k], data_i[k+1]) for k in fit_inds}

	print('Processing ' + name + '...', end=''); sys.stdout.flush()
	g, tau_array, feh_array = gfunc2D(isogrid, fitparams, alpha, isodict=isodict)

	outfile = os.path.join(outputdir, name)
	np.savez(outfile, g=g, tau_array=tau_array, feh_array=feh_array)

	print(' ' + str(round((i+1) / len(data) * 100)) + '%')
