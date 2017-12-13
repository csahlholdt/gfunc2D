import h5py
import numpy as np
from gfunc2d.gfunc2dmain import gfunc2D
from gfunc2d.gridtools import get_isochrone, get_afa_arrays, load_as_dict
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.style as mpls
mpls.use('classic')

import time

# Isochrone grid path
isogrid = '/Users/christian/isochrones/grids/YY_grid.h5'

# Define the true parameters (indicated by '0' in the variable name):
feh0 = -0.5  # True [Fe/H]
alpha0 = 0.0 # True [alpha/Fe]
tau0 = 7.0   # True age in Gyr
Mini0 = 1.0  # True initial mass in Msun
#Mini0 = 1.0666478  # True initial mass in Msun
plx0 = 5.0   # True parallax in mas
print('Assumed true parameter values:')
print('[Fe/H]            =', feh0)
print('alpha             =', alpha0)
print('age (tau)         =', tau0, 'Gyr')
print('initial mass      =', Mini0,'Msun')
print('parallax          =', plx0, 'mas')
print('')

q0 = {}
# Calculate the isochrone (q0) for the true age and metallicity
with h5py.File(isogrid, 'r') as gridfile:
    q0 = get_isochrone(gridfile, alpha0, feh0, tau0)

Mini = q0['Mini']
if Mini0 > max(Mini):
    raise ValueError('Initial mass too high for the age')

# Select the isochrone mass closest to the true value
k = np.argmin(abs(Mini-Mini0))
Mini_used = Mini[k]

# True distance modulus
mu0 = 5 * np.log10(100/plx0)

# True absolute and observed G magnitude, colours, and other parameters
Gabs0 = q0['G'][k]
Gapp0 = Gabs0 + mu0
UmB0 = q0['U'][k] - q0['B'][k]
BmV0 = q0['B'][k] - q0['V'][k] 
VmR0 = q0['V'][k] - q0['R'][k]
VmI0 = q0['V'][k] - q0['I'][k]
GmJ0 = q0['G'][k] - q0['J'][k]
GmH0 = q0['G'][k] - q0['H'][k]
GmK0 = q0['G'][k] - q0['Ks'][k]
JmH0 = q0['J'][k] - q0['H'][k]
HmK0 = q0['H'][k] - q0['Ks'][k]
logT0 = q0['logT'][k]
logL = q0['logL'][k]
logg = q0['logg'][k]
print('Calculated true observables:')
print('Absolute G mag    =', Gabs0)
print('Apparent G mag    =', Gapp0)
print('log(T)            =', logT0)
print('U - B             =', UmB0)
print('B - V             =', BmV0)
print('V - R             =', VmR0)
print('V - I             =', VmI0)
print('G - J             =', GmJ0)
print('G - H             =', GmH0)
print('G - Ks            =', GmK0)
print('J - H             =', JmH0)
print('H - Ks            =', HmK0)

# Draw an HR diagram with the relevant isochrone(s)
Gabs = q0['G']
CI = q0['G'] - q0['Ks']

fig, ax = plt.subplots()

ax.plot(CI, Gabs, '-k')
ax.plot(CI[k], Gabs[k], 'ok')

ax.set_xlabel(r'$G - Ks$ [mag]')
ax.set_ylabel(r'$M_{G}$ [mag]')

# Dashed = +/-0.7 Gyr
# Red/blue = +/-0.5 dex in [Fe/H]
dtau = 0.7
dfeh = 0.5
#dtau = 0.5
#dfeh = 0.4
with h5py.File(isogrid, 'r') as gridfile:
    q1 = get_isochrone(gridfile, alpha0, feh0, tau0+dtau)
    q2 = get_isochrone(gridfile, alpha0, feh0, tau0-dtau)
    q3 = get_isochrone(gridfile, alpha0, feh0+dfeh, tau0)
    q4 = get_isochrone(gridfile, alpha0, feh0-dfeh, tau0)
    q5 = get_isochrone(gridfile, alpha0, feh0+dfeh, tau0+dtau)
    q6 = get_isochrone(gridfile, alpha0, feh0+dfeh, tau0-dtau)
    q7 = get_isochrone(gridfile, alpha0, feh0-dfeh, tau0+dtau)
    q8 = get_isochrone(gridfile, alpha0, feh0-dfeh, tau0-dtau)

isochrones = [q1, q2, q3, q4, q5, q6, q7, q8]
lines = ['--k', '--k', '-r', '-b', '--r', '--r', '--b', '--b']

for isochrone, line in zip(isochrones, lines):
    Gabs1 = isochrone['G']
    CI1 = isochrone['G'] - isochrone['Ks']
    ax.plot(CI1, Gabs1, line)

ax.set_xlim(CI[k]-0.5, CI[k]+0.5)
ax.set_ylim(Gabs0-2.5, Gabs0+2.5)
ax.invert_yaxis()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.grid()

##### calculate G function vs. tau and feh

# Observed parallax and its uncertainty (mas)
plx_obs = plx0
plx_err = 0.3

# Define input dict (type, value, and uncertainty)
fitparams = {'G': (round(Gapp0, 3), 0.01),
             #'G-Ks': (GmK0, 0.01),
             #'G-H': (GmH0, 0.01),
             #'G-J': (GmJ0, 0.01),
             #'V-I': (VmI0, 0.01),
             'FeH': (feh0, 0.20),
             'logT': (int(10**(logT0)), 100),
             'plx': (plx_obs, plx_err)}

# Assumed [alpha/Fe]
alpha_assumed = alpha0

# Load isogrid as dict
print('\nLoading datafile...')
with h5py.File(isogrid, 'r') as gridfile:
    isodict = load_as_dict(gridfile, (alpha0, alpha0))

# Calculate the 2D G-function
print('\nComputing 2D G-function...')
t0 = time.time()
g, tau_array, feh_array = gfunc2D(isogrid, fitparams, alpha_assumed, isodict=isodict)
t1 = time.time()
print('Calculation time:', round(t1-t0, 3), 'seconds')

# Add small number to allow logarithmic scale
eps = 1e-20;
A = g.T + eps

kernel = np.array([0.25, 0.5, 0.25])

func = lambda x: np.convolve(x, kernel, mode='same')
B = np.apply_along_axis(func, 0, A)
C = np.apply_along_axis(func, 1, B)

C /= np.amax(C)
# C = A / np.amax(A)

fig, ax = plt.subplots()

cax = ax.imshow(np.log10(C), extent=(tau_array[0], tau_array[-1], feh_array[-1], feh_array[0]), aspect='auto', interpolation='none')
cbar = fig.colorbar(cax)

ax.scatter(tau0, feh0, s=20**2, c='w', marker='+')
ax.contour(tau_array, feh_array, np.log10(C), [-1.01], colors='w', linestyles='solid')

ax.set_xlabel('Age [Gyr]')
ax.set_ylabel('[Fe/H]')
ax.invert_yaxis()
ax.grid()

ax.set_xlim(tau_array[0], tau_array[-1])
ax.set_ylim(feh_array[0], feh_array[-1])

#plt.show()


fig = plt.figure()

ax0 = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=3, sharex=ax0)
ax2 = plt.subplot2grid((4, 4), (1, 3), rowspan=3, sharey=ax0)

percent = [80, 90, 95, 99]
percentiles = []
for p in percent:
    percentiles.append(np.percentile(np.log10(C[C > 1e-17]), p))
ax0.contour(tau_array, feh_array, np.log10(C), percentiles, colors='k', linestyles='solid')

ax0.set_xlabel('Age [Gyr]')
ax0.set_ylabel('[Fe/H]')
ax0.invert_yaxis()
ax0.grid()

ax0.set_xlim(tau_array[0], tau_array[-1])
ax0.set_ylim(feh_array[0], feh_array[-1])

tau_dist = np.sum(C, axis=0)
tau_dist /= np.amax(tau_dist)

# log
#ax1.plot(tau_array, np.log10(tau_dist))
#ax1.set_ylim([1.2*np.amin(np.log10(tau_dist)), 0])

# Not log
ax1.plot(tau_array, tau_dist)
ax1.set_ylim([0, 1.2*np.amax(tau_dist)])
ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

feh_dist = np.sum(C, axis=1)
feh_dist /= np.amax(feh_dist)

# log
#ax2.plot(np.log10(feh_dist), feh_array)
#ax2.set_xlim([1.2*np.amin(np.log10(feh_dist)), 0])

# Not log
ax2.plot(feh_dist, feh_array)
ax2.set_xlim([0, 1.2*np.amax(feh_dist)])
ax2.set_xticks([0.0, 0.5, 1.0])

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)

plt.show()
