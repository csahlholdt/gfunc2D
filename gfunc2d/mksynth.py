import h5py
import numpy as np
from gfunc2d.gridtools import get_isochrone, get_gridparams

known_filters = ['J', 'H', 'Ks', #2MASS
                 'u', 'v', 'g', 'r', 'i' ,'z', #SkyMapper
                 'G', 'G_BPbr', 'G_BPft', 'G_RP'] #Gaia (DR2)

def generate_synth_stars(isogrid, outputfile, t_bursts, ns, m_min, feh_params,
                         IMF_alpha=2.3, rand_seed=1):
    """
    Generate synthetic sample of stars, save stellar parameters in hdf5-format.

    Parameters
    ----------
    isogrid : str
        Isochrone grid file (hdf5) to sample from.

    outputfile : str
        Name of file (hdf5) to store the parameters of the synthetic sample in.

    t_bursts : array
        Array with dimension (n, 3), where each row gives
        [t_low, t_high, probability] of a star formation burst (in Gyr).
        Can also be 1D array [t_low, t_high, probability] in which case the
        probability is ignored (since all stars must come from the one burst).

    ns : int
        Number of stars to generate.

    m_min : float
        Minimum stellar mass generated.

    feh_params : array
        An array giving the mean and dispersion of the metallicites of the
        synthetic stars [feh_mean, feh_dispersion].
        The metallicities are drawn from a normal distribution
        N(feh_mean, feh_dispersion).

    IMF_alpha : float, optional
        Exponent to use for the initial mass function when drawing masses.
        Default is 2.3 (Salpeter IMF, probability \propto m^[-2.3]).

    rand_seed : int, optional
        Seed for np.random. Ensures that samples can be reproduced.
        Default value is 1.
    """

    # This initialises the random number generator to a given state so that
    # results can be repeated
    np.random.seed(rand_seed)

    # Settings for the synthetic dataset
    single_burst = True if len(t_bursts.shape) == 1 else False
    feh_mean, feh_disp = feh_params
    config = {'t_bursts': t_bursts, 'm_min': m_min, 'alpha': IMF_alpha,
              'ns': ns, 'feh_mean': feh_mean, 'feh_disp': feh_disp,
              'seed': rand_seed, 'gridpath': isogrid}

    # Arrays to store true parameters
    tau = np.zeros(ns) # True ages (yr)
    feh = np.zeros(ns) # True [Fe/H]

    # Auxiliary arrays for generating true ages
    if not single_burst:
        prob = t_bursts[:, 2]
        prob = prob / np.sum(prob)
        n_bursts = len(prob)

    # Open isochrone grid
    gridfile = h5py.File(isogrid, 'r')

    # Get isochrone parameters and prepare dictionary with real data
    params = get_gridparams(gridfile)[0]
    data = {}
    # Prepare arrays for each parameter + the age
    for param in params + ['age']:
        data[param] = np.zeros(ns)

    iv = 0 # Number of generated stars with valid isochrone
    ne = 0 # Number of generated stars that have evolved beyond isochrones
    while iv < ns:
        # Define true age
        if single_burst:
            age = t_bursts[0] + (t_bursts[1]-t_bursts[0]) * np.random.rand()
        else:
            i_burst = np.random.choice(range(n_bursts), p=prob)
            age = t_bursts[i_burst, 0] +\
                  (t_bursts[i_burst, 1]-t_bursts[i_burst, 0]) * np.random.rand()

        m = m_min * np.random.rand()**(-1/(IMF_alpha-1)) # True initial mass
        feh_test = np.random.normal(feh_mean, feh_disp)
        # Get the isochrone for [Fe/H], age
        q, afa = get_isochrone(gridfile, 0.0, feh_test, age)
        iso_age = afa[2] # True age
        q_mass = q['Mini']

        # If initial mass is in the valid range for the age
        if m < q_mass[-1]:
            # Interpolate along the isochrone to the given initial mass
            im = np.where(q_mass <= m)[0][-1]
            ip = np.where(q_mass > m)[0][0]
            # Now q_mass[im] <= m < q_mass[ip]
            h = (m - q_mass[im]) / (q_mass[ip] - q_mass[im])
            # Save the interpolated isochrone parameters for the chosen model
            for param in params:
                data[param][iv] = (1-h)*q[param][im] + h*q[param][ip]
            data['age'][iv] = iso_age

            iv += 1
        else:
            ne += 1

    print('Number of valid stars = ', iv)
    print('Number of discarded stars (too massive for the age) = ', ne)
    gridfile.close()

    # Open the file that the synthetic sample is saved in
    outfile = h5py.File(outputfile, 'w')

    # Save the config information
    for cparam in config:
        outfile.create_dataset('config/'+cparam, data=config[cparam])

    # Save the stellar data
    for sparam in data:
        outfile.create_dataset('data/'+sparam,
                                data=data[sparam])
    outfile.close()


def make_synth_obs(synthfile, outputfile, obs_params, plx_distribution='exp'):
    '''
    Generate an input file for gfunc2D based on synthetic sample of stars.

    Parameters
    ----------
    synthfile : str
        File (hdf5) with data for a synthetic sample of stars.

    outputfile : str
        Output file for "observed" stellar parameters (text-file).

    obs_params : dict
        Dictionary with names of parameters to observe as keys and their
        uncertainties as values e.g. {'FeHini': 0.1, 'logg': 0.2, ...}.
        Parameter names must match the names in the isochrone grid file (this
        means that for temperatures, the name is 'logT'; The temperature will
        be saved in Kelvin, not as logarithm).

    plx_distribution : float, optional
        Value of the parallax. Can also give the string 'exp' in which case the
        parallaxes are given an exponential distribution to mimic the density
        of observed stars in the solar neighborhood.
        Default value is 'exp'.
    '''

    # Check whether observed magnitudes should be calculated
    obs_mags = list(set(known_filters) & set(obs_params))
    if len(obs_mags) > 0 and 'plx' not in obs_params:
        raise ValueError('"plx" must be in obs_params to observe magnitudes')

    # Get the 'true' parameters of the synthetic stars
    true_data = {}
    synth_data = h5py.File(synthfile, 'r')
    ns = synth_data['config/ns'].value

    for oparam in obs_params:
        if oparam == 'plx':
            continue
        try:
            if oparam == 'logT':
                true_data[oparam] = 10**(synth_data['data/'+oparam][:])
            else:
                true_data[oparam] = synth_data['data/'+oparam][:]
        except:
            raise KeyError('Parameter ' + oparam + ' not in synthetic data...')
    synth_data.close()

    # If parallaxes are to be fitted, the true values are assumed based on
    # the input plx_distribution
    if 'plx' in obs_params:
        if plx_distribution == 'exp':
            # Approximate distance distribution of stars observed by Gaia (?)
            plx_true = np.exp(np.random.normal(0.5637, 0.8767, ns))
        else:
            # Else a constant value (given in plx_distribution)
            plx_true = plx_distribution*np.ones(ns)

        true_data['plx'] = plx_true

        # True distance modulus and magnitudes
        mu_true = 5 * np.log10(100/plx_true)
        app_mags_true = {x: [] for x in obs_mags}
        for mag in obs_mags:
            app_mags_true[mag] = true_data[mag] + mu_true

    # Prepare dictionary with observed parameters
    obs_data = {x: [] for x in obs_params}

    # Make observed data assuming Gaussian uncertainties
    for oparam in obs_data:
        if oparam in obs_mags:
            obs_data[oparam] = app_mags_true[oparam] + \
                               np.random.normal(0, obs_params[oparam], ns)
        else:
            obs_data[oparam] = true_data[oparam] + \
                               np.random.normal(0, obs_params[oparam], ns)

    # Use pandas to organize the data and print it to a text file
    pd_data = pd.DataFrame.from_dict(obs_data)
    for i, column in enumerate(list(pd_data)[::-1]):
        pd_data.insert(len(obs_data)-i, column+'_unc',
                       obs_params[column]*np.ones(ns))
    pd_data.to_csv(outputfile, index_label='#sid', sep='\t',
                   float_format='%10.4f')