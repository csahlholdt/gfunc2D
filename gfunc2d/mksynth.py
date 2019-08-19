import h5py
import numpy as np
import pandas as pd
from gfunc2d.gridtools import get_isochrone, get_gridparams

known_filters = ['J', 'H', 'Ks', #2MASS
                 'u', 'v', 'g', 'r', 'i' ,'z', #SkyMapper
                 'G', 'G_BPbr', 'G_BPft', 'G_RP'] #Gaia (DR2)

def generate_synth_stars(isogrid, outputfile, t_bursts, ns, feh_params,
                         IMF_alpha=2.35, rand_seed=1, extra_giants=0):
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

    feh_params : array
        An array giving the mean and dispersion of the metallicites of the
        synthetic stars [feh_mean, feh_dispersion].
        The metallicities are drawn from a normal distribution
        N(feh_mean, feh_dispersion).

    IMF_alpha : float, optional
        Power law exponent to use for the initial mass function.
        Default is 2.35 (Salpeter IMF).

    rand_seed : int, optional
        Seed for np.random. Ensures that samples can be reproduced.
        Default value is 1.

    extra_giants : float, optional
        Option to artificially increase the number of giants in the sample.
        A number between 0 and 1 setting the final fraction of the total sample
        which will be forced to be giants.
        Default is 0 in which case no extra giants are added.
    """

    # This initialises the random number generator to a given state so that
    # results can be repeated
    np.random.seed(rand_seed)

    # Settings for the synthetic dataset
    single_burst = True if len(t_bursts.shape) == 1 else False
    feh_mean, feh_disp = feh_params
    config = {'t_bursts': t_bursts, 'IMF_alpha': IMF_alpha, 'ns': ns,
              'feh_mean': feh_mean, 'feh_disp': feh_disp,
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
    for param in params + ['age', 'phase']:
        data[param] = np.zeros(ns)

    iv = 0 # Number of generated stars with valid isochrone
    ne = 0 # Number of generated stars that have evolved beyond isochrones
    # The evolutionary phase of the current star (simple dwarf or giant)
    phase_i = 0
    while iv < ns:
        # Define true age
        if single_burst:
            age = t_bursts[0] + (t_bursts[1]-t_bursts[0]) * np.random.rand()
        else:
            i_burst = np.random.choice(range(n_bursts), p=prob)
            age = t_bursts[i_burst, 0] +\
                  (t_bursts[i_burst, 1]-t_bursts[i_burst, 0]) * np.random.rand()
        feh_test = np.random.normal(feh_mean, feh_disp)

        # Get the isochrone for [Fe/H], age
        q, afa = get_isochrone(gridfile, 0.0, feh_test, age)

        # Find indices of lowest model-to-model temperature difference
        low_inds = np.argsort(np.diff(10**q['logT']))[:5]
        # Split between dwarf and giant at this index
        split_ind = int(np.median(low_inds))

        # Set the minimum mass depending on whether a star is forced to be
        # a giant
        if iv < ns*(1-extra_giants):
            # Minimum temperature to include (setting the minimum mass also)
            Teffmin_dwarf = 5300-500*feh_test
            idx_dwarf = np.argmin((np.abs(10**q['logT'][:split_ind]-Teffmin_dwarf)))
            m_min = q['Mini'][idx_dwarf]
            phase_i = 0
        else:
            m_min = q['Mini'][split_ind]
            phase_i = 1

        m = m_min * np.random.rand()**(-1/(IMF_alpha-1)) # True initial mass

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
            data['phase'][iv] = phase_i

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
        if config[cparam] is None:
            config[cparam] = 'None'
        outfile.create_dataset('config/'+cparam, data=config[cparam])

    # Save the stellar data
    for sparam in data:
        outfile.create_dataset('data/'+sparam,
                                data=data[sparam])
    outfile.close()


def make_synth_obs(synthfile, outputfile, obs_params, plx_distribution=1,
                   perturb_true_values=True):
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
        Value of the parallax in mas. Can also give the string 'Skymapper'
        which mimics the parallax distribution in SkyMapper data.
        Default value is 1.
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

    # Whether or not these magnitudes are to be observed, they are loaded
    # to be used when defining parallax uncertainties
    J_mag_true = synth_data['data/J'][:]
    G_mag_true = synth_data['data/G'][:]
    synth_data.close()

    # If parallaxes are to be fitted, the true values are assumed based on
    # the input plx_distribution
    if 'plx' in obs_params:
        plx_true = np.zeros(ns)
        if plx_distribution == 'Skymapper':
            # Approximate parallax distribution of stars in the SkyMapper survey
            for i in range(ns):
                plx_true[i] = SM_parallax(J_mag_true[i])
        else:
            # Else a constant value (given in plx_distribution)
            plx_true = plx_distribution*np.ones(ns)

        true_data['plx'] = plx_true

        # True distance modulus and magnitudes
        mu_true = 5 * np.log10(100/plx_true)
        app_mags_true = {x: [] for x in obs_mags}
        for mag in obs_mags:
            app_mags_true[mag] = true_data[mag] + mu_true
        G_app_mag_true = G_mag_true + mu_true

    # Prepare dictionary with observed parameters
    obs_data = {x: [] for x in obs_params}

    # Make observed data assuming Gaussian uncertainties
    for oparam in obs_data:
        if oparam in obs_mags:
            if obs_params[oparam][1] == 'abs':
                obs_data[oparam] = app_mags_true[oparam]
                if perturb_true_values:
                    obs_data[oparam] += np.random.normal(0, obs_params[oparam][0], ns)
            else:
                obs_data[oparam] = app_mags_true[oparam]
                if perturb_true_values:
                    obs_data[oparam] += np.random.normal(0, app_mags_true[oparam]*obs_params[oparam][0], ns)
        elif oparam == 'plx' and obs_params[oparam][1] == 'Gaia':
            plx_true_err = np.zeros(ns)
            for i in range(ns):
                plx_true_err[i] = SM_parallax_err(G_app_mag_true[i])
            obs_data[oparam] = true_data[oparam]
            if perturb_true_values:
                obs_data[oparam] += np.random.normal(0, plx_true_err)
        else:
            if obs_params[oparam][1] == 'abs':
                obs_data[oparam] = true_data[oparam]
                if perturb_true_values:
                    obs_data[oparam] += np.random.normal(0, obs_params[oparam][0], ns)
            else:
                obs_data[oparam] = true_data[oparam]
                if perturb_true_values:
                    obs_data[oparam] += np.random.normal(0, true_data[oparam]*obs_params[oparam][0], ns)

    # Use pandas to organize the data and print it to a text file
    pd_data = pd.DataFrame.from_dict(obs_data)
    for i, column in enumerate(list(pd_data)[::-1]):
        if column == 'plx' and obs_params[oparam][1] == 'Gaia':
            pd_data.insert(len(obs_data)-i, column+'_unc',
                           plx_true_err)
        else:
            if obs_params[column][1] == 'abs':
                pd_data.insert(len(obs_data)-i, column+'_unc',
                               obs_params[column][0]*np.ones(ns))
            else:
                pd_data.insert(len(obs_data)-i, column+'_unc',
                               obs_params[column][0]*true_data[column])
    pd_data.to_csv(outputfile, index_label='#sid', sep='\t',
                   float_format='%10.4f')


def SM_parallax(J_abs):
    '''
    Based on the absolute J_magnitude,
    return the parallax for a single star
    '''
    plx_ints = np.arange(-5.25, 6, 0.5)

    plx_dists = np.array([[-0.41, 0.12, -0.66],
                          [-0.36, 0.14, -0.70],
                          [-0.33, 0.16, -0.70],
                          [-0.31, 0.18, -0.70],
                          [-0.31, 0.21, -0.70],
                          [-0.33, 0.23, -0.70],
                          [-0.33, 0.24, -0.70],
                          [-0.39, 0.24, -0.70],
                          [-0.29, 0.24, -0.70],
                          [-0.27, 0.23, -0.70],
                          [-0.25, 0.23, -0.70],
                          [-0.22, 0.23, -0.70],
                          [-0.18, 0.23, -0.67],
                          [-0.14, 0.24, -0.64],
                          [-0.11, 0.24, -0.59],
                          [-0.08, 0.23, -0.53],
                          [-0.05, 0.22, -0.48],
                          [ 0.01, 0.22, -0.43],
                          [ 0.08, 0.22, -0.36],
                          [ 0.18, 0.21, -0.28],
                          [ 0.23, 0.23, -0.20],
                          [ 0.49, 0.30, -0.02]])

    plx_index = np.digitize(J_abs, plx_ints)

    if plx_index == 0:
        plx_index = 1
    elif plx_index > len(plx_dists):
        plx_index = len(plx_dists)

    plx_params = plx_dists[plx_index-1]

    plx = 0
    while plx < 10**(plx_params[2]):
        plx = 10**(np.random.normal(plx_params[0], plx_params[1]))

    return plx


def SM_parallax_err(G_app):
    '''
    Based on the apparent G_mag,
    return the parallax uncertainty for a single star
    '''
    plxerr_ints = np.arange(8, 19, 1)

    plx_errs = np.array([[-1.4, 0.17, -1.70],
                         [-1.4, 0.19, -1.70],
                         [-1.39, 0.2 , -1.70],
                         [-1.42, 0.18, -1.70],
                         [-1.45, 0.17, -1.70],
                         [-1.61, 0.17, -1.70],
                         [-1.52, 0.15, -1.70],
                         [-1.4, 0.13, -1.70],
                         [-1.3, 0.1 , -1.55],
                         [-0.92, 0.15, -1.21]])

    plxerr_index = np.digitize(G_app, plxerr_ints)

    if plxerr_index == 0:
        plxerr_index = 1
    elif plxerr_index > len(plx_errs):
        plxerr_index = len(plx_errs)

    plxerr_params = plx_errs[plxerr_index-1]

    plxerr = 0
    while plxerr < 10**(plxerr_params[2]):
        plxerr = 10**(np.random.normal(plxerr_params[0], plxerr_params[1]))

    return plxerr
