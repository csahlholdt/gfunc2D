import numpy as np
from scipy.optimize import newton, bisect


def mu_mode(plx_obs, plx_sigma, mu0, w, method='newton'):
    '''
    Finds the mode of P(mu), i.e. the value of mu that maximises
    P(mu) = exp(-0.5*((plx_obs-p(mu))/plx_sigma)^2 - 0.5*w*(mu-mu0)^2)
    where p(mu) = (100 mas)*10^(-0.2*mu).

    Parameters
    ----------
    plx_obs : float
        Observed parallax in milliarcseconds.

    plx_sigma : float
        Uncertainty on observed parallax.

    mu0 : float
        Weighted mean of observed and prior distance modulus.

    w : float
        Sum of weights of observed and prior distance modulus.

    method : str, optional
        Method used to find zero-point of derivative.
        Can be either 'newton' or 'bisect'.
        Default is 'newton'

    Returns
    -------
    mu_mode : float
        Value of the distance modulus (mu) which maximises the function
        P(mu) = exp(-0.5*((plx_obs-p(mu))/plx_sigma)^2 - 0.5*w*(mu-mu0)).
        If no value is found, None is returned instead.
    '''

    # Define some constants used to rewrite the function P(mu)
    kappa = 0.2 * np.log(10)
    plx0 = 100 * np.exp(-kappa*mu0)
    wp = (plx0 / plx_sigma)**2
    p = plx_obs / plx0
    b = w / (kappa**2 * wp)

    # Function which is zero for maximum P(mu) and its derivative
    # Note: x = exp(-kappa*(mu-mu0))
    fx = lambda x: b*np.log(x) + x*(x-p)

    if method == 'newton':
        fx_prime = lambda x: b/x + 2*x - p

        if p > (np.exp(1)/(np.exp(1) - 1) + b/np.exp(1)):
            x1 = p - b/np.exp(1)
        else:
            x1 = (p - b + np.sqrt((p-b)**2 + 4*b)) / 2

        # Find x so f(x)=0
        # If no solution is found None is returned instead.
        with np.errstate(invalid='ignore'):
            try:
                x0 = newton(fx, x1, fx_prime)
            except:
                return None
    elif method == 'bisect':
        xtest = np.linspace(0.001, 10)
        fxtest = fx(xtest)

        # Find xb1, xb2 which bound f(x)=0
        # If no zero point exists return None
        for i in range(len(xtest)-1):
            if fxtest[i]*fxtest[i+1] < 0:
                xb1, xb2 = xtest[i], xtest[i+1]
                break
        else:
            return None

        # Find f(x)=0 by bisection
        x0 = bisect(fx, xb1, xb2)

    # Convert from x to mu
    mu_mode = -np.log(x0) / kappa + mu0

    return mu_mode


def mu_log_lik(mu, plx_obs, plx_sigma,
               mag_obs, mag_sigma, mag_abs, mu_prior, mu_prior_w):
    '''
    For an array of distance moduli (mu values), this function returns the
    log-likelihood -0.5*X2, where
    X2 = ((plx_obs-p(mu))/plx_sigma)^2 + ((mu_obs-mu)/mag_sigma)^2
                                       + mu_prior_w*(mu-mu_prior)^2.

    Parameters
    ----------
    mu : array of float
        Distance moduli for which to calculate the log-likelihood.

    plx_obs : float
        Observed parallax in milliarcseconds.

    plx_sigma : float
        Uncertainty on observed parallax.

    mag_obs : float
        Observed apparent magnitude.

    mag_sigma : float
        Uncertainty on observed apparent magnitude.

    mag_abs : float
        Absolute magnitude from the isochrone (same passband as mag_obs).

    mu_prior : float
        Prior mean distance modulus.

    mu_prior_w : float
        Statistical weight of mu_prior (= sigma_mu_priot^(-2)).

    Returns
    -------
    log_lik : array of float
        Array of log-likelihood for each value of mu in the input array.
    '''

    # Theoretical parallaxes for the different mu values
    plx = 10**(2 - 0.2 * mu)

    # chi2 for prior on mu
    x2 = mu_prior_w * (mu - mu_prior)**2

    # Add chi2 for mag_obs
    x2 += ((mag_obs - mag_abs - mu) / mag_sigma)**2

    # Add chi2 for plx_obs
    x2 += ((plx_obs - plx) / plx_sigma)**2

    log_lik = -0.5 * x2

    return log_lik


def marginalise_mu(plx_obs, plx_sigma,
                   mag_obs, mag_sigma, mag_abs, mu_prior, mu_prior_w):
    '''
    Returns the integral over mu (distance modulus) of the relative likelihood
    function L(mu) = exp(-0.5*X2), where
    X2 = ((plx_obs-p(mu))/plx_sigma)^2 + ((mu_obs-mu)/mag_sigma)^2
                                       + mu_prior_w*(mu-mu_prior)^2.

    Parameters
    ----------
    plx_obs : float
        Observed parallax in milliarcseconds.

    plx_sigma : float
        Uncertainty on observed parallax.

    mag_obs : float
        Observed apparent magnitude.

    mag_sigma : float
        Uncertainty on observed apparent magnitude.

    mag_abs : float
        Absolute magnitude from the isochrone (same passband as mag_obs).

    mu_prior : float
        Prior mean distance modulus.

    mu_prior_w : float
        Statistical weight of mu_prior (= sigma_mu_priot^(-2)).

    Returns
    -------
    lik_int_mu : float
        Integral over distance modulus of the relative likelihood function
        L(mu).
    '''

    # Combine mu_prior and mu_obs = mag_obs - mag_abs
    mag_w = mag_sigma**(-2)
    w0 = mag_w + mu_prior_w
    mu_obs = mag_obs - mag_abs
    mu0 = (mu_prior * mu_prior_w + mu_obs * mag_w) / w0

    # Calculate the mode of L(mu)
    mode = mu_mode(plx_obs, plx_sigma, mu0, w0)
    if mode is None:
        return 0

    # Calculate the log-likelihood at the mode of L(mu) and the mode+-0.1 mas
    delta_mu = 0.1
    mu_min = mode - delta_mu
    mu_max = mode + delta_mu
    mu = np.array([mu_min, mode, mu_max])
    log_lik = mu_log_lik(mu, plx_obs, plx_sigma,
                         mag_obs, mag_sigma, mag_abs, mu_prior, mu_prior_w)

    # If the log-likelihood at the mode of L(mu) is larger than the smallest of
    # log_lik(mode+-0.1mas) by at least 100, the interval (+-0.1 mas) is taken
    # to be large enough. Otherwise the interval is doubled in range until the
    # log-likelihood at the interval edges is low enough.
    while (log_lik[1] - min(log_lik[0], log_lik[2]) < 100):
        delta_mu *= 2
        mu_min = mode - delta_mu
        mu_max = mode + delta_mu
        mu = np.array([mu_min, mode, mu_max])
        log_lik = mu_log_lik(mu, plx_obs, plx_sigma,
                             mag_obs, mag_sigma, mag_abs, mu_prior, mu_prior_w)

    # The relative likelihood function L(mu) is integrated over the final
    # interval in mu (mu_mode +- delta_mu)
    mu_step = delta_mu * 0.01
    mu = np.arange(mu_min, mu_max, mu_step)
    log_lik = mu_log_lik(mu, plx_obs, plx_sigma,
                         mag_obs, mag_sigma, mag_abs, mu_prior, mu_prior_w)
    lik_int_mu = sum(np.exp(log_lik)) * mu_step

    return lik_int_mu


def marginalise_mu_fast(plx_obs, plx_sigma,
                        mag_obs, mag_sigma, mag_abs, mu_prior, mu_prior_w):
    '''
    Returns the integral over mu (distance modulus) of the relative likelihood
    function L(mu) = exp(-0.5*X2), where
    X2 = ((plx_obs-p(mu))/plx_sigma)^2 + ((mu_obs-mu)/mag_sigma)^2
                                       + mu_prior_w*(mu-mu_prior)^2.
    This is a simpler and faster version of marginalise_mu. It probably only
    works when the uncertainty on the observed magnitude is low (~0.02).

    Parameters
    ----------
    plx_obs : float
        Observed parallax in milliarcseconds.

    plx_sigma : float
        Uncertainty on observed parallax.

    mag_obs : float
        Observed apparent magnitude.

    mag_sigma : float
        Uncertainty on observed apparent magnitude.

    mag_abs : float
        Absolute magnitude from the isochrone (same passband as mag_obs).

    mu_prior : float
        Prior mean distance modulus.

    mu_prior_w : float
        Statistical weight of mu_prior (= sigma_mu_priot^(-2)).

    Returns
    -------
    lik_int_mu : float
        Integral over distance modulus of the relative likelihood function
        L(mu).
    '''

    # Combine mu_prior and mu_obs = mag_obs - mag_abs
    mag_w = mag_sigma**(-2)
    w0 = mag_w + mu_prior_w
    mu_obs = mag_obs - mag_abs
    mu0 = (mu_prior * mu_prior_w + mu_obs * mag_w) / w0

    mag_int = [mu0-5*w0**(-2), mu0+5*w0**(-2)]

    if plx_sigma / plx_obs <= 0.1 and plx_obs > 0:
        plx_int = [plx_obs-5*plx_sigma, plx_obs+5*plx_sigma]
        mu_plx_int = [-5*np.log10(plx_int[1]/100),
                      -5*np.log10(plx_int[0]/100)]
        mu_min = min(mag_int[0], mu_plx_int[0])
        mu_max = max(mag_int[1], mu_plx_int[1])
        mu_step = (mu_max - mu_min)*0.02
    else:
        mu_min = mag_int[0]
        mu_max = mag_int[1]
        mu_step = (mag_int[1]-mag_int[0])*0.02

    mu = np.arange(mu_min, mu_max, mu_step)
    log_lik = mu_log_lik(mu, plx_obs, plx_sigma,
                         mag_obs, mag_sigma, mag_abs, mu_prior, mu_prior_w)
    lik_int_mu = sum(np.exp(log_lik)) * mu_step

    return lik_int_mu
