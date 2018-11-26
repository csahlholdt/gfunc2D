import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mpls
mpls.use('classic')

from gfunc2d.gridtools import get_isochrone
from gfunc2d.gstats import smooth_gfunc2d, norm_gfunc


def loglik_plot(ax, g, tau_array, feh_array, smooth=True):
    # Add small number to allow logarithmic scale
    eps = 1e-20
    A = g.T + eps

    if smooth:
        A = smooth_gfunc2d(A)
    A = norm_gfunc(A)

    dtau = tau_array[1] - tau_array[0]
    dfeh = feh_array[1] - feh_array[0]
    plot_lims = (tau_array[0]-dtau, tau_array[-1]+dtau,
                 feh_array[0]-dfeh, feh_array[-1]+dfeh)
    cax = ax.imshow(np.log10(A), origin='lower', extent=plot_lims,
                    aspect='auto', interpolation='none')
    cbar = plt.gcf().colorbar(cax)
    cbar.set_label('log10(G-function)')

    ax.set_xlabel('Age [Gyr]')
    ax.set_ylabel('[Fe/H]')
    ax.grid()

    ax.set_xlim(plot_lims[:2])
    ax.set_ylim(plot_lims[2:])


def loglik_save(g, tau_array, feh_array, savename, smooth=True):
    fig, ax = plt.subplots()
    loglik_plot(ax, g, tau_array, feh_array, smooth)

    fig.tight_layout()
    fig.savefig(savename)
    plt.close(fig)


def contour_plot(axes, g, tau_array, feh_array, smooth=True):
    assert len(axes) == 3

    # Add small number to allow logarithmic scale
    eps = 1e-20
    A = g.T + eps

    if smooth:
        A = smooth_gfunc2d(A)
    A = norm_gfunc(A)

    ax0, ax1, ax2 = axes

    percent = [70, 80, 90, 95, 99]
    percentiles = np.percentile(np.log10(A[A > 1e-15]), percent)
    try:
        ax0.contour(tau_array, feh_array, np.log10(A), percentiles, colors='k',
                    linestyles='solid')
    except ValueError:
        ax0.contour(tau_array, feh_array, np.log10(A), [percentiles[-1]],
                    colors='k', linestyles='solid')

    ax0.set_xlabel('Age [Gyr]')
    ax0.set_ylabel('[Fe/H]')
    ax0.invert_yaxis()
    ax0.grid()

    dtau = tau_array[1] - tau_array[0]
    dfeh = feh_array[1] - feh_array[0]
    ax0.set_xlim(tau_array[0]-dtau, tau_array[-1]+dtau)
    ax0.set_ylim(feh_array[0]-dfeh, feh_array[-1]+dfeh)

    tau_dist = np.sum(A, axis=0)
    tau_dist /= np.amax(tau_dist)

    ax1.plot(tau_array, tau_dist)
    ax1.set_ylim([0, 1.2*np.amax(tau_dist)])
    ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax1.grid()

    feh_dist = np.sum(A, axis=1)
    feh_dist /= np.amax(feh_dist)

    ax2.plot(feh_dist, feh_array)
    ax2.set_xlim([0, 1.2*np.amax(feh_dist)])
    ax2.set_xticks([0.0, 0.5, 1.0])
    ax2.grid()

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)


def contour_save(g, tau_array, feh_array, savename, smooth=True):
    fig = plt.figure()

    ax0 = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=3, sharex=ax0)
    ax2 = plt.subplot2grid((4, 4), (1, 3), rowspan=3, sharey=ax0)

    contour_plot((ax0, ax1, ax2), g, tau_array, feh_array, smooth)

    fig.tight_layout()
    fig.savefig(savename)
    plt.close(fig)


def hr_plot(ax, isodict, sid, hr_axes, hr_vals, hr_units, par=None,
            alpha=0, feh=0, ages=[0.5, 1, 3, 6, 10, 15]):
    xax, yax = hr_axes
    xval, yval = hr_vals
    xunit, yunit = hr_units

    if xax == 'logT':
        xval_plot = np.log10(xval)
    else:
        xval_plot = xval

    if yunit == 'mag':
        yval_plot = -5*np.log10(par/100)
        legend_loc = 3
    else:
        yval_plot = yval
        legend_loc = 2

    isos = []
    act_ages =  []
    for age in ages:
        iso, act_afa = get_isochrone(isodict, alpha, feh, age)
        isos.append(iso)
        act_ages.append(act_afa[2])

    for i, iso in enumerate(isos):
        if yunit == 'mag':
            ax.plot(iso[xax], yval - iso[yax], '.', markersize=3, zorder=0,
                    label=str(act_ages[i]) + ' Gyr')
        else:
            ax.plot(iso[xax], iso[yax], '.', markersize=3, zorder=0,
                    label=str(act_ages[i]) + ' Gyr')

    ax.scatter(xval_plot, yval_plot, marker='*', c='k', s=50, zorder=1)
    if xax == 'logT':
        xlim_low = min(xval_plot-0.05, max(xval_plot-0.2, min(isos[-1][xax])))
        xlim_high = max(xval_plot+0.05, min(xval_plot+0.2, max(isos[0][xax])))
        ax.set_xlim(xlim_low, xlim_high)
        ax.invert_xaxis()
        ax.set_xlabel(r'$\log(T_{\mathrm{eff}}[\mathrm{K}])$')
    else:
        ax.set_xlabel(xax)
    if yax == 'logg':
        ylim_low = min(yval_plot-0.5, max(yval_plot-2, min(isos[-1][yax])))
        ylim_high = max(yval_plot+0.5, min(yval_plot+2, max(isos[-1][yax])))
        ax.set_ylim(ylim_low, ylim_high)
        ax.invert_yaxis()
        ax.set_ylabel(r'$\log g\;\;[\log(\mathrm{cm}/\mathrm{s}^2)]$')
    else:
        ax.set_ylabel(yax)
    if yunit == 'mag':
        ylim_low = min(yval_plot-1, max(yval_plot-4, yval-max(isos[-1][yax])))
        ylim_high = max(yval_plot+1, min(yval_plot+4, yval-min(isos[-1][yax])))
        if np.isfinite(ylim_low) and np.isfinite(ylim_high):
            ax.set_ylim(ylim_low, ylim_high)
        ax.set_ylabel(r'$\mu$ (Distance modulus)')

    ax.set_title(sid + ', [Fe/H] = ' + str(act_afa[1]))
    #ax.legend(loc=legend_loc, fontsize=11, ncol=2)


def hr_save(isodict, sid, hr_axes, hr_vals, hr_units, savename, par=None,
            alpha=0, feh=0, ages=[0.5, 1, 3, 6, 10, 15]):
    fig, ax = plt.subplots(figsize=(4.5,5.5))

    hr_plot(ax, isodict, sid, hr_axes, hr_vals, hr_units, par=par,
            alpha=alpha, feh=feh, ages=ages)

    fig.tight_layout()
    fig.savefig(savename)
    plt.close(fig)
