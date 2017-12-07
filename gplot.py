import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mpls
mpls.use('classic')


def gplot_loglik(g, tau_array, feh_array, smooth=True, savename=None, show=True):
    # Add small number to allow logarithmic scale
    eps = 1e-20
    A = g.T + eps

    if smooth:
        kernel = np.array([0.25, 0.5, 0.25])
        func = lambda x: np.convolve(x, kernel, mode='same')
        B = np.apply_along_axis(func, 0, A)
        C = np.apply_along_axis(func, 1, B)
    else:
        C = A / np.amax(A)
    
    C /= np.amax(C)

    fig, ax = plt.subplots()

    dtau = tau_array[1] - tau_array[0]
    dfeh = feh_array[1] - feh_array[0]
    plot_lims = (tau_array[0]-dtau, tau_array[-1]+dtau, feh_array[-1]+dfeh, feh_array[0]-dfeh)
    cax = ax.imshow(np.log10(C), extent=plot_lims, aspect='auto', interpolation='none')
    cbar = fig.colorbar(cax)
    cbar.set_label('log10(G-function)')

    ax.set_xlabel('Age [Gyr]')
    ax.set_ylabel('[Fe/H]')
    ax.invert_yaxis()
    ax.grid()

    ax.set_xlim(tau_array[0]-dtau, tau_array[-1]+dtau)
    ax.set_ylim(feh_array[0]-dfeh, feh_array[-1]+dfeh)

    if savename is not None:
        fig.savefig(savename)

    if show:
        plt.show()

    plt.close(fig)


def gplot_contour(g, tau_array, feh_array, smooth=True, savename=None, show=True):
    # Add small number to allow logarithmic scale
    eps = 1e-20
    A = g.T + eps

    if smooth:
        kernel = np.array([0.25, 0.5, 0.25])
        func = lambda x: np.convolve(x, kernel, mode='same')
        B = np.apply_along_axis(func, 0, A)
        C = np.apply_along_axis(func, 1, B)
    else:
        C = A / np.amax(A)
    
    C /= np.amax(C)

    fig = plt.figure()

    ax0 = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=3, sharex=ax0)
    ax2 = plt.subplot2grid((4, 4), (1, 3), rowspan=3, sharey=ax0)

    percent = [70, 80, 90, 95, 99]
    for p in percent:
        percentiles = np.percentile(np.log10(C[C > 1e-15]), percent)
    ax0.contour(tau_array, feh_array, np.log10(C), percentiles, colors='k', linestyles='solid')

    ax0.set_xlabel('Age [Gyr]')
    ax0.set_ylabel('[Fe/H]')
    ax0.invert_yaxis()
    ax0.grid()

    dtau = tau_array[1] - tau_array[0]
    dfeh = feh_array[1] - feh_array[0]
    ax0.set_xlim(tau_array[0]-dtau, tau_array[-1]+dtau)
    ax0.set_ylim(feh_array[0]-dfeh, feh_array[-1]+dfeh)

    tau_dist = np.sum(C, axis=0)
    tau_dist /= np.amax(tau_dist)

    ax1.plot(tau_array, tau_dist)
    ax1.set_ylim([0, 1.2*np.amax(tau_dist)])
    ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    feh_dist = np.sum(C, axis=1)
    feh_dist /= np.amax(feh_dist)

    ax2.plot(feh_dist, feh_array)
    ax2.set_xlim([0, 1.2*np.amax(feh_dist)])
    ax2.set_xticks([0.0, 0.5, 1.0])

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)

    if savename is not None:
        fig.savefig(savename)

    if show:
        plt.show()

    plt.close(fig)
