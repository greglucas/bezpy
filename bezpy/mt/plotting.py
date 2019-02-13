"""Plotting routines for MT data."""

__all__ = ["plot_apparent_resistivity"]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import container
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .utils import apparent_resistivity


def plot_apparent_resistivity(periods, Z, Z_var=None, fig=None,
                              ax_res=None, ax_phase=None, alpha=0.25,
                              markersize=8, figsize=(8, 6), xlim=None, ylim=None):
    """Plot generator for standard MT apparent resistivity and phase."""
    # pylint: disable=invalid-name,too-many-locals,too-many-arguments
    if ax_res is None:
        fig, ax_res = plt.subplots(figsize=figsize)
    if ax_phase is None:
        locator = make_axes_locatable(ax_res)
        ax_phase = locator.append_axes("bottom", size="50%", pad="20%", sharex=ax_res)

    resistivity, resistivity_var, phase, phase_var = apparent_resistivity(periods, Z, Z_var)

    # Default entries for each component of the tensor
    labels = ['xx', 'xy', 'yx', 'yy']
    colors = ['C1', 'C0', 'C3', 'C2']
    alphas = [alpha, 1., 1., alpha]
    # Store background color to fill the inside of the markers
    bg_color = ax_res.get_facecolor()

    for i in range(4):
        if Z_var is None:
            ax_res.plot(periods, resistivity[i, :], c=colors[i],
                        label=labels[i], alpha=alphas[i])
            ax_phase.plot(periods, phase[i, :], c=colors[i],
                          label=labels[i], alpha=alphas[i])
        else:
            # Ignoring nans
            good_vals = ~np.isnan(resistivity[i, :])
            x = periods[good_vals]
            y = resistivity[i, good_vals]
            yerr = 2*resistivity_var[i, good_vals]

            ax_res.errorbar(x, y, yerr=yerr,
                            label=labels[i], color=colors[i], marker='o', fmt='o',
                            markersize=markersize, alpha=alphas[i],
                            markerfacecolor=bg_color, markeredgecolor=colors[i], markeredgewidth=1)
            y = phase[i, good_vals]
            yerr = 2*phase_var[i, good_vals]
            ax_phase.errorbar(x, y, yerr=yerr,
                              color=colors[i], marker='o', fmt='o', alpha=alphas[i],
                              markersize=markersize, markerfacecolor=bg_color,
                              markeredgecolor=colors[i], markeredgewidth=1)

    # get handles and remove error bars
    handles, leg_labels = ax_res.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    # Don't plot line/circle separate if interpolation and points are used
    # combine the handles together to produce a nicer legend
    if len(handles) == 8:
        handles = ((handles[0], handles[4]), (handles[1], handles[5]),
                   (handles[2], handles[6]), (handles[3], handles[7]))
    ax_res.legend(handles, leg_labels[:4], ncol=1, fancybox=False, numpoints=1)

    ax_res.set_xscale('log')
    ax_res.set_yscale('log')
    ax_phase.set_xscale('log')
    ax_phase.set_yscale('linear')

    if xlim is None:
        xlim = [10**np.floor(np.log10(np.min(periods))), 10**np.ceil(np.log10(np.max(periods)))]
    if ylim is None:
        good_vals = np.logical_and(~np.isnan(resistivity), ~(resistivity == 0.))
        ylim = [10**np.floor(np.log10(np.min(resistivity[good_vals]))),
                10**np.ceil(np.log10(np.max(resistivity[good_vals])))]
    ax_res.set_xlim(xlim)
    ax_res.set_ylim(ylim)
    ax_phase.set_xlim(xlim)
    ax_phase.set_ylim(-90, 90)
    ax_phase.set_yticks([-90, -45, 0, 45, 90])

    ax_res.set_ylabel(r"$\rho_a$ ($\Omega m$)")
    ax_res.set_title("Apparent Resistivity")
    ax_res.tick_params(labelbottom=False)

    ax_phase.set_xlabel("Period (s)")
    ax_phase.set_ylabel(r"$\phi$ (deg)")
    ax_phase.set_title("Phase Angle")

    return (fig, ax_res, ax_phase)
