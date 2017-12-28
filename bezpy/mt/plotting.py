"""Plotting routines for MT data."""

__all__ = ["plot_apparent_resistivity"]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .utils import apparent_resistivity

def plot_apparent_resistivity(periods, Z, Z_var=None, fig=None, ax_res=None, ax_phase=None, alpha=0.25,
                              markersize=8, figsize=(12,10), xlim=None, ylim=None):
    if ax_res is None:
        fig, ax_res = plt.subplots(figsize=figsize)
    if ax_phase is None:
        ax_phase = make_axes_locatable(ax_res).append_axes("bottom", size=2.5, pad=0.5, sharex=ax_res)

    resistivity, resistivity_var, phase, phase_var = apparent_resistivity(periods, Z, Z_var)

    # No variance, so this should be just a standard line plot
    if Z_var is None:
        ax_res.plot(periods, resistivity[1,:], c='C0', linewidth=1.5, label='xy')
        ax_res.plot(periods, resistivity[2,:], c='C3', linewidth=1.5, label='yx')

        ax_res.plot(periods, resistivity[0,:], c='C1', linewidth=1.5, label='xx', alpha=alpha)
        ax_res.plot(periods, resistivity[3,:], c='C2', linewidth=1.5, label='yy', alpha=alpha)

        ax_phase.plot(periods, phase[1,:], c='C0', linewidth=1.5, label='xy')
        ax_phase.plot(periods, phase[2,:], c='C3', linewidth=1.5, label='yx')

        ax_phase.plot(periods, phase[0,:], c='C1', linewidth=1.5, label='xx', alpha=alpha)
        ax_phase.plot(periods, phase[3,:], c='C2', linewidth=1.5, label='yy', alpha=alpha)

    # Got the variance, so plot it as scatter points with error bars
    else:
        ax_res.errorbar(periods, resistivity[1,:], yerr=2*resistivity_var[1,:], label='xy',
                    color='C0', marker='o', fmt='o', markersize=markersize, elinewidth=2,
                    markerfacecolor='none', markeredgecolor='C0', markeredgewidth=2)
        ax_res.errorbar(periods, resistivity[2,:], yerr=2*resistivity_var[2,:], label='yx',
                    color='C3', marker='X', fmt='o', markersize=markersize, elinewidth=2)

        ax_res.errorbar(periods, resistivity[0,:], yerr=2*resistivity_var[0,:], label='xx',
                    color='C1', marker='*', fmt='o', markersize=markersize, elinewidth=2, alpha=alpha)
        ax_res.errorbar(periods, resistivity[3,:], yerr=2*resistivity_var[3,:], label='yy',
                    color='C2', marker='^', fmt='o', markersize=markersize, elinewidth=2, alpha=alpha)

        ax_phase.errorbar(periods, phase[1,:], yerr=2*phase_var[1,:],
                    color='C0', marker='o', fmt='o', markersize=markersize, elinewidth=2,
                    markerfacecolor='none', markeredgecolor='C0', markeredgewidth=2)
        ax_phase.errorbar(periods, phase[2,:], yerr=2*phase_var[2,:],
                    color='C3', marker='X', fmt='o', markersize=markersize, elinewidth=2)

        ax_phase.errorbar(periods, phase[0,:], yerr=2*phase_var[0,:],
                    color='C1', marker='*', fmt='o', markersize=markersize, elinewidth=2, alpha=alpha)
        ax_phase.errorbar(periods, phase[3,:], yerr=2*phase_var[3,:],
                    color='C2', marker='^', fmt='o', markersize=markersize, elinewidth=2, alpha=alpha)
        # get handles
        #ax_res.legend(ncol=1, fancybox=False)
        handles, labels = ax_res.get_legend_handles_labels()
        # remove the errorbars
        handles = [h[0] for h in handles]
        # use them in the legend
        ax_res.legend(handles, labels, ncol=1, fancybox=False, numpoints=1)

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
    ax_res.tick_params(labelbottom='off')

    ax_phase.set_xlabel("Period (s)")
    ax_phase.set_ylabel(r"$\phi$ (deg)")
    ax_phase.set_title("Phase Angle")

    return (fig, ax_res, ax_phase)
