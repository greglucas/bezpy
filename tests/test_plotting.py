"""Smoke tests for the optional matplotlib plotting routines."""
import numpy as np
import pytest

import bezpy

# Skip the whole module if matplotlib is not installed (it is an optional extra).
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def test_plot_apparent_resistivity_3d(site3d):
    fig, ax_res, ax_phase = site3d.plot_apparent_resistivity()
    assert ax_res.get_ylabel() != ""
    matplotlib.pyplot.close(fig)


def test_plot_apparent_resistivity_3d_with_interp(site3d):
    interp_freqs = np.logspace(-3, -1, 20)
    fig, ax_res, ax_phase = site3d.plot_apparent_resistivity(interp_freqs=interp_freqs)
    matplotlib.pyplot.close(fig)


def test_plot_apparent_resistivity_1d():
    site = bezpy.mt.get_1d_site("CP-1")
    fig = site.plot_apparent_resistivity(interp_freqs=np.logspace(-3, -1, 20))
    matplotlib.pyplot.close(fig)


def test_plot_depth_1d():
    site = bezpy.mt.get_1d_site("CP-1")
    ax = site.plot_depth()
    assert ax is not None
    matplotlib.pyplot.close(ax.figure)
