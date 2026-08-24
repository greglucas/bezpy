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


# ---------------------------------------------------------------------
# bezpy.plot map helpers (Agg smoke tests; cartopy parts skip if absent)
# ---------------------------------------------------------------------

import shapely.geometry  # noqa: E402

from bezpy.plot import (  # noqa: E402
    add_map_features,
    plot_hazard_map,
    plot_lines,
    plot_vectors,
    scale_vectors,
)
from bezpy.tl import TransmissionLine  # noqa: E402

LATLONS = np.array([[40.0, -105.0], [41.0, -104.0], [42.0, -103.0]])


def test_scale_vectors_preserves_angle():
    x, y = np.array([3.0, 0.0, 1000.0]), np.array([4.0, -200.0, 0.0])
    sx, sy = scale_vectors(x, y)
    np.testing.assert_allclose(np.arctan2(sy, sx), np.arctan2(y, x))
    # log10 magnitudes, with sub-1 magnitudes clamped to 1 (-> 0)
    np.testing.assert_allclose(np.hypot(sx, sy), [np.log10(5.0), np.log10(200.0), 3.0])
    zx, zy = scale_vectors(np.array([0.0]), np.array([0.0]))
    assert zx[0] == 0.0 and zy[0] == 0.0


def test_plot_vectors():
    fig, ax = matplotlib.pyplot.subplots()
    quiver = plot_vectors(ax, LATLONS, np.array([[10.0, 5.0], [0.0, -30.0], [100.0, 0.0]]))
    assert quiver is not None
    with pytest.raises(ValueError, match="scale must be"):
        plot_vectors(ax, LATLONS, np.ones((3, 2)), scale="cubic")
    matplotlib.pyplot.close(fig)


def test_plot_lines_colored_by_value():
    lines = [TransmissionLine(shapely.geometry.LineString([(-105.0, 40.0), (-104.0, 41.0)]))]
    fig, ax = matplotlib.pyplot.subplots()
    collection = plot_lines(ax, lines, values=[3.5], linewidths=2)
    np.testing.assert_allclose(collection.get_array(), [3.5])
    matplotlib.pyplot.close(fig)


def test_plot_hazard_map():
    fig, ax = matplotlib.pyplot.subplots()
    artist = plot_hazard_map(ax, LATLONS, [1.0, 10.0, 100.0], colorbar_label="E (mV/km)")
    assert artist.norm.__class__.__name__ == "LogNorm"
    matplotlib.pyplot.close(fig)


def test_plot_hazard_map_fades_uncertain_sites():
    """Sites at/above uncertainty_limit are faded, others keep
    full opacity; no effect without both relative_uncertainty and the limit."""
    fig, ax = matplotlib.pyplot.subplots()
    artist = plot_hazard_map(
        ax,
        LATLONS,
        [1.0, 10.0, 100.0],
        relative_uncertainty=[0.05, 0.5, 0.9],
        uncertainty_limit=0.3,
    )
    np.testing.assert_array_equal(artist.get_alpha(), [1.0, 0.25, 0.25])
    matplotlib.pyplot.close(fig)

    fig, ax = matplotlib.pyplot.subplots()
    plain = plot_hazard_map(ax, LATLONS, [1.0, 10.0, 100.0])
    assert plain.get_alpha() is None
    matplotlib.pyplot.close(fig)


def test_add_map_features_requires_cartopy():
    pytest.importorskip("cartopy")
    import cartopy.crs as ccrs

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    features = add_map_features(ax)
    assert len(features) == 3
    matplotlib.pyplot.close(fig)
