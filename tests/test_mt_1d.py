"""Physics tests for the 1D magnetotelluric impedance recursion (Site1d)."""

import numpy as np
import pytest

import bezpy
from bezpy.mt.utils import apparent_resistivity


def test_halfspace_apparent_resistivity(halfspace_site):
    # For a uniform half-space the recovered apparent resistivity must equal the
    # layer resistivity at every period, independent of frequency.
    freqs = np.array([1e-4, 1e-3, 1e-2, 1e-1])
    Z = halfspace_site.calcZ(freqs)
    periods = 1.0 / freqs
    resistivity, _, phase, _ = apparent_resistivity(periods, Z)
    # Off-diagonal components carry the signal for a 1D earth.
    assert np.allclose(resistivity[1, :], 100.0, rtol=1e-6)
    assert np.allclose(resistivity[2, :], 100.0, rtol=1e-6)


def test_halfspace_phase_is_45_degrees(halfspace_site):
    freqs = np.array([1e-3, 1e-2, 1e-1])
    Z = halfspace_site.calcZ(freqs)
    _, _, phase, _ = apparent_resistivity(1.0 / freqs, Z)
    assert np.allclose(phase[1, :], 45.0, atol=1e-6)


def test_1d_diagonal_components_are_zero(halfspace_site):
    # A 1D earth has no xx/yy response.
    Z = halfspace_site.calcZ(np.array([1e-3, 1e-2, 1e-1]))
    assert np.allclose(Z[0, :], 0.0)
    assert np.allclose(Z[3, :], 0.0)


def test_1d_antisymmetry(halfspace_site):
    # Z is returned as [0, Zxy; -Zxy, 0]; Zyx must be the negative of Zxy.
    Z = halfspace_site.calcZ(np.array([1e-3, 1e-2, 1e-1]))
    assert np.allclose(Z[2, :], -Z[1, :])


def test_1d_dc_frequency_is_zero(halfspace_site):
    # The DC (zero-frequency) impedance is defined as zero.
    Z = halfspace_site.calcZ(np.array([0.0, 1e-2]))
    assert np.allclose(Z[:, 0], 0.0)


def test_1d_dc_frequency_is_zero_in_any_position(halfspace_site):
    Z = halfspace_site.calcZ(np.array([1e-2, 0.0, 1e-3]))
    assert np.allclose(Z[:, 1], 0.0)
    assert np.isfinite(Z).all()


def test_resistivity_increases_impedance():
    # A more resistive half-space produces a larger impedance magnitude.
    low = bezpy.mt.Site1d("low", thicknesses=[1e9], resistivities=[10.0])
    high = bezpy.mt.Site1d("high", thicknesses=[1e9], resistivities=[1000.0])
    freqs = np.array([1e-2])
    assert np.abs(high.calcZ(freqs)[1, 0]) > np.abs(low.calcZ(freqs)[1, 0])


def test_depths_are_cumulative():
    site = bezpy.mt.Site1d(
        "layers", thicknesses=[100.0, 200.0, 300.0], resistivities=[10.0, 20.0, 30.0]
    )
    assert np.allclose(site.depths, [100.0, 300.0, 600.0])


def test_optional_halfspace_plot_thickness_is_not_an_extra_layer():
    pytest.importorskip("matplotlib")
    site = bezpy.mt.Site1d(
        "profile",
        thicknesses=[100.0, 200.0, 300.0],
        resistivities=[10.0, 20.0, 30.0],
    )
    ax = site.plot_depth()
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0.0, 0.1, 0.3])


def test_too_many_thicknesses_rejected():
    with pytest.raises(ValueError, match="optional plotting thickness"):
        bezpy.mt.Site1d(
            "profile",
            thicknesses=[1.0, 2.0, 3.0],
            resistivities=[10.0, 20.0],
        )
