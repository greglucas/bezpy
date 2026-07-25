"""Tests for bezpy.mt.utils apparent resistivity calculations."""

import numpy as np

from bezpy.mt.utils import apparent_resistivity


def test_apparent_resistivity_known_value():
    # Z = 1+0j [mV/km]/[nT] at T=5s -> rho_a = T/5 * |Z|^2 = 1.0, phase = 0.
    Z = np.ones((4, 1), dtype=complex)
    periods = np.array([5.0])
    resistivity, res_var, phase, phase_var = apparent_resistivity(periods, Z)
    assert np.allclose(resistivity, 1.0)
    assert np.allclose(phase, 0.0)


def test_apparent_resistivity_scaling_with_period():
    # rho_a scales linearly with period for fixed Z.
    Z = 2 * np.ones((4, 3), dtype=complex)
    periods = np.array([5.0, 10.0, 50.0])
    resistivity, _, _, _ = apparent_resistivity(periods, Z)
    # rho_a = T/5 * |Z|^2 = T/5 * 4
    expected = periods / 5.0 * 4.0
    assert np.allclose(resistivity[1, :], expected)


def test_apparent_resistivity_phase_45deg():
    # Z = 1+1j -> angle 45 deg.
    Z = (1 + 1j) * np.ones((4, 1), dtype=complex)
    _, _, phase, _ = apparent_resistivity(np.array([10.0]), Z)
    assert np.allclose(phase, 45.0)


def test_apparent_resistivity_without_variance_reports_unknown_errors():
    # Missing uncertainty must not be turned into a fabricated finite error.
    Z = (1 + 1j) * np.ones((4, 2), dtype=complex)
    resistivity, res_var, phase, phase_var = apparent_resistivity(np.array([10.0, 20.0]), Z)
    assert np.all(np.isfinite(resistivity))
    assert np.isnan(res_var).all()
    assert np.isnan(phase_var).all()
