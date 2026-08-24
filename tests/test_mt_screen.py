"""Tests for bezpy.mt.screen (impedance + spatial screening)."""

import numpy as np

from bezpy.mt import Site3d, screen_impedance, screen_sites


def _site(name, rho=100.0, lat=40.0, lon=-105.0, rel_err=0.01, periods=None, phase_deg=45.0):
    """A synthetic site: uniform apparent resistivity ``rho`` (ohm-m), first-
    quadrant phase, small relative error -- a clean 1D-like transfer function.
    Z organised (xx, xy, yx, yy); off-diagonals carry the signal."""
    if periods is None:
        periods = np.logspace(1, 4, 24)
    # rho = T/5 * |Z|^2  ->  |Z| = sqrt(5*rho/T)
    absZ = np.sqrt(5.0 * rho / periods)
    zxy = absZ * np.exp(1j * np.radians(phase_deg))
    Z = np.zeros((4, len(periods)), dtype=complex)
    Z[1] = zxy
    Z[2] = -zxy  # yx = -xy, folds to the same phase
    Z_var = (rel_err * np.abs(Z)) ** 2
    Z_var[Z_var == 0] = (rel_err * np.median(absZ)) ** 2
    s = Site3d(name)
    s.periods, s.Z, s.Z_var = periods, Z, Z_var
    s.latitude, s.longitude = lat, lon
    return s


def test_clean_site_passes():
    m = screen_impedance(_site("good"))
    assert m["imp_flag"] == "ok"
    assert m["frac_rho_out"] == 0.0
    assert m["rel_err_med"] < 0.05


def test_out_of_bounds_resistivity_is_bad():
    s = _site("hot")
    # Push apparent resistivity absurdly high everywhere (|Z| x1000 -> rho x1e6)
    s.Z = s.Z * 1000.0
    assert screen_impedance(s)["imp_flag"] == "bad"


def test_huge_relative_error_is_bad():
    s = _site("noisy", rel_err=2.0)  # |dZ|/|Z| = 2 everywhere
    assert screen_impedance(s)["imp_flag"] == "bad"


def test_oscillating_resistivity_flagged():
    s = _site("rough")
    rng = np.random.default_rng(0)
    # multiply |Z| by a wild per-period factor -> jagged log-rho
    factor = 10.0 ** rng.uniform(-2, 2, s.Z.shape[1])
    s.Z = s.Z * factor
    m = screen_impedance(s)
    assert m["imp_flag"] in ("bad", "review")
    assert m["rho_roughness"] > 0.6


def test_out_of_quadrant_phase_is_bad():
    s = _site("flipped", phase_deg=-60.0)  # third/fourth-quadrant phase
    assert screen_impedance(s)["imp_flag"] == "bad"


def test_screen_sites_spatial_outlier():
    # a compact cluster of clean sites, all fine on impedance
    sites = [_site(f"s{i}", lat=40 + 0.1 * i, lon=-105 + 0.1 * (i % 3)) for i in range(9)]
    values = np.array([1000.0] * 8 + [80000.0])  # last one is a wild outlier
    table = screen_sites(sites, values=values, k=6, loo_decades=0.74)
    table = table.set_index("code")
    assert table.loc["s8", "spatial_flag"] == "bad"
    assert table.loc["s8", "verdict"] == "bad"
    # the clean interior sites survive
    assert (table.loc[[f"s{i}" for i in range(8)], "verdict"] == "ok").all()


def test_screen_sites_impedance_only():
    sites = [_site("a"), _site("b")]
    sites[1].Z = sites[1].Z * 1000.0  # b is non-physical
    table = screen_sites(sites).set_index("code")
    assert table.loc["a", "verdict"] == "ok"
    assert table.loc["b", "verdict"] == "bad"
    assert (table["spatial_flag"] == "ok").all()  # no spatial screen without values
