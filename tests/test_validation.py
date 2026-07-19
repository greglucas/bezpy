"""Tests for bezpy.validation: leave-one-out skill and the synthetic-truth
benchmark that arbitrates method defaults (D6/D7)."""
import numpy as np
import pytest
import shapely.geometry

pysecs = pytest.importorskip("pysecs")

from bezpy.tl import TransmissionLine  # noqa: E402
from bezpy.validation import (  # noqa: E402
    leave_one_out,
    score_analytic_calibration,
    score_ensemble_calibration,
    score_extrapolation,
    score_interpolation_methods,
    synthetic_scenario,
)

RNG = np.random.default_rng(9)


@pytest.fixture(scope="module")
def scenario():
    return synthetic_scenario(seed=0, ntimes=2048)


def make_lines():
    coords = [
        [(-106.5, 38.0), (-105.0, 38.8), (-103.5, 39.6), (-102.0, 40.5)],
        [(-104.5, 42.5), (-103.0, 41.5), (-101.5, 41.0)],
    ]
    return [TransmissionLine(shapely.geometry.LineString(c)) for c in coords]


def test_true_e_matches_pipeline_at_sites(scenario):
    """At the MT sites themselves, the production path must reproduce the
    analytic truth (same Z, same convolution)."""
    lats = [s.latitude for s in scenario.sites]
    lons = [s.longitude for s in scenario.sites]
    truth = scenario.true_e(lats, lons)
    produced = scenario.site_e()
    np.testing.assert_allclose(produced, truth, atol=1e-9*np.abs(truth).max())


def test_interpolation_scores(scenario):
    """The harness measures rankings; it must not presuppose them (see
    VALIDATION.md for the current results). Verify completeness and sanity:
    every option lands well under 100% error against truth."""
    lines = make_lines()
    table = score_interpolation_methods(scenario, lines,
                                        how=("nn", "delaunay"),
                                        resample_km=(None, 2.0))
    assert len(table) == 4
    assert (table["rel_rms_error"] < 0.5).all()
    assert (table["rel_rms_error"] > 0.).all()


def test_extrapolation_asymptotic_beats_zeros(scenario):
    """The D6 arbiter: on truncated-band impedances, the asymptotic
    continuation must reduce the E error against full-band truth."""
    table = score_extrapolation(scenario)
    assert table.loc["asymptotic", "rel_rms_error"] < \
        table.loc["zeros", "rel_rms_error"]


def test_ensemble_calibration_covers_truth_near_nominal_rate():
    """The roadmap 1.2 phase 1 acceptance test: the n_realizations ensemble's
    +-1 std band must cover known truth at approximately the nominal 68%
    rate -- not exactly (the production spline-smoothing path damps
    propagated noise a bit, see score_ensemble_calibration's docstring and
    VALIDATION.md), but nowhere near 0% (no propagation happening) or 100%
    (grossly overstated uncertainty)."""
    result = score_ensemble_calibration(n_trials=15, n_realizations=60,
                                        ntimes=1024, seed=0)
    assert 0.45 < result["coverage_1sigma"] < 0.80
    assert result["n_samples"] > 1000


def test_analytic_calibration_is_conservative_but_not_broken():
    """The roadmap 1.2 phase 2 acceptance test: analytic (return_var=True)
    coverage does NOT hit the nominal 68% -- it lands around 0.32-0.37,
    substantially more conservative than phase 1's ensemble (~0.6-0.65),
    because the Jacobian is frozen at the one noisy measured impedance and
    misses the smoothing spline's knot-selection variability that only
    shows up when it is actually refit under different noise draws (see
    VALIDATION.md 1.2 for the full write-up and per-site evidence this is
    systematic, not a few outliers). This test enforces the OBSERVED,
    reproducible range -- not the aspirational nominal rate -- so it fails
    if a future change breaks propagation entirely (near 0%) or makes it
    wildly over-wide (near 100%), without chasing the documented gap as if
    it were a regression."""
    result = score_analytic_calibration(n_trials=10, ntimes=1024, seed=0)
    assert 0.20 < result["coverage_1sigma"] < 0.55
    assert result["n_samples"] > 1000


def test_leave_one_out_scores():
    from bezpy.mag.secs import R_EARTH

    obs_latlons = np.array([[lat, lon]
                            for lat in np.linspace(34, 46, 5)
                            for lon in np.linspace(-110, -96, 5)], dtype=float)
    poles = np.array([[lat, lon, R_EARTH + 110e3]
                      for lat in np.linspace(30, 50, 4)
                      for lon in np.linspace(-115, -90, 5)])
    source = pysecs.SECS(sec_df_loc=poles)
    t = np.linspace(0, 4*np.pi, 60)[:, None]
    source.sec_amps = 1e13*np.sin(t + RNG.uniform(0, 2*np.pi, len(poles)))
    B = source.predict(np.column_stack([obs_latlons,
                                        np.full(len(obs_latlons), R_EARTH)]))

    scores = leave_one_out(B, obs_latlons=obs_latlons,
                           metrics=("rmse", "correlation"))
    assert list(scores.columns) == ["rmse", "correlation"]
    assert len(scores) == len(obs_latlons)
    assert (scores["rmse"] >= 0).all()
    # Interior observatories are well predicted by their neighbors
    assert scores["correlation"].median() > 0.9


def test_leave_one_out_coherence_bands():
    times = np.arange(512)*60.
    obs_latlons = np.array([[lat, lon]
                            for lat in np.linspace(38, 42, 3)
                            for lon in np.linspace(-106, -100, 3)], dtype=float)
    # Smooth, spatially coherent synthetic field
    B = np.zeros((len(times), len(obs_latlons), 3))
    for k, (lat, lon) in enumerate(obs_latlons):
        B[:, k, 0] = 50*np.sin(2*np.pi*times/3600.) * (1 + 0.01*(lat - 40))
        B[:, k, 1] = 30*np.cos(2*np.pi*times/1800.) * (1 + 0.01*(lon + 103))

    scores = leave_one_out(B, obs_latlons=obs_latlons, metrics=("rmse",),
                           bands=[(600., 7200.)], dt=60.)
    coherence_column = [c for c in scores.columns if c.startswith("coherence")]
    assert len(coherence_column) == 1
    assert scores[coherence_column[0]].median() > 0.9


def test_score_fitting_methods_smoke():
    from bezpy.validation import score_fitting_methods
    table = score_fitting_methods(seed=0, ntimes=48)
    assert set(table.index) >= {"lstsq", "kalman"}
    assert (table["rmse_nT"] > 0).all() and np.isfinite(table["rmse_nT"]).all()


def test_coherence_per_component_perfect_prediction():
    """Perfect predictions must score coherence ~1 in every band (A2).

    Regression: the ravel of (ntimes, 2) interleaved Bx/By into one fake
    series and scored ~0.7 for identical inputs.
    """
    import pandas as pd
    import xarray as xr

    from bezpy.mag import SECSInterpolator
    from bezpy.mag.secs import R_EARTH  # noqa: F401 (import parity with above)
    from bezpy.validation import leave_one_out

    pysecs = pytest.importorskip("pysecs")
    rng = np.random.default_rng(7)
    ntimes = 512
    # Dense enough that a held-out observatory is well constrained by the
    # rest -- with only corner stations the LOO prediction error is real.
    obs_latlons = np.array([[lat, lon]
                            for lat in (38.0, 40.0, 42.0)
                            for lon in (-108.0, -105.0, -102.0)])
    # A smooth SEC source so left-out observatories are well predicted
    poles = np.array([[lat, lon, 6371e3 + 110e3]
                      for lat in np.linspace(34, 46, 4)
                      for lon in np.linspace(-112, -98, 4)])
    source = pysecs.SECS(sec_df_loc=poles)
    # Broadband amplitudes with power *inside* the scored 300-7200 s band --
    # coherence over a band the signal does not occupy is meaningless.
    t_sec = 60.0 * np.arange(ntimes)[:, None]
    phases = rng.uniform(0, 2 * np.pi, (3, len(poles)))
    source.sec_amps = 1e13 * sum(
        np.sin(2 * np.pi * t_sec / period + phase)
        for period, phase in zip((600.0, 1800.0, 4800.0), phases))
    b = source.predict(np.column_stack([obs_latlons,
                                        np.full(len(obs_latlons), 6371e3)]))
    times = pd.date_range("2024-01-01", periods=ntimes, freq="1min")
    ds = xr.Dataset(
        {name: (("time", "obs"), b[:, :, i])
         for i, name in enumerate(("Bx", "By", "Bz"))},
        coords={"time": times, "obs": [f"O{i}" for i in range(len(obs_latlons))],
                "latitude": ("obs", obs_latlons[:, 0]),
                "longitude": ("obs", obs_latlons[:, 1])})
    scores = leave_one_out(ds, interp=SECSInterpolator(grid="auto"),
                           metrics=("rmse",), bands=[(300.0, 7200.0)])
    coherence = scores["coherence_300-7200s"]
    # Noise-free smooth source: near-perfect held-out predictions
    assert (coherence > 0.95).all()


def test_coherence_bands_identity_and_independence():
    """_coherence_bands must be ~1 for identical series and low for
    independent noise (A2 regression: interleaved Bx/By ravel scored ~0.7
    for identical inputs; a single-segment Welch estimate scores ~1 for
    *any* inputs, hiding real prediction error)."""
    from bezpy.validation import _coherence_bands

    rng = np.random.default_rng(3)
    x = rng.standard_normal(2000)
    same = _coherence_bands(x, x, 60.0, [(300.0, 7200.0)])
    assert same[0] > 0.999
    other = _coherence_bands(x, rng.standard_normal(2000), 60.0,
                             [(300.0, 7200.0)])
    assert other[0] < 0.6
