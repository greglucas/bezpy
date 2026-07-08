"""Tests for bezpy.validation: leave-one-out skill and the synthetic-truth
benchmark that arbitrates method defaults (D6/D7)."""
import numpy as np
import pytest
import shapely.geometry

pysecs = pytest.importorskip("pysecs")

from bezpy.tl import TransmissionLine  # noqa: E402
from bezpy.validation import (  # noqa: E402
    leave_one_out,
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
