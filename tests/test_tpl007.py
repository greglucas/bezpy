"""Tests for bezpy.tpl007: NERC TPL-007 alpha/beta scaling factors.

The exact numbers in the standard's Table 3 cannot be reproduced here: beta
is defined against the reference geomagnetic field waveform NERC distributes
with TPL-007, which bezpy does not redistribute. What is tested instead is
everything that does not depend on having that file --

* the closed-form pieces (eq. 1-3, Tables 2 and 3) exactly;
* the invariants that make ``normalize="reference"`` the robust default:
  beta of the reference model is identically 1, and beta is unchanged by any
  overall rescaling of the waveform;
* the physics ordering, against the published Table 3 -- a resistive model
  must give a larger beta than a conductive one, and driving the shipped USGS
  1-D profiles through the real pipeline must rank them the same way the
  standard's own table ranks them.

The last one is the substantive check: it exercises the whole
impedance -> convolution -> peak -> normalize path against 20 published
values, and only the storm spectrum differs.
"""

import numpy as np
import pandas as pd
import pytest

import bezpy
from bezpy.tpl007 import (
    ALPHA_TABLE,
    BENCHMARK_PEAK_E,
    BETA_TABLE,
    QUEBEC_REFERENCE_MODEL,
    alpha_factor,
    beta_factor,
    peak_geoelectric_field,
    tabulated_beta,
)


def _storm_like(seed=1989, n=16384, dt=10.0, band=(100.0, 10000.0), exponent=1.0, same=False):
    """A reproducible stand-in for the NERC reference waveform.

    Band-limited to the periods a real storm carries its power in (and that
    the MT impedances are measured over). Leaving the band open lets
    near-Nyquist content dominate the peak of E, which is not how the
    reference storm behaves and measurably degrades the agreement with
    Table 3 (see docs/validation.md item 1.3).

    :param same: drive both components with the identical series, isolating
        the ground's response from the source's polarization
    """
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n, d=dt)
    periods = np.divide(1.0, freqs, out=np.full_like(freqs, np.inf), where=freqs > 0)
    amplitude = np.zeros_like(freqs)
    in_band = (freqs > 0) & (periods >= band[0]) & (periods <= band[1])
    amplitude[in_band] = freqs[in_band] ** -exponent

    def draw():
        spectrum = amplitude * np.exp(1j * rng.uniform(0, 2 * np.pi, len(freqs)))
        series = np.fft.irfft(spectrum, n)
        return 1000.0 * series / np.abs(series).max()

    first = draw()
    return (first, first.copy()) if same else (first, draw())


@pytest.fixture(name="waveform")
def waveform_fixture():
    """A broadband, storm-like (Bx, By) plane wave at 10 s sampling."""
    return _storm_like()


@pytest.fixture(name="calibrated_waveform")
def calibrated_waveform_fixture(waveform):
    """``waveform`` rescaled so the Quebec model peaks at exactly 8 V/km.

    This is how NERC built the reference waveform, so a waveform calibrated
    this way makes ``normalize="absolute"`` and ``"reference"`` agree.
    """
    mag_x, mag_y = waveform
    Ex, Ey = QUEBEC_REFERENCE_MODEL.convolve_fft(mag_x, mag_y, dt=10)
    peak = np.max(np.hypot(Ex, Ey)) / 1e3  # V/km
    scale = BENCHMARK_PEAK_E / peak
    return mag_x * scale, mag_y * scale


# ---------------------------------------------------------------- closed form


@pytest.mark.parametrize(
    "latitude,expected",
    [(60, 0.992), (54, 0.498), (50, 0.314), (45, 0.177), (40, 0.1), (20, 0.1), (75, 1.0)],
)
def test_alpha_formula(latitude, expected):
    """Equation 3, including the [0.1, 1] clip at both ends.

    Note eq. 3 reaches only 0.992 at the 60-degree reference latitude where
    Table 2 says 1.0 -- the standard's two routes genuinely disagree, and
    this pins which one this function implements.
    """
    assert alpha_factor(latitude) == pytest.approx(expected, abs=5e-4)


def test_alpha_table_matches_published_values():
    """method='table' returns Table 2 verbatim at the tabulated latitudes."""
    got = alpha_factor(ALPHA_TABLE["geomagnetic_latitude"].values, method="table")
    np.testing.assert_allclose(got, ALPHA_TABLE["alpha"].values)


def test_alpha_table_takes_next_latitude_at_or_above():
    """Between tabulated rows the conservative (larger) factor is used."""
    assert alpha_factor(55.0, method="table") == 0.6  # the 56-degree row
    assert alpha_factor(56.0, method="table") == 0.6  # exactly on the row
    assert alpha_factor(70.0, method="table") == 1.0  # saturates at >= 60


def test_alpha_sign_and_validation():
    assert alpha_factor(-60) == alpha_factor(60)
    with pytest.raises(ValueError, match="within"):
        alpha_factor(120)
    with pytest.raises(ValueError, match="finite"):
        alpha_factor(np.nan)
    with pytest.raises(ValueError, match="formula"):
        alpha_factor(60, method="nonsense")


def test_peak_geoelectric_field():
    """Equations 1 and 2."""
    assert peak_geoelectric_field(1.0, 1.0) == 8.0
    assert peak_geoelectric_field(0.5, 0.5) == 2.0
    assert peak_geoelectric_field(1.0, 1.0, event="supplemental") == 12.0
    with pytest.raises(ValueError, match="benchmark"):
        peak_geoelectric_field(1.0, 1.0, event="halloween")


def test_tabulated_beta():
    assert tabulated_beta("SHIELD") == 1.0
    assert tabulated_beta("AP1") == 0.33
    assert tabulated_beta("AP1", event="supplemental") == 0.30
    assert tabulated_beta("PT1") == 1.17  # the only model above the reference
    with pytest.raises(KeyError, match="Table 3"):
        tabulated_beta("NOT_A_REGION")


# ------------------------------------------------------------------ invariants


def test_reference_model_beta_is_one(waveform):
    """The Quebec model is the definition of beta = 1, for any waveform."""
    with pytest.warns(UserWarning, match="not the scaled TPL-007 reference"):
        df = beta_factor(QUEBEC_REFERENCE_MODEL, *waveform, dt=10)
    assert df["beta"].iloc[0] == pytest.approx(1.0)


def test_beta_invariant_to_waveform_amplitude(waveform):
    """normalize='reference' is a ratio, so rescaling B cannot move it.

    This is the whole reason the ratio is the default: a mis-scaled waveform
    corrupts a literal E/8 in direct proportion, and leaves this untouched.
    """
    mag_x, mag_y = waveform
    site = bezpy.mt.get_1d_site("AP1")
    with pytest.warns(UserWarning):
        base = beta_factor(site, mag_x, mag_y, dt=10)["beta"].iloc[0]
        scaled = beta_factor(site, 3.7 * mag_x, 3.7 * mag_y, dt=10)["beta"].iloc[0]
    assert scaled == pytest.approx(base, rel=1e-12)


def test_absolute_normalization_needs_a_calibrated_waveform(waveform, calibrated_waveform):
    """The two normalizations agree only when Quebec really does peak at 8."""
    site = bezpy.mt.get_1d_site("AP1")
    ratio = beta_factor(site, *calibrated_waveform, dt=10, normalize="reference")
    absolute = beta_factor(site, *calibrated_waveform, dt=10, normalize="absolute")
    assert absolute["beta"].iloc[0] == pytest.approx(ratio["beta"].iloc[0], rel=1e-10)
    assert ratio["reference_peak_E"].iloc[0] == pytest.approx(BENCHMARK_PEAK_E, rel=1e-10)

    # ... and diverge by exactly the miscalibration when it does not.
    with pytest.warns(UserWarning, match="not the scaled TPL-007 reference"):
        off = beta_factor(site, *waveform, dt=10, normalize="absolute")
    factor = off["reference_peak_E"].iloc[0] / BENCHMARK_PEAK_E
    assert off["beta"].iloc[0] == pytest.approx(ratio["beta"].iloc[0] * factor, rel=1e-6)


def test_calibration_warning_is_silent_when_calibrated(calibrated_waveform):
    """No warning when the waveform reproduces the 8 V/km definition."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        beta_factor(bezpy.mt.get_1d_site("AP1"), *calibrated_waveform, dt=10)


def test_beta_validation(waveform):
    mag_x, mag_y = waveform
    site = bezpy.mt.get_1d_site("AP1")
    with pytest.raises(ValueError, match="1-D arrays"):
        beta_factor(site, mag_x[:, None], mag_y[:, None], dt=10)
    with pytest.raises(ValueError, match="normalize"):
        beta_factor(site, mag_x, mag_y, dt=10, normalize="nonsense")
    with pytest.raises(ValueError, match="at least one MT site"):
        beta_factor([], mag_x, mag_y, dt=10)


# ------------------------------------------------------------------- physics


def test_resistive_ground_gives_larger_beta(waveform):
    """A resistive half-space induces a larger surface E than a conductive one."""
    resistive = bezpy.mt.Site1d("resistive", thicknesses=[100e3], resistivities=[3000.0, 3000.0])
    conductive = bezpy.mt.Site1d("conductive", thicknesses=[100e3], resistivities=[3.0, 3.0])
    with pytest.warns(UserWarning):
        df = beta_factor([resistive, conductive], *waveform, dt=10)
    assert df.loc["resistive", "beta"] > df.loc["conductive", "beta"]


def test_beta_reproduces_the_published_table(waveform):
    """Drive the shipped USGS 1-D profiles through the full pipeline and
    compare against TPL-007-4 Table 3.

    Table 3 was computed with the reference storm's spectrum and this uses a
    synthetic stand-in, and the standard itself notes the factors "are
    dependent upon the frequency content of the reference storm" -- so exact
    agreement is not available without NERC's waveform file. What is
    available, and asserted here, is that the whole
    impedance -> convolution -> peak -> normalize path lands on the
    standard's own numbers to within ~0.1 in beta and reproduces their
    ordering. A sign error, a units error, or a broken normalization all
    destroy both. Measured across seeds: Spearman 0.90-0.99 (median 0.96),
    median absolute error ~0.11 (docs/validation.md item 1.3).
    """
    models = [m for m in BETA_TABLE.index if m in set(bezpy.mt.list_1d_sites())]
    assert len(models) >= 20, f"expected most of Table 3 to be shipped, got {models}"

    sites = [bezpy.mt.get_1d_site(m) for m in models]
    with pytest.warns(UserWarning):
        computed = beta_factor(sites, *waveform, dt=10)["beta"].values
    published = BETA_TABLE.loc[models, "beta_b"].values

    rank_correlation = pd.Series(computed).corr(pd.Series(published), method="spearman")
    assert rank_correlation > 0.85, f"rank correlation only {rank_correlation:.3f}"
    assert np.median(np.abs(computed - published)) < 0.15
    # The reference model anchors the scale: nothing should land wildly off.
    assert np.all(computed > 0.02) and np.all(computed < 2.0)


def test_beta_components_equal_for_an_isotropic_ground(waveform):
    """A 1-D ground is isotropic: driven by the same series in both
    components, beta_x and beta_y must come out identical.

    They differ under ``waveform`` only because its two components are
    independent draws -- for a 1-D site the component split reports the
    *source* polarization, not the ground. Reading beta_x/beta_y as a
    property of the Earth is only valid for a 3-D impedance.
    """
    site = bezpy.mt.get_1d_site("IP1")
    with pytest.warns(UserWarning):
        same = beta_factor(site, *_storm_like(same=True), dt=10)
        independent = beta_factor(site, *waveform, dt=10)
    assert same["beta_x"].iloc[0] == pytest.approx(same["beta_y"].iloc[0], rel=1e-12)
    assert independent["beta_x"].iloc[0] != pytest.approx(
        independent["beta_y"].iloc[0], rel=1e-3
    )
    # |E| >= max(|Ex|, |Ey|) pointwise, so the same holds for the peaks.
    for df in (same, independent):
        assert df["beta"].iloc[0] >= max(df["beta_x"].iloc[0], df["beta_y"].iloc[0]) - 1e-12


def test_beta_frame_is_a_self_contained_deliverable(waveform):
    """Every row carries the denominator it was normalized by."""
    models = ["AP1", "IP3", "CP2"]
    with pytest.warns(UserWarning):
        df = beta_factor([bezpy.mt.get_1d_site(m) for m in models], *waveform, dt=10)
    assert list(df.index) == models
    assert df.index.name == "site"
    assert set(df.columns) == {
        "latitude",
        "longitude",
        "peak_E",
        "beta",
        "beta_x",
        "beta_y",
        "reference_peak_E",
    }
    assert df["reference_peak_E"].nunique() == 1
    np.testing.assert_allclose(df["beta"], df["peak_E"] / df["reference_peak_E"])
