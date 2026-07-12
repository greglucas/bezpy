"""Tests for bezpy.mt.impulse (DTIR)."""
import numpy as np
import pytest

from bezpy.mt import Site1d, Site3d
from bezpy.mt.impulse import DTIR


@pytest.fixture
def banded_z():
    """Periods and a physical (half-space-like) impedance sampled on them."""
    periods = np.logspace(1.5, 4, 25)
    Z = Site1d("hs", [1e9], [100.0]).calcZ(1.0 / periods)
    return periods, Z


def test_dtir_accepts_periods_and_z(banded_z):
    """The documented keyword path works without a Site3d (B3)."""
    periods, Z = banded_z
    # nmax spans the longest period; small decay so the (underdetermined)
    # data-space fit reproduces the input band closely
    dtir = DTIR(dt=60, nmin=-5, nmax=300, decay_factor=1e-8)
    ns, zn = dtir.calculate(periods=periods, Z=Z)
    assert zn.shape == (4, len(ns))
    assert np.all(np.isfinite(zn))
    # The recovered transfer function reproduces the input band
    zf = dtir.get_zf(periods)
    mid = slice(5, -5)  # away from the band edges
    rel = np.abs(zf[1, mid] - Z[1, mid]) / np.abs(Z[1, mid])
    assert np.median(rel) < 0.05


def test_dtir_site_without_variance(banded_z):
    """A Site3d with Z_var=None must not crash (B3)."""
    periods, Z = banded_z
    site = Site3d("novar", latitude=40.0, longitude=-105.0,
                  periods=periods, Z=Z)
    dtir = DTIR(dt=60, nmin=-5, nmax=60)
    ns, zn = dtir.calculate(site=site)
    assert np.all(np.isfinite(zn))


def test_dtir_requires_input():
    with pytest.raises(ValueError, match="periods and Z or a Site3d"):
        DTIR().calculate()
