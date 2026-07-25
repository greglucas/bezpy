"""Tests for the legacy MT readers (EMTF Z-files and EDI) in bezpy.mt.legacy.

The fixtures use identity channel orientations and zero file rotations, so
the values written in the files pass straight through to the Site3d and can
be asserted exactly.
"""

import os

import numpy as np
import pytest

from bezpy.mt import Site3d, read_edi, read_zfile
from bezpy.mt.legacy import EDIFile, ZFile

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ZSS = os.path.join(DATA_DIR, "sample.zss")
EDI = os.path.join(DATA_DIR, "sample.edi")


def test_read_zfile_site():
    site = read_zfile(ZSS)
    assert isinstance(site, Site3d)
    assert site.name == "TST01"
    assert site.latitude == 40.0
    assert site.longitude == -105.0
    np.testing.assert_array_equal(site.periods, [10.0, 31.623, 100.0, 316.23, 1000.0])

    # Values from the fixture: identity orientations pass straight through
    np.testing.assert_allclose(site.Z[:, 0], [1 + 0.5j, 2 - 0.5j, -2 + 0.1j, -1 + 0.2j])
    np.testing.assert_allclose(
        site.Z[:, 2], [0.25 + 0.125j, 0.5 - 0.125j, -0.5 + 0.025j, -0.25 + 0.05j]
    )
    # Var(Z) = sigma_e(E channel) * sigma_s(H channel); identity sigma_s
    np.testing.assert_allclose(site.Z_var[:, 0], [0.09, 0.09, 0.16, 0.16])
    # calc_resisitivity ran (resistivity available for plotting)
    assert site.resistivity is not None


def test_zfile_rotation_invariants():
    zfile = ZFile(ZSS)
    z0, err0 = zfile.impedance(angle=0.0)
    z45, err45 = zfile.impedance(angle=45.0)
    # Rotation preserves the tensor invariants (determinant, sum of squares)
    np.testing.assert_allclose(np.linalg.det(z45), np.linalg.det(z0), rtol=1e-10)
    np.testing.assert_allclose(
        np.sum(np.abs(z45) ** 2, axis=(1, 2)), np.sum(np.abs(z0) ** 2, axis=(1, 2)), rtol=1e-10
    )


def test_zfile_tippers():
    zfile = ZFile(ZSS)
    tipper, error = zfile.tippers()
    np.testing.assert_allclose(tipper[0], [0.1, 0.2])
    np.testing.assert_allclose(error[0], [0.2, 0.2])  # sqrt(0.04 * 1)


def test_read_edi_site():
    site = read_edi(EDI)
    assert site.name == "TST02"
    assert site.latitude == 40.5
    assert site.longitude == -105.25
    # ORDER=DEC frequencies -> ascending periods
    np.testing.assert_allclose(site.periods, [10.0, 100.0])
    np.testing.assert_allclose(site.Z[:, 0], [1 + 0.5j, 2 - 0.5j, -2 + 0.1j, -1 + 0.2j])
    np.testing.assert_allclose(site.Z[:, 1], [0.5 + 0.25j, 1 - 0.25j, -1 + 0.05j, -0.5 + 0.1j])
    np.testing.assert_allclose(site.Z_var[0], [0.09, 0.09])
    np.testing.assert_allclose(site.Z_var[2], [0.16, 0.16])


def test_read_edi_rotation_invariant():
    z0 = read_edi(EDI, angle=0.0)
    z30 = read_edi(EDI, angle=30.0)
    det = lambda s: s.Z[0] * s.Z[3] - s.Z[1] * s.Z[2]  # noqa: E731
    np.testing.assert_allclose(det(z30), det(z0), rtol=1e-10)
    assert not np.allclose(z30.Z, z0.Z)


def test_edi_variances_rotate_as_independent_variances():
    edi = EDIFile.__new__(EDIFile)
    edi.Z = np.zeros((1, 2, 2), dtype=complex)
    edi.Zvar = np.array([[[1.0, 4.0], [9.0, 16.0]]])
    _, error = edi.impedance(angle=45.0)
    # At 45 degrees every squared rotation coefficient is 1/2, so every
    # output element receives one quarter of the sum of all input variances.
    np.testing.assert_allclose(error**2, np.full((1, 2, 2), 7.5))


def test_edi_negative_fourier_sign_is_conjugated(tmp_path):
    path = tmp_path / "negative_sign.edi"
    text = (
        open(EDI)
        .read()
        .replace(
            r"SIGNCONVENTION=exp(+ i\omega t)",
            r"SIGNCONVENTION=exp(- i\omega t)",
        )
    )
    path.write_text(text)
    plus = read_edi(EDI)
    minus = read_edi(path)
    np.testing.assert_array_equal(minus.Z, np.conj(plus.Z))


def test_legacy_sites_work_in_the_pipeline():
    """The whole point: legacy sites drop into the standard machinery."""
    site = read_zfile(ZSS)
    freqs = np.fft.rfftfreq(256, d=60.0)
    Z = site.calcZ(freqs)
    assert Z.shape == (4, len(freqs))
    ex, ey = site.convolve_fft(np.random.default_rng(0).standard_normal(200), np.zeros(200), dt=60)
    assert np.all(np.isfinite(ex))


HZ_ONLY_ZSS = """\
 TRANSFER FUNCTIONS IN MEASUREMENT COORDINATES
 ********** WITH FULL ERROR COVARIANCE **********
 (test fixture: vertical-field-only site)
station    :TST03
coordinate    41.000  -104.000 declination    0.00
number of channels   3   number of frequencies   1
 orientations and tilts of each channel
    1     0.00     0.00 TST  Hx
    2    90.00     0.00 TST  Hy
    3     0.00     0.00 TST  Hz

period :      10.000    decimation level   1    freq. band from   1 to   5
number of data point   1000 sampling freq.   1.000 Hz
 Transfer Functions
  0.1000E+00  0.0000E+00  0.2000E+00  0.0000E+00
 Inverse Coherent Signal Power Matrix
  0.1000E+01  0.0000E+00
  0.0000E+00  0.0000E+00  0.1000E+01  0.0000E+00
 Residual Covariance
  0.4000E-01  0.0000E+00
"""


def test_missing_channels_raise(tmp_path):
    path = tmp_path / "hz_only.zss"
    path.write_text(HZ_ONLY_ZSS)

    zfile = ZFile(path)
    with pytest.raises(ValueError, match="electric fields"):
        zfile.impedance()
    tipper, error = zfile.tippers()
    np.testing.assert_allclose(tipper[0], [0.1, 0.2])
    np.testing.assert_allclose(error[0], [0.2, 0.2])  # sqrt(0.04 * 1)
