"""Tests for the ModEM gridded-impedance reader (bezpy.mt.modem).

The committed fixtures (tests/data/gridded_impedance_modem.dat and
tests/data/gridded_impedance_plain.dat) were written by the generators in
this file (``modem_text`` / ``plain_text``) and describe the same 4 sites --
a 2x2 grid at 0.25 degree spacing (lats 40.0/40.25, lons 255.0/255.25 in the
0-360 east convention of the gridded products) -- at 3 periods (10, 100,
1000 s). Values follow a fixed, binary-exact formula so the tests can assert
exact equality:

    site index s (0..3, grid order), component c (0..3 = xx, xy, yx, yy),
    period index p (ascending):
        Re(Z) = (s + 1) + 0.25*c + 0.0625*p
        Im(Z) = -(0.5*(s + 1) + 0.25*c + 0.0625*p)
        error = 0.5 + 0.125*c + 0.03125*p     (modem-style fixture only)

All coefficients are negative powers of two, so the decimal values written
to the files round-trip exactly through float parsing.
"""

import os

import numpy as np
import pytest

from bezpy.mt import SiteCollection, read_gridded_impedance

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEM = os.path.join(DATA_DIR, "gridded_impedance_modem.dat")
PLAIN = os.path.join(DATA_DIR, "gridded_impedance_plain.dat")

LATS = (40.0, 40.25)
LONS = (255.0, 255.25)  # 0-360 east convention, as in the gridded products
PERIODS = (10.0, 100.0, 1000.0)
COMPONENTS = ("ZXX", "ZXY", "ZYX", "ZYY")


def z_value(s, c, p):
    """Impedance for site s, component c, period index p (binary-exact)."""
    return complex((s + 1) + 0.25 * c + 0.0625 * p, -(0.5 * (s + 1) + 0.25 * c + 0.0625 * p))


def z_error(s, c, p):  # noqa: ARG001 - s kept for signature symmetry
    """Error bar for component c, period index p (binary-exact)."""
    return 0.5 + 0.125 * c + 0.03125 * p


def site_grid():
    """(code, lat, lon) for the 2x2 grid, in site-index order."""
    grid = []
    for lat in LATS:
        for lon in LONS:
            grid.append((f"S{len(grid) + 1:03d}", lat, lon))
    return grid


def _data_rows(periods, errors, lowercase=False, period_major=True):
    """Impedance data rows; period-major files list the periods descending."""
    rows = []
    period_order = list(reversed(range(len(periods))))
    if period_major:
        loops = [(p, s) for p in period_order for s in range(4)]
    else:
        loops = [(p, s) for s in range(4) for p in period_order]
    for p, s in loops:
        code, lat, lon = site_grid()[s]
        x, y = (lat - 40.125) * 111000.0, (lon - 255.125) * 85000.0
        for c, comp in enumerate(COMPONENTS):
            value = z_value(s, c, p)
            row = (
                f"{periods[p]:.6E}  {code}  {lat:.4f}  {lon:.4f}  "
                f"{x:.3f}  {y:.3f}  0.000  "
                f"{comp.lower() if lowercase else comp}  "
                f"{value.real:.6f}  {value.imag:.6f}"
            )
            if errors:
                row += f"  {z_error(s, c, p):.6f}"
            rows.append(row)
    return rows


def modem_text(periods=PERIODS, units="[mV/km]/[nT]", errors=True):
    """'#'-comment + '>'-metadata style ModEM file, with a second header
    block and TX/TY tipper rows after the impedance section."""
    lines = [
        "# ModEM impedance responses: synthetic bezpy test fixture",
        "# Written by tests/test_mt_modem.py:modem_text",
        "> Full_Impedance",
        "> exp(-i\\omega t)",
        f"> {units}",
        "> 0.00",
        "> 40.125 255.125",
        f"> {len(periods)} 4",
    ]
    lines += _data_rows(periods, errors=errors, period_major=True)
    # Second header block + tipper section: both must be ignored
    lines += [
        "# Synthetic tipper section (ignored by the impedance reader)",
        "> Full_Vertical_Components",
        "> exp(-i\\omega t)",
        "> []",
        "> 0.00",
        "> 40.125 255.125",
        f"> {len(periods)} 4",
    ]
    for code, lat, lon in site_grid():
        for comp in ("TX", "TY"):
            lines.append(
                f"{periods[0]:.6E}  {code}  {lat:.4f}  {lon:.4f}  "
                f"0.000  0.000  0.000  {comp}  0.010000  0.020000  0.100000"
            )
    return "\n".join(lines) + "\n"


def plain_text(periods=PERIODS):
    """Plain descriptive-text header (13 lines, no '#' or '>'), rows grouped
    by site, no error column, lowercase component names."""
    lines = [
        "Gridded impedance product: synthetic bezpy test fixture",
        "Written by tests/test_mt_modem.py:plain_text",
        "Survey: synthetic 2x2 grid at 0.25 degree spacing",
        "Origin: 40.125 deg latitude, 255.125 deg east longitude",
        "Periods: 3 values, 10 s to 1000 s",
        "Sign convention: exp(-i omega t)",
        "Units: practical, mV/km per nT",
        "Orientation: x = geographic north, y = geographic east",
        "Components: zxx zxy zyx zyy",
        "Errors: not included in this file",
        "Reference: tests for bezpy.mt.read_gridded_impedance",
        "Columns: period code lat lon x y z component real imag",
        "Data follow below this thirteen-line description",
    ]
    lines += _data_rows(periods, errors=False, lowercase=True, period_major=False)
    return "\n".join(lines) + "\n"


def check_collection(collection, periods=PERIODS):
    """Assert the 4-site grid parsed exactly per the value formula."""
    assert isinstance(collection, SiteCollection)
    assert len(collection.sites) == 4
    for s, (site, (code, lat, lon)) in enumerate(zip(collection.sites, site_grid())):
        assert site.name == code
        assert site.latitude == lat
        assert site.longitude == lon - 360.0  # converted to -180..180
        np.testing.assert_array_equal(site.periods, periods)
        assert site.Z.shape == (4, len(periods))
        assert site.Z.dtype == complex
        for c in range(4):
            for p in range(len(periods)):
                assert site.Z[c, p] == z_value(s, c, p)


def test_read_modem_style_fixture():
    collection = read_gridded_impedance(MODEM)
    check_collection(collection)
    for s, site in enumerate(collection.sites):
        # Errors pass through exactly as variances
        assert site.Z_var.shape == (4, 3)
        for c in range(4):
            for p in range(3):
                assert site.Z_var[c, p] == z_error(s, c, p) ** 2
        # Tipper rows and the second header block contributed nothing
        assert not np.any(np.isnan(site.Z))


def test_read_plain_header_fixture():
    collection = read_gridded_impedance(PLAIN)
    check_collection(collection)
    for site in collection.sites:
        assert site.Z_var is None


def test_units_mismatch_warns(tmp_path):
    path = tmp_path / "other_units.dat"
    path.write_text(modem_text(units="[V/m]/[T]"))
    with pytest.warns(UserWarning, match="units"):
        collection = read_gridded_impedance(path)
    # Values are still read as-is
    check_collection(collection)


def test_matching_units_do_not_warn(recwarn):
    read_gridded_impedance(MODEM)
    assert len(recwarn) == 0


def test_no_impedance_rows_raises(tmp_path):
    path = tmp_path / "tippers_only.dat"
    # Headers and tipper rows only: no impedance content
    text = "\n".join(
        line
        for line in modem_text().splitlines()
        if not (len(line.split()) > 7 and line.split()[7] in COMPONENTS)
    )
    path.write_text(text + "\n")
    with pytest.raises(ValueError, match="[Nn]o.*impedance"):
        read_gridded_impedance(path)


def test_missing_component_is_nan(tmp_path):
    path = tmp_path / "missing_component.dat"
    text = modem_text()
    # Drop site S001's ZXY row at 10 s
    lines = [
        line
        for line in text.splitlines()
        if not (line.startswith("1.000000E+01") and "S001" in line and " ZXY " in line)
    ]
    path.write_text("\n".join(lines) + "\n")
    site = read_gridded_impedance(path).sites[0]
    assert np.isnan(site.Z[1, 0])
    assert np.isnan(site.Z_var[1, 0])
    assert site.Z[0, 0] == z_value(0, 0, 0)


def test_partial_error_column(tmp_path):
    path = tmp_path / "partial_errors.dat"
    lines = modem_text().splitlines()
    # Strip the error column from site S001's ZXX row at 10 s
    for i, line in enumerate(lines):
        if line.startswith("1.000000E+01") and "S001" in line and " ZXX " in line:
            lines[i] = line.rsplit(None, 1)[0]
            break
    path.write_text("\n".join(lines) + "\n")
    site = read_gridded_impedance(path).sites[0]
    # Some rows have errors, so Z_var exists; the stripped entry is NaN
    assert site.Z_var is not None
    assert np.isnan(site.Z_var[0, 0])
    assert site.Z_var[1, 0] == z_error(0, 1, 0) ** 2


def test_collection_convolve_fft(tmp_path):
    """End-to-end: parsed sites drop straight into the FFT machinery."""
    # Site3d's cubic-spline interpolation needs more than 3 periods, so
    # generate a 5-period variant with the same formula
    path = tmp_path / "five_periods.dat"
    path.write_text(modem_text(periods=(10.0, 31.6, 100.0, 316.0, 1000.0)))
    collection = read_gridded_impedance(path)
    check_collection(collection, periods=(10.0, 31.6, 100.0, 316.0, 1000.0))

    rng = np.random.default_rng(0)
    ex, ey = collection.convolve_fft(rng.standard_normal(200), rng.standard_normal(200), dt=60)
    assert ex.shape == (4, 200)
    assert ey.shape == (4, 200)
    assert np.all(np.isfinite(ex))
    assert np.all(np.isfinite(ey))
