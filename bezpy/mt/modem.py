"""Reader for impedance data files in the ModEM format.

ModEM data/response files -- including the gridded CONUS impedance products
derived from them (e.g. ``USA_impedance_gridded_45per_0.25x0.25-SP2.dat``) --
interleave one row per (period, site, component):

    period(s)  site_code  latitude  longitude  x(m)  y(m)  z(m)  component  real  imag  [error]

with ``component`` one of ``ZXX/ZXY/ZYX/ZYY`` for impedances and ``TX/TY``
for tippers. Header layouts are inconsistent across product versions: some
files carry ``#`` comment lines followed by ``>`` metadata lines (units, sign
convention, period/site counts), others 10-13 lines of plain descriptive
text, and a second header block can appear mid-file before a tipper section.
The parser here is therefore content-driven: it recognizes impedance data
rows by their shape and skips everything else, rather than assuming a fixed
header length.
"""

__all__ = ["read_gridded_impedance"]

import re
import warnings

import numpy as np

from .site import Site3d, SiteCollection

# Impedance component -> row of the (4, nperiods) bezpy impedance array
_COMPONENTS = {"ZXX": 0, "ZXY": 1, "ZYX": 2, "ZYY": 3}
# Units token on ">" metadata lines, e.g. [mV/km]/[nT] or [V/m]/[T]
_UNITS_RE = re.compile(r"\[[^\[\]]*\]/\[[^\[\]]*\]")
_EXPECTED_UNITS = "[mV/km]/[nT]"


def _parse_impedance_row(fields):
    """Parse whitespace-split fields as a ModEM impedance data row.

    A line is an impedance row iff it has at least 10 fields, the first
    field is a number (the period), and field 7 is an impedance component.
    Returns ``(period, lat, lon, component_index, value, error)`` or ``None``
    if the line is anything else (header text, comments, tipper rows, ...).
    The error is ``None`` when the row has no (parseable) error column.
    """
    if len(fields) < 10:
        return None
    component = fields[7].upper()
    if component not in _COMPONENTS:
        return None
    try:
        period = float(fields[0])
        lat = float(fields[2])
        lon = float(fields[3])
        value = complex(float(fields[8]), float(fields[9]))
    except ValueError:
        return None
    error = None
    if len(fields) >= 11:
        try:
            error = float(fields[10])
        except ValueError:
            error = None
    return period, lat, lon, _COMPONENTS[component], value, error


def read_gridded_impedance(filename):
    """Read a ModEM-format impedance data file into a :class:`SiteCollection`.

    Covers ModEM data/response files and the gridded CONUS impedance
    products distributed in that format (e.g. Anna Kelbert's
    ``USA_impedance_gridded_45per_0.25x0.25-SP2.dat``). See the module
    docstring for the row layout. Parsing is content-driven: any line that
    is not an impedance data row (comments, ``>`` metadata, plain-text
    headers, tipper rows, mid-file header blocks) is skipped, so files with
    any of the header conventions in the wild parse identically.

    Impedance values are read as-is and assumed to be in practical units,
    (mV/km)/nT -- the standard ``> [mV/km]/[nT]`` declaration. If a ``>``
    metadata line declares different units, a warning is emitted and the
    values are still read unchanged. No sign-convention conversion is
    applied. The gridded products use 0-360 east longitude; longitudes
    greater than 180 are converted to bezpy's -180..180 convention.

    Each site becomes a :class:`Site3d` with periods sorted ascending, ``Z``
    shaped ``(4, nperiods)`` complex in ``(xx, xy, yx, yy)`` order (missing
    components are NaN), and ``Z_var = error**2`` when the file carries an
    error column (``None`` when no row in the file has one; NaN for
    individual rows that lack it).

    :param filename: path to the ModEM data file
    :return: :class:`SiteCollection` of the sites in first-seen file order,
        usable directly in :func:`bezpy.calc_geoelectric_field` or via
        :meth:`SiteCollection.convolve_fft`
    :raises ValueError: if the file contains no impedance data rows

    Example
    -------
    >>> collection = bezpy.mt.read_gridded_impedance(
    ...     "USA_impedance_gridded_45per_0.25x0.25-SP2.dat")  # doctest: +SKIP
    >>> ex, ey = collection.convolve_fft(mag_x, mag_y, dt=60)  # doctest: +SKIP
    """
    # site code -> [latitude, longitude, list of (period, comp, value, error)]
    sites = {}
    file_has_errors = False

    with open(filename) as f:
        for line in f:
            fields = line.split()
            row = _parse_impedance_row(fields)
            if row is None:
                # Non-data line: check ">" metadata lines for a units token
                stripped = line.lstrip()
                if stripped.startswith(">"):
                    match = _UNITS_RE.search(stripped)
                    if match is not None and match.group(0) != _EXPECTED_UNITS:
                        warnings.warn(
                            f"File declares units {match.group(0)!r} instead of "
                            f"the expected {_EXPECTED_UNITS!r}; impedance values "
                            "are read as-is without conversion.",
                            stacklevel=2,
                        )
                continue

            period, lat, lon, comp, value, error = row
            if error is not None:
                file_has_errors = True
            entry = sites.setdefault(fields[1], [lat, lon, []])
            entry[2].append((period, comp, value, error))

    if not sites:
        raise ValueError(f"No ModEM impedance data rows found in {filename}")

    collection = []
    for code, (lat, lon, rows) in sites.items():
        periods = np.array(sorted({period for period, *_ in rows}))
        index = {period: i for i, period in enumerate(periods)}

        Z = np.full((4, periods.size), np.nan + np.nan * 1j, dtype=complex)
        Z_var = np.full((4, periods.size), np.nan)
        for period, comp, value, error in rows:
            Z[comp, index[period]] = value
            if error is not None and np.isfinite(error):
                Z_var[comp, index[period]] = error**2

        site = Site3d(code)
        site.latitude = lat
        # 0-360 east longitude -> bezpy's -180..180 convention
        site.longitude = lon - 360.0 if lon > 180.0 else lon
        site.periods = periods
        site.Z = Z
        site.Z_var = Z_var if file_has_errors else None
        with np.errstate(invalid="ignore", divide="ignore"):
            site.calc_resistivity()
        collection.append(site)

    return SiteCollection(collection)
