"""NERC TPL-007 hazard quantities computed from measured Earth conductivity.

TPL-007 sets the peak geoelectric field a GMD Vulnerability Assessment must
plan against as

    E_peak = 8 x alpha x beta   (V/km, benchmark GMD event)
    E_peak = 12 x alpha x beta  (V/km, supplemental GMD event)

where ``alpha`` scales for geomagnetic latitude and ``beta`` scales for the
local Earth conductivity structure. The standard offers ``beta`` two ways:
look it up from a table of physiographic regions, or -- the path this module
implements -- compute it from a "technically justified" Earth model, i.e.

    beta_b = E / 8     beta_s = E / 12                     (TPL-007-4 eq. 4, 5)

with ``E`` the peak geoelectric field that model produces when driven by the
reference geomagnetic field waveform.

The tabulated route assumes a 1-D layered Earth averaged over a large
physiographic region. Real conductivity is 3-D, and the resulting ground
response spans a considerably wider range than the 1-D regions imply, so a
utility whose ``beta`` is set by a table is planning against a number that
may be conservative or optimistic by a large factor -- with no way to tell
which. Driving a measured 3-D magnetotelluric impedance through the same
reference waveform replaces that table lookup with a site-specific,
defensible number, in the units the standard is written in.

Two practical notes on using :func:`beta_factor` in a compliance filing:

* **You supply the waveform.** ``beta`` is defined against the reference
  geomagnetic field time series (the March 13-14 1989 Ottawa record scaled to
  60 degrees geomagnetic latitude, 10 s sampling), which NERC distributes as
  a data file with TPL-007. bezpy does not redistribute it.
* **The result carries its own calibration check.** The reference waveform is
  scaled precisely so that the reference Quebec Earth model yields 8 V/km,
  which makes ``beta`` a *ratio* of two peak fields computed from the same
  waveform. That is how this module computes it by default, and the
  denominator is reported in every row: if you fed in the correct benchmark
  waveform, ``reference_peak_E`` comes back as 8.0 V/km. If it does not, the
  waveform is not the benchmark one -- an error that a literal ``E / 8``
  would have silently absorbed into ``beta``.

Reference: NERC Reliability Standard TPL-007-4, "Transmission System Planned
Performance for Geomagnetic Disturbance Events", Attachment 1.
"""

__all__ = [
    "ALPHA_TABLE",
    "BETA_TABLE",
    "BENCHMARK_PEAK_E",
    "SUPPLEMENTAL_PEAK_E",
    "QUEBEC_REFERENCE_MODEL",
    "alpha_factor",
    "beta_factor",
    "peak_geoelectric_field",
    "tabulated_beta",
]

import warnings

import numpy as np
import pandas as pd

from .mt.site import Site1d, SiteCollection

#: Reference peak geoelectric field for the benchmark GMD event, V/km.
BENCHMARK_PEAK_E = 8.0

#: Reference peak geoelectric field for the supplemental GMD event, V/km.
SUPPLEMENTAL_PEAK_E = 12.0

_REFERENCE_PEAK_E = {"benchmark": BENCHMARK_PEAK_E, "supplemental": SUPPLEMENTAL_PEAK_E}

#: The reference Quebec Earth model, TPL-007-4 Attachment 1 Table 4. The
#: benchmark waveform is scaled so that this model produces a peak
#: geoelectric field of exactly 8 V/km, which makes it the denominator that
#: defines ``beta``.
QUEBEC_REFERENCE_MODEL = Site1d(
    "TPL-007 reference (Quebec)",
    thicknesses=[15e3, 10e3, 125e3, 200e3],
    resistivities=[20000.0, 200.0, 1000.0, 100.0, 3.0],
)

# Relative tolerance on the reference peak before warning that the supplied
# waveform is probably not the scaled benchmark record.
_REFERENCE_RTOL = 0.02

#: Geomagnetic field scaling factors, TPL-007-4 Attachment 1 Table 2. The
#: tabulated alternative to :func:`alpha_factor`'s eq. 3; see that function
#: for where the two disagree.
ALPHA_TABLE = pd.DataFrame(
    {
        "geomagnetic_latitude": [40.0, 45.0, 50.0, 54.0, 56.0, 57.0, 58.0, 59.0, 60.0],
        "alpha": [0.10, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
)

#: Earth conductivity scaling factors, TPL-007-4 Attachment 1 Table 3 -- the
#: published ``beta`` for each USGS/NRCan 1-D physiographic Earth model, for
#: the benchmark (``beta_b``) and supplemental (``beta_s``) events. The
#: standard notes these "are dependent upon the frequency content of the
#: reference storm", which is why the two columns differ.
BETA_TABLE = pd.DataFrame(
    [
        ("AK1A", 0.56, 0.51),
        ("AK1B", 0.56, 0.51),
        ("AP1", 0.33, 0.30),
        ("AP2", 0.82, 0.78),
        ("BR1", 0.22, 0.22),
        ("CL1", 0.76, 0.73),
        ("CO1", 0.27, 0.25),
        ("CP1", 0.81, 0.77),
        ("CP2", 0.95, 0.86),
        ("FL1", 0.76, 0.73),
        ("CS1", 0.41, 0.37),
        ("IP1", 0.94, 0.90),
        ("IP2", 0.28, 0.25),
        ("IP3", 0.93, 0.90),
        ("IP4", 0.41, 0.35),
        ("NE1", 0.81, 0.77),
        ("PB1", 0.62, 0.55),
        ("PB2", 0.46, 0.39),
        ("PT1", 1.17, 1.19),
        ("SL1", 0.53, 0.49),
        ("SU1", 0.93, 0.90),
        ("BOU", 0.28, 0.24),
        ("FBK", 0.56, 0.56),
        ("PRU", 0.21, 0.22),
        ("BC", 0.67, 0.62),
        ("PRAIRIES", 0.96, 0.88),
        ("SHIELD", 1.0, 1.0),
        ("ATLANTIC", 0.79, 0.76),
    ],
    columns=["model", "beta_b", "beta_s"],
).set_index("model")


def alpha_factor(geomagnetic_latitude, method="formula"):
    """Geomagnetic latitude scaling factor (TPL-007-4 eq. 3 / Table 2).

    The standard gives two routes and they are **not** identical: the
    empirical expression ``alpha = 0.001 * exp(0.115 * L)`` clipped to
    ``[0.1, 1]``, and the coarse tabulation in Table 2. They differ by up to
    0.026 (at 45 degrees the formula gives 0.177 against a tabulated 0.2, at
    56 degrees 0.626 against 0.6). Either is permitted; which you used is
    worth stating in a filing, because it moves the planned-against field by
    a few percent.

    :param geomagnetic_latitude: **geomagnetic** (not geographic) latitude in
        degrees. The magnitude is used, so southern-hemisphere latitudes may
        be passed as negative.
    :param method: ``"formula"`` (default, eq. 3) or ``"table"`` (Table 2,
        which steps at the tabulated latitudes -- the standard's own
        conservative reading is to take the factor for the next latitude at
        or above the site, so that is what this does)
    :return: the scaling factor, same shape as the input
    """
    latitude = np.abs(np.asarray(geomagnetic_latitude, dtype=float))
    if not np.isfinite(latitude).all():
        raise ValueError("geomagnetic_latitude must be finite")
    if np.any(latitude > 90):
        raise ValueError("geomagnetic_latitude must lie within [-90, 90] degrees")
    if method == "formula":
        return np.clip(0.001 * np.exp(0.115 * latitude), 0.1, 1.0)
    if method == "table":
        edges = np.asarray(ALPHA_TABLE["geomagnetic_latitude"], dtype=float)
        values = np.asarray(ALPHA_TABLE["alpha"], dtype=float)
        return values[np.searchsorted(edges, latitude, side="left").clip(0, len(values) - 1)]
    raise ValueError(f"method must be 'formula' or 'table'; got {method!r}")


def tabulated_beta(model, event="benchmark"):
    """The published ``beta`` for a named Earth model (TPL-007-4 Table 3).

    This is the number a utility uses when it does *not* have a
    technically-justified Earth model of its own -- the 1-D physiographic
    region average. Compare it against :func:`beta_factor` on measured
    impedances from the same footprint: the gap between the two is the
    difference between the hazard the region-average implies and the hazard
    the ground actually presents.

    :param model: Earth model name, e.g. ``"AP1"``, ``"IP3"``, ``"SHIELD"``
    :param event: ``"benchmark"`` (default) or ``"supplemental"``
    :return: the tabulated scaling factor
    """
    column = "beta_b" if event == "benchmark" else "beta_s"
    _reference_amplitude(event)  # validates the event name
    try:
        return float(BETA_TABLE.loc[model, column])
    except KeyError:
        raise KeyError(
            f"{model!r} is not in TPL-007-4 Table 3; available models are "
            f"{sorted(BETA_TABLE.index)}"
        ) from None


def peak_geoelectric_field(alpha, beta, event="benchmark"):
    """Peak geoelectric field to plan against (TPL-007-4 eq. 1, 2).

    ``E_peak = 8 * alpha * beta`` for the benchmark event, ``12 * alpha *
    beta`` for the supplemental event.

    :param alpha: geomagnetic latitude factor, see :func:`alpha_factor`
    :param beta: Earth conductivity factor, see :func:`beta_factor`
    :param event: ``"benchmark"`` (default) or ``"supplemental"``
    :return: peak geoelectric field in V/km
    """
    return (
        _reference_amplitude(event) * np.asarray(alpha, dtype=float) * np.asarray(beta, dtype=float)
    )


def _reference_amplitude(event):
    """The 8 or 12 V/km reference amplitude for an event type."""
    try:
        return _REFERENCE_PEAK_E[event]
    except KeyError:
        raise ValueError(f"event must be 'benchmark' or 'supplemental'; got {event!r}") from None


def _peak_magnitude(Ex, Ey):
    """Peak of |E| over time, and of each component, in the input units.

    ``Ex``/``Ey`` are ``(nsites, ntimes)``. NaNs are ignored so a site with a
    partially masked series still reports the peak of what it does have.
    """
    # pylint: disable=invalid-name
    with warnings.catch_warnings():
        # An all-NaN site is a real possibility (no valid impedance); let it
        # come back as NaN rather than raising on the empty slice.
        warnings.simplefilter("ignore", RuntimeWarning)
        peak = np.nanmax(np.hypot(Ex, Ey), axis=-1)
        peak_x = np.nanmax(np.abs(Ex), axis=-1)
        peak_y = np.nanmax(np.abs(Ey), axis=-1)
    return peak, peak_x, peak_y


def beta_factor(
    sites,
    mag_x,
    mag_y,
    *,
    dt=10,
    event="benchmark",
    normalize="reference",
    extrapolate="asymptotic",
):
    """Earth conductivity scaling factor from a measured impedance.

    Drives every site's impedance with the reference geomagnetic field
    waveform as a spatially uniform plane wave -- the plane wave method the
    standard points to -- and normalizes the resulting peak geoelectric field
    by the reference Quebec model's peak from the same waveform.

    :param sites: an MT site, a sequence of sites, or a ``SiteCollection``.
        ``Site3d`` gives the full 3-D tensor response; ``Site1d`` gives the
        layered-Earth response (use it to reproduce a tabulated ``beta``).
    :param mag_x: northward magnetic field in nT, shape ``(ntimes,)`` -- the
        reference geomagnetic field waveform distributed with TPL-007
    :param mag_y: eastward magnetic field in nT, same shape
    :param dt: waveform sample interval in seconds (default 10, the sampling
        rate of the NERC benchmark waveform)
    :param event: ``"benchmark"`` (default) or ``"supplemental"``. Which one
        applies is determined by *which waveform you supply*; this argument
        only sets the nominal reference amplitude used for the calibration
        check and for ``normalize="absolute"``.
    :param normalize: ``"reference"`` (default) divides by the Quebec model's
        peak field computed from the same waveform, which is exact by the
        definition of the reference waveform and is insensitive to any
        overall rescaling of it. ``"absolute"`` divides by the nominal 8 (or
        12) V/km, the literal reading of eq. 4 -- identical when the waveform
        is scaled correctly, and wrong in proportion to the error when it is
        not.
    :param extrapolate: impedance band-edge handling (see ``Site3d.calcZ``).
        The default ``"asymptotic"`` continues the impedance past the
        measured band instead of zeroing it; ``"zeros"`` band-stops, which
        discards real out-of-band signal and biases ``beta`` low.
    :return: ``pandas.DataFrame`` indexed by site name with columns
        ``latitude``, ``longitude``, ``peak_E`` (V/km), ``beta``, ``beta_x``,
        ``beta_y``, and ``reference_peak_E`` (V/km, the denominator -- carried
        on every row so an exported table stands on its own).

        ``beta`` is built from the peak of ``|E|``; ``beta_x``/``beta_y``
        from the peaks of the individual components, for a planner applying
        the field directionally. For a 1-D model driven by a broadband
        waveform these are all close; for a 3-D impedance with strong
        anisotropy they are not, and the difference is the point.
    """
    # pylint: disable=invalid-name,too-many-locals
    if normalize not in ("reference", "absolute"):
        raise ValueError(f"normalize must be 'reference' or 'absolute'; got {normalize!r}")
    nominal = _reference_amplitude(event)

    mag_x = np.asarray(mag_x, dtype=float)
    mag_y = np.asarray(mag_y, dtype=float)
    if mag_x.ndim != 1 or mag_x.shape != mag_y.shape:
        raise ValueError(
            "mag_x and mag_y must be matching 1-D arrays -- the reference "
            "waveform is a single plane wave applied uniformly to every site"
        )

    if hasattr(sites, "sites"):  # a SiteCollection
        collection = sites
        site_list = list(sites.sites)
    else:
        site_list = [sites] if hasattr(sites, "calcZ") else list(sites)
        if not site_list:
            raise ValueError("at least one MT site is required")
        collection = SiteCollection(site_list)

    Ex, Ey = collection.convolve_fft(mag_x, mag_y, dt=dt, extrapolate=extrapolate)
    peak, peak_x, peak_y = _peak_magnitude(Ex, Ey)

    ref_Ex, ref_Ey = QUEBEC_REFERENCE_MODEL.convolve_fft(mag_x, mag_y, dt=dt)
    reference_peak = np.nanmax(np.hypot(ref_Ex, ref_Ey))

    # mV/km -> V/km
    peak, peak_x, peak_y = peak / 1e3, peak_x / 1e3, peak_y / 1e3
    reference_peak = reference_peak / 1e3

    if not np.isfinite(reference_peak) or reference_peak <= 0:
        raise ValueError(
            "the reference Quebec model produced no finite peak field from "
            "this waveform; check mag_x/mag_y and dt"
        )
    if abs(reference_peak - nominal) > _REFERENCE_RTOL * nominal:
        warnings.warn(
            f"the reference Quebec Earth model yields {reference_peak:.3f} V/km from "
            f"this waveform, not the {nominal:.1f} V/km the {event} waveform is "
            f"defined to produce. This is not the scaled TPL-007 reference waveform "
            f"(or dt={dt} s is wrong). beta is still the correct ratio of peak fields, "
            f"but it is not the standard's beta unless the waveform matches.",
            stacklevel=2,
        )

    denominator = reference_peak if normalize == "reference" else nominal

    return pd.DataFrame(
        {
            "latitude": [site.latitude for site in site_list],
            "longitude": [site.longitude for site in site_list],
            "peak_E": peak,
            "beta": peak / denominator,
            "beta_x": peak_x / denominator,
            "beta_y": peak_y / denominator,
            "reference_peak_E": reference_peak,
        },
        index=pd.Index([site.name for site in site_list], name="site"),
    )
