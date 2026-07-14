"""Screen MT transfer functions for hazard use.

A geoelectric-hazard estimate is only as trustworthy as the impedance behind
it: a transfer function with non-physical apparent resistivity, out-of-quadrant
phase, or a huge relative error produces a nonsense E field that a downstream
extreme-value fit will happily turn into a giant "hazard". Two independent
screens catch these:

* **Impedance screen** -- per-site physical plausibility of the off-diagonal
  (``xy``/``yx``) response: apparent resistivity within crustal/mantle bounds,
  smooth in log-period (real Earth response is smooth; noisy Z oscillates),
  phase in the first quadrant, and a bounded relative error.
* **Spatial screen** -- a hazard value (e.g. a 100-year |E| level) that is a
  large-decade outlier against a site's nearest neighbours is flagged, since
  the true field varies smoothly over the MT grid.

Each site gets a ``verdict`` in ``{"ok", "review", "bad"}``: ``bad`` fails a
hard physical check or is a spatial outlier and should be dropped; ``review``
trips a soft flag and is kept but worth a look; ``ok`` passes cleanly.
"""

__all__ = ["screen_impedance", "screen_sites"]

import numpy as np
import pandas as pd

from .utils import apparent_resistivity

# Off-diagonal rows of Z organised (xx, xy, yx, yy).
_OFFDIAG = (1, 2)
R_EARTH_KM = 6371.0


def _great_circle_km(lat, lon, lats, lons):
    lat, lon, lats, lons = map(np.radians, (lat, lon, lats, lons))
    d = (np.sin((lats - lat) / 2) ** 2
         + np.cos(lat) * np.cos(lats) * np.sin((lons - lon) / 2) ** 2)
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(d, 0, 1)))


def screen_impedance(site, rho_bounds=(3e-2, 3e5), max_rel_err=0.75,
                     max_roughness=0.6, frac_out_soft=0.2, frac_out_hard=0.5,
                     min_band=5):
    """Physical-plausibility metrics and a flag for one site's impedance.

    Uses the off-diagonal (``xy``/``yx``) apparent resistivity and phase. The
    phase is folded to ``(-90, 90)`` by :func:`apparent_resistivity`, so a
    physical (first-quadrant) response sits in ``(0, 90)`` -- values below zero
    are out of quadrant.

    :param site: a ``Site3d``/``Site1d`` with ``periods``, ``Z``, ``Z_var``.
    :param rho_bounds: (low, high) apparent resistivity in ohm-m considered
        physical (default 0.03 - 3e5, brine-saturated crust to resistive mantle).
    :param max_rel_err: median relative error ``|dZ|/|Z|`` above which the
        transfer function is a hard fail (default 0.75).
    :param max_roughness: max std of the second difference of ``log10(rho)`` vs
        ``log10(period)`` (log-log curvature) before the curve is "rough"
        (default 0.6 decade).
    :param frac_out_soft/frac_out_hard: fraction of samples out of the
        resistivity bounds or the phase quadrant for a soft / hard flag.
    :param min_band: fewest valid off-diagonal periods to screen at all.
    :return: dict of metrics plus ``imp_flag`` in ``{"ok", "review", "bad"}``.
    """
    periods = np.asarray(site.periods, dtype=float)
    Z = np.asarray(site.Z)
    Z_var = None if site.Z_var is None else np.asarray(site.Z_var, dtype=float)
    rho, _, phase, _ = apparent_resistivity(periods, Z, Z_var)

    valid = np.isfinite(Z[list(_OFFDIAG)]) & np.isfinite(rho[list(_OFFDIAG)])
    if Z_var is not None:
        valid &= np.isfinite(Z_var[list(_OFFDIAG)]) & (Z_var[list(_OFFDIAG)] > 0)
    n_band = int(valid.any(axis=0).sum())

    lo, hi = rho_bounds
    rho_out, phase_out, rel_errs, roughs, phase_roughs = [], [], [], [], []
    for r in _OFFDIAG:
        m = valid[_OFFDIAG.index(r)]
        if m.sum() < 3:
            continue
        rr = rho[r, m]
        rho_out.append(((rr < lo) | (rr > hi)).mean())
        ph = phase[r, m]
        phase_out.append((ph < -10.0).mean())      # out of first quadrant
        if Z_var is not None:
            rel_errs.append(np.median(np.sqrt(Z_var[r, m]) / np.abs(Z[r, m])))
        # log-log curvature of resistivity vs period (sorted by period)
        order = np.argsort(periods[m])
        lr = np.log10(np.clip(rr[order], 1e-6, None))
        if lr.size >= 3:
            roughs.append(np.std(np.diff(lr, 2)))
            phase_roughs.append(np.std(np.diff(ph[order], 2)))

    def _agg(x, default=np.nan):
        return float(np.max(x)) if len(x) else default

    frac_rho_out = _agg(rho_out, 1.0)
    frac_phase_out = _agg(phase_out, 1.0)
    rel_err_med = _agg(rel_errs, np.nan)
    roughness = _agg(roughs, np.nan)
    phase_rough = _agg(phase_roughs, np.nan)

    s_band = n_band < min_band
    s_rho_hard = frac_rho_out > frac_out_hard
    s_rho_soft = frac_rho_out > frac_out_soft
    s_phase_hard = frac_phase_out > frac_out_hard
    s_phase_soft = frac_phase_out > frac_out_soft
    s_rel_err = np.isfinite(rel_err_med) and rel_err_med > max_rel_err
    s_rough = np.isfinite(roughness) and roughness > max_roughness

    hard = s_band or s_rho_hard or s_phase_hard or s_rel_err
    soft = s_rho_soft or s_phase_soft or s_rough
    imp_flag = "bad" if hard else ("review" if soft else "ok")

    return {
        "n_band": n_band, "frac_rho_out": frac_rho_out,
        "frac_phase_out": frac_phase_out, "rel_err_med": rel_err_med,
        "rho_roughness": roughness, "phase_roughness": phase_rough,
        "imp_flag": imp_flag,
    }


def screen_sites(sites, values=None, k=8, loo_decades=0.74, min_neighbors=4,
                 **imp_kwargs):
    """Screen a collection of MT sites; return a per-site verdict table.

    :param sites: sequence of ``Site3d``/``Site1d`` (need ``name``,
        ``latitude``, ``longitude``, and impedance).
    :param values: optional per-site hazard proxy (e.g. 100-year |E|) for the
        spatial leave-one-out screen; same length/order as ``sites``. Omit to
        run the impedance screen only.
    :param k: nearest neighbours for the spatial screen (default 8).
    :param loo_decades: a site is a spatial outlier if ``|log10(value) -
        median(log10(neighbour values))|`` exceeds this (default 0.74 decade,
        ~a factor of 5.5).
    :param min_neighbors: fewest finite neighbours to run the spatial screen
        for a site.
    :param imp_kwargs: forwarded to :func:`screen_impedance`.
    :return: DataFrame indexed by site name with the impedance metrics,
        ``loo_resid``/``spatial_flag`` (if ``values`` given), and a combined
        ``verdict`` in ``{"ok", "review", "bad"}``.
    """
    rows = []
    lat = np.array([s.latitude for s in sites], dtype=float)
    lon = np.array([s.longitude for s in sites], dtype=float)
    for s in sites:
        m = screen_impedance(s, **imp_kwargs)
        m["code"] = s.name
        m["latitude"] = s.latitude
        m["longitude"] = s.longitude
        rows.append(m)
    df = pd.DataFrame(rows).set_index("code")

    df["loo_resid"] = np.nan
    df["spatial_flag"] = "ok"
    if values is not None:
        vals = np.asarray(values, dtype=float)
        logv = np.log10(np.where(vals > 0, vals, np.nan))
        for i in range(len(sites)):
            if not np.isfinite(logv[i]):
                continue
            d = _great_circle_km(lat[i], lon[i], lat, lon)
            d[i] = np.inf
            order = np.argsort(d)
            neigh = order[np.isfinite(logv[order])][:k]
            if len(neigh) < min_neighbors:
                continue
            resid = logv[i] - np.median(logv[neigh])
            df.iloc[i, df.columns.get_loc("loo_resid")] = resid
            if abs(resid) > loo_decades:
                df.iloc[i, df.columns.get_loc("spatial_flag")] = "bad"

    def verdict(row):
        if row["imp_flag"] == "bad" or row["spatial_flag"] == "bad":
            return "bad"
        if row["imp_flag"] == "review":
            return "review"
        return "ok"

    df["verdict"] = df.apply(verdict, axis=1)
    return df.reset_index()
