"""Storm and event cataloging.

Two complementary primitives:

* :func:`find_storms` -- the index-based catalog used by the published
  studies: threshold Dst / Kp, expand each trigger by a window, merge
  overlaps. Still the right tool for case-study selection ("give me the
  big storms of this solar cycle").
* :func:`find_events` -- threshold exceedance with runs declustering on
  *any* series (|dB/dt|, |E|, line voltage, ...). This is the primitive
  that turns continuous-scan output into an event catalog and feeds
  peaks-over-threshold statistics, without assuming a global index
  responded to a local event.

Both return :class:`pandas.DataFrame` catalogs with one row per event.
"""

__all__ = ["find_events", "find_storms"]

import numpy as np
import pandas as pd


def _to_series(data, name=None):
    """Accept pandas Series or xarray DataArray with a time coordinate."""
    if hasattr(data, "to_pandas") and not isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.to_pandas()  # xarray DataArray -> Series
    if isinstance(data, pd.DataFrame):
        if name is None or name not in data:
            raise ValueError(f"column {name!r} not found in DataFrame")
        data = data[name]
    if not isinstance(data, pd.Series):
        raise TypeError("expected a pandas Series or 1-D xarray DataArray")
    return data.sort_index()


def _cluster(times, gap):
    """Group sorted timestamps into clusters separated by more than ``gap``.

    :return: list of (first, last) timestamps per cluster
    """
    if len(times) == 0:
        return []
    times = pd.DatetimeIndex(times)
    breaks = np.nonzero(np.diff(times.values) > np.timedelta64(gap))[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(times) - 1]
    return [(times[s], times[e]) for s, e in zip(starts, ends)]


def find_events(
    series, threshold, direction="above", decluster="runs", gap="12h", min_duration=None
):
    """Catalog threshold exceedances of a series with runs declustering.

    :param series: pandas Series (or 1-D xarray DataArray) indexed by time
    :param threshold: exceedance level, in the series' own units
    :param direction: ``"above"`` catalogs samples > threshold (e.g. |E|),
        ``"below"`` samples < threshold (e.g. Dst)
    :param decluster: ``"runs"`` (exceedances separated by less than ``gap``
        form one event -- the standard runs method) or None (every
        contiguous exceedance run is its own event)
    :param gap: minimum separation between independent events
        (Timedelta-like, default 12 h)
    :param min_duration: optionally require the exceedance (first to last
        sample) to span at least this long
    :return: DataFrame with one row per event: ``start``, ``end`` (first and
        last exceeding sample), ``peak_time``, ``peak_value``, ``duration``
    """
    series = _to_series(series)
    if direction == "above":
        exceeding = series.index[series.values > threshold]
    elif direction == "below":
        exceeding = series.index[series.values < threshold]
    else:
        raise ValueError("direction must be one of: above, below")

    if decluster == "runs":
        gap_td = pd.Timedelta(gap)
    elif decluster is None:
        # Merge only contiguous samples (one native step apart)
        steps = series.index.to_series().diff().dropna()
        gap_td = steps.min() if len(steps) else pd.Timedelta(0)
    else:
        raise ValueError("decluster must be 'runs' or None")

    rows = []
    for start, end in _cluster(exceeding, gap_td):
        chunk = series.loc[start:end]
        peak_idx = chunk.idxmax() if direction == "above" else chunk.idxmin()
        duration = end - start
        if min_duration is not None and duration < pd.Timedelta(min_duration):
            continue
        rows.append(
            {
                "start": start,
                "end": end,
                "peak_time": peak_idx,
                "peak_value": chunk.loc[peak_idx],
                "duration": duration,
            }
        )
    return pd.DataFrame(rows, columns=["start", "end", "peak_time", "peak_value", "duration"])


def find_storms(index, dst_threshold=-140, kp_threshold=8, window="1.5D", min_duration=None):
    """Catalog geomagnetic storms from index data (the papers' method).

    Samples where Dst drops below ``dst_threshold`` or Kp reaches
    ``kp_threshold`` trigger a storm; each trigger contributes a
    ``+-window`` interval and overlapping intervals merge into one storm.

    :param index: DataFrame (or dict of Series) with ``dst`` and/or ``kp``
        columns indexed by time, or a single named Series. Only thresholds
        whose series are present are applied.
    :param dst_threshold: Dst trigger level in nT (default -140, as in the
        2019 hazard study); None disables
    :param kp_threshold: Kp trigger level (default 8); None disables
    :param window: half-width of the storm window around triggering samples
        (default 1.5 days)
    :param min_duration: optionally require the triggering condition to span
        at least this long (first to last trigger)
    :return: DataFrame with one row per storm: ``start``, ``end`` (window
        edges), ``trigger_start``, ``trigger_end``, and the observed
        extremes (``min_dst`` and/or ``max_kp``)
    """
    if isinstance(index, pd.Series):
        index = index.to_frame(index.name)
    elif isinstance(index, dict):
        index = pd.DataFrame({k: _to_series(v) for k, v in index.items()})
    columns = {str(c).lower(): c for c in index.columns}

    triggers = []
    if dst_threshold is not None and "dst" in columns:
        dst = index[columns["dst"]].dropna()
        triggers.append(dst.index[dst.values < dst_threshold])
    if kp_threshold is not None and "kp" in columns:
        kp = index[columns["kp"]].dropna()
        triggers.append(kp.index[kp.values >= kp_threshold])
    if not triggers:
        raise ValueError(
            "index has neither a 'dst' nor a 'kp' column matching the "
            f"requested thresholds (columns: {list(index.columns)})"
        )

    window_td = pd.Timedelta(window)
    trigger_times = (
        pd.DatetimeIndex(np.concatenate([t.values for t in triggers])).sort_values().unique()
    )

    rows = []
    # Two triggers' windows overlap iff they are less than 2*window apart
    for first, last in _cluster(trigger_times, 2 * window_td):
        if min_duration is not None and last - first < pd.Timedelta(min_duration):
            continue
        start, end = first - window_td, last + window_td
        row = {"start": start, "end": end, "trigger_start": first, "trigger_end": last}
        if "dst" in columns:
            row["min_dst"] = index[columns["dst"]].loc[start:end].min()
        if "kp" in columns:
            row["max_kp"] = index[columns["kp"]].loc[start:end].max()
        rows.append(row)
    base = ["start", "end", "trigger_start", "trigger_end"]
    extremes = [name for name, col in (("min_dst", "dst"), ("max_kp", "kp")) if col in columns]
    return pd.DataFrame(rows, columns=base + extremes)
