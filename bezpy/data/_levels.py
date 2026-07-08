"""Geomagnetic data quality levels (provenance).

Observatory data is published at several quality levels that trade timeliness
for accuracy. bezpy stores each level separately so that, for example,
years-delayed *definitive* data never overwrites the *variation* (real-time)
data a user already has -- and so that the original, un-culled real-time series
remains available even after definitive data (which sometimes removes spikes
that are real signal in other contexts) is published.

Levels, from lowest to highest quality/latency:

``variation``
    Raw real-time / reported data, available immediately. The "original" data.
``adjusted``
    Baseline-adjusted real-time data.
``provisional``
    Partially checked data (weeks-to-months latency).
``quasi-definitive``
    Near-definitive quality, published within ~3 months. Good for rapid
    post-storm analysis.
``definitive``
    Final, baseline-corrected, peer-reviewed data. Published yearly, so it can
    lag by 1+ years, and may cull data points that are valid in other contexts.

The numeric rank is used when merging overlapping data: higher rank wins.
"""
__all__ = ["canonical_level", "level_rank", "LEVELS", "BEST"]

# Sentinel meaning "merge everything, best quality wins" on load.
BEST = "best"

# Canonical level -> quality/latency rank (higher = better quality).
LEVELS = {
    "variation": 0,
    "adjusted": 1,
    "provisional": 2,
    "quasi-definitive": 3,
    "definitive": 4,
}

# Accepted spellings/synonyms mapped onto the canonical names above. Keys are
# compared case-insensitively with spaces/underscores/hyphens normalized.
_ALIASES = {
    "variation": "variation",
    "reported": "variation",
    "rep": "variation",
    "raw": "variation",
    "realtime": "variation",
    "adjusted": "adjusted",
    "adj": "adjusted",
    "adjorrep": "adjusted",
    "provisional": "provisional",
    "prov": "provisional",
    "quasidefinitive": "quasi-definitive",
    "quasi": "quasi-definitive",
    "qd": "quasi-definitive",
    "definitive": "definitive",
    "def": "definitive",
}


def _normalize(level):
    """Lower-case and strip separators so aliases match regardless of style."""
    return "".join(ch for ch in str(level).lower() if ch.isalnum())


def canonical_level(level, allow_best=False):
    """Return the canonical level name for a user-supplied spelling.

    Parameters
    ----------
    level : str
        A level name or synonym (e.g. ``"Quasi-definitive"``, ``"reported"``).
    allow_best : bool, optional
        If ``True``, the special value ``"best"`` is passed through unchanged
        (used by :func:`load_observatory` to request a merged view).

    Raises
    ------
    ValueError
        If the level is not recognized.
    """
    normalized = _normalize(level)
    if allow_best and normalized == "best":
        return BEST
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    raise ValueError(
        f"Unknown data level {level!r}. Valid levels: "
        f"{', '.join(LEVELS)}" + (" (or 'best')" if allow_best else ""))


def level_rank(level):
    """Return the quality rank of a canonical level (higher = better)."""
    return LEVELS[canonical_level(level)]
