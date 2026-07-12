"""Map plotting helpers for geoelectric / GIC results.

Thin wrappers around matplotlib (and cartopy where installed): every
function takes an existing axes and returns the artists it added, so these
compose with any figure layout. Both libraries are optional -- matplotlib
via the ``plotting`` extra, cartopy separately (its map features are simply
skipped when it is absent and a plain axes is passed).
"""

__all__ = ["add_map_features", "scale_vectors", "plot_vectors", "plot_lines",
           "plot_hazard_map"]

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib
    except ImportError:
        raise ImportError(
            "bezpy.plot requires matplotlib; install it with "
            "`pip install 'bezpy[plotting]'`") from None
    return matplotlib


def _crs_kwargs(ax):
    """Pass data coordinates as lon/lat when the axes is a cartopy GeoAxes."""
    try:
        import cartopy.crs as ccrs
    except ImportError:
        return {}
    return {"transform": ccrs.PlateCarree()} if hasattr(ax, "projection") else {}


def add_map_features(ax, scale="50m", states=True):
    """Add coastlines, borders, and (optionally) state lines to a GeoAxes.

    The boilerplate block from every paper's map figure. Requires cartopy;
    raises with an actionable message otherwise.

    :param ax: a cartopy GeoAxes
    :param scale: Natural Earth feature scale (``"110m"``, ``"50m"``, ``"10m"``)
    :param states: include state/province lines
    :return: list of the added feature artists
    """
    try:
        import cartopy.feature as cfeature
    except ImportError:
        raise ImportError(
            "add_map_features requires cartopy (pip install cartopy); "
            "the other bezpy.plot functions work on plain matplotlib axes"
        ) from None
    features = [
        cfeature.NaturalEarthFeature("physical", "coastline", scale,
                                     edgecolor="black", facecolor="none"),
        cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land",
                                     scale, edgecolor="gray", facecolor="none"),
    ]
    if states:
        features.append(cfeature.NaturalEarthFeature(
            "cultural", "admin_1_states_provinces_lines", scale,
            edgecolor="gray", facecolor="none", linewidth=0.5))
    return [ax.add_feature(feature) for feature in features]


def scale_vectors(x, y, scale=np.log10):
    """Rescale vector magnitudes while preserving their angles.

    The papers' map idiom for fields spanning decades: magnitudes below 1
    are clamped to 1 (so the scaling never flips a vector), then the
    magnitude is passed through ``scale`` (log10 by default) and the
    components are rebuilt at the original angle.

    :param x, y: vector components (any consistent units)
    :param scale: magnitude mapping (default ``np.log10``)
    :return: ``(x_scaled, y_scaled)``
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    magnitude = np.hypot(x, y)
    clamped = np.maximum(magnitude, 1.)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(magnitude > 0., scale(clamped)/magnitude, 0.)
    return x*factor, y*factor


def plot_vectors(ax, latlons, vectors, scale="log", color="black", **kwargs):
    """Quiver a horizontal vector field (B or E) at point locations.

    :param ax: matplotlib axes (or cartopy GeoAxes; lon/lat handled)
    :param latlons: ``(n, 2)`` [latitude, longitude]
    :param vectors: ``(n, 2)`` [x (northward), y (eastward)] components
    :param scale: ``"log"`` (magnitude-preserving-angle log scaling, the
        papers' default), ``"linear"``, or a callable applied to magnitudes
    :param kwargs: forwarded to ``ax.quiver``
    :return: the Quiver artist
    """
    latlons = np.atleast_2d(np.asarray(latlons, dtype=float))
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    vec_x, vec_y = vectors[:, 0], vectors[:, 1]
    if scale == "log":
        vec_x, vec_y = scale_vectors(vec_x, vec_y)
    elif callable(scale):
        vec_x, vec_y = scale_vectors(vec_x, vec_y, scale=scale)
    elif scale != "linear":
        raise ValueError("scale must be 'log', 'linear', or a callable")
    # Quiver is (east, north) in plot axes: u = y component, v = x component
    return ax.quiver(latlons[:, 1], latlons[:, 0], vec_y, vec_x,
                     color=color, **_crs_kwargs(ax), **kwargs)


def plot_lines(ax, lines, values=None, cmap="viridis", norm=None, **kwargs):
    """Draw transmission lines, optionally colored by a per-line value.

    :param ax: matplotlib axes (or cartopy GeoAxes)
    :param lines: sequence of TransmissionLine objects
    :param values: optional per-line scalars (e.g. peak |V|); colored via
        ``cmap``/``norm`` when given, plain ``color`` kwarg otherwise
    :param kwargs: forwarded to LineCollection (``linewidths``, ``color``...)
    :return: the LineCollection artist
    """
    _require_matplotlib()
    from matplotlib.collections import LineCollection
    segments = [line.pts for line in lines]
    collection = LineCollection(segments, cmap=cmap if values is not None else None,
                                norm=norm, **_crs_kwargs(ax), **kwargs)
    if values is not None:
        collection.set_array(np.asarray(values, dtype=float))
    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def plot_hazard_map(ax, latlons, values, log=True, cmap="magma_r",
                    add_colorbar=True, colorbar_label=None, **kwargs):
    """Scatter per-site hazard values (e.g. 100-year |E|) on a map.

    :param ax: matplotlib axes (or cartopy GeoAxes)
    :param latlons: ``(n, 2)`` [latitude, longitude]
    :param values: per-site scalars
    :param log: color on a log scale (hazard values span decades)
    :param add_colorbar: attach a colorbar to the axes' figure
    :param kwargs: forwarded to ``ax.scatter`` (``s``, ``marker``, ...);
        ``vmin``/``vmax`` set the log color limits (values outside saturate
        at the colormap ends rather than being dropped), and a ``norm``
        kwarg overrides the log scaling entirely
    :return: the PathCollection artist
    """
    matplotlib = _require_matplotlib()
    latlons = np.atleast_2d(np.asarray(latlons, dtype=float))
    values = np.asarray(values, dtype=float)
    norm = kwargs.pop("norm", None)
    if log and norm is None:
        positive = values[np.isfinite(values) & (values > 0)]
        if positive.size:
            vmin = kwargs.pop("vmin", None)
            vmax = kwargs.pop("vmax", None)
            norm = matplotlib.colors.LogNorm(
                vmin=positive.min() if vmin is None else vmin,
                vmax=positive.max() if vmax is None else vmax)
    artist = ax.scatter(latlons[:, 1], latlons[:, 0], c=values, cmap=cmap,
                        norm=norm, **_crs_kwargs(ax), **kwargs)
    if add_colorbar:
        ax.figure.colorbar(artist, ax=ax, label=colorbar_label)
    return artist
