"""Helper functions"""

__all__ = ["haversine_distance", "haversine_dl", "fast_interp_weights", "fast_interpolate",
           "delaunay_weights", "filter_bbox", "filter_polygon"]

import numpy as np
import shapely
from scipy.spatial import Delaunay


def gnomonic_forward(lats, lons, lat0, lon0, what="points"):
    """Forward gnomonic projection of (lats, lons) radians about (lat0, lon0).

    Returns ``(x, y)`` plane coordinates. Only valid within 90 degrees of the
    projection center; ``what`` names the offending point set in the error.
    """
    cos_c = np.sin(lat0) * np.sin(lats) + np.cos(lat0) * np.cos(lats) * np.cos(lons - lon0)
    if np.any(cos_c <= 0.0):
        raise ValueError(
            f"gnomonic projection is invalid for {what} 90 degrees or more "
            "from the site network's mean location; use use_gnomic=False or "
            "split the domain"
        )
    x = (np.cos(lats) * np.sin(lons - lon0)) / cos_c
    y = (np.cos(lat0) * np.sin(lats) - np.sin(lat0) * np.cos(lats) * np.cos(lons - lon0)) / cos_c
    return x, y


def filter_polygon(lines, polygon, predicate="intersects"):
    """Select the transmission lines that intersect a polygon.

    Uses a spatial index (STRtree) so large inventories filter quickly, with
    no geopandas required.

    :param lines: sequence of TransmissionLine objects
    :param polygon: a shapely geometry (Polygon, MultiPolygon, ...)
    :param predicate: spatial predicate to apply (``"intersects"`` default,
        ``"within"`` for fully-contained lines, ...)
    :return: list of the matching TransmissionLine objects
    """
    geometries = [shapely.LineString(line.pts) for line in lines]
    tree = shapely.STRtree(geometries)
    idx = np.sort(tree.query(polygon, predicate=predicate))
    return [lines[i] for i in idx]


def filter_bbox(lines, bbox, predicate="intersects"):
    """Select the transmission lines that intersect a lon/lat bounding box.

    :param lines: sequence of TransmissionLine objects
    :param bbox: ``(west, south, east, north)`` in degrees
    :return: list of the matching TransmissionLine objects
    """
    west, south, east, north = bbox
    return filter_polygon(lines, shapely.box(west, south, east, north),
                          predicate=predicate)


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance, in km, between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    # Distance in km between points in x1 and points in x2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def haversine_dl(lon1, lat1, lon2, lat2):
    """Calculate haversine vectors in the lon/lat directions between each point.
       dl is returned as an array of npts x 2, where the two coordinates
       dx and dy == dLat and dLon respectively (x == North, y == East)."""

    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # First time through, we will calculate the longitudinal variation,
    # so lat1 == lat2

    # Full equation:
    # a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    # Longitude calculation: dlat == 0
    a = np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)**2

    # Distance in km between points in x1 and points in x2
    # multiply by sign of dlon to get vector pointing in the right direction
    dl_lon = (6371 * 2 * np.arcsin(np.sqrt(a))) * np.sign(dlon)

    # Latitudinal variation: dlon == 0
    a = np.sin(dlat/2.)**2

    # Distance in km between points in x1 and points in x2
    # multiply by sign of dlat to get vector pointing in the right direction
    dl_lat = (6371 * 2 * np.arcsin(np.sqrt(a))) * np.sign(dlat)

    # Return it in the shape of: (npts, 2)
    return np.vstack([dl_lat, dl_lon]).T


def fast_interp_weights(xyz, uvw, deg=2):
    """Set fast interpolation weights for Delaunay triangulation.
    :param xyz: data locations
    :param uvw: interpolation locations
    """
    tri = Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, deg]
    bary = np.einsum('njk,nk->nj', temp[:, :deg, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def fast_interpolate(values, vtx, wts, fill_value=np.nan):
    """A method to perform fast Delaunay interpolation."""
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret


def delaunay_weights(source_latlons, target_latlons, use_gnomic=True):
    """Barycentric weights interpolating from source points onto target points.

    Both point sets go through **one** projection before triangulation --
    projecting only one of them skews the weights (the pre-0.5.0
    ``set_delaunay_weights`` bug), so this is the single place that
    projection is done for every Delaunay interpolation in bezpy.

    :param source_latlons: ``(nsources, 2)`` array of ``(latitude, longitude)``
        in degrees -- the locations values are known at (e.g. MT sites)
    :param target_latlons: ``(ntargets, 2)`` array of ``(latitude, longitude)``
        in degrees -- the locations to interpolate onto
    :param use_gnomic: project through a gnomonic projection centered on the
        source network (default), which maps great circles to straight lines;
        requires every point within 90 degrees of the network's mean location.
        Otherwise longitudes are shrunk by ``cos(latitude)``.
    :return: ``(vtx, wts)`` for :func:`fast_interpolate`. A target outside the
        source convex hull has a negative weight, which ``fast_interpolate``
        turns into ``fill_value``.
    """
    source = np.asarray(source_latlons, dtype=float)
    target = np.asarray(target_latlons, dtype=float)
    if source.ndim != 2 or source.shape[1] != 2 or len(source) < 3:
        raise ValueError(
            "source_latlons must have shape (nsources, 2) [latitude, longitude] "
            "with at least 3 points for a Delaunay triangulation"
        )
    if target.ndim != 2 or target.shape[1] != 2 or len(target) == 0:
        raise ValueError("target_latlons must have shape (ntargets, 2) [latitude, longitude]")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("latitudes and longitudes must all be finite")

    src = np.deg2rad(source)
    tgt = np.deg2rad(target)
    if use_gnomic:
        lat0 = np.mean(src[:, 0])
        lon0 = np.mean(src[:, 1])
        src_xy = np.column_stack(
            gnomonic_forward(src[:, 0], src[:, 1], lat0, lon0, what="source points")
        )
        tgt_xy = np.column_stack(
            gnomonic_forward(tgt[:, 0], tgt[:, 1], lat0, lon0, what="target points")
        )
    else:
        # (lon, lat) axis order, longitudes shrunk by cos(latitude) to adjust
        # for the converging meridians near the poles.
        src_xy = src[:, ::-1].copy()
        tgt_xy = tgt[:, ::-1].copy()
        src_xy[:, 0] *= np.abs(np.cos(src_xy[:, 1]))
        tgt_xy[:, 0] *= np.abs(np.cos(tgt_xy[:, 1]))

    return fast_interp_weights(src_xy, tgt_xy)
