"""Helper functions"""

__all__ = ["haversine_distance", "haversine_dl", "fast_interp_weights", "fast_interpolate"]

import numpy as np
import scipy.spatial.qhull as qhull


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
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)  # pylint: disable=no-member
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, deg]
    bary = np.einsum('njk,nk->nj', temp[:, :deg, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def fast_interpolate(values, vtx, wts, fill_value=np.nan):
    """A method to perform fast Delaunay interpolation."""
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret
