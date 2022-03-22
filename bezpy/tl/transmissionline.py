"""
High Voltage Electric Transmission Lines

author: Greg Lucas

e-mail: greg.m.lucas@gmail.com
"""

__all__ = ["TransmissionLine"]

import numpy as np
import shapely

from .utils import haversine_distance, haversine_dl, fast_interp_weights


class TransmissionLine:
    """A high voltage transmission line object."""

    def __init__(self, line):
        # Shapely LineString object as input
        # shape: npts x 2
        # The points are stored in longitude/latitude order
        # Everything else is stored opposite with x == North, y == East
        self.pts = np.array(line.geometry.coords)

        # Calculate the length of the line in km
        # calculate the distance between each successive points and then sum that
        self.length = np.sum(haversine_distance(self.pts[:-1, 0], self.pts[:-1, 1],
                                                self.pts[1:, 0], self.pts[1:, 1]))

        # Calculate the dl: x coordinate is latitude
        # (dLat, dLon) == (dx, dy)
        # dl has the shape: npts x 2
        self.dl = haversine_dl(self.pts[:-1, 0], self.pts[:-1, 1],
                               self.pts[1:, 0], self.pts[1:, 1])

        # Store the nearest MT Site to each point
        # shape: npts
        self.nearest_sites = None

        # What region is the point in
        # shape: npts
        self.regions1d = None

        # Delaunay weights for fast interpolation
        # Shapes are: npts x 3
        self.delaunay_vtx = None
        self.delaunay_wts = None

    @property
    def npts(self):
        """Number of points along the line."""
        return len(self)

    def __len__(self):
        return len(self.pts)

    def set_nearest_sites(self, site_xys):
        """Sets the nearest site to each point along the transmission line."""
        # Using Haversine distance to calculate nearest sites
        distances = haversine_distance(self.pts[:, 0][:, np.newaxis],
                                       self.pts[:, 1][:, np.newaxis],
                                       site_xys[:, 1][np.newaxis, :],
                                       site_xys[:, 0][np.newaxis, :])
        self.nearest_sites = np.argmin(distances, axis=1)

    def set_1d_regions(self, region_polygons, default_region=18):
        """Sets the 1d region that each point along the transmission line is in."""
        # Start out with a default value of 18,
        # so if there is no point in the polygon it goes there
        # 18 is CP-1 in the conductivity 1D profiles currently
        self.regions1d = np.ones(self.npts, dtype=np.int)*default_region
        for i in range(self.npts):
            point = shapely.geometry.Point(self.pts[i, :])
            possible_polygons = [j for j, cond_poly in enumerate(region_polygons)
                                 if point.within(cond_poly)]
            if possible_polygons:
                self.regions1d[i] = possible_polygons[0]

            # for j, cond_poly in enumerate(region_polygons):
            #     if p.within(cond_poly):
            #         self.regions1d[i] = j
            #         break # stops after the first within is satisfied

        # TODO: Use geopandas spatial joins instead
        # line_gdf = gpd.GeoDataFrame(crs={'init': 'epsg:4326'},
        #                     geometry=[shapely.geometry.Point(self.pts[i,:]) for i in
        #                     range(len(self.pts))])
        # now that we have a geodataframe of the points along the line
        # we can do a spatial join
        # After the spatial join, we need to groupby the index in case the point
        # is within multiple regions, and only keep the first index
        # joined_gdf = gpd.sjoin(
        #    line_gdf, region_polygons, how='left', op='within').groupby(lambda x: x).first()
        # There could be nans inside the joined gdf if the point was outside
        # of a region
        # index_right is which index it was in
        # self.regions1d = np.array(joined_gdf.index_right.fillna(default_region).values,
        #     dtype=np.int)

    def set_delaunay_weights(self, site_xys, use_gnomic=True):
        """Sets the delaunay weights for the given sites for each point along the line."""
        # function takes data first, and then the interpolation locations
        # We are trying to go from the site_xys (data_xy) to our transmission line
        # points (interp_xy)

        # Turn everything into radians
        xys = np.deg2rad(site_xys)
        pts = np.deg2rad(self.pts)
        if use_gnomic:
            # Use a gnomonic projection (every line is a great circle distance)

            # Make the middle lat/lon point be the mean of our data points
            lon0 = np.mean(xys[:, 1])
            lat0 = np.mean(xys[:, 0])

            # Calculate the x/y locations of the data points
            lons = xys[:, 1]
            lats = xys[:, 0]

            cos_c = np.sin(lat0)*np.sin(lats) + np.cos(lat0)*np.cos(lats)*np.cos(lons-lon0)
            x = (np.cos(lats)*np.sin(lons-lon0))/cos_c
            y = (np.cos(lat0)*np.sin(lats) - np.sin(lat0)*np.cos(lats)*np.cos(lons-lon0))
            xys[:, 0], xys[:, 1] = x, y

            # Now calculate the x/y locations of the interpolation points
            lons = pts[:, 0]
            lats = pts[:, 1]

            cos_c = np.sin(lat0)*np.sin(lats) + np.cos(lat0)*np.cos(lats)*np.cos(lons-lon0)
            x = (np.cos(lats)*np.sin(lons-lon0))/cos_c
            y = (np.cos(lat0)*np.sin(lats) - np.sin(lat0)*np.cos(lats)*np.cos(lons-lon0))/cos_c
            pts[:, 0], pts[:, 1] = x, y
        else:
            # multiply by the cosine of latitude to adjust for the distances near the poles.
            xys[:, 0] = xys[:, 0]*np.abs(np.cos(xys[:, 1]))
            pts[:, 0] = pts[:, 0]*np.abs(np.cos(pts[:, 1]))

        self.delaunay_vtx, self.delaunay_wts = fast_interp_weights(xys, pts)

    def calc_voltages(self, E, how=None):
        """Calculates the voltages across this power line, given E(t).

        :param E: Electric field over time shape: (ntimes, nlocs, 2)

        :param how: Method of interpolation, one of (nn, 1d, delaunay)
        """
        # pylint: disable=invalid-name
        if how == "nn":
            if self.nearest_sites is None:
                raise ValueError("The closest regions were not defined yet, call:\n" +
                                 "    .set_nearest_sites(site_xys)\n" +
                                 "to initialize the closest regions")
            E3d = np.atleast_3d(E)[:, self.nearest_sites[:-1], :]

        elif how == "1d":
            if self.regions1d is None:
                raise ValueError("The closest regions were not defined yet, call:\n" +
                                 "    .set_1d_regions(region_polygons)\n" +
                                 "to initialize the closest regions")
            E3d = np.atleast_3d(E)[:, self.regions1d[:-1], :]

        elif how == "delaunay":
            if self.delaunay_vtx is None or self.delaunay_wts is None:
                raise ValueError("The interpolation weights were not defined yet, call:\n" +
                                 "    .set_delaunay_weights(site_xys)\n" +
                                 "to initialize the interpolation weights")

            # NOTE: this implements fast_interpolate here rather than calling it
            #       due to the dimensionality of the E-fields

            # vtx shape: (npts, 3)
            # that turns the first E into a shape of: (ntimes, npts, 3, 2) when
            # doing the fancy-indexing
            # Need to sum over the 3 weights, which are along axis 1 to make it a
            # 3d E field again
            # If any weights are less than 0, the interpolation is out of the boundary
            # so return np.nan
            E3d = np.sum(np.atleast_3d(E)[:, self.delaunay_vtx[:-1], :] *
                         self.delaunay_wts[np.newaxis, :-1, :, np.newaxis], axis=2)
            E3d[:, np.any(self.delaunay_wts[:-1] < 0, axis=1), :] = np.nan

        else:
            raise ValueError("how must be a string of: nn, 1d, or delaunay")

        # E3d in units of mV/km
        # dl in units of km
        # divide by 1000 to change to V
        voltages = np.sum(E3d*self.dl[np.newaxis, :, :], axis=(1, 2)) / 1000.
        # if there is only one voltage, just return that as a float
        # Otherwise return the entire array of voltages, which is the
        # length of the final dimension of E input (ntimes)
        if len(voltages) == 1:
            return float(voltages)

        return voltages

    def get_coords(self, transform=None):
        """Returns the coordinates of this line

           :param transform: function that transforms the x/y coordinates
                             which could be a basemap instance or pyproj projection
        """

        if transform is None:
            return self.pts

        # Otherwise we have a transform function, so apply it to the points
        # transform takes a list of x and a list of y coordinates
        # it outputs a list of new x and new y coordinates, so we need to
        # stack them back into a single array of shape: (npts, 2)
        return np.vstack(transform(self.pts[:, 0], self.pts[:, 1])).T
