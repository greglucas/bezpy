import argparse
#-----------------------------
# Command Line Options
#-----------------------------

description = "Create real-time maps of the magnetic field across the US."
parser = argparse.ArgumentParser(description=description)

parser.add_argument('--outdir', action='store',
                    required=False, default='.',
                    help='Where the output files will be written to. (default: "."')

parser.add_argument('--timeseries', action='store_true',
                    default=False, required=False,
                    help='Make a plot of time series')

parser.add_argument('--secmap', action='store_true',
                    default=False, required=False,
                    help='Make the most recent map of SEC predicted magnetic fields')

parser.add_argument('--movie', action='store_true',
                    default=False, required=False,
                    help='Make a movie of the magnetic field changing over the past day')

parser.add_argument('--server', action='store',
                    default='https://geomag.usgs.gov',
                    required=False,
                    help='Server url to query for data (default: %(default)s)')

parser.add_argument('--verbose', action='store_true',
                    default=False, required=False,
                    help='Print out timing information')

args = parser.parse_args()

SERVER_URL = args.server

# server requests
import requests
import datetime
import time

# Numerical
import numpy as np
import pandas as pd
from pySECS import SECS
import bezpy

# Plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

#------------------
# Defaults
#------------------
t0 = time.time()

R_earth = 6378000
Bscale = 12
arrow_width = 0.004
cmap = mpl.cm.plasma
norm = mpl.colors.LogNorm(vmin=1, vmax=1000)

# Mapping library
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

bbox = [-135, -65, 20, 75]
projection = ccrs.LambertAzimuthalEqualArea(central_latitude=42.5, central_longitude=-95)
proj_data = ccrs.PlateCarree()

def scale_vectors(x, y, scale=np.log10):
    """Scale vectors while preserving the angle.

    Parameters
    ==========
    x: Cartesian x coordinate(s)
    y: Cartesian y coordinate(s)

    scale: function to scale the magnitude by (Default: log10)
    """
    mag = np.sqrt(x**2 + y**2)
    # Prevent logs of values <1, which could be large negatives and flip the true sign
    with np.errstate(invalid='ignore'):
        mag[mag<1] = 1.
    angle = np.arctan2(y, x)
    newx = scale(mag)*np.cos(angle)
    newy = scale(mag)*np.sin(angle)

    return (newx, newy)

#------------------------------------------------
# Routines for downloading geomag data from USGS
#------------------------------------------------
USGS_CODES = ['BRW', 'DED', 'CMO', 'SIT', 'SHU',
              'NEW', 'BOU', 'FRD', 'FRN', 'TUC',
              'BSL', 'HON', 'SJG', 'GUA']

# Actual sites
NRCAN_CODES = ['ALE', 'BLC', 'BRD', 'CBB', 'EUA',
               'FCC', 'IQA', 'MEA', 'OTT', 'RES',
               'STJ', 'SNK', 'VIC', 'YKC']

# Removed ALE, EUA, SNK
NRCAN_CODES = ['BLC', 'BRD', 'CBB',
               'FCC', 'IQA', 'MEA', 'OTT', 'RES',
               'STJ', 'VIC', 'YKC']

class GeomagObservatory:
    def __init__(self, name='BOU'):
        """Object to download and store geomagnetic observatory data"""

        self.name = name
        # Location code on USGS servers
        self.loc_code = 'R0'
        # Canadian data is in R1 (GOES)
        if name in NRCAN_CODES:
            self.loc_code = 'R1'

        self.dt = datetime.timedelta(days=1)
        self._update_times()
        self.df = None
        self.update_data()

    def _update_times(self):
        self.end_time = datetime.datetime.utcnow()
        self.start_time = self.end_time - self.dt

    def update_data(self):
        self._update_times()

        url = ("{server_url}/ws/edge/?format=json".format(server_url=SERVER_URL) +
                "&id={name}".format(name=self.name) +
                "&type={loc_code}".format(loc_code=self.loc_code) +
                "&starttime={starttime}".format(starttime=self.start_time.strftime("%Y-%m-%dT%H:%M:%SZ")) +
                "&endtime={endtime}".format(endtime=self.end_time.strftime("%Y-%m-%dT%H:%M:%SZ")) +
                "&sampling_period=60" +
                "&elements=X,Y,Z") # other codes: ,F,SV,SQ,Dist

        try:
            r = requests.get(url)
            data = r.json()
            self.df = pd.DataFrame(index=pd.to_datetime(data['times']),
                     data={'X': data['values'][0]['values'],
                           'Y': data['values'][1]['values'],
                           'Z': data['values'][2]['values']},
                     dtype=np.float).dropna()

            self.df['Direction'] = np.rad2deg(np.arctan2(self.df['X'], self.df['Y']))
        except:
            print(f"Failed to update {self.name}")

observatories = [GeomagObservatory(obs) for obs in USGS_CODES] + \
                [GeomagObservatory(obs) for obs in NRCAN_CODES]

good_observatories = [obs for obs in observatories
                      if ((obs.df is not None) and (not obs.df.empty))]
missing_observatories = [obs for obs in observatories if obs not in good_observatories]

good_xy = []
missing_xy = []

for obs in observatories:
    site = bezpy.mag.get_iaga_observatory(obs.name)
    lonlat = (site['longitude'], site['latitude'])
    if obs in good_observatories:
        good_xy.append(lonlat)
    else:
        missing_xy.append(lonlat)

good_xy = np.array(good_xy)
missing_xy = np.array(missing_xy)

nobs = len(good_observatories)
B = pd.concat([obs.df[['X', 'Y', 'Z']] for obs in good_observatories], axis=1)
# Resample, then interpolate
B = B.resample('15Min').median().interpolate(limit_direction='both')
# Detrend
B = B - B.median()

# Create numpy array
B_obs = np.zeros((len(B), nobs, 3))
B_obs[:,:,0] = B['X']
B_obs[:,:,1] = B['Y']
B_obs[:,:,2] = B['Z']

# Save the most recent time string for file naming
time_string = B.index[-1].strftime("%Y%m%d-%H%M")

if args.verbose:
    print("Downloading data: {:.2f} s".format(time.time()-t0))

#-----------------------
# Set up regular grids
#-----------------------
delta_x = delta_y = 2

xs = np.arange(-180, -20, delta_x)
ys = np.arange(0, 89, delta_y)
yy, xx = np.meshgrid(ys, xs)
# Change it to an N x 2 matrix
gridded_xy = np.vstack([np.ravel(xx), np.ravel(yy)]).T

# Plot grid
# NOTE: This specifies box edges for pcolormesh, which must be
#       one longer than the 'data points'
grid_shape = xx.shape

xs = np.append((xs - delta_x/2), xs[-1] + delta_x/2)
ys = np.append((ys - delta_y/2), ys[-1] + delta_y/2)
yy_edges, xx_edges = np.meshgrid(ys, xs)

obs_lat_lon_r = np.zeros((len(good_xy), 3))
obs_lat_lon_r[:,0] = good_xy[:,1]
obs_lat_lon_r[:,1] = good_xy[:,0]
obs_lat_lon_r[:,2] = R_earth

obs_var = np.ones(obs_lat_lon_r.shape)
# Don't include Z component in fits for now
obs_var[:,2] = np.inf

gridded_lat_lon_r = np.zeros((len(gridded_xy), 3))
gridded_lat_lon_r[:,0] = gridded_xy[:,1]
gridded_lat_lon_r[:,1] = gridded_xy[:,0]
gridded_lat_lon_r[:,2] = R_earth

# specify the SECS grid
lat, lon, r = np.meshgrid(np.linspace(15,85,36),
                          np.linspace(-175,-25,76),
                          R_earth+110000, indexing='ij')
secs_lat_lon_r = np.hstack((lat.reshape(-1,1),
                            lon.reshape(-1,1),
                            r.reshape(-1,1)))

#--------------
# SECS
#--------------
secs = SECS(sec_df_loc=secs_lat_lon_r)
secs.fit(obs_loc=obs_lat_lon_r, obs_B=B_obs,
         obs_var=obs_var, epsilon=0.05)

B_gridded = secs.predict_B(gridded_lat_lon_r)

if args.verbose:
    print("SECS Interpolation: {:.2f} s".format(time.time()-t0))

#---------------
# Plotting
#---------------
def setup_map():
    scale = '10m'
    coast = cfeature.NaturalEarthFeature(category='physical', scale=scale,
                                         edgecolor='k',
                                         facecolor='none', name='coastline')
    states = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale=scale,
            facecolor='none',
            edgecolor='k')
    countries = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_countries',
            scale=scale,
            facecolor='none',
            edgecolor='k')

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(projection=projection))

    ax.set_extent(bbox, proj_data)
    ax.add_feature(coast)
    ax.add_feature(states, alpha=0.8)
    ax.add_feature(countries)

    return (fig, ax)


if args.timeseries:
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 8))

    for obs in good_observatories:
        ax1.plot(obs.df.index, obs.df['X']-obs.df['X'].median(), label=obs.name)
        ax2.plot(obs.df.index, obs.df['Y']-obs.df['Y'].median(), label=obs.name)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=5, fancybox=True, shadow=True)
    ax1.set_ylabel('B$_x$ (nT)')
    ax2.set_ylabel('B$_y$ (nT)')

    ylim = ax1.get_ylim()
    max_y = np.max(np.abs(ylim))
    ax1.set_ylim(-max_y, max_y)

    ylim = ax2.get_ylim()
    max_y = np.max(np.abs(ylim))
    ax2.set_ylim(-max_y, max_y)

    fig.savefig(args.outdir + '/timeseries_{}.png'.format(time_string), bbox_inches='tight')

    if args.verbose:
        print("Time series figure: {:.2f} s".format(time.time()-t0))

if args.secmap:
    # Make a single map and save it
    t = -1
    Bx = B_gridded[t,:,0]
    By = B_gridded[t,:,1]
    B_mag = np.sqrt(Bx**2 + By**2)
    By_log, Bx_log = scale_vectors(By, Bx)

    fig, ax = setup_map()

    cax = ax.pcolormesh(xx_edges, yy_edges, B_mag.reshape(grid_shape),
                               cmap=cmap, norm=norm,
                               transform=proj_data)

    cbar = plt.colorbar(cax, orientation='horizontal', label='B (nT)')

    Q = ax.quiver(gridded_xy[:,0], gridded_xy[:,1],
                  By_log, Bx_log, color='k',
                  transform=proj_data,
                  scale_units='inches', scale=Bscale, width=arrow_width,
                  alpha=1., regrid_shape=15)

    Bx = B_obs[t,:,0]
    By = B_obs[t,:,1]
    B_mag = np.sqrt(Bx**2 + By**2)

    By_log, Bx_log = scale_vectors(By, Bx)

    #obs_scatter = ax.scatter(good_xy[:,0], good_xy[:,1],
    #                         c='g', s=50, alpha=0.75,
    #                         zorder=9, transform=proj_data)
    # Plot the missing observatories in Red
    missing_scatter = ax.scatter(missing_xy[:,0], missing_xy[:,1],
                                    c='r', s=50, alpha=0.75,
                                    zorder=9, transform=proj_data)

    Q_obs = ax.quiver(good_xy[:,0], good_xy[:,1],
                      By_log, Bx_log, color='g',
                      transform=proj_data,
                      scale_units='inches', scale=Bscale, width=arrow_width,
                      alpha=1., zorder=10)

    title = ax.set_title("SECS Magnetic Field\n{0}".format(
                    B.index[t].strftime("%Y/%m/%d %H:%M")))

    fig.savefig(args.outdir + "/secs_map_{}.png".format(time_string), bbox_inches='tight')

    if args.verbose:
        print("SECS map: {:.2f} s".format(time.time()-t0))

if args.movie:
    # Animations
    from matplotlib import animation

    # Make a single map and save it
    fig, ax = setup_map()

    t = -1
    Bx = B_gridded[t,:,0]
    By = B_gridded[t,:,1]
    B_mag = np.sqrt(Bx**2 + By**2)
    By_log, Bx_log = scale_vectors(By, Bx)

    cax = ax.pcolormesh(xx_edges, yy_edges, B_mag.reshape(grid_shape),
                               cmap=cmap, norm=norm,
                               transform=proj_data)

    cbar = plt.colorbar(cax, orientation='horizontal', label='B (nT)')

    Q = ax.quiver(gridded_xy[:,0], gridded_xy[:,1],
                  By_log, Bx_log, color='k',
                  transform=proj_data,
                  scale_units='inches', scale=Bscale, width=arrow_width,
                  alpha=1., regrid_shape=15)

    #----------------------------------------
    # Regrid in map space and
    # calculate B-fields on this new grid.
    #----------------------------------------
    # Get the uniform xy coordinates of the quiver grid and then delete that quiver
    xy = Q.get_offsets()
    Q.remove()

    uniform_xy = proj_data.transform_points(projection, xy[:,0], xy[:,1])
    uniform_xy[:,2] = R_earth
    # Need to send lat/lon/r, so swap axis 0/1
    B_uniform = secs.predict_B(uniform_xy[:,[1,0,2]])

    Bx = B_uniform[t,:,0]
    By = B_uniform[t,:,1]

    By_log, Bx_log = scale_vectors(By, Bx)

    Q = ax.quiver(uniform_xy[:,0], uniform_xy[:,1],
                  By_log, Bx_log, color='k',
                  transform=proj_data,
                  scale_units='inches', scale=Bscale, width=arrow_width,
                  alpha=1.)

    Bx = B_obs[t,:,0]
    By = B_obs[t,:,1]

    By_log, Bx_log = scale_vectors(By, Bx)

    #obs_scatter = ax.scatter(good_xy[:,0], good_xy[:,1],
    #                         c='g', s=50, alpha=0.75,
    #                         zorder=9, transform=proj_data)
    # Plot the missing observatories in Red
    missing_scatter = ax.scatter(missing_xy[:,0], missing_xy[:,1],
                                    c='r', s=50, alpha=0.75,
                                    zorder=9, transform=proj_data)

    Q_obs = ax.quiver(good_xy[:,0], good_xy[:,1],
                      By_log, Bx_log, color='g',
                      transform=proj_data,
                      scale_units='inches', scale=Bscale, width=arrow_width,
                      alpha=1., zorder=10)

    title = ax.set_title("SECS Magnetic Field\n{0}".format(
                    B.index[t].strftime("%Y/%m/%d %H:%M")))

    def animate(t):
        title.set_text("{0}".format(B.index[t].strftime("%Y/%m/%d %H:%M")))

        # pcolormesh
        Bx = B_gridded[t,:,0]
        By = B_gridded[t,:,1]
        B_mag = np.sqrt(Bx**2 + By**2)

        cax.set_array(B_mag)

        # Uniform grid vectors
        Bx = B_uniform[t,:,0]
        By = B_uniform[t,:,1]
        By_log, Bx_log = scale_vectors(By, Bx)
        Q.set_UVC(By_log, Bx_log)

        # Observations
        Bx = B_obs[t,:,0]
        By = B_obs[t,:,1]

        By_log, Bx_log = scale_vectors(By, Bx)
        Q_obs.set_UVC(By_log, Bx_log)

    anim = animation.FuncAnimation(fig, animate, frames=range(len(B)), interval=150)
    anim.save('daily_movie_{}.mp4'.format(time_string))

    if args.verbose:
        print("SECS animation: {:.2f} s".format(time.time()-t0))
