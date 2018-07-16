import argparse
#-----------------------------
# Command Line Options
#-----------------------------

description = "Grid and save the magnetic field data from geomagnetic observatories"
parser = argparse.ArgumentParser(description=description)

parser.add_argument('--outdir', action='store',
                    required=False, default='.',
                    help='Where the output files will be written to. (default: ".")')

parser.add_argument('--outfile', action='store',
                    required=False, default='secs_data.npz',
                    help='The output filename. (default: "secs_data.npz")')

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
from secs import SECS
import bezpy

#------------------
# Defaults
#------------------
t0 = time.time()

R_earth = 6378000

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

times = np.array(B.index)

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

outfile = args.outdir + '/' + args.outfile
np.savez(outfile, B_gridded=B_gridded, gridded_xy=gridded_xy,
                  B_obs=B_obs, obs_xy=good_xy, times=times)
