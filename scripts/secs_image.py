import argparse
#-----------------------------
# Command Line Options
#-----------------------------

description = "Create real-time maps of the magnetic field across the US."
parser = argparse.ArgumentParser(description=description)

parser.add_argument('infile', action='store',
                    help='Input filename with the secs data stored in npz format.')

parser.add_argument('--outdir', action='store',
                    required=False, default='.',
                    help='Where the output files will be written to. (default: ".")')

parser.add_argument('--outfile', action='store',
                    required=False, default='secs_map.png',
                    help='The output filename. (default: "secs_map.png")')

parser.add_argument('--verbose', action='store_true',
                    default=False, required=False,
                    help='Print out timing information')

args = parser.parse_args()

import datetime
import time

# Numerical
import numpy as np

# Plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

# Mapping library
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#------------------
# Defaults
#------------------
t0 = time.time()

R_earth = 6378000
Bscale = 12
arrow_width = 0.004
cmap = mpl.cm.plasma
norm = mpl.colors.LogNorm(vmin=1, vmax=1000)

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

#---------------------------
# Load the most recent data
#---------------------------

data_load = np.load(args.infile)
B_gridded = data_load['B_gridded']
B_obs = data_load['B_obs']
gridded_xy = data_load['gridded_xy']
obs_xy = data_load['obs_xy']
times = data_load['times'] # np.datetime64 array

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

#-----------------------
# Set up regular grids
#-----------------------
xs = np.unique(gridded_xy[:,0])
delta_x = xs[1]-xs[0]
ys = np.unique(gridded_xy[:,1])
delta_y = ys[1]-ys[0]
yy, xx = np.meshgrid(ys, xs)
grid_shape = xx.shape

xs = np.append((xs - delta_x/2), xs[-1] + delta_x/2)
ys = np.append((ys - delta_y/2), ys[-1] + delta_y/2)
yy_edges, xx_edges = np.meshgrid(ys, xs)

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

Q_obs = ax.quiver(obs_xy[:,0], obs_xy[:,1],
                  By_log, Bx_log, color='g',
                  transform=proj_data,
                  scale_units='inches', scale=Bscale, width=arrow_width,
                  alpha=1., zorder=10)

t = times[-1].astype(datetime.datetime)/1000000000
time_string = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M')
title = ax.set_title(f'SECS Magnetic Field\n{time_string}')

fig.savefig(args.outdir + '/' + args.outfile, bbox_inches='tight')

if args.verbose:
    print('SECS map: {:.2f} s'.format(time.time()-t0))
