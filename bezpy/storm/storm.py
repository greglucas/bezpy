import warnings
import os
from datetime import timedelta
import datetime

from typing import Tuple, List

import numpy as np
import xarray as xr

from bezpy.mag import GeomagObservatory, USGS_CODES, NRCAN_CODES
from bezpy.mag import get_iaga_observatory

from pysecs import SECS

EARTH_RADIUS = 6371e3

def get_obs_location_from_iaga_data(observatories, good_observatories) -> Tuple[np.ndarray, np.ndarray]:
    good_xy = []
    missing_xy = []

    for obs in observatories:
        site = get_iaga_observatory(obs.name)
        lonlat = (site['longitude'], site['latitude'])
        if obs in good_observatories:
            good_xy.append(lonlat)
        else:
            missing_xy.append(lonlat)

    good_xy = np.array(good_xy)
    missing_xy = np.array(missing_xy)

    return good_xy, missing_xy


class Storm:
    def __str__(self) -> str:
        return f"<Storm - {self.date}>"

    def __init__(self,
                 date : datetime.datetime, 
                 solar_cycle : int,
                 obs_data_dir : str,
                 kp : float = None,
                 dt : float = None,
                 storm_window_size_days : int = 6,
                 call_setup:bool=True):
        self.solar_cycle : int = solar_cycle
        self.cycle : int = solar_cycle
        self.kp : float = kp
        self.dt : float = dt
        self.storm_window_size : int = timedelta(days=storm_window_size_days)

        self._center_date : datetime.datetime = date
        self._start_datetime : datetime.datetime = None
        self._end_datetime : datetime.datetime = None

        self.data_location : str = obs_data_dir
        # Internal Book Keeping TODO: Rename these to something more explicit
        self.ds : xr.Dataset = None
        self.ds_pred : xry.Dataset = None
        self.obs : List[GeomagObservatory] = []
        self.include_year_in_obs_path : bool = False

        if call_setup:
            self.setup()

    def setup(self) -> None:
        """ Setup the Storm

        Currently just caclulate out start and end dates from the window
        """
        self._center_date.replace(hour=0, minute=0, second=0)
        self.start_datetime = self._center_date - self.storm_window_size
        self.end_datetime = self._center_date + self.storm_window_size

    @property
    def date(self) -> str:
        return self._center_date.strftime("%Y-%m-%d")

    @property
    def start_date(self) -> str:
        return self.start_datetime.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def end_date(self) -> str:
        return self.end_datetime.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def year(self) -> int:
        return self._center_date.year

    @property
    def month(self) -> int:
        return self._center_date.month

    @property
    def day(self) -> int:
        return self._center_date.day

    @property
    def magnetic(self) -> xr.Dataset:
        return self.ds

    @property
    def magnetic_data(self) -> xr.Dataset:
        return self.ds

    def save_data_to_csv(self, fname : str) -> str:
        self.csv_fname = fname

        print(f'Writing output into: {self.csv_fname}')
        with open(self.csv_fname, 'w') as file:
            file.write('site,latitude,longitude,Emax,E Norm-of-Mean,E Mean-of-Norm\n')
            for site in self.ds_pred.site.values:
                ds_slice = self.ds_pred.sel(site=site)
                line = f"{ds_slice['site'].values}, {ds_slice['latitude'].values}, {ds_slice['longitude'].values}, {ds_slice['Emax'].values}, {ds_slice['E_norm_of_mean'].values}, {ds_slice['E_mean_of_norm'].values}"
                file.write(line+'\n')

            file.write('Metadata\n')
            file.write(f'Date Created:, {str(datetime.datetime.now())}')
            file.write(f'Created By:, {'Miles Curry'}')

        return self.csv_fname

    def save_storm_to_netcdf(self, fname) -> str:
        """ Save the predicted storm data to netcdf """
        self.ds_pred.to_netcdf(fname)
        return fname

    def close(self) -> None:
        """ Close all open data objects related to this storm """

        # TODO: This somehwat works and needs to be updated
        for obs in self.obs:
            obs.close()

        self.ds.close() 

    def setup_obs_magnetic_data(self, server_url="https://geomag.usgs.gov",
                                download_from_iris : bool=False,
                                percent_nan_crit : float = 10.0) -> Tuple[List[str], List[str], dict, np.ndarray, np.ndarray]:
        """ Retrieve the magnetic data by either downloading them from iris or reading them 
        from NetCDF files at self.data_location.
        """

        start_date = self.start_date
        end_date = self.end_date
    
        if download_from_iris:
            timedelta_days = end_date - start_date
            observatories, _ = download_from_iris(sever=server_url, start_date=start_date, timedelta_days=timedelta_days)
        else:
            observatories, missing_obs = self.read_from_data(nan_percent_criteria=percent_nan_crit)
    
        good_observatories = []
        obs_dict = {}
    
        good_xy, missing_xy = get_obs_location_from_iaga_data(observatories, good_observatories)   

        print("MISSING OBS:", missing_obs)
    
        # We need to clean up this return...

        return good_observatories, missing_obs, obs_dict, good_xy, missing_xy

    # TODO: Think of a better name than this
    def read_from_data(self, 
                       extra_delta_t : timedelta=timedelta(days=7),
                       nan_percent_criteria : float = 50.0) -> Tuple[List[GeomagObservatory], List[GeomagObservatory]]:
        """ Reall all observatories for this year that are stored on disk in NetCDF files """

        all_obs_dirs = os.listdir(self.data_location)
        missing_obs = []

        print(f"Storm - {self.date} reading data from {self.start_date} to {self.end_date}... ")
        print(f"Plus grabbing {extra_delta_t} of extra days..")
        print(f"Reading: ", end=' ')

        obs_list = []
        obs_data = {}

        for obs_code in all_obs_dirs:
            obs_code : str

            fname = self.generate_obs_file(obs_code)
            
            obs = GeomagObservatory(obs_code)
    
            if not os.path.isfile(fname):
                print(f"{fname} does not exist :(")
                missing_obs.append(f'{obs_code} (NA);')
                continue
    
            df = xr.load_dataset(fname)
            df = df.sel(time=slice(self.start_datetime - extra_delta_t, 
                                   self.end_datetime + extra_delta_t))

            is_valid, obs_missing_percent = self._is_geo_mag_obs_valid(obs_code, df, nan_percent_criteria)

            if not is_valid:
                missing_obs.append(f'{obs_code} ({obs_missing_percent:.0f}%);')
                print(f"Skipping - Invalid: {obs_code} - obs_missing_percent")
                df.close()
                continue

            obs.df = df
            obs_list.append(obs)
            obs_data[obs_code] = obs.df

        new_data = [ds.sel(time=slice(self._center_date - extra_delta_t, self._center_date + extra_delta_t)).drop_duplicates(dim="time") for ds in obs_data.values()]
        for obs_ds in new_data:
            # If the slice isn't empty
            if obs_ds["Bx"].size == 0 or np.all(np.isnan(obs_ds["Bx"])):
                continue

        if len(new_data) == 0:
            print("This storm is completly empty!")
            raise Exception("Storm is completly empty!")

        ds = xr.concat(new_data, dim="observatory")
        ds = ds.assign_coords(observatory=list(obs_data.keys()))
        ds = ds.transpose("time", "observatory")
        ds = ds.interpolate_na(dim="time", method="linear", fill_value="extrapolate")
        ds = self.rm_poly(ds, dim="time", deg=1)

        self.ds = ds 
        self.obs_data = obs_data

        return obs_list, missing_obs

    # Probably can be deleted as we are now interpolaiting...
    def _is_geo_mag_obs_valid(self, obs_code : str, df : xr.Dataset, nan_criteria : float = 50.0) -> bool:
        """ Removes obs if they have are missing, missing data completly, or have nan_critical or greater percent nans
        missing """
        bx_is_empty = (df['Bx'].isnull()).all().item()
        by_is_empty = (df['By'].isnull()).all().item()
        bz_is_empty = (df['Bz'].isnull()).all().item()

        something_empty = False

        if bx_is_empty:
            print(f"{obs_code}'s Bx is empty!", end=' ')
            something_empty = True
        
        if by_is_empty:
            print(f"{obs_code}'s By is empty!", end=' ')
            something_empty = True

        if bz_is_empty:
            print(f"{obs_code}'s Bz is empty!", end=' ')
            something_empty = True

        if something_empty:
            return False, 100
            
        total_times = len(df['time'])
        nnan_bx_percent = (df['Bx'].isnull().sum().values / total_times) * 100
        nnan_by_percent = (df['By'].isnull().sum().values / total_times) * 100
        nnan_bz_percent = (df['Bz'].isnull().sum().values / total_times) * 100

        too_many_nans = False

        if nnan_bx_percent >= nan_criteria:
            print(f'Bx has {nnan_bx_percent:.1f}% NaNs! Total Times: {total_times}')
            too_many_nans = True
            
        if nnan_by_percent >= nan_criteria:
            print(f'By has {nnan_by_percent:.1f}% NaNs! Total Times: {total_times}')
            too_many_nans = True
            
        if nnan_bz_percent >= nan_criteria:
            print(f'Bz has {nnan_bz_percent:.1f}% NaNs! Total Times: {total_times}')
            too_many_nans = True

        if too_many_nans:
            return False, max(nnan_bx_percent, nnan_by_percent, nnan_bz_percent)

        return True, 0

    
    def generate_obs_file(self, obs_code : str):
        """ Given an obs_code - generate the NetCDF file name for it """
        fname = f'{obs_code.lower()}_min_{self.year}.nc'
        if self.include_year_in_obs_path:
            # TODO: This is a hold-over for data that I stored that included the year in the file name
            fname_path = os.path.join(self.data_location, obs_code, str(self.year), fname)
        else:
            fname_path = os.path.join(self.data_location, obs_code, fname)
        
        return fname_path

    
    def rm_poly(self, ds, dim='time', deg=3, **kwargs):
        """Remove degree polynomial along dimension dim from ds."""
        coefficients = ds.polyfit(dim, deg=deg, **kwargs)
        coord = ds[dim]
        fits = []
        if isinstance(ds,xr.Dataset):
            print('RM Poly dataset')
            for v in coefficients:
                name = v.replace("_polyfit_coefficients", "")
                fit = xr.polyval(coord, coefficients[v]).rename(name)
                fits.append(fit)
            fits = xr.merge(fits)
        elif isinstance(ds,xr.DataArray):
            print('RM Poly dataset')
            name = ds.name
            v=list(coefficients.data_vars)[0]
            fits = xr.polyval(coord, coefficients[v]).rename(name)
        ds_rm_poly =  ds - fits
        return ds_rm_poly

    def setup_secs(self,
                   lat_bounds: List[float],
                   lon_bounds : List[float],
                   nLat : int = 36, 
                   nLon : int = 76,
                   EARTH_RADIUS : float = EARTH_RADIUS+110000, 
                   indexing : str='ij') -> SECS:
        """ Setup and define SECS """

        lat, lon, r = np.meshgrid(np.linspace(lat_bounds[0], lat_bounds[1], nLat),
                                  np.linspace(lon_bounds[0], lon_bounds[1], nLon),
                                  EARTH_RADIUS+110000, indexing=indexing)
        secs_lat_lon_r = np.hstack((lat.reshape(-1, 1),
                                    lon.reshape(-1, 1),
                                    r.reshape(-1,1)))
        self.SECS = SECS(sec_df_loc=secs_lat_lon_r)

        return self.SECS

    def caclulate_SECS(self, secs, B, obs_xy, pred_xy, J=False):
        """
        Calculate SECS output magnetic field
        
        secs:          
        B shape:       (ntimes, nobs, 3 (Bx By Bz)) or
                       (ntimes, nobs, 6 (Bx By Bz Varx Vary Varz))
                       input B observations, and optionally variances
                       
        obs_xy shape:  (nobs, 2 (lat, lon))
                       input location of B observations
                       
        pred_xy shape: (npred, 2 (lat, lon))
                       desired output locations
        
        J:             (Boolean) 
                       if True, calculate current densities, default is False
        """
        if obs_xy.shape[0] != B.shape[1]:
            raise ValueError(f"Number of observation points [{obs_xy.shape[0]}] "
                             f"doesn't match B input [{B.shape[1]}]")
            
        obs_lat_lon_r = np.zeros((len(obs_xy), 3))
        obs_lat_lon_r[:, 0] = obs_xy[:, 0]
        obs_lat_lon_r[:, 1] = obs_xy[:, 1]
        obs_lat_lon_r[:, 2] = EARTH_RADIUS
        
        if B.shape[2] == 3:
            # observation error standard deviation is 1nT
            # Don't include Z component in fits for now
            obs_std = np.ones(B.shape)
            obs_std[:, :, 2] = np.inf
                    
        elif B.shape[2] == 6:
            # use observation errors passed with B
            obs_std = B[:, :, 3:]
            B = B[:, :, :3]
            
        else:
            raise ValueError("Observations should have 3 (BxByBz) or 6 (BxByBzVxVyVz) elements")


        print('Nan obs:', nan_obs)

        # TODO: Filtering OBS here is dangerious, as we can lose obs here..
        # Filter out observatories with any NaNs
        # maybe we should remove?
        # nan_obs = np.any(np.isnan(B), axis=(0, 2))
        # B = B[:, ~nan_obs, :]
        # obs_std = obs_std[:, ~nan_obs, :]
        # obs_lat_lon_r = obs_lat_lon_r[~nan_obs, :]
        
        # specify prediction points
        pred_lat_lon_r = np.zeros((len(pred_xy), 3))
        pred_lat_lon_r[:, 0] = pred_xy[:, 0]
        pred_lat_lon_r[:, 1] = pred_xy[:, 1]
        pred_lat_lon_r[:, 2] = EARTH_RADIUS

        # fit secs to data
        secs.fit(obs_loc=obs_lat_lon_r, obs_B=B,
                 obs_std=obs_std, epsilon=0.05)
        
        # predict at desired locations
        B_pred = secs.predict_B(pred_lat_lon_r)
        if J:
            J_pred = secs.predict_J(pred_lat_lon_r + [[0, 0, 110000]])
        
        if J:
            return B_pred, J_pred
        else:
            return B_pred

    def calculate_geoelectric_field_at_mt_sites(self,
                                                secs : SECS = None,
                                                MT_sites : List[Site3D], obs_data, drive_by_obs : str = None):
        """ Calculate the electric field at MT sites"""
        ds = self.ds

        input_B = np.stack([ds["Bx"], ds["By"], ds["Bz"]], axis=-1)
        input_xy = np.array([(obs.attrs["Latitude"], obs.attrs["Longitude"]) for obs in self.obs_data.values()])
        site_xys = np.array([(site.latitude, site.longitude) for site in MT_sites])

        if secs is None:
            self.secs = self.setup_secs()
        else:
            self.secs = secs
        
        pred_B = self.calculate_SECS(secs, input_B, input_xy, site_xys)
        pred_E = np.zeros((len(input_B), len(site_xys), 2))

        for i, site in enumerate(MT_sites):
            site.data = site.data.sort_index()
            with (warnings.catch_warnings(), np.errstate(divide="ignore")):
                warnings.simplefilter("ignore")
                # Convolve B to impedance sites
                Ex, Ey = site.convolve_fft(pred_B[:, i, 0], pred_B[:, i, 1], dt=60)

            pred_E[:, i, 0] = Ex
            pred_E[:, i, 1] = Ey

        nclip = 24 * 60
        self.ds_pred = xr.Dataset(
            {
                "Bx": (("time", "site"), pred_B[..., 0]),
                "By": (("time", "site"), pred_B[..., 1]),
                "Ex": (("time", "site"), pred_E[..., 0]),
                "Ey": (("time", "site"), pred_E[..., 1]),
                # It might be more appropriate to put these calculations in another function?
                "Bmax": (("site",), np.sqrt(pred_B[nclip:-nclip, :, 0]**2 + pred_B[nclip:-nclip, :, 1]**2).max(axis=0)),
                "Emax": (("site",), np.sqrt(pred_E[nclip:-nclip, :, 0]**2 + pred_E[nclip:-nclip, :, 1]**2).max(axis=0)),
            },
            coords={
                "time": ds.time,
                "site": [site.name for site in MT_sites],
                "latitude": (("site",), site_xys[:, 0]),
                "longitude": (("site",), site_xys[:, 1]),
            },
        )
    
        # Assign Units
        self.ds_pred['Bx'].attrs['units'] = 'nT'
        self.ds_pred['By'].attrs['units'] = 'nT'
        self.ds_pred['Ex'].attrs['units'] = 'V/km'
        self.ds_pred['Ey'].attrs['units'] = 'V/km'
        self.ds_pred['Bmax'].attrs['units'] = 'nT'
        self.ds_pred['Emax'].attrs['units'] = 'nT'

        self.ds_pred = ds_pred
        return self.ds_pred, obs_used

    def calculate_boxcar_norm_of_mean(self, time_window_size=60, center=False) -> Tuple[np.ndarray, np.ndarray]:
        """ Calculate the norm_of_mean boxcar for a given time window """
        Ex = self.ds_pred['Ex']
        Ey = self.ds_pred['Ey']

        nclip = 24 * 60
        avg_Ex = Ex.rolling(time=time_window_size, center=center).mean()
        avg_Ey = Ey.rolling(time=time_window_size, center=center).mean()
        e_norm_of_mean_rolling = np.hypot(avg_Ex, avg_Ey)
        max_e_norm_of_mean = e_norm_of_mean_rolling.isel(time=slice(nclip, -nclip)).max(dim='time', skipna=True)

        self.ds_pred['E_norm_of_mean'] = max_e_norm_of_mean
        self.ds_pred['E_norm_of_mean_rolling'] = e_norm_of_mean_rolling
        
        return max_e_norm_of_mean, e_norm_of_mean_rolling

    def calculate_boxcar_mean_of_norm(self,
                                      time_window_size=60, # Minutes
                                      center=False):
        Ex = self.ds_pred['Ex']
        Ey = self.ds_pred['Ey']
        
        nclip = 24 * 60
        norm = np.hypot(Ex[:,:], Ey[:,:])
        rolling_avg_norm = norm.rolling(time=time_window_size, center=center).mean()
        max_avg_norm = rolling_avg_norm.isel(time=slice(nclip, -nclip)).max(dim='time', skipna=True)
        self.ds_pred['E_mean_of_norm'] = max_avg_norm
        self.ds_pred['E_mean_of_norm_rolling'] = rolling_avg_norm
        return max_avg_norm, rolling_avg_norm

    def plot_mag(self, mag_data, show=True, fname='./output/timesries-obs-plot.png'):
        pass

    def add_emax_to_plot(self, var, ax, proj_data, norm, cmap):
        print(f"Adding storm {self} to plot for ({var})...")
        Eh = self.ds_pred[var]
        lat = self.ds_pred['latitude']
        lon = self.ds_pred['longitude']
        
        return ax.scatter(lon, lat,
                s=10,
                c=Eh,
                norm=norm,
                transform=proj_data,
                cmap=cmap,
                plotnonfinite=True
                )
        
    def plot_emax(self, show_fig=False, save_fig=False):
        """ """
        pass

    def plot_mean_of_norm(self):
        """ """
        pass

    def plot_norm_of_mean(self):
        """ """
        pass

class SolarCycle:
    """ A class to hold Storms and allow for easier plotting.

    Currently, this class really only does plotting, but perhaps it could be used
    later to kick off storm processing, although, somtimes it's eaiser to just have
    storms do their own processing.

    I actually don't think I am even the plotting functions here in my code, at the
    moment. I am just eseentially plotting in manually, but this would be very helpful.

    Perhaps too this might be more general as 'StormCollection'.
    """ 
    def __init__(self, cycle_number : int,
                 storms : List[Storm]):
        self.cycle_number : int = cycle_number
        self.storms : List[Storm] = storms

        # Plotting Things
        self.nCols : int = 0
        self.nRows : int = 0
        self.iCol  : int = 0
        self.iRow  : int = 0


    def __str__(self) -> str:
        return f'<SolarCycle: {self.cycle_number} - nStorms: {self.nStorms}>'

    @property
    def nStorms(self) -> int:
        return len(self.storms)

    @property
    def cycle(self) -> int:
        return self.cycle_number

    def add_storm(self, storm : Storm):
        self.storms.append(storm)

    def setup_emax_plot(self, nCols=4):
        title = f"Emax's Solar Cycle #{self.cycle_number} - nStroms: {self.storms}"
        # No more than nCols columns
        self.nCols = nCols
        self.nRows = self.nStorms // self.nCols

    def add_storm_to_plot(self, storm : Storm, array_name : str):
        """  """
        print(f'C: {self.nRows} R: {self.nCols} - Index: {self.iRow} {self.iCol}')
        
        ax = self.fig.add_subplot(self.gs[self.iRow, self.iCol], projection=self.proj_data)
        storm.add_emax_to_plot(array_name, 
                               ax,
                               self.proj_data, self.norm, self.cmap)

        self.iCol += 1
        if self.iCol == self.nCols:
            self.iCol = 0
            self.iRow += 1

    def show_plot(self):
        self.fig.save(f'test_{self.cycle_number}.png')

    def save_plot(self, fname):
        self.fig.save(f'test_{self.cycle_number}.png')
        
    def plot_hourly_mean_of_norm(self):
        pass

    def plot_hourly_norm_of_mean(self):
        pass
