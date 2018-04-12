"""
Magnetotelluric site classes.

"""
__all__ = ["Site", "Site1d", "Site3d"]


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import scipy.interpolate

from .utils import apparent_resistivity
from .plotting import plot_apparent_resistivity

#----------------------------
# Constants
#----------------------------
MU = 4*np.pi*1e-7; # Magnetic Permeability (H/m)

# Putting this here, so we don't have to reinstantiate a client
# every time we download waveforms
_IRIS_client = None

class Site:
    """Magnetotelluric Site base class."""

    def __init__(self, name):
        self.name = name
        self.data = None
        self.longitude = None
        self.latitude = None
        #self.impulse = ImpulseResponse()

        self.periods = None
        self.Z = None
        self.Z_var = None
        self.resistivity = None
        self.resistivity_var = None
        self.phase = None
        self.phase_var = None

        self.min_period = None
        self.max_period = None

    def convolve_fft(self, magX, magY, dt=60, extrapolate=0):
        """Convolution in frequency space."""

        # Note that I use rfft, because the input is real-valued. This eliminates
        # the need to calculate complex conjugates of negative frequencies.
        # To utilize normal fft you can do the following:
        #   magX_fft_c = np.fft.fft(magX, n=N)
        #   freqs_c = np.fft.fftfreq(N, d=dt)
        #   neg_freqs = freqs_c < 0.
        #   Z_c = self.calcZ(np.abs(freqs_c))
        #   Z_c[:,neg_freqs] = np.conj(Z_c[:,neg_freqs])
        #   Ex_t = np.real(np.fft.ifft(Z_c[0,:]*magX_fft_c)) ...
        # That produces the same results, accounting for complex conjugates of
        # the impedances at negative frequencies.

        original_N = len(magX)
        # Round N to the next highest power of 2 (+1 (makes it 2) to prevent circular convolution)
        N = 2**(int(np.log2(original_N))+2)

        freqs = np.fft.rfftfreq(N, d=dt)
        # Z needs to be organized as: xx, xy, yx, yy
        Z_interp = self.calcZ(freqs)

        magX_fft = np.fft.rfft(magX, n=N)
        magY_fft = np.fft.rfft(magY, n=N)

        Ex_fft = Z_interp[0,:]*magX_fft + Z_interp[1,:]*magY_fft
        Ey_fft = Z_interp[2,:]*magX_fft + Z_interp[3,:]*magY_fft

        Ex_t = np.real(np.fft.irfft(Ex_fft)[:original_N])
        Ey_t = np.real(np.fft.irfft(Ey_fft)[:original_N])

        return (Ex_t, Ey_t)

    def calcZ(self):
        raise NotImplementedError("calcZ not implemented for this object yet.")

#---------------------------------
# 3d EarthScope Sites
#---------------------------------
class Site3d(Site):
    """EarthScope USArray Site"""

    def __init__(self, name):
        # Call the base site class and initialize all those variables
        super(Site3d, self).__init__(name)

        self.waveforms = None

    def _repr_html_(self):
        return "<p style=\"font-size:22px; color:blue\"><b>Site 3d: {0}</b></p>".format(self.name) + \
               self.data._repr_html_()

    def spline_interp(self, freqs, logspace=True, extrapolate=1,
                      mag_phase_interp=False, use_min_max_period=False):
        # if ext=0 or ‘extrapolate’, return the extrapolated value.
        # if ext=1 or ‘zeros’, return 0
        # if ext=2 or ‘raise’, raise a ValueError
        # if ext=3 of ‘const’, return the boundary value.

        interp_func = scipy.interpolate.UnivariateSpline

        ncomponents = self.Z.shape[0]

        with np.errstate(divide='ignore', invalid='ignore'):
            interp_periods = 1./np.asarray(freqs)

        # Output Z
        Z_interp = np.zeros((ncomponents, len(interp_periods)), dtype=np.complex)

        if logspace:
            periods = np.log10(self.periods)
            x = np.log10(interp_periods)
        else:
            periods = self.periods
            x = interp_periods

        for i in range(ncomponents):
            #z_interp[i,:] = np.interp(np.log10(periods), np.log10(Z_periods), Z[i,:], left=0, right=0)
            # Do a spline interpolation in log-space instead

            # Need to drop nans
            good_vals = ~np.isnan(self.Z[i,:])
            # Limit the period range if the min and max periods are given
            # Note that we need to work with self.periods because the local
            # variable periods could be log-transformed
            if use_min_max_period == True:
                if self.min_period is not None:
                    good_vals = np.logical_and(good_vals, self.min_period < self.periods)
                if self.max_period is not None:
                    good_vals = np.logical_and(good_vals, self.periods < self.max_period)

            if self.Z_var is None:
                weights = None
                smooth_factor = 0. # This forces interpolation through every point
            else:
                # NOTE: Z_var should never be negative, but some files have negatives
                #       in them and this is a simple fix
                weights = 1./np.sqrt(np.abs(self.Z_var[i,good_vals]))
                smooth_factor = None # This lets the interpolation function determine the smoothing

            # Whether to interpolate on magnitude/phase of Z,
            # or on the real/imaginary components of Z
            if mag_phase_interp:
                spl_mag = interp_func(periods[good_vals], np.abs(self.Z[i,good_vals]),
                                      w=weights, ext=extrapolate, s=smooth_factor)
                spl_phase = interp_func(periods[good_vals], np.unwrap(np.angle(self.Z[i,good_vals])),
                                      w=weights, ext=extrapolate, s=smooth_factor)
                with np.errstate(invalid='ignore'):
                    Z_interp[i,:] = spl_mag(x) * np.exp(1j*np.unwrap(spl_phase(x)))
            else:
                spl = interp_func(periods[good_vals], self.Z[i,good_vals].real,
                                  w=weights, ext=extrapolate, s=smooth_factor)
                splc = interp_func(periods[good_vals], self.Z[i,good_vals].imag,
                                   w=weights, ext=extrapolate, s=smooth_factor)
                with np.errstate(invalid='ignore'):
                    Z_interp[i,:] = spl(x) + 1j*splc(x)

        # First column in the Z_interp is the DC component with infinite period, zero that out
        Z_interp[:,freqs == 0.] = 0.

        return Z_interp

    def plot_apparent_resistivity(self, interp_freqs=None):
        xlim = [10**np.floor(np.log10(np.min(self.periods))), 10**np.ceil(np.log10(np.max(self.periods)))]
        fig, ax_res, ax_phase = plot_apparent_resistivity(self.periods, self.Z, self.Z_var, xlim=xlim)

        if interp_freqs is not None:
            Z = self.calcZ(interp_freqs)
            with np.errstate(divide='ignore', invalid='ignore'):
                periods = 1./interp_freqs
            # Make a line plot with the interpolated values
            plot_apparent_resistivity(periods, Z, fig=fig, ax_res=ax_res, ax_phase=ax_phase, xlim=xlim)

        # Need to leave raw string out of newlines for latex
        ax_res.set_title(r"{name}".format(name=self.name) +
                     "\n" +
                     r"Longitude: {lon:4.1f}$^\circ$ ".format(lon=self.longitude) +
                     r"Latitude: {lat:4.1f}$^\circ$".format(lat=self.latitude) +
                     "\n" +
                     ax_res.get_title())
        return (fig, ax_res, ax_phase)

    def calcZ(self, freqs):
        # extrapolate=1: bandpass filter to only interpolate between data points (no extrapolation)
        return self.spline_interp(freqs, logspace=True, extrapolate=1)

    def calc_resisitivity(self):
        self.resistivity, self.resistivity_var, self.phase, self.phase_var = \
        apparent_resistivity(self.periods,self.Z, self.Z_var)

    def download_waveforms(self):
        # Only reason we are putting a global in here is because
        # of how long it takes to create the Client, and with multiple
        # downloads from different sites, we really only need one client
        # to be stored somewhere in the code
        global _IRIS_client
        if _IRIS_client is None:
            from obspy.clients.fdsn import Client
            _IRIS_client = Client("IRIS")

        # Download the stream
        # The channels are
        # E: LQN/LQE
        # B: LFN/LFE/LFZ
        stream = _IRIS_client.get_waveforms("EM", self.name, "*", "*",
                                        self.start_time, self.end_time)
        # Convert the stream to a pandas DataFrame
        self.waveforms = convert_stream_to_df(stream)
        # Channel conversion factors and renaming
        # The conversion factors were obtained from Anna Kelbert
        # and are standard for the IRIS database

        # Magnetic Field
        self.waveforms[["LFE", "LFN", "LFZ"]] *= 0.01 # nT
        # Electric Field
        self.waveforms[["LQN","LQE"]] *= 2.44141221047903e-05 # mV/km
        # Renaming
        self.waveforms.rename(columns={"LFE": "By", "LFN": "Bx", "LFZ": "Bz",
                                       "LQE": "Ey", "LQN": "Ex"},
                              inplace=True)

    def load_waveforms(self, directory="./"):
        """Load the waveform data that has already been downloaded."""
        self.waveforms = pd.read_hdf(directory + self.name + ".hdf", "waveforms")

    def save_waveforms(self, directory="./"):
        """Save the waveform data to the specified file location."""
        if self.waveforms is None:
            raise ValueError("There are no waveforms to save for site: " + site.name +
                             "\nYou can download waveforms from the IRIS database by calling " +
                             ".download_waveforms()")
        self.waveforms.to_hdf(directory + self.name + ".hdf", "waveforms")

class Site1d(Site):
    """MT class for a 1D Fernberg profile"""
    def __init__(self, name, thicknesses, resistivities):
        # Call the base site class and initialize all those variables
        super(Site1d, self).__init__(name)

        self.thicknesses = np.array(thicknesses)
        self.resistivities = np.array(resistivities)

    @property
    def depths(self):
        return np.cumsum(self.thicknesses)

    def _repr_html_(self):
        return "<p style=\"font-size:22px; color:blue\"><b>1d Site: {0}</b></p>".format(self.name)

    def plotDepth(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.step(self.resistivities, np.insert(self.depths, 0, 0)/1000., label=self.name)
        return ax

# Following NERC report
# http://www.nerc.com/comm/PC/Geomagnetic%20Disturbance%20Task%20Force%20GMDTF%202013/GIC%20Application%20Guide%202013_approved.pdf
    def calcZ(self, freqs):
        freqs = np.asarray(freqs)
        resistivities = self.resistivities
        thicknesses = self.thicknesses

        n = len(resistivities)
        nfreq = len(freqs)

        omega =  2*np.pi*freqs
        complex_factor = 1j*omega*MU

        # eq. 5
        k = np.sqrt(1j*omega[np.newaxis,:]*MU/resistivities[:,np.newaxis])

        # eq. 6
        Z = np.zeros(shape=(n, nfreq), dtype=np.complex)
        # DC frequency produces divide by zero errors
        with np.errstate(divide='ignore', invalid='ignore'):
            Z[-1,:] = complex_factor/k[-1,:]

            # eq. 7 (reflection coefficient at interface)
            r = np.zeros(shape=(n, nfreq), dtype=np.complex)

            for i in range(n-2,-1,-1):
                r[i,:] = (1-k[i,:]*Z[i+1,:]/complex_factor) / (1+k[i,:]*Z[i+1,:]/complex_factor)
                Z[i,:] = complex_factor*(1-r[i,:]*np.exp(-2*k[i,:]*thicknesses[i]))/(k[i,:]*
                                              (1+r[i,:]*np.exp(-2*k[i,:]*thicknesses[i])))

        # Fill in the DC impedance as zero
        if freqs[0] == 0.:
            Z[:,0] = 0.

        # Return a 3d impedance [0, Z; -Z, 0]
        Z_output = np.zeros(shape=(4,nfreq), dtype=np.complex)
        # Only return the top layer impedance
        # Z_factor is conversion from H->B, 1.e-3/MU
        Z_output[1,:] = Z[0,:]*(1.e-3/MU)
        Z_output[2,:] = -Z_output[1,:]
        return Z_output

    def plot_apparent_resistivity(self, interp_freqs=None):

        if interp_freqs is None:
            raise ValueError("Need interpolation frequencies to plot the resistivity at.")

        with np.errstate(divide='ignore', invalid='ignore'):
            periods = 1./interp_freqs

        Z = self.calcZ(interp_freqs)
        xlim = [10**np.floor(np.log10(np.min(periods))), 10**np.ceil(np.log10(np.max(periods)))]
        fig, ax_res, ax_phase = plot_apparent_resistivity(periods, Z, xlim=xlim)

        fig.suptitle(r"1D Region: {name}".format(name=self.name), size=20)
        return fig

def convert_trace_to_df(trace):
    dt = datetime.timedelta(seconds=trace.stats.sampling_rate)
    index = pd.to_datetime(trace.stats.starttime.datetime + trace.times()*dt)
    df = pd.DataFrame(index=index, data={trace.stats.channel: trace.data})
    return df

def convert_stream_to_df(stream):
    cols = {}
    for trace in stream:
        df_trace = convert_trace_to_df(trace)
        if trace.stats.channel in cols:
            cols[trace.stats.channel] = pd.concat([cols[trace.stats.channel], df_trace])
        else:
            cols[trace.stats.channel] = df_trace

    df = pd.concat(cols.values(), axis=1)
    return df
