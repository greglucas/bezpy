"""
Magnetotelluric site classes.

"""
__all__ = ["Site", "Site1d", "Site3d"]

import sys
import datetime
import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.signal import zpk2tf, filtfilt, bilinear_zpk
import matplotlib.pyplot as plt

from .utils import apparent_resistivity
from .plotting import plot_apparent_resistivity

# ----------------------------
# Constants
# ----------------------------
MU = 4*np.pi*1e-7  # Magnetic Permeability (H/m)

# Putting this here, so we don't have to reinstantiate a client
# every time we download waveforms
_IRIS_CLIENT = None


class Site:
    """Magnetotelluric Site base class."""
    # pylint: disable=too-many-instance-attributes
    def __init__(self, name):
        self.name = name
        self.data = None
        self.longitude = None
        self.latitude = None
        # self.impulse = ImpulseResponse()

        self.periods = None
        self.Z = None
        self.Z_var = None
        self.resistivity = None
        self.resistivity_var = None
        self.phase = None
        self.phase_var = None

        self.min_period = None
        self.max_period = None

        self.samplingrate = None
        self.nimsid = None
        self.zpk = None
        self.timedelays = None

    def convolve_fft(self, mag_x, mag_y, dt=60):
        """Convolution in frequency space."""
        # pylint: disable=invalid-name

        # Note that I use rfft, because the input is real-valued. This eliminates
        # the need to calculate complex conjugates of negative frequencies.
        # To utilize normal fft you can do the following:
        #   mag_x_fft_c = np.fft.fft(mag_x, n=N)
        #   freqs_c = np.fft.fftfreq(N, d=dt)
        #   neg_freqs = freqs_c < 0.
        #   Z_c = self.calcZ(np.abs(freqs_c))
        #   Z_c[:,neg_freqs] = np.conj(Z_c[:,neg_freqs])
        #   Ex_t = np.real(np.fft.ifft(Z_c[0,:]*mag_x_fft_c)) ...
        # That produces the same results, accounting for complex conjugates of
        # the impedances at negative frequencies.

        N0 = len(mag_x)
        # Round N to the next highest power of 2 (+1 (makes it 2) to prevent circular convolution)
        N = 2**(int(np.log2(N0))+2)

        freqs = np.fft.rfftfreq(N, d=dt)
        # Z needs to be organized as: xx, xy, yx, yy
        Z_interp = self.calcZ(freqs)

        mag_x_fft = np.fft.rfft(mag_x, n=N)
        mag_y_fft = np.fft.rfft(mag_y, n=N)

        Ex_fft = Z_interp[0, :]*mag_x_fft + Z_interp[1, :]*mag_y_fft
        Ey_fft = Z_interp[2, :]*mag_x_fft + Z_interp[3, :]*mag_y_fft

        Ex_t = np.real(np.fft.irfft(Ex_fft)[:N0])
        Ey_t = np.real(np.fft.irfft(Ey_fft)[:N0])

        return (Ex_t, Ey_t)

    def calcZ(self, freqs):  # pylint: disable=invalid-name
        """Calculates transfer function, Z, from the input frequencies."""
        raise NotImplementedError("calcZ not implemented for this object yet.")

    def sysidcheck(self):
        """Returns logger and hardware, backbone from sysid and sampling rate (Hz)"""
        sysid = self.nimsid
        samplingrate = self.samplingrate

        # first search for an 'H' for the hourly time series, and get rid of it for
        # the further parsing
        logger = 'MT1'
        i = sysid.find('H')
        if not i == -1:
            logger = 'HP200'
            sysid = sysid[0:i] + sysid[i+2:]

        # parse the system ID. If it does not make sense, return an empty
        # response structure.
        j = sysid.find('-')
        if j == -1:
            sys.exit('Invalid system ID in NIMsysRsp')

        nim1 = int(sysid[0:j])
        # we know the following NIMS series
        nimlist1 = [2106, 2311, 2405, 2406, 2501, 2502, 2503, 2508, 2509, 2606, 2611,
                    2612, 1303, 1305, 1105]

        if np.isnan(nim1):
            sys.exit('NIMS ID ' + sysid + ' does not seem to be valid. Please correct.')
        elif nim1 not in nimlist1:
            sys.exit('We do not know NIMS series ' + str(nim1) + '. Check the system ID.')

        nim2str = sysid[j+2:]
        nim2 = int(nim2str)

        # and assume that the possible numbers are 1-30
        nimlist2 = range(1, 31)
        backbone = 0

        if nim2str[0] == 'B' or nim2str[0] == 'b':
            # recognized backbone NIMS ID
            backbone = 1
        elif nim2str[0] == 'A' or nim2str[0] == 'a':
            # recognized new experimental NIMS
            print('NIMS ID ' + sysid + ' is a new experimental system. Look out for clock drift.')
        elif np.isnan(nim2):
            sys.exit('NIMS ID ' + sysid + ' does not seem to be valid. Please correct.')
        elif nim2 not in nimlist2:
            sys.exit('NIMS ID ' + sysid + ' does not seem to be valid. Please correct.')

        # if 2106-1/10, assume PC104 hardware of v2001 or v2006
        hardware = 'STE'
        if (nim1 == 2106) & (nim2 <= 10):
            hardware = 'PC104'

        # if 2106-1/10 and the sampling rate is 4 Hz, assume HP200 (hourly files)
        if (nim1 == 2106) & (nim2 <= 10) & (samplingrate == 4):
            logger = 'HP200'

        # verify HP200 data logger: assuming these can only be 2106-1/10
        if logger == 'HP200':
            if (nim1 != 2106) or (nim2 > 10):
                print('A possible problem with the system ID detected. HP200 data \
                logger has been inferred, but the system is not 2106-1/10.')
                sys.exit('Please make sure ' + sysid + ' does not have an H character.')

        return logger, hardware, backbone

    def nim_sys_rsp(self):
        """reads NIMS id and sampling rate and set parameters of Butterworth filter."""

        # get logger and hardware, backbone from sysid and sampling rate (Hz)
        logger, hardware, backbone = self.sysidcheck()

        # This overrides anything about time delays in John Booker's nimsread.
        # This is a product of lengthy correspondence between Gary Egbert and
        # Barry Narod, with reference to diagrams on the NIMS firmware, and is
        # believed to be correct.
        if logger == 'HP200':    # 1 hour files, 4 Hz after decimation by nimsread
            timedelays = [-0.0055, -0.0145, -0.0235, 0.1525, 0.0275]
        elif self.samplingrate == 1:    # MT1 data logger
            timedelays = [-0.1920, -0.2010, -0.2100, -0.2850, -0.2850]
        elif self.samplingrate == 8:    # MT1 data logger
            timedelays = [0.2455, 0.2365, 0.2275, 0.1525, 0.1525]
        else:
            sys.exit('Unknown sampling rate, please check!')

        z1mag = []
        p1mag = [-6.28319+1j*10.8825, -6.28319-1j*10.8825, -12.5664]
        k1mag = 1984.31

        # based on the NIMS hardware, we determine the filter characteristics.
        if hardware == 'PC104':
            z1ele = [0.0]
            p1ele = [-3.333333E-05]
            k1ele = 1.0
        else:
            z1ele = [0.0]
            p1ele = [-1.666670E-04]
            k1ele = 1.0

        z2ele = []
        p2ele = [-3.88301+1j*11.9519, -3.88301-1j*11.9519, -10.1662+1j*7.38651,
                 -10.1662-1j*7.38651, -12.5664]
        k2ele = 313384

        # z: zero, p: pole, k:gain
        self.zpk = dict.fromkeys(['FN', 'FE', 'FZ'], {'F1': {'z': z1mag, 'p': p1mag, 'k': k1mag}})

        if backbone:   # no high pass filters
            self.zpk.update(dict.fromkeys(['QN', 'QE'],
                                          {'F1': {'z': z2ele, 'p': p2ele, 'k': k2ele}}))
        else:
            self.zpk.update(dict.fromkeys(['QN', 'QE'],
                                          {'F1': {'z': z1ele, 'p': p1ele, 'k': k1ele},
                                           'F2': {'z': z2ele, 'p': p2ele, 'k': k2ele}}))

        self.timedelays = timedelays


# ---------------------------------
# 3d EarthScope Sites
# ---------------------------------
class Site3d(Site):
    """EarthScope USArray Site"""

    def __init__(self, name):
        # Call the base site class and initialize all those variables
        super(Site3d, self).__init__(name)

        self.waveforms = None
        self.start_time = None
        self.end_time = None

        self.runlist = []
        self.runinfo = {}

    def _repr_html_(self):
        return f"<p style=\"font-size:22px; color:blue\"><b>Site 3d: {self.name}</b></p>" + \
               self.data._repr_html_()  # pylint: disable=protected-access

    # pylint: disable=too-many-arguments,too-many-locals
    def spline_interp(self, freqs, logspace=True, extrapolate=1,
                      mag_phase_interp=False, use_min_max_period=False):
        """Performs spline interpolation through the site object.

        :param logspace: Boolean, whether to interpolate in logspace or linear
        :param extrapolate: What to do with frequencies outside of the data.
            if ext=0 or ‘extrapolate’, return the extrapolated value.
            if ext=1 or ‘zeros’, return 0
            if ext=2 or ‘raise’, raise a ValueError
            if ext=3 of ‘const’, return the boundary value.
        :param mag_phase_interp: Whether to interpolate on magnitude and phase
            versus real/imaginary component
        :param use_min_max_period: Whether to use the min/max period information
            contained in the transfer function xml
        """
        # pylint: disable=unsubscriptable-object
        # pylint: disable=invalid-name
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
            # Do a spline interpolation in log-space instead

            # Need to drop nans
            good_vals = ~np.isnan(self.Z[i, :])
            # Limit the period range if the min and max periods are given
            # Note that we need to work with self.periods because the local
            # variable periods could be log-transformed
            if use_min_max_period:
                if self.min_period is not None:
                    good_vals = np.logical_and(good_vals, self.min_period < self.periods)
                if self.max_period is not None:
                    good_vals = np.logical_and(good_vals, self.periods < self.max_period)

            if self.Z_var is None:
                weights = None
                smooth_factor = 0.  # This forces interpolation through every point
            else:
                # NOTE: Z_var should never be negative, but some files have negatives
                #       in them and this is a simple fix
                weights = 1./np.sqrt(np.abs(self.Z_var[i, good_vals]))
                smooth_factor = None  # This lets the interpolation function determine the smoothing

            # Whether to interpolate on magnitude/phase of Z,
            # or on the real/imaginary components of Z
            if mag_phase_interp:
                spl_mag = interp_func(periods[good_vals], np.abs(self.Z[i, good_vals]),
                                      w=weights, ext=extrapolate, s=smooth_factor)
                spl_phase = interp_func(periods[good_vals],
                                        np.unwrap(np.angle(self.Z[i, good_vals])),
                                        w=weights, ext=extrapolate, s=smooth_factor)
                with np.errstate(invalid='ignore'):
                    Z_interp[i, :] = spl_mag(x) * np.exp(1j*np.unwrap(spl_phase(x)))
            else:
                spl = interp_func(periods[good_vals], self.Z[i, good_vals].real,
                                  w=weights, ext=extrapolate, s=smooth_factor)
                splc = interp_func(periods[good_vals], self.Z[i, good_vals].imag,
                                   w=weights, ext=extrapolate, s=smooth_factor)
                with np.errstate(invalid='ignore'):
                    Z_interp[i, :] = spl(x) + 1j*splc(x)

        # First column in the Z_interp is the DC component with infinite period, zero that out
        Z_interp[:, freqs == 0.] = 0.

        return Z_interp

    def plot_apparent_resistivity(self, interp_freqs=None):
        """Plot the apparent resistivity and phase of the transfer function."""
        xlim = [10**np.floor(np.log10(np.min(self.periods))),
                10**np.ceil(np.log10(np.max(self.periods)))]
        fig, ax_res, ax_phase = plot_apparent_resistivity(self.periods, self.Z, self.Z_var,
                                                          xlim=xlim)

        if interp_freqs is not None:
            Z = self.calcZ(interp_freqs)
            with np.errstate(divide='ignore', invalid='ignore'):
                periods = 1./interp_freqs
            # Make a line plot with the interpolated values
            plot_apparent_resistivity(periods, Z, fig=fig,
                                      ax_res=ax_res, ax_phase=ax_phase, xlim=xlim)

        # Need to leave raw string out of newlines for latex
        ax_res.set_title(r"{name}".format(name=self.name) +
                         "\n" +
                         r"Longitude: {lon:4.1f}$^\circ$ ".format(lon=self.longitude) +
                         r"Latitude: {lat:4.1f}$^\circ$".format(lat=self.latitude) +
                         "\n" +
                         ax_res.get_title())
        return (fig, ax_res, ax_phase)

    def calcZ(self, freqs):
        """Calculates transfer function, Z, from the input frequencies."""
        # extrapolate=1: bandpass filter to only interpolate between data points
        #                (no extrapolation)
        return self.spline_interp(freqs, logspace=True, extrapolate=1)

    def calc_resisitivity(self):
        """Calculate the apparent resistivity and phase of the transfer function."""
        self.resistivity, self.resistivity_var, self.phase, self.phase_var = \
            apparent_resistivity(self.periods, self.Z, self.Z_var)

    def download_waveforms(self):
        """Download the site's collected data 'waveforms' from the IRIS servers."""
        # The only reason we are putting a global in here is because
        # of how long it takes to create the Client, and with multiple
        # downloads from different sites, we really only need one client
        # to be stored somewhere in the code
        global _IRIS_CLIENT  # pylint: disable=global-statement
        if _IRIS_CLIENT is None:
            from obspy.clients.fdsn import Client
            _IRIS_CLIENT = Client("IRIS")

        # Download the stream
        # The channels are
        # E: LQN/LQE
        # B: LFN/LFE/LFZ
        stream = _IRIS_CLIENT.get_waveforms("EM", self.name, "*", "*",
                                            self.start_time, self.end_time)
        # Convert the stream to a pandas DataFrame
        self.waveforms = convert_stream_to_df(stream)
        # Channel conversion factors and renaming
        # The conversion factors were obtained from Anna Kelbert
        # and are standard for the IRIS database

        # Magnetic Field
        self.waveforms[["FE", "FN", "FZ"]] *= 0.01  # nT
        # Electric Field (assuming the dipole length as 100m)
        self.waveforms[["QN", "QE"]] *= 2.44141221047903e-05  # mV/km
        # Correcting electric Field with actual length of dipole
        for runid in self.runlist:
            try:
                mask = ((self.waveforms.index > self.runinfo[runid]['Start']) &
                        (self.waveforms.index < self.runinfo[runid]['End']))
                self.waveforms["QN"].loc[mask] *= (100.0/self.runinfo[runid]['Ex'])  # mV/km
                self.waveforms["QE"].loc[mask] *= (100.0/self.runinfo[runid]['Ey'])  # mV/km

            # If there's no info of length, then use default length.
            except KeyError:
                pass

        # Filtering waveforms by NIMS system response
        for name in self.waveforms:   # loop for FE, FN, FZ, QN, QE
            for filt in self.zpk[name]:   # loop for filter
                zval = self.zpk[name][filt]['z']
                pval = self.zpk[name][filt]['p']
                kval = self.zpk[name][filt]['k']

                # convert analog zpk to digital filter
                zval, pval, kval = bilinear_zpk(zval, pval, kval, self.samplingrate)
                b, a = zpk2tf(zval, pval, kval)

                self.waveforms[name] = filtfilt(b, a, self.waveforms[name].interpolate())

        # Renaming
        self.waveforms.rename(columns={"FE": "BE", "FN": "BN", "FZ": "BZ",
                                       "QE": "EE", "QN": "EN"},
                              inplace=True)

    def load_waveforms(self, directory="./"):
        """Load the waveform data that has already been downloaded."""
        self.waveforms = pd.read_hdf(directory + self.name + ".hdf", "waveforms")

    def save_waveforms(self, directory="./"):
        """Save the waveform data to the specified file location."""
        if self.waveforms is None:
            raise ValueError("There are no waveforms to save for site: " + self.name +
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
        """The depth of the profiles."""
        return np.cumsum(self.thicknesses)

    def _repr_html_(self):
        return "<p style=\"font-size:22px; color:blue\"><b>1d Site: {0}</b></p>".format(self.name)

    def plot_depth(self, ax=None):
        """Plots the resistivity vs. depth profile."""
        if ax is None:
            _, ax = plt.subplots()
        ax.step(self.resistivities, np.insert(self.depths, 0, 0)/1000., label=self.name)
        return ax

# Following NERC report
# http://www.nerc.com/comm/PC/Geomagnetic%20Disturbance%20Task%20Force%20GMDTF%202013/GIC%20Application%20Guide%202013_approved.pdf
    def calcZ(self, freqs):
        # pylint: disable=invalid-name
        freqs = np.asarray(freqs)
        resistivities = self.resistivities
        thicknesses = self.thicknesses

        n = len(resistivities)
        nfreq = len(freqs)

        omega = 2*np.pi*freqs
        complex_factor = 1j*omega*MU

        # eq. 5
        k = np.sqrt(1j*omega[np.newaxis, :]*MU/resistivities[:, np.newaxis])

        # eq. 6
        Z = np.zeros(shape=(n, nfreq), dtype=np.complex)
        # DC frequency produces divide by zero errors
        with np.errstate(divide='ignore', invalid='ignore'):
            Z[-1, :] = complex_factor/k[-1, :]

            # eq. 7 (reflection coefficient at interface)
            r = np.zeros(shape=(n, nfreq), dtype=np.complex)

            for i in range(n-2, -1, -1):
                r[i, :] = ((1-k[i, :]*Z[i+1, :]/complex_factor) /
                           (1+k[i, :]*Z[i+1, :]/complex_factor))
                Z[i, :] = (complex_factor*(1-r[i, :]*np.exp(-2*k[i, :]*thicknesses[i])) /
                           (k[i, :]*(1+r[i, :]*np.exp(-2*k[i, :]*thicknesses[i]))))

        # Fill in the DC impedance as zero
        if freqs[0] == 0.:
            Z[:, 0] = 0.

        # Return a 3d impedance [0, Z; -Z, 0]
        Z_output = np.zeros(shape=(4, nfreq), dtype=np.complex)
        # Only return the top layer impedance
        # Z_factor is conversion from H->B, 1.e-3/MU
        Z_output[1, :] = Z[0, :]*(1.e-3/MU)
        Z_output[2, :] = -Z_output[1, :]
        return Z_output

    def plot_apparent_resistivity(self, interp_freqs=None):
        """Plots the apparent resistivity and phase for the site."""
        if interp_freqs is None:
            raise ValueError("Need interpolation frequencies to plot the resistivity at.")

        with np.errstate(divide='ignore', invalid='ignore'):
            periods = 1./interp_freqs

        Z = self.calcZ(interp_freqs)
        xlim = [10**np.floor(np.log10(np.min(periods))), 10**np.ceil(np.log10(np.max(periods)))]
        fig, _, _ = plot_apparent_resistivity(periods, Z, xlim=xlim)

        fig.suptitle(r"1D Region: {name}".format(name=self.name), size=20)
        return fig


def convert_stream_to_df(stream):
    """Converts an obspy stream (and traces within the stream) to a dataframe."""
    cols = {}
    for trace in stream:
        # The first letter is sampling frequency, which is already in
        # the stats object. A stream could contain multiple sampling rates
        # which would name these different columns, which we don't want.
        channel = trace.stats.channel[1:]
        index = pd.date_range(start=trace.stats.starttime.datetime,
                              freq=1./trace.stats.sampling_rate*datetime.timedelta(seconds=1),
                              periods=trace.stats.npts)
        df_trace = pd.DataFrame(index=index,
                                data={channel: trace.data})

        if channel in cols:
            cols[channel] = pd.concat([cols[channel], df_trace])
        else:
            cols[channel] = df_trace

    df = pd.concat(cols.values(), axis=1)
    return df
