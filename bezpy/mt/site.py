"""
Magnetotelluric site classes.

"""
__all__ = ["Site", "Site1d", "Site3d", "SiteCollection", "ConductivityModel"]

import pathlib
import warnings

import numpy as np
import pandas as pd
import scipy.fft
import scipy.interpolate

from .utils import apparent_resistivity

# ----------------------------
# Constants
# ----------------------------
MU = 4*np.pi*1e-7  # Magnetic Permeability (H/m)


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

        freqs = scipy.fft.rfftfreq(N, d=dt)
        # Z needs to be organized as: xx, xy, yx, yy
        Z_interp = self.calcZ(freqs)

        mag_x_fft = scipy.fft.rfft(mag_x, n=N, workers=-1)
        mag_y_fft = scipy.fft.rfft(mag_y, n=N, workers=-1)

        Ex_fft = Z_interp[0, :]*mag_x_fft + Z_interp[1, :]*mag_y_fft
        Ey_fft = Z_interp[2, :]*mag_x_fft + Z_interp[3, :]*mag_y_fft

        Ex_t = np.real(scipy.fft.irfft(Ex_fft, workers=-1)[:N0])
        Ey_t = np.real(scipy.fft.irfft(Ey_fft, workers=-1)[:N0])

        return (Ex_t, Ey_t)

    def calcZ(self, freqs):  # pylint: disable=invalid-name
        """Calculates transfer function, Z, from the input frequencies."""
        raise NotImplementedError("calcZ not implemented for this object yet.")


# ---------------------------------
# 3d EarthScope Sites
# ---------------------------------
class Site3d(Site):
    """EarthScope USArray Site"""

    def __init__(self, name):
        # Call the base site class and initialize all those variables
        super().__init__(name)

        self.xml = None
        self.product_id = None
        self.waveforms = None
        self.start_time = None
        self.end_time = None

        # Metadata filled in by the readers (read_xml/read_zfile/read_edi);
        # initialized here so every Site3d has a well-defined shape.
        self.network_code = "EM"
        self.elevation = None
        self.declination = None
        self.channel_orientation = {}
        self.rating = None
        self.quality_flag = None
        self.sign_convention = 1

        self.datalogger = None

        # Keep track of the previous calculated Z values
        # to speed up multiple calls with the same frequency range
        self._prev_freqs = None
        self._prev_Z = None
        self._prev_extrapolate = None

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
        Z_interp = np.zeros((ncomponents, len(interp_periods)), dtype=complex)

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
        # lazy-load this so we don't require matplotlib to be installed by default
        from .plotting import plot_apparent_resistivity
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
        ax_res.set_title(rf"{self.name}" +
                         "\n" +
                         rf"Longitude: {self.longitude:4.1f}$^\circ$ " +
                         rf"Latitude: {self.latitude:4.1f}$^\circ$" +
                         "\n" +
                         ax_res.get_title())
        return (fig, ax_res, ax_phase)

    def calcZ(self, freqs, extrapolate="zeros"):
        """Calculates transfer function, Z, from the input frequencies.

        :param extrapolate: how to treat frequencies outside the measured
            period band.

            * ``"zeros"`` (default, the historical behavior): a hard band-stop,
              Z = 0 outside the band.
            * ``"asymptotic"``: physical band-edge continuation. |Z| follows
              the half-space sqrt(omega) asymptote anchored at the band-edge
              value (phase held at the edge), decaying to Z = 0 at DC on the
              long-period side and growing as sqrt(f) on the short-period
              side. This preserves the short-period content that dominates
              dB/dt-driven E instead of truncating it.
        """
        if extrapolate not in ("zeros", "asymptotic"):
            raise ValueError("extrapolate must be one of: zeros, asymptotic")
        if not np.array_equal(freqs, self._prev_freqs) or extrapolate != self._prev_extrapolate:
            # If the inputs are different, then recalculate the Z values
            # and store them for later use. Store a copy so that later in-place
            # mutation of the caller's array can't silently invalidate the cache.
            # ext=1 in the spline: zero outside the measured band, then
            # optionally fill the out-of-band frequencies with the asymptote.
            Z_interp = self.spline_interp(freqs, logspace=True, extrapolate=1)
            if extrapolate == "asymptotic":
                Z_interp = self._extrapolate_asymptotic(np.asarray(freqs, dtype=float), Z_interp)
            self._prev_Z = Z_interp
            self._prev_freqs = np.array(freqs)
            self._prev_extrapolate = extrapolate

        return self._prev_Z

    def _extrapolate_asymptotic(self, freqs, Z_interp):
        """Fill out-of-band frequencies with a sqrt(f) continuation of Z.

        For a half space, |Z| grows exactly as sqrt(omega); anchoring that
        asymptote at each component's band-edge value (keeping the edge phase)
        is continuous at the edge and has the right DC limit (Z -> 0).
        """
        # pylint: disable=invalid-name
        for i in range(Z_interp.shape[0]):
            good_periods = self.periods[~np.isnan(self.Z[i, :])]
            f_lo, f_hi = 1./np.max(good_periods), 1./np.min(good_periods)
            # Evaluate the spline just inside the band so ext=1 can't zero it
            Z_lo, Z_hi = self.spline_interp([f_lo*(1 + 1e-9), f_hi*(1 - 1e-9)],
                                            logspace=True, extrapolate=1)[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                high = freqs > f_hi
                Z_interp[i, high] = Z_hi * np.sqrt(freqs[high]/f_hi)
                low = (freqs < f_lo) & (freqs > 0.)
                Z_interp[i, low] = Z_lo * np.sqrt(freqs[low]/f_lo)
        return Z_interp

    def calc_resistivity(self):
        """Calculate the apparent resistivity and phase of the transfer function."""
        self.resistivity, self.resistivity_var, self.phase, self.phase_var = \
            apparent_resistivity(self.periods, self.Z, self.Z_var)

    def calc_resisitivity(self):
        """Deprecated misspelling of :meth:`calc_resistivity`."""
        warnings.warn("calc_resisitivity is deprecated (misspelling); "
                      "use calc_resistivity", DeprecationWarning, stacklevel=2)
        self.calc_resistivity()

    def download_waveforms(self):
        """Download the site's collected data 'waveforms' from the IRIS servers."""
        # If there is no runlist in the datalogger, then it wasn't able to be
        # initialized, so raise that this is unavailable.
        if not self.datalogger.runlist:
            raise ValueError("Cannot download waveforms because the DataLogger has " +
                             "not been initialized properly. Possibly because this is " +
                             "not an IRIS MT Station.")

        self.waveforms = self.datalogger.download_iris_waveforms(self.name,
                                                                 self.start_time,
                                                                 self.end_time,
                                                                 network_code=self.network_code)
        self._rotate_waveforms()

    def _waveform_path(self, directory):
        return pathlib.Path(directory) / f"{self.name}.hdf"

    def load_waveforms(self, directory="./"):
        """Load the waveform data that has already been downloaded.

        :param directory: directory (str or path-like) containing
            ``<site name>.hdf``
        """
        self.waveforms = pd.read_hdf(self._waveform_path(directory),
                                     key="waveforms")
        if "Bx" not in self.waveforms.columns:
            # Only rotate the waveforms if they haven't been rotated/don't exist yet
            self._rotate_waveforms()

    def save_waveforms(self, directory="./"):
        """Save the waveform data as ``<site name>.hdf`` in ``directory``."""
        if self.waveforms is None:
            raise ValueError("There are no waveforms to save for site: " + self.name +
                             "\nYou can download waveforms from the IRIS database by calling " +
                             ".download_waveforms()")
        self.waveforms.to_hdf(self._waveform_path(directory), key="waveforms")

    def _rotate_waveforms(self):
        """Rotate the waveforms to the geographic coordinate system.

        This is added as an extra method to add the components to the in-memory
        objects, rather than saving the transformed values to disk.
        """
        # Magnetic field components rotated by the channel orientation
        self.waveforms["Bx"] = (self.waveforms["BN"] * np.cos(np.deg2rad(self.channel_orientation["Bx"]))
                                + self.waveforms["BE"] * np.cos(np.deg2rad(self.channel_orientation["By"])))
        self.waveforms["By"] = (self.waveforms["BN"] * np.sin(np.deg2rad(self.channel_orientation["Bx"]))
                                + self.waveforms["BE"] * np.sin(np.deg2rad(self.channel_orientation["By"])))

        # Electric field components rotated by the channel orientation
        self.waveforms["Ex"] = (self.waveforms["EN"] * np.cos(np.deg2rad(self.channel_orientation["Ex"]))
                                + self.waveforms["EE"] * np.cos(np.deg2rad(self.channel_orientation["Ey"])))
        self.waveforms["Ey"] = (self.waveforms["EN"] * np.sin(np.deg2rad(self.channel_orientation["Ex"]))
                                + self.waveforms["EE"] * np.sin(np.deg2rad(self.channel_orientation["Ey"])))


class Site1d(Site):
    """MT class for a 1D Fernberg profile"""
    def __init__(self, name, thicknesses, resistivities):
        # Call the base site class and initialize all those variables
        super().__init__(name)

        self.thicknesses = np.array(thicknesses)
        self.resistivities = np.array(resistivities)

    @property
    def depths(self):
        """The depth of the profiles."""
        return np.cumsum(self.thicknesses)

    def _repr_html_(self):
        return f"<p style=\"font-size:22px; color:blue\"><b>1d Site: {self.name}</b></p>"

    def plot_depth(self, ax=None):
        """Plots the resistivity vs. depth profile."""
        # lazy-load this so we don't require matplotlib to be installed by default
        import matplotlib.pyplot as plt
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
        Z = np.zeros(shape=(n, nfreq), dtype=complex)
        # DC frequency produces divide by zero errors
        with np.errstate(divide='ignore', invalid='ignore'):
            Z[-1, :] = complex_factor/k[-1, :]

            # eq. 7 (reflection coefficient at interface)
            r = np.zeros(shape=(n, nfreq), dtype=complex)

            for i in range(n-2, -1, -1):
                r[i, :] = ((1-k[i, :]*Z[i+1, :]/complex_factor) /
                           (1+k[i, :]*Z[i+1, :]/complex_factor))
                Z[i, :] = (complex_factor*(1-r[i, :]*np.exp(-2*k[i, :]*thicknesses[i])) /
                           (k[i, :]*(1+r[i, :]*np.exp(-2*k[i, :]*thicknesses[i]))))

        # Fill in the DC impedance as zero
        if freqs[0] == 0.:
            Z[:, 0] = 0.

        # Return a 3d impedance [0, Z; -Z, 0]
        Z_output = np.zeros(shape=(4, nfreq), dtype=complex)
        # Only return the top layer impedance
        # Z_factor is conversion from H->B, 1.e-3/MU
        Z_output[1, :] = Z[0, :]*(1.e-3/MU)
        Z_output[2, :] = -Z_output[1, :]
        return Z_output

    def plot_apparent_resistivity(self, interp_freqs=None):
        """Plots the apparent resistivity and phase for the site."""
        # lazy-load this so we don't require matplotlib to be installed by default
        from .plotting import plot_apparent_resistivity
        if interp_freqs is None:
            raise ValueError("Need interpolation frequencies to plot the resistivity at.")

        with np.errstate(divide='ignore', invalid='ignore'):
            periods = 1./interp_freqs

        Z = self.calcZ(interp_freqs)
        xlim = [10**np.floor(np.log10(np.min(periods))), 10**np.ceil(np.log10(np.max(periods)))]
        fig, _, _ = plot_apparent_resistivity(periods, Z, xlim=xlim)

        fig.suptitle(rf"1D Region: {self.name}", size=20)
        return fig

class SiteCollection:
    """Collection of MT sites
    
    Useful for calculating FFTs of all sites at once
    rather than individually in a slower for-loop.

    Parameters
    ----------
    sites : list of Site objects
        List of MT sites to include in the collection.
    """
    def __init__(self, sites):
        self.sites = sites
        self._N0 = None
        self._dt = None
        self._extrapolate = None

    def convolve_fft(self, mag_x, mag_y, dt=60, extrapolate="zeros"):
        """Convolution in frequency space for all sites at once.

        :param mag_x: northward magnetic field in nT; shape ``(ntimes,)``
            applies the same (spatially uniform) field to every site, shape
            ``(ntimes, nsites)`` gives each site its own local field (e.g.
            SECS-interpolated)
        :param mag_y: eastward magnetic field, same shape as ``mag_x``
        :param dt: sample interval in seconds
        :param extrapolate: band-edge handling passed through to each site's
            ``calcZ`` where supported (see ``Site3d.calcZ``)
        :return: ``(Ex, Ey)`` in mV/km, each with shape ``(nsites, ntimes)``
        """
        mag_x, mag_y = np.asarray(mag_x, dtype=float), np.asarray(mag_y, dtype=float)
        if mag_x.shape != mag_y.shape:
            raise ValueError("mag_x and mag_y must have the same shape")
        if np.isnan(mag_x).any() or np.isnan(mag_y).any():
            raise ValueError(
                "magnetic field input contains NaNs, which would corrupt the "
                "FFT; interpolate them (bezpy.mag.prepare(gaps='interpolate')) "
                "or predict gap-free fields with SECS first")

        N0 = mag_x.shape[0]
        # Round N to the next highest power of 2 (+1 (makes it 2) to prevent circular convolution)
        N = 2**(int(np.log2(N0))+2)

        freqs = scipy.fft.rfftfreq(N, d=dt)

        if N0 != self._N0 or dt != self._dt or extrapolate != self._extrapolate:
            # Only recalculate the interpolated impedances when the frequencies
            # (or the band-edge policy) change.
            self._N0 = N0
            self._dt = dt
            self._extrapolate = extrapolate
            # Z needs to be organized as: xx, xy, yx, yy
            # Z_interp final dimension is N frequencies
            self._Z_interp = np.zeros((4, len(self.sites), len(freqs)), dtype=complex)
            for i, site in enumerate(self.sites):
                try:
                    self._Z_interp[:, i, :] = site.calcZ(freqs, extrapolate=extrapolate)
                except TypeError:
                    # Physical sites (Site1d) are valid at all frequencies
                    self._Z_interp[:, i, :] = site.calcZ(freqs)
        Z_interp = self._Z_interp

        if mag_x.ndim == 1:
            # One shared field: fft shape (nfreqs,), broadcasts over sites
            mag_x_fft = scipy.fft.rfft(mag_x, n=N, workers=-1)
            mag_y_fft = scipy.fft.rfft(mag_y, n=N, workers=-1)
        else:
            if mag_x.shape[1] != len(self.sites):
                raise ValueError(
                    f"per-site input must be (ntimes, {len(self.sites)} sites); "
                    f"got {mag_x.shape}")
            # Per-site fields: batch the FFTs, then organize as (nsites, nfreqs)
            mag_x_fft = scipy.fft.rfft(mag_x, n=N, axis=0, workers=-1).T
            mag_y_fft = scipy.fft.rfft(mag_y, n=N, axis=0, workers=-1).T

        Ex_fft = Z_interp[0, :, :]*mag_x_fft + Z_interp[1, :, :]*mag_y_fft
        Ey_fft = Z_interp[2, :, :]*mag_x_fft + Z_interp[3, :, :]*mag_y_fft

        Ex_t = np.real(scipy.fft.irfft(Ex_fft, axis=-1, workers=-1)[:, :N0])
        Ey_t = np.real(scipy.fft.irfft(Ey_fft, axis=-1, workers=-1)[:, :N0])

        return (Ex_t, Ey_t)


class ConductivityModel(SiteCollection):
    """Collection of MT sites from a conductivity model
    
    This reads in the ModEM model files and creates a SiteCollection

    Parameters
    ----------
    fname : str or Path
        Path to the model output file
    """
    def __init__(self, filename):
        sites = []
        with open(filename) as f:
            while True:
                line = f.readline()
                if line.startswith(">"):
                    break
            for i in range(4):
                # 5 ">" lines
                f.readline()
            # Now we are at the line with the number of periods
            # which we need to keep track of so we know how big
            # of an array we need to allocate
            nperiods = int(f.readline().split()[1])

            # Now start parsing the actual site content
            prev_site_name = None
            for line in f:
                if line.startswith('#'):
                    # Hit the TX / TY section which we don't need
                    break
                elements = line.split()
                period = float(elements[0])
                name = elements[1]
                lat = float(elements[2])
                lon = float(elements[3])
                component = elements[7]
                val = float(elements[8]) + float(elements[9])*1j

                if name != prev_site_name:
                    prev_site_name = name
                    site = Site3d(name)
                    sites.append(site)
                    site.latitude = lat
                    site.longitude = lon
                    site.periods = np.zeros(nperiods)
                    site.Z = np.zeros((4, nperiods), dtype=complex)
                    old_period = 0.
                    period_counter = -1

                if period != old_period:
                    period_counter += 1
                    old_period = period
                    site.periods[period_counter] = period

                if component == 'ZXX':
                    loc = 0
                elif component == 'ZXY':
                    loc = 1
                elif component == 'ZYX':
                    loc = 2
                elif component == 'ZYY':
                    loc = 3

                site.Z[loc, period_counter] = val
        super().__init__(sites)
