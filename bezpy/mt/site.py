"""
Magnetotelluric site classes.

"""
__all__ = ["Site", "Site1d", "Site3d", "SiteCollection", "ConductivityModel",
           "sites_to_dataset", "sites_from_dataset"]

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

# SiteCollection.convolve_fft processes sites in blocks whose frequency-domain
# workspace (interpolated impedances + spectra) stays under this budget; the
# interpolated-impedance cache is only kept when the whole collection fits.
_CONVOLVE_BUDGET_BYTES = 512 * 2**20


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

        # rfft, because the input is real-valued: the negative-frequency
        # half-spectrum is implied (conjugate symmetry), so Z is only ever
        # evaluated at non-negative frequencies.

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
    """EarthScope USArray Site

    :param name: site name
    :param latitude: site latitude in degrees (optional)
    :param longitude: site longitude in degrees (optional)
    :param periods: periods of the impedance tensor in seconds, ascending (optional)
    :param Z: complex impedance tensor in (mV/km)/nT, organized
        (xx, xy, yx, yy) along the first axis; shape ``(4, nperiods)`` (optional)
    :param Z_var: variance of the impedance tensor, same shape as ``Z`` (optional)
    """

    # pylint: disable=too-many-arguments
    def __init__(self, name, latitude=None, longitude=None,
                 periods=None, Z=None, Z_var=None):
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

        # Optional keyword data, filling in what the readers would otherwise set
        self.latitude = latitude
        self.longitude = longitude
        if periods is not None:
            self.periods = np.asarray(periods, dtype=float)
        if Z is not None:
            self.Z = np.asarray(Z, dtype=complex)
        if Z_var is not None:
            self.Z_var = np.asarray(Z_var, dtype=float)

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
            # When variances weight the spline, a non-finite or zero variance
            # would poison every interpolated value (NaN weights propagate
            # through the whole spline; zero variance is an infinite weight).
            # Drop those periods from the fit instead -- same policy read_xml
            # applies at parse time, enforced here so every entry path
            # (gridded products, user-supplied Z_var) is safe.
            if self.Z_var is not None:
                good_vals &= np.isfinite(self.Z_var[i, :]) & (self.Z_var[i, :] != 0.)
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

            # The spline degree needs at least k+1 points; short period lists
            # (e.g. downsampled gridded products) fall back to lower degrees
            # rather than failing inside scipy.
            ngood = int(np.count_nonzero(good_vals))
            if ngood < 2:
                raise ValueError(
                    f"site {self.name}: component {i} ({'xx xy yx yy'.split()[i]}) "
                    f"has {ngood} valid impedance estimate(s); need at least 2 "
                    "to interpolate Z")
            k = min(3, ngood - 1)

            # Whether to interpolate on magnitude/phase of Z,
            # or on the real/imaginary components of Z
            if mag_phase_interp:
                spl_mag = interp_func(periods[good_vals], np.abs(self.Z[i, good_vals]),
                                      w=weights, k=k, ext=extrapolate, s=smooth_factor)
                spl_phase = interp_func(periods[good_vals],
                                        np.unwrap(np.angle(self.Z[i, good_vals])),
                                        w=weights, k=k, ext=extrapolate, s=smooth_factor)
                with np.errstate(invalid='ignore'):
                    Z_interp[i, :] = spl_mag(x) * np.exp(1j*np.unwrap(spl_phase(x)))
            else:
                spl = interp_func(periods[good_vals], self.Z[i, good_vals].real,
                                  w=weights, k=k, ext=extrapolate, s=smooth_factor)
                splc = interp_func(periods[good_vals], self.Z[i, good_vals].imag,
                                   w=weights, k=k, ext=extrapolate, s=smooth_factor)
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

        Uses the projection (transpose) rather than the exact inverse of the
        channel-orientation matrix, which is identical when the two channels
        of each pair are orthogonal -- the standard installation.
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

    def __len__(self):
        return len(self.sites)

    def __iter__(self):
        return iter(self.sites)

    def __getitem__(self, key):
        """Integer indexing returns a site; slicing returns a new SiteCollection."""
        if isinstance(key, slice):
            return SiteCollection(self.sites[key])
        return self.sites[key]

    def __repr__(self):
        counts = {}
        for site in self.sites:
            cls = type(site).__name__
            counts[cls] = counts.get(cls, 0) + 1
        breakdown = ", ".join(f"{n} {cls}" for cls, n in sorted(counts.items()))
        return f"SiteCollection({len(self.sites)} sites: {breakdown})"

    @property
    def latitudes(self):
        """Site latitudes as a float array (NaN where a site has none)."""
        return np.array([np.nan if site.latitude is None else site.latitude
                         for site in self.sites], dtype=float)

    @property
    def longitudes(self):
        """Site longitudes as a float array (NaN where a site has none)."""
        return np.array([np.nan if site.longitude is None else site.longitude
                         for site in self.sites], dtype=float)

    def to_dataset(self):
        """Convert the collection to the canonical xarray Dataset.

        See :func:`sites_to_dataset` for the dataset convention.
        """
        return sites_to_dataset(self.sites)

    def to_netcdf(self, path):
        """Save the collection to a netCDF file (see :func:`sites_to_dataset`)."""
        self.to_dataset().to_netcdf(path)

    @classmethod
    def from_dataset(cls, ds):
        """Build a collection from a canonical dataset (see :func:`sites_from_dataset`)."""
        return cls(sites_from_dataset(ds))

    @classmethod
    def from_netcdf(cls, path):
        """Load a collection saved by :meth:`to_netcdf`."""
        import xarray as xr
        with xr.open_dataset(path) as ds:
            ds.load()
        return cls.from_dataset(ds)

    def _calc_block_Z(self, sites, freqs, extrapolate):
        """Interpolated impedances (4, len(sites), nfreqs) for a site block."""
        Z = np.zeros((4, len(sites), len(freqs)), dtype=complex)
        for i, site in enumerate(sites):
            try:
                Z[:, i, :] = site.calcZ(freqs, extrapolate=extrapolate)
            except TypeError:
                # Physical sites (Site1d) are valid at all frequencies
                Z[:, i, :] = site.calcZ(freqs)
        return Z

    def convolve_fft(self, mag_x, mag_y, dt=60, extrapolate="zeros",
                     site_chunk=None):
        """Convolution in frequency space for all sites at once.

        Memory scales with ``nsites * nfft`` (``nfft`` is roughly 4x the input
        length): the frequency-domain workspace is ~200 bytes per site per
        input sample, and the returned E arrays are ``16 * nsites * ntimes``
        bytes. Sites are processed in blocks sized to keep the workspace
        under a fixed budget, so large collections (e.g. gridded impedance
        products) trade the impedance cache for bounded memory; the output
        size is up to the caller (shorten the window or chunk the scan).

        :param mag_x: northward magnetic field in nT; shape ``(ntimes,)``
            applies the same (spatially uniform) field to every site, shape
            ``(ntimes, nsites)`` gives each site its own local field (e.g.
            SECS-interpolated)
        :param mag_y: eastward magnetic field, same shape as ``mag_x``
        :param dt: sample interval in seconds
        :param extrapolate: band-edge handling passed through to each site's
            ``calcZ`` where supported (see ``Site3d.calcZ``)
        :param site_chunk: number of sites processed per block (default:
            sized from a ~512 MB workspace budget). The interpolated-impedance
            cache is kept only when all sites fit in one block.
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
        nsites = len(self.sites)
        if mag_x.ndim == 2 and mag_x.shape[1] != nsites:
            raise ValueError(
                f"per-site input must be (ntimes, {nsites} sites); "
                f"got {mag_x.shape}")

        if site_chunk is None:
            # Z (4) + E spectra (2) + per-site B spectra (2) complex128 rows
            per_site = 8 * len(freqs) * 16
            site_chunk = int(max(1, min(nsites, _CONVOLVE_BUDGET_BYTES // per_site)))

        use_cache = site_chunk >= nsites
        if use_cache:
            if N0 != self._N0 or dt != self._dt or extrapolate != self._extrapolate:
                # Only recalculate the interpolated impedances when the
                # frequencies (or the band-edge policy) change.
                self._N0 = N0
                self._dt = dt
                self._extrapolate = extrapolate
                # Z needs to be organized as: xx, xy, yx, yy
                self._Z_interp = self._calc_block_Z(self.sites, freqs, extrapolate)
        else:
            # Blocked mode never keeps the (too large) impedance cache
            self._N0 = self._dt = self._extrapolate = None
            self._Z_interp = None

        if mag_x.ndim == 1:
            # One shared field: fft shape (nfreqs,), broadcasts over sites
            shared_x_fft = scipy.fft.rfft(mag_x, n=N, workers=-1)
            shared_y_fft = scipy.fft.rfft(mag_y, n=N, workers=-1)

        Ex_t = np.empty((nsites, N0))
        Ey_t = np.empty((nsites, N0))
        for i0 in range(0, nsites, site_chunk):
            sel = slice(i0, min(i0 + site_chunk, nsites))
            Z_interp = self._Z_interp[:, sel, :] if use_cache else \
                self._calc_block_Z(self.sites[sel.start:sel.stop], freqs, extrapolate)

            if mag_x.ndim == 1:
                mag_x_fft, mag_y_fft = shared_x_fft, shared_y_fft
            else:
                # Per-site fields: batch the FFTs, organize as (nblock, nfreqs)
                mag_x_fft = scipy.fft.rfft(mag_x[:, sel], n=N, axis=0, workers=-1).T
                mag_y_fft = scipy.fft.rfft(mag_y[:, sel], n=N, axis=0, workers=-1).T

            Ex_fft = Z_interp[0, :, :]*mag_x_fft + Z_interp[1, :, :]*mag_y_fft
            Ey_fft = Z_interp[2, :, :]*mag_x_fft + Z_interp[3, :, :]*mag_y_fft

            Ex_t[sel] = scipy.fft.irfft(Ex_fft, axis=-1, workers=-1)[:, :N0]
            Ey_t[sel] = scipy.fft.irfft(Ey_fft, axis=-1, workers=-1)[:, :N0]

        return (Ex_t, Ey_t)


class ConductivityModel(SiteCollection):
    """Collection of MT sites from a ModEM data/response file.

    Compatibility wrapper around :func:`bezpy.mt.read_gridded_impedance`,
    which replaced the original header-counting parser here: it is
    content-driven (tolerant of the header variants across product
    versions), reads the error column into ``Z_var`` when present, and
    normalizes longitudes to -180..180. Prefer calling
    ``read_gridded_impedance`` directly in new code.

    Parameters
    ----------
    filename : str or Path
        Path to the ModEM-format data file
    """
    def __init__(self, filename):
        from .modem import read_gridded_impedance
        super().__init__(read_gridded_impedance(filename).sites)


# ---------------------------------
# Canonical dataset representation
# ---------------------------------
COMPONENTS = ["xx", "xy", "yx", "yy"]


def sites_to_dataset(sites):
    """Convert MT sites to the canonical xarray Dataset.

    This is the package-wide interface convention for site collections:

    * dims: ``site``, ``component`` (length 4), and ``period`` for 3D
      impedance sites; ``site`` and ``layer`` for 1D layered sites.
    * coords: ``site`` = site names (str), ``latitude(site)`` and
      ``longitude(site)`` floats (NaN when a site has none), ``period`` =
      float seconds ascending with attrs ``{"units": "s"}``, ``component`` =
      ``["xx", "xy", "yx", "yy"]`` (x = North, y = East).
    * 3D (:class:`Site3d`) variables: ``Z_real`` and ``Z_imag``, dims
      ``("site", "component", "period")``, units (mV/km)/nT; ``Z_var``
      (real, same dims) is included only if at least one site has a
      non-None ``Z_var`` (NaN rows for sites without one).
    * 1D (:class:`Site1d`) variables: ``thickness`` and ``resistivity``,
      dims ``("site", "layer")``, units m and ohm-m, NaN-padded to the
      deepest profile.
    * Dataset attrs: ``{"title": "bezpy MT site collection",
      "site_type": "3d" or "1d"}``.

    Sites with different period grids share the sorted union of all periods
    on the ``period`` axis; a site's missing periods are NaN in
    ``Z_real``/``Z_imag``/``Z_var``. This is safe downstream because
    :meth:`Site3d.spline_interp` drops NaN Z values per component.

    Requires xarray (install the ``bezpy[data]`` extra).

    :param sites: iterable of :class:`Site3d` or :class:`Site1d` objects;
        a single dataset cannot mix the two site types
    :return: ``xarray.Dataset`` following the convention above
    :raises ValueError: for an empty collection or mixed 1D/3D site types
    """
    import xarray as xr
    sites = list(sites)
    if not sites:
        raise ValueError("Cannot build a dataset from an empty site collection.")
    n_3d = sum(isinstance(site, Site3d) for site in sites)
    n_1d = sum(isinstance(site, Site1d) for site in sites)
    if n_3d and n_1d:
        raise ValueError(
            f"Cannot mix Site1d and Site3d objects in one dataset (got {n_3d} "
            f"Site3d and {n_1d} Site1d); split the collection by site type first.")
    if n_3d + n_1d != len(sites):
        raise ValueError("All sites must be Site3d or Site1d objects.")

    coords = {"site": [site.name for site in sites],
              "latitude": ("site", np.array([np.nan if site.latitude is None else site.latitude
                                             for site in sites], dtype=float)),
              "longitude": ("site", np.array([np.nan if site.longitude is None else site.longitude
                                              for site in sites], dtype=float))}

    if n_3d:
        # Sorted union of every site's period grid; per-site gaps stay NaN
        periods = np.unique(np.concatenate([np.asarray(site.periods, dtype=float)
                                            for site in sites]))
        Z = np.full((len(sites), 4, len(periods)), np.nan + 1j*np.nan, dtype=complex)
        Z_var = np.full((len(sites), 4, len(periods)), np.nan)
        any_var = False
        for i, site in enumerate(sites):
            idx = np.searchsorted(periods, site.periods)
            Z[i][:, idx] = site.Z
            if site.Z_var is not None:
                any_var = True
                Z_var[i][:, idx] = site.Z_var
        z_attrs = {"units": "(mV/km)/nT"}
        data_vars = {"Z_real": (("site", "component", "period"), Z.real, z_attrs),
                     "Z_imag": (("site", "component", "period"), Z.imag, z_attrs)}
        if any_var:
            data_vars["Z_var"] = (("site", "component", "period"), Z_var,
                                  {"units": "((mV/km)/nT)^2"})
        coords["period"] = ("period", periods, {"units": "s"})
        coords["component"] = COMPONENTS
        site_type = "3d"
    else:
        nlayers = max(len(site.resistivities) for site in sites)
        thickness = np.full((len(sites), nlayers), np.nan)
        resistivity = np.full((len(sites), nlayers), np.nan)
        for i, site in enumerate(sites):
            thickness[i, :len(site.thicknesses)] = site.thicknesses
            resistivity[i, :len(site.resistivities)] = site.resistivities
        data_vars = {"thickness": (("site", "layer"), thickness, {"units": "m"}),
                     "resistivity": (("site", "layer"), resistivity, {"units": "ohm-m"})}
        site_type = "1d"

    return xr.Dataset(data_vars, coords=coords,
                      attrs={"title": "bezpy MT site collection", "site_type": site_type})


def sites_from_dataset(ds):
    """Rebuild MT site objects from a canonical dataset.

    Inverts :func:`sites_to_dataset` (see there for the convention):

    * 3D sites get ``periods`` equal to the dataset's (union) period axis,
      with NaN Z entries preserved; ``Z_var`` is None for a site whose
      variance is all-NaN or when the ``Z_var`` variable is absent.
    * 1D sites have the NaN layer padding stripped.
    * NaN ``latitude``/``longitude`` become None on the site.

    :param ds: ``xarray.Dataset`` following the :func:`sites_to_dataset`
        convention
    :return: list of :class:`Site3d` or :class:`Site1d` objects
    :raises ValueError: if the dataset has neither 3D nor 1D site variables
    """
    names = [str(name) for name in ds["site"].values]
    latitudes = np.asarray(ds["latitude"].values, dtype=float)
    longitudes = np.asarray(ds["longitude"].values, dtype=float)
    latitudes = [None if np.isnan(lat) else float(lat) for lat in latitudes]
    longitudes = [None if np.isnan(lon) else float(lon) for lon in longitudes]

    sites = []
    if "Z_real" in ds:
        periods = np.asarray(ds["period"].values, dtype=float)
        Z = ds["Z_real"].values + 1j*ds["Z_imag"].values
        Z_var = ds["Z_var"].values if "Z_var" in ds else None
        for i, name in enumerate(names):
            site_var = None
            if Z_var is not None and not np.isnan(Z_var[i]).all():
                site_var = Z_var[i]
            sites.append(Site3d(name, latitude=latitudes[i], longitude=longitudes[i],
                                periods=periods, Z=Z[i], Z_var=site_var))
    elif "thickness" in ds:
        thickness = np.asarray(ds["thickness"].values, dtype=float)
        resistivity = np.asarray(ds["resistivity"].values, dtype=float)
        for i, name in enumerate(names):
            keep = ~np.isnan(thickness[i])
            site = Site1d(name, thickness[i][keep], resistivity[i][keep])
            site.latitude = latitudes[i]
            site.longitude = longitudes[i]
            sites.append(site)
    else:
        raise ValueError(
            "Dataset does not follow the bezpy site convention: expected "
            "Z_real/Z_imag (3d) or thickness/resistivity (1d) variables.")
    return sites
