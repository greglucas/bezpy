"""Module to compute and use impulse response functions."""

# pylint: disable=invalid-name
__all__ = ["DTIR"]

import functools
import time

import numpy as np
from scipy.linalg import toeplitz

from .site import Site3d
from .utils import apparent_resistivity


@functools.lru_cache(maxsize=4)
def _cached_q(nmin, nmax, q_choice, decay_factor):
    """Build and cache the regularization matrix shared by equal DTIRs."""
    ns = np.arange(nmin, nmax + 1)
    if q_choice == 1:
        n = ns[:, np.newaxis]
        m = ns[np.newaxis, :]
        nm = n * m
        odds = np.mod(n - m, 2) == 1
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = nm / ((n - m) ** 2)
            np.fill_diagonal(alpha, 0)
        Q = nm * np.pi**2 * (1 + 3 * alpha)
        Q[odds] = ((2 - 3 * np.pi**2 * nm) * alpha + 12 * alpha**2 - np.pi**2 * nm)[odds]
        np.fill_diagonal(Q, (ns**2 * np.pi**2 / 2 + ns**4 * np.pi**4 / 4))
        n0 = ns == 0
        Q[n0, n0] = 1.0
    elif q_choice == 2:
        Q = np.diag(ns**4)
    else:  # q_choice == 3
        second_diff = np.zeros(len(ns))
        second_diff[0] = 2
        if len(ns) > 1:
            second_diff[1] = -1
        Q = toeplitz(second_diff, r=second_diff)
        Q *= decay_factor * ns[:, np.newaxis] ** 4
    Q.setflags(write=False)
    return Q


@functools.lru_cache(maxsize=4)
def _cached_qinv(nmin, nmax, q_choice, decay_factor):
    """Pseudo-inverse paired with :func:`_cached_q`.

    Building this dense inverse dominates construction time. A Nowcast has
    one DTIR per site with identical horizons, so sharing it avoids repeating
    the same cubic-cost factorization hundreds of times.
    """
    Q_inv = np.linalg.pinv(_cached_q(nmin, nmax, q_choice, decay_factor))
    Q_inv.setflags(write=False)
    return Q_inv


class DTIR:
    """Discrete Time Impulse Response (DTIR) class

    Calculates the DTIR following the paper:

    Kelbert, A., C. C. Balch, A. Pulkkinen, G. D. Egbert,
    J. J. Love, E. J. Rigler, and I. Fujii (2017),
    Methodology for time-domain estimation of storm time geoelectric fields
    using the 3-D magnetotelluric response tensors,
    Space Weather, 15, 874–894, doi:10.1002/2017SW001594.
    """

    # pylint: disable=too-many-instance-attributes,too-many-arguments
    def __init__(
        self,
        *,
        dt=1,
        nmin=-60,
        nmax=600,
        decay_factor=1.0e-2,
        Q_choice=1,
        model_space=False,
        verbose=False,
    ):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if int(nmin) != nmin or int(nmax) != nmax or nmin > nmax:
            raise ValueError("nmin and nmax must be integers with nmin <= nmax")
        if not np.isfinite(decay_factor) or decay_factor <= 0:
            raise ValueError("decay_factor must be finite and positive")
        if Q_choice not in (1, 2, 3):
            raise ValueError("Q_choice must be one of: 1, 2, 3")
        self.dt = float(dt)  # In seconds
        self.nmin = int(nmin)
        self.nmax = int(nmax)

        self.decay_factor = decay_factor
        self.Q_choice = Q_choice
        self.model_space = model_space
        self.verbose = verbose

        self.Q = None
        self.Q_inv = None
        self.zn = None

        # If these change we need to recalculate Q,
        # all other variables can be modified though
        self._Qnmin = nmin
        self._Qnmax = nmax
        self._Q_choice = Q_choice
        self._Q_decay_factor = decay_factor

    def __repr__(self):
        return (
            f"DTIR(dt={self.dt}, nmin={self.nmin}, nmax={self.nmax}, "
            f"decay_factor={self.decay_factor}, Q_choice={self.Q_choice}, "
            f"model_space={self.model_space})"
        )

    @property
    def ns(self):
        """Return a list of the time indices."""
        return np.arange(self._Qnmin, self._Qnmax + 1)

    def __len__(self):
        # n of the output arrays
        return self._Qnmax - self._Qnmin + 1

    def get_zf(self, periods):
        """Calculate Z at the given set of periods"""
        if self.zn is None:
            raise ValueError("Need to call dtir.calculate(...) first.")

        A = self._calcA(periods)
        zf = (A @ self.zn.T).T
        return zf

    def get_apparent_resistivity(self, periods):
        """Calculates the apparent resistivity for the given periods."""
        return apparent_resistivity(periods, self.get_zf(periods))

    def calculate(self, periods=None, Z=None, Z_var=None, site=None):
        """Calculates the DTIR for the given periods and Z, or the site object."""
        # pylint: disable=too-many-locals
        if isinstance(site, Site3d):
            periods, Z, Z_var = site.periods, site.Z, site.Z_var
        elif periods is None or Z is None:
            raise ValueError("Must pass in either periods and Z or a Site3d object")
        periods = np.asarray(periods, dtype=float)
        Z = np.asarray(Z, dtype=complex)
        if periods.ndim != 1 or periods.size < 2:
            raise ValueError("periods must be a 1-D array with at least 2 values")
        if not np.all(np.isfinite(periods)) or np.any(periods <= 0):
            raise ValueError("periods must be finite and positive")
        if Z.shape != (4, periods.size):
            raise ValueError(f"Z must have shape (4, {periods.size}); got {Z.shape}")
        if Z_var is not None:
            Z_var = np.asarray(Z_var, dtype=float)
            if Z_var.shape != Z.shape:
                raise ValueError(f"Z_var must have shape {Z.shape}; got {Z_var.shape}")

        t0 = time.time()
        ncomponents = Z.shape[0]

        self._calcQs()

        # Output data storage
        self.zn = np.zeros((ncomponents, len(self)))

        for k in range(ncomponents):
            t1 = time.time()
            good = np.isfinite(Z[k]) & np.isfinite(periods)
            if Z_var is not None:
                good &= np.isfinite(Z_var[k]) & (Z_var[k] > 0)
            if np.count_nonzero(good) < 2:
                raise ValueError(
                    f"component {k} has fewer than 2 impedance estimates "
                    "with finite Z and positive finite variance"
                )

            component_periods = periods[good]
            component_z = Z[k, good]
            variance = np.ones(good.sum()) if Z_var is None else Z_var[k, good]
            A = self._calcA(component_periods)
            # Split real/imaginary data into one real-valued inverse problem.
            A = np.vstack([A.real, A.imag])

            # Stack the real and imaginary components
            b = np.hstack([component_z.real, component_z.imag])
            data_variance = np.hstack([variance, variance])
            precision = np.diag(1.0 / data_variance)
            covariance = np.diag(data_variance)

            if self.model_space:
                # MODEL-SPACE
                # Gramian A^T Sigma A + lambda Q
                G = A.T @ precision @ A + self.decay_factor * self.Q

                # Solve A*zn = b for zn
                self.zn[k, :] = np.linalg.solve(G, A.T @ precision @ b)

            else:
                # DATA-SPACE
                # Gramian (A Q^-1 A^T + lambda Sigma^-1)
                G = A @ self.Q_inv @ A.T + self.decay_factor * covariance

                b_lambda = np.linalg.solve(G, b)

                self.zn[k, :] = self.Q_inv @ A.T @ b_lambda

            if self.verbose:
                print("Iteration", k, " complete", time.time() - t1)

        if self.verbose:
            print("Total time for impulse response:", time.time() - t0)

        # Return list of ns and the impulse response at those points
        return (self.ns, self.zn)

    def _calcA(self, periods):
        # input periods in seconds, then turn it into an angular frequency
        # in the exponential

        # Setting up a vectorized version of A
        # Need to make sure the indices are proper on this
        # m matrix to get the change along 2nd dimension
        # m is shape (1, ns) // m = self.ns[np.newaxis,:]
        # periods is shape (nper, 1)
        # broadcast together makes A shape (nper, ns)
        return np.exp(-1j * self.ns[np.newaxis, :] * self.dt * (2 * np.pi / periods[:, np.newaxis]))

    def _calcQs(self):
        """Goes through logic on whether we need to calculate Q/Q_inv"""

        # If any of these variables were modified, then we need
        # to recalculate Q/Qinv
        if (
            self._Qnmin != self.nmin
            or self._Qnmax != self.nmax
            or self._Q_choice != self.Q_choice
            or self._Q_decay_factor != self.decay_factor
        ):
            self.Q = None
            self.Q_inv = None
            self._Qnmin = self.nmin
            self._Qnmax = self.nmax
            self._Q_choice = self.Q_choice
            self._Q_decay_factor = self.decay_factor

        # Set up the Q matrix if not present
        if self.model_space and self.Q is None:
            self._calcQ()

        if not self.model_space and self.Q_inv is None:
            # Now in data space, but no Q inverse yet
            self._calcQinv()

    def _calcQ(self):
        t0 = time.time()
        self.Q = _cached_q(self.nmin, self.nmax, self.Q_choice, float(self.decay_factor))

        if self.verbose:
            print("Q Matrix Size:", self.Q.shape)
            print("Q Matrix Complete:", time.time() - t0)

    def _calcQinv(self):
        t0 = time.time()

        if self.Q is None:
            self._calcQ()
        self.Q_inv = _cached_qinv(self.nmin, self.nmax, self.Q_choice, float(self.decay_factor))

        if self.verbose:
            print("Q Inversion Complete:", time.time() - t0)

    # FFT of magnetic field time series and the MT site
    def convolve(self, mag_x, mag_y):
        """Convolution of the DTIR with the magnetic field in time domain."""

        if self.zn is None:
            raise ValueError("Need to calculate the DTIR before convolving.")
        mag_x = np.asarray(mag_x, dtype=float)
        mag_y = np.asarray(mag_y, dtype=float)
        if mag_x.ndim != 1 or mag_x.shape != mag_y.shape:
            raise ValueError("mag_x and mag_y must be equal-length 1-D arrays")
        if not np.isfinite(mag_x).all() or not np.isfinite(mag_y).all():
            raise ValueError("mag_x and mag_y must contain only finite values")

        # Pad the ns/zn to perform the convolution
        max_n = np.max([np.abs(self._Qnmin), np.abs(self._Qnmax)])
        # This will pad 0's before and after zn to make it symmetric
        # about zero. No padding on the component axis, so the shape
        # of zn becomes: (4 x 2*maxN+1)
        zn_pad = np.pad(
            self.zn, [(0, 0), (np.abs(-max_n - self._Qnmin), max_n - self._Qnmax)], "constant"
        )
        if len(mag_x) < zn_pad.shape[1]:
            raise ValueError(
                f"magnetic input ({len(mag_x)} samples) is shorter than the "
                f"DTIR kernel ({zn_pad.shape[1]} samples)"
            )

        # Convolve the impulse response with the magnetic field to get the electric field
        # in the time domain
        Ex_t = np.convolve(mag_x, zn_pad[0, :].real, mode="same") + np.convolve(
            mag_y, zn_pad[1, :].real, mode="same"
        )
        Ey_t = np.convolve(mag_x, zn_pad[2, :].real, mode="same") + np.convolve(
            mag_y, zn_pad[3, :].real, mode="same"
        )

        return (Ex_t, Ey_t)
