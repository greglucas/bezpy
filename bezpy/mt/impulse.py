"""Module to compute and use impulse response functions."""
# pylint: disable=invalid-name
__all__ = ["DTIR"]

import time
import numpy as np
from scipy.linalg import toeplitz

from .site import Site3d
from .utils import apparent_resistivity


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
    def __init__(self, dt=1, nmin=-60, nmax=600, decay_factor=1.e-2,
                 Q_choice=1, model_space=False, verbose=False):
        self.dt = dt  # In seconds
        self.nmin = nmin
        self.nmax = nmax

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

    def __repr__(self):
        return f"DTIR(dt={self.dt}, nmin={self.nmin}, nmax={self.nmax}, " \
               f"decay_factor={self.decay_factor}, Q_choice={self.Q_choice}, " \
               f"model_space={self.model_space})"

    @property
    def ns(self):
        """Return a list of the time indices."""
        return np.arange(self._Qnmin, self._Qnmax+1)

    def __len__(self):
        # n of the output arrays
        return self._Qnmax - self._Qnmin + 1

    def get_zf(self, periods):
        """Calculate Z at the given set of periods"""
        if self.zn is None:
            raise ValueError("Need to call dtir.calculate(...) first.")

        A = self._calcA(periods)
        zf = (A@self.zn.T).T
        return zf

    def get_apparent_resistivity(self, periods):
        """Calculates the apparent resistivity for the given periods."""
        return apparent_resistivity(periods, self.get_zf(periods))

    def calculate(self, periods=None, Z=None, Z_var=None, site=None):
        """Calculates the DTIR for the given periods and Z, or the site object."""
        # pylint: disable=too-many-locals
        if isinstance(site, Site3d):
            # Drop any nan's along the entire column.
            good_locs = ~np.any(np.isnan(site.Z), axis=0)
            periods = site.periods[good_locs]
            Z = site.Z[:, good_locs]
            Z_var = site.Z_var[:, good_locs]
        else:
            # elif periods is None or Z is None:
            raise ValueError("Must pass in either periods and Z or a Site3d object")

        t0 = time.time()
        # Number of data points
        nper = len(periods)

        ncomponents = Z.shape[0]
        if ncomponents != 4:
            raise ValueError("Error in number of componenets, possibly because of a 1D array " +
                             "Expecting Z to be of the shape (4 x N)")

        self._calcQs()

        A = self._calcA(periods)
        # split out the real and imaginary components to solve in real space
        A = np.vstack([A.real, A.imag])

        # Scaling by variances
        if Z_var is None:
            variance = np.ones((ncomponents, nper))
        else:
            variance = Z_var

        # Output data storage
        self.zn = np.zeros((ncomponents, len(self)))

        for k in range(ncomponents):
            t1 = time.time()

            # Stack the real and imaginary components
            b = np.hstack([Z[k, :].real, Z[k, :].imag])
            sigma = 1./np.hstack([variance[k, :].real, variance[k, :].real])
            # No off-diagonal terms for now, so just need
            # to take 1/diagonal
            # sigma_inv = np.linalg.inv(sigma)
            sigma_inv = np.diag(1./sigma)
            sigma = np.diag(sigma)

            if self.model_space:
                # MODEL-SPACE
                # Gramian A^T Sigma A + lambda Q
                G = A.T@sigma@A + self.decay_factor*self.Q

                # Solve A*zn = b for zn
                self.zn[k, :] = np.linalg.solve(G, A.T@sigma@b)

            else:
                # DATA-SPACE
                # Gramian (A Q^-1 A^T + lambda Sigma^-1)
                G = A@self.Q_inv@A.T + self.decay_factor*sigma_inv

                b_lambda = np.linalg.solve(G, b)

                self.zn[k, :] = self.Q_inv @ A.T @ b_lambda

            if self.verbose:
                print("Iteration", k, " complete", time.time()-t1)

        if self.verbose:
            print("Total time for impulse response:", time.time()-t0)

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
        return np.exp(-1j*self.ns[np.newaxis, :]*self.dt*(2*np.pi/periods[:, np.newaxis]))

    def _calcQs(self):
        """Goes through logic on whether we need to calculate Q/Q_inv"""

        # If any of these variables were modified, then we need
        # to recalculate Q/Qinv
        if (self._Qnmin != self.nmin or
                self._Qnmax != self.nmax or
                self._Q_choice != self.Q_choice):

            self.Q = None
            self.Q_inv = None
            self._Qnmin = self.nmin
            self._Qnmax = self.nmax
            self._Q_choice = self.Q_choice

        # Set up the Q matrix if not present
        if self.model_space and self.Q is None:
            self._calcQ()

        if not self.model_space and self.Q_inv is None:
            # Now in data space, but no Q inverse yet
            self._calcQinv()

    def _calcQ(self):
        t0 = time.time()

        if self.Q_choice == 1:
            n = self.ns[:, np.newaxis]
            m = self.ns[np.newaxis, :]
            nm = n*m
            # Contains indices for odd values
            odds = np.mod(n-m, 2) == 1

            # Ignore the division by zero along the diagonal for now and fill it later
            with np.errstate(divide='ignore', invalid='ignore'):
                alpha = nm/((n-m)**2)
                np.fill_diagonal(alpha, 0)

            # Fill everything first to setup the matrix
            # k - l == even
            self.Q = (nm*np.pi**2*(1+3*alpha))
            # k - l == odd
            self.Q[odds] = ((2-3*np.pi**2*nm)*alpha + 12*alpha**2 - np.pi**2*nm)[odds]
            # k == l : diagonal
            np.fill_diagonal(self.Q, (self.ns**2*np.pi**2/2 + self.ns**4*np.pi**4/4))
            # k == l == 0
            n0 = self.ns == 0
            self.Q[n0, n0] = 1.

        elif self.Q_choice == 2:
            # Simple diagonal matrix
            self.Q = np.diag(self.ns**4)
            # Q[-nmin,-nmin] = 1.
            # np.fill_diagonal(Q, 1./np.arange(nmin, nmax)**4)
            # Q[0,0] = 1e-4

        elif self.Q_choice == 3:
            # Second order difference, with decay for n away from 0
            second_diff = np.zeros(len(self))
            second_diff[0] = 2
            second_diff[1] = -1
            self.Q = toeplitz(second_diff, r=second_diff)
            self.Q *= self.decay_factor * self.ns[:, np.newaxis]**4

        if self.verbose:
            print("Q Matrix Size:", self.Q.shape)
            print("Q Matrix Complete:", time.time()-t0)

    def _calcQinv(self):
        t0 = time.time()

        if self.Q is None:
            # No Q matrix present yet, so make one
            self._calcQ()

        # In dataspace we need to calculate the inverse of Q
        # Q is singular so calculate the pseudo-inverse instead
        self.Q_inv = np.linalg.inv(self.Q)

        if self.verbose:
            print("Q Inversion Complete:", time.time()-t0)

    # FFT of magnetic field time series and the MT site
    def convolve(self, mag_x, mag_y):
        """Convolution of the DTIR with the magnetic field in time domain."""

        if self.zn is None:
            raise ValueError("Need to calculate the DTIR before convolving.")

        # Pad the ns/zn to perform the convolution
        max_n = np.max([np.abs(self._Qnmin), np.abs(self._Qnmax)])
        # This will pad 0's before and after zn to make it symmetric
        # about zero. No padding on the component axis, so the shape
        # of zn becomes: (4 x 2*maxN+1)
        zn_pad = np.pad(self.zn,
                        [(0, 0), (np.abs(-max_n-self._Qnmin), max_n-self._Qnmax)],
                        'constant')

        # Convolve the impulse response with the magnetic field to get the electric field
        # in the time domain
        Ex_t = np.convolve(mag_x, zn_pad[0, :].real, mode='same') + \
            np.convolve(mag_y, zn_pad[1, :].real, mode='same')
        Ey_t = np.convolve(mag_x, zn_pad[2, :].real, mode='same') + \
            np.convolve(mag_y, zn_pad[3, :].real, mode='same')

        return (Ex_t, Ey_t)
