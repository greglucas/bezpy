
__all__ = ["DTIR"]

import time
import numpy as np
import scipy.linalg

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

    def __init__(self, dt=1, nmin=-60, nmax=600, decay_factor=1.e-2,
                 Q_choice=1, model_space=False, verbose=False):
        self.dt = dt # In seconds
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
               f"decay_factor={self.decay_factor}, Q_choice={self.Q_choice}, model_space={self.model_space})"

    @property
    def ns(self):
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
        zf = self.get_zf(periods)

        return apparent_resistivity(periods, zf)

    def calculate(self, periods=None, Z=None, Z_var=None, site=None):
        """Calculates the DTIR for the given periods and Z, or the site object."""

        if isinstance(site, Site3d):
            periods = site.periods
            Z = site.Z
            Z_var = site.Z_var
        elif periods is None or Z is None:
            raise ValueError("Must pass in either periods and Z or a Site3d object")

        t0 = time.time()
        # Number of data points
        nper = len(periods)

        ncomponents = Z.shape[0]
        if ncomponents != 4:
            raise ValueError("Error in number of componenets, possibly because of a 1D array " +
                             "Expecting Z to be of the shape (4 x N)")

        #TODO: Put the A matrix calculation here instead of inside the loop later
        #      Need it inside loop currently because of nans and changing size of
        #      the data array
        # Generate the A matrix
        #A = self._calcA(periods)

        self._calcQs()

        # Scaling by variances
        if Z_var is None:
            variance = np.ones((ncomponents, nper), dtype=np.complex) + 1j
        else:
            # Put variance in both real and complex plane
            variance = Z_var + Z_var*1j
            # TODO: Fix this, by possibly dropping the values in the solver
            #       and then repopulating the matrix?
            # Where variance is nan, make it infinite
            #variance[np.isnan(variance)] = 1.e9
            #Z[np.isnan(Z)] = 0.

        # Output data storage
        zn = np.zeros((ncomponents,len(self)), dtype=np.complex)

        for k in range(ncomponents):
            t1 = time.time()

            b = Z[k,:]
            nan_locs = np.isnan(b)
            b = b[~nan_locs]
            # Make sigma our variances
            sigma = np.diag(variance[k,~nan_locs])

            #TODO: Can put the A calculation outside of
            #      the loop if it weren't for the nan possibility...
            A = self._calcA(periods[~nan_locs])


            if self.model_space:
                ## MODEL-SPACE

                # Gramian A^T Sigma A + lambda Q
                #TODO: Do we need Q to be in complex space too?
                #      Q = Q*(1. + 1j) ???
                G = A.T@sigma@A + self.decay_factor*self.Q

                # Solve A*zn = b for zn
                zn[k,:] = np.linalg.solve(G, A.T@sigma@b)

            else:
                ## DATA-SPACE

                # No off-diagonal terms for now, so just need
                # to take 1/diagonal
                #sigma_inv = np.linalg.inv(sigma)
                sigma_inv = np.diag(1./variance[k,~nan_locs])

                # Gramian (A Q^-1 A^T + lambda Sigma^-1)
                G = A@self.Q_inv@A.T + self.decay_factor*sigma_inv

                b_lambda = np.linalg.solve(G, b)

                zn[k,:] = self.Q_inv @ A.T @ b_lambda

            if self.verbose:
                print("Iteration", k, " complete", time.time()-t1)

        if self.verbose:
            print("Total time for impulse response:", time.time()-t0)

        self.zn = zn
        # Return list of ns and the impulse response at those points
        return (self.ns, zn)

    def _calcA(self, periods):
        # input periods in seconds, then turn it into an angular frequency
        # in the exponential

        # Setting up a vectorized version of A
        # Need to make sure the indices are proper on this
        # m matrix to get the change along 2nd dimension
        # m is shape (1, ns) // m = self.ns[np.newaxis,:]
        # periods is shape (nper, 1)
        # broadcast together makes A shape (nper, ns)

        return np.exp(-1j*self.ns[np.newaxis,:]*self.dt*(2*np.pi/periods[:,np.newaxis]))

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
            n = self.ns[:,np.newaxis]
            m = self.ns[np.newaxis,:]

            # Ignore the division by zero along the diagonal for now and fill it later
            with np.errstate(divide='ignore', invalid='ignore'):
                alpha = n*m/((n-m)**2)
                np.fill_diagonal(alpha, 0)

            # Fill everything first to setup the matrix
            # k - l == odd
            self.Q = ((2-3*np.pi**2*n*m)*alpha + 12*alpha**2 - np.pi**2*n*m).copy()
            # k - l == even
            self.Q[::2,::2] = (n*m*np.pi**2*(1+3*alpha))[::2,::2].copy()
            # k == l : diagonal
            np.fill_diagonal(self.Q, (self.ns**2*np.pi**2/2 + self.ns**4*np.pi**4/4).copy())
            # k == l == 0
            self.Q[self.ns==0,self.ns==0] = 1.

            # Random scaling from appendix, to get similar results?
            self.Q *= 1e9

        elif self.Q_choice == 2:
            # Simple diagonal matrix
            self.Q = np.diag(self.ns**4)
            #Q[-nmin,-nmin] = 1.
            #np.fill_diagonal(Q, 1./np.arange(nmin, nmax)**4)
            #Q[0,0] = 1e-4

        elif self.Q_choice == 3:
            # Second order difference, with decay for n away from 0
            second_diff = np.zeros(len(self))
            second_diff[0] = 2
            second_diff[1] = -1
            self.Q = scipy.linalg.toeplitz(second_diff, r=second_diff)

            self.Q *= self.decay_factor * self.ns[:,np.newaxis]**4

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
        self.Q_inv = np.linalg.pinv(self.Q)

        if self.verbose:
            print("Q Inversion Complete:", time.time()-t0)

    # FFT of magnetic field time series and the MT site
    def convolve(self, magX, magY):
        """Convolution of the DTIR with the magnetic field in time domain."""

        if self.zn is None:
            raise ValueError("Need to calculate the DTIR before convolving.")

        # Pad the ns/zn to perform the convolution
        maxN = np.max([np.abs(self._Qnmin), np.abs(self._Qnmax)])
        # This will pad 0's before and after zn to make it symmetric
        # about zero. No padding on the component axis, so the shape
        # of zn becomes: (4 x 2*maxN+1)
        zn_pad = np.pad(self.zn, [(0,0), (np.abs(-maxN-self._Qnmin), maxN-self._Qnmax)], 'constant')

        # Convolve the impulse response with the magnetic field to get the electric field
        # in the time domain
        Ex_t = np.convolve(magX, zn_pad[0,:].real, mode='same') + \
               np.convolve(magY, zn_pad[1,:].real, mode='same')
        Ey_t = np.convolve(magX, zn_pad[2,:].real, mode='same') + \
               np.convolve(magY, zn_pad[3,:].real, mode='same')

        return (Ex_t, Ey_t)
