"""Helper functions for magnetotelluric data."""

__all__ = ["apparent_resistivity"]

import numpy as np


def apparent_resistivity(periods, Z, Z_var=None):
    """Calculates the apparent resistivity for the given periods and Z."""
    if Z_var is None:
        Z_var = np.ones((Z.shape)) + np.ones((Z.shape))*1j

    # Ignore warnings because of nan possibilities. Just push those through
    with np.errstate(divide='ignore', invalid='ignore'):
        Z_std = np.sqrt(Z_var/2.)  # pylint: disable=invalid-name
        resistivity = periods/5. * np.abs(Z)**2
        resistivity_std = 2*Z_std*periods/5. * np.abs(Z)

        phase = np.rad2deg(np.arctan(np.tan(np.angle(Z))))
        # Y components adjusted by pi
        # phase[2:,:] = np.rad2deg(np.arctan(np.tan(np.angle(Z[2:,:]) + np.pi)))
        phase_std = np.rad2deg(np.abs(Z_std / Z))

    return (resistivity, resistivity_std, phase, phase_std)
