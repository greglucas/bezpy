"""Helper functions for magnetotelluric data."""

__all__ = ["apparent_resistivity"]

import numpy as np


def apparent_resistivity(periods, Z, Z_var=None):
    """Calculate apparent resistivity, phase, and their 1-sigma errors.

    ``Z_var`` is the variance of the complex impedance, with half assigned to
    each of its real and imaginary parts. When it is unavailable, the returned
    uncertainty arrays are NaN: absence of an error estimate is not evidence
    for a unit or zero error.

    Notes
    -----
    The second and fourth return values were historically stored on
    ``Site`` as ``resistivity_var``/``phase_var``. They are standard errors,
    not variances; those attribute names remain for compatibility.

    :param periods: positive periods in seconds, shape ``(nperiods,)``
    :param Z: complex impedance, shape ``(..., nperiods)``
    :param Z_var: optional non-negative impedance variance, same shape as ``Z``
    :return: ``(resistivity, resistivity_std, phase, phase_std)``
    """
    periods = np.asarray(periods, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    if periods.ndim != 1 or Z.ndim < 1 or Z.shape[-1] != len(periods):
        raise ValueError("Z's last dimension must match the 1-D periods array")
    if not np.all(np.isfinite(periods)) or np.any(periods <= 0):
        raise ValueError("periods must be finite and positive")
    if Z_var is None:
        Z_var = np.full(Z.shape, np.nan)
    else:
        Z_var = np.asarray(Z_var, dtype=float)
        if Z_var.shape != Z.shape:
            raise ValueError(f"Z_var must have shape {Z.shape}; got {Z_var.shape}")
        Z_var = np.where(Z_var >= 0, Z_var, np.nan)

    # Ignore warnings because of nan possibilities. Just push those through
    with np.errstate(divide="ignore", invalid="ignore"):
        Z_std = np.sqrt(Z_var / 2.0)  # pylint: disable=invalid-name
        resistivity = periods / 5.0 * np.abs(Z) ** 2
        resistivity_std = 2 * Z_std * periods / 5.0 * np.abs(Z)

        phase = np.rad2deg(np.arctan(np.tan(np.angle(Z))))
        # Y components adjusted by pi
        # phase[2:,:] = np.rad2deg(np.arctan(np.tan(np.angle(Z[2:,:]) + np.pi)))
        phase_std = np.rad2deg(np.abs(Z_std / Z))

    return (resistivity, resistivity_std, phase, phase_std)
