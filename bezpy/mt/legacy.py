"""Readers for legacy MT transfer-function formats: EMTF Z-files and EDI.

Pre-EMTF-XML-era data (e.g. surveys contemporary with the March 1989 storm)
is distributed in these formats. The parsers are ported from the
Anomalies1989 study's ``data_readers.py``; :func:`read_zfile` and
:func:`read_edi` wrap them to return :class:`bezpy.mt.Site3d` objects in
bezpy's conventions (Z organized ``(xx, xy, yx, yy) x nperiods``, x = North,
periods ascending). Impedance values are taken from the files as-is -- both
formats conventionally store practical units, (mV/km)/nT.
"""

__all__ = ["read_zfile", "read_edi"]

import re
import warnings

import numpy as np

from .site import Site3d


class ZFile:
    """Parsed EMTF Z-file (.zss/.zrr/.zmm): transfer functions + covariances.

    Exposes ``impedance(angle)`` and ``tippers(angle)`` which rotate the
    predicted/predictor channels to ``angle`` degrees east of geographic
    north using the recorded channel orientations.
    """

    def __init__(self, filename):
        with open(filename) as f:
            self._parse(f)

    def _parse(self, f):
        # skip header lines
        for _ in range(3):
            f.readline()

        # station ID
        line = f.readline()
        if line.lower().startswith("station"):
            self.station = line.strip().split(":", 1)[1].strip()
        else:
            self.station = line.strip()

        # coordinates and declination
        line = f.readline().strip().lower()
        match = re.match(r"\s*coordinate\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+"
                         r"declination\s+(-?\d+\.?\d*)", line)
        self.coordinates = (float(match.group(1)), float(match.group(2)))
        self.declination = float(match.group(3))

        # number of channels / frequencies
        line = f.readline().strip().lower()
        match = re.match(r"\s*number\s+of\s+channels\s+(\d+)\s+number\s+of"
                         r"\s+frequencies\s+(\d+)", line)
        nchannels = int(match.group(1))
        nfreqs = int(match.group(2))

        f.readline()

        # channel orientations and names
        self.orientation = np.zeros((nchannels, 2))
        self.channels = []
        for i in range(nchannels):
            line = f.readline().strip()
            match = re.match(r"\s*\d+\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+\w*\s+"
                             r"(\w+)", line)
            self.orientation[i, 0] = float(match.group(1))
            self.orientation[i, 1] = float(match.group(2))
            # sometimes the channel ID comes out with extra characters
            self.channels.append(match.group(3)[:2].title())

        f.readline()  # blank line

        # EMTF assumes two predictor channels (the horizontal magnetics), so
        # nchannels - 2 is the number of predicted channels
        npredicted = nchannels - 2
        self.periods = np.zeros(nfreqs)
        self.transfer_functions = np.zeros((nfreqs, npredicted, 2),
                                           dtype=complex)
        # residual covariance (predicted x predicted) and inverse coherent
        # signal power (predictor x predictor)
        self.sigma_e = np.zeros((nfreqs, npredicted, npredicted), dtype=complex)
        self.sigma_s = np.zeros((nfreqs, 2, 2), dtype=complex)

        for i in range(nfreqs):
            line = f.readline().strip()
            self.periods[i] = float(re.match(
                r"\s*period\s*:\s+(\d+\.?\d*)\s+decimation\s+level",
                line).group(1))
            f.readline()
            f.readline()

            for j in range(npredicted):
                c1r, c1i, c2r, c2i = f.readline().split()
                self.transfer_functions[i, j, 0] = float(c1r) + 1j*float(c1i)
                self.transfer_functions[i, j, 1] = float(c2r) + 1j*float(c2i)

            f.readline()  # label line
            v1r, v1i = f.readline().split()
            v2r, v2i, v3r, v3i = f.readline().split()
            self.sigma_s[i, 0, 0] = float(v1r) + 1j*float(v1i)
            self.sigma_s[i, 1, 0] = float(v2r) + 1j*float(v2i)
            self.sigma_s[i, 0, 1] = float(v2r) - 1j*float(v2i)
            self.sigma_s[i, 1, 1] = float(v3r) + 1j*float(v3i)

            f.readline()  # label line
            for j in range(npredicted):
                values = f.readline().split()
                for k in range(j + 1):
                    self.sigma_e[i, j, k] = \
                        float(values[2*k]) + 1j*float(values[2*k + 1])
                    if j != k:
                        self.sigma_e[i, k, j] = \
                            float(values[2*k]) - 1j*float(values[2*k + 1])

    def _predictor_rotation(self, angle):
        """Inverse rotation of the horizontal-magnetic predictor channels."""
        hx, hy = self.channels.index("Hx"), self.channels.index("Hy")
        u = np.eye(2)
        u[hx, hx] = np.cos(np.deg2rad(self.orientation[hx, 0] - angle))
        u[hx, hy] = np.sin(np.deg2rad(self.orientation[hx, 0] - angle))
        u[hy, hx] = np.cos(np.deg2rad(self.orientation[hy, 0] - angle))
        u[hy, hy] = np.sin(np.deg2rad(self.orientation[hy, 0] - angle))
        return np.linalg.inv(u), hx, hy

    def impedance(self, angle=0.):
        """Impedance tensor and its standard error, rotated to ``angle``.

        :return: ``(z, error)`` with shapes ``(nperiods, 2, 2)``
        """
        if "Ex" not in self.channels or "Ey" not in self.channels:
            raise ValueError(
                "These transfer functions do not contain electric fields as "
                "predicted channels, so no impedance can be formed.")
        u, hx, hy = self._predictor_rotation(angle)

        ex = self.channels.index("Ex")
        ey = self.channels.index("Ey")
        npredicted = self.transfer_functions.shape[1]
        v = np.eye(npredicted)
        v[ex-2, ex-2] = np.cos(np.deg2rad(self.orientation[ex, 0] - angle))
        v[ey-2, ex-2] = np.sin(np.deg2rad(self.orientation[ex, 0] - angle))
        v[ex-2, ey-2] = np.cos(np.deg2rad(self.orientation[ey, 0] - angle))
        v[ey-2, ey-2] = np.sin(np.deg2rad(self.orientation[ey, 0] - angle))

        tf = v @ self.transfer_functions @ u.T
        sigma_s = u @ self.sigma_s @ u.T
        sigma_e = v @ self.sigma_e @ v.T

        z = np.zeros((self.periods.size, 2, 2), dtype=complex)
        z[:, 0, 0] = tf[:, ex-2, hx]   # Zxx
        z[:, 0, 1] = tf[:, ex-2, hy]   # Zxy
        z[:, 1, 0] = tf[:, ey-2, hx]   # Zyx
        z[:, 1, 1] = tf[:, ey-2, hy]   # Zyy

        var = np.zeros((self.periods.size, 2, 2))
        for row, e_idx in ((0, ex), (1, ey)):
            for col, h_idx in ((0, hx), (1, hy)):
                var[:, row, col] = np.real(sigma_e[:, e_idx-2, e_idx-2]
                                           * sigma_s[:, h_idx, h_idx])
        return z, np.sqrt(var)

    def tippers(self, angle=0.):
        """Vertical-field transfer functions (Tx, Ty) and standard errors."""
        if "Hz" not in self.channels:
            raise ValueError(
                "These transfer functions do not contain the vertical "
                "magnetic field as a predicted channel, so no tippers exist.")
        u, hx, hy = self._predictor_rotation(angle)
        hz = self.channels.index("Hz")

        tf = self.transfer_functions @ u.T
        sigma_s = u @ self.sigma_s @ u.T

        tipper = np.zeros((self.periods.size, 2), dtype=complex)
        tipper[:, 0] = tf[:, hz-2, hx]
        tipper[:, 1] = tf[:, hz-2, hy]
        var = np.zeros((self.periods.size, 2))
        var[:, 0] = np.real(self.sigma_e[:, hz-2, hz-2]*sigma_s[:, hx, hx])
        var[:, 1] = np.real(self.sigma_e[:, hz-2, hz-2]*sigma_s[:, hy, hy])
        return tipper, np.sqrt(var)


def _dms_to_degrees(text, positive_east=True):
    """``DD:MM:SS.S`` to decimal degrees (sign on the degrees field)."""
    deg, minutes, seconds = text.split(":")
    magnitude = abs(float(deg)) + (float(minutes) + float(seconds)/60.)/60.
    return np.copysign(magnitude, float(deg))


def _edi_block(dat, header, count):
    """Extract the float array following an EDI ``>HEADER`` line."""
    match = re.search(header + rf"\s*//\s*{count}\s*\n"
                      r"([\d\.E\-\+\s\n]+)\n"
                      r"\s*\n?"
                      r"\s*>", dat)
    if match is None:
        raise AttributeError(f"EDI block not found: {header}")
    return np.array([float(v) for v in match.group(1).split()])


def _rotation_matrices(angles_deg):
    """Per-period 2x2 rotation matrices, (nper, 2, 2). MT Methods, pg. 127."""
    radians = np.deg2rad(np.asarray(angles_deg, dtype=float))
    r = np.array([[np.cos(radians), np.sin(radians)],
                  [-np.sin(radians), np.cos(radians)]])
    return np.moveaxis(r, -1, 0)


class EDIFile:
    """Parsed EDI file: impedances (de-rotated to geographic) and tippers."""

    def __init__(self, filename):
        with open(filename) as f:
            dat = f.read()

        match = re.search(r'DATAID="?(.+?)"?\s*$', dat, flags=re.MULTILINE)
        if match is None:
            raise ValueError("Error in EDI parsing at the DATAID line.")
        self.station = match.group(1)
        lat = re.search(r"LAT=(-?\d+:\d\d:\d\d\.?\d*)", dat).group(1)
        lon = re.search(r"LONG=(-?\d+:\d\d:\d\d\.?\d*)", dat).group(1)
        self.coordinates = (_dms_to_degrees(lat), _dms_to_degrees(lon))

        sign = re.search(r"SIGNCONVENTION=(exp\(\+?\-? i\\omega t\))", dat)
        if sign is None:
            warnings.warn("EDI file has no sign convention information; "
                          "assuming exp(+ i omega t)", stacklevel=3)
        elif sign.group(1) != r"exp(+ i\omega t)":
            warnings.warn(f"EDI sign convention {sign.group(1)!r} is opposite "
                          "the assumed exp(+ i omega t)", stacklevel=3)

        nper = int(re.search(r"NFREQ=(\d+)", dat).group(1))
        frequencies = _edi_block(
            dat, r"\s*>!\*\*\*\*FREQUENCIES\*\*\*\*!\s*\n"
            r"\s*>FREQ\s*(?:NFREQ=\d+)?\s*(?:ORDER=DEC)?(?:ORDER=INC)?", nper)
        self.periods = 1./frequencies

        z_rot_angles = _edi_block(
            dat, r"\s*>!\*\*\*\*IMPEDANCE ROTATION ANGLES\*\*\*\*!\s*\n"
            r"\s*>ZROT", nper)
        r_z = _rotation_matrices(z_rot_angles)

        self.Z = np.zeros((nper, 2, 2), dtype=complex)
        self.Zerr = np.zeros((nper, 2, 2))
        for comp, i, j in zip(("ZXX", "ZXY", "ZYX", "ZYY"),
                              (0, 0, 1, 1), (0, 1, 0, 1)):
            real = _edi_block(dat, rf">{comp}R\s+ROT=ZROT", nper)
            imag = _edi_block(dat, rf">{comp}I\s+ROT=ZROT", nper)
            var = _edi_block(dat, rf">{comp}\.VAR\s+ROT=ZROT", nper)
            self.Z[:, i, j] = real + 1j*imag
            self.Zerr[:, i, j] = np.sqrt(var)
        # Undo (de-rotate) whatever rotation has been applied in the file so
        # the stored tensor is in geographic coordinates
        self.Z = r_z.swapaxes(1, 2) @ self.Z @ r_z
        # NOTE: rotating standard errors linearly like the tensor is the
        # legacy behavior, not statistically correct (variances rotate with
        # squared coefficients); entries it drives <= 0 are discarded in
        # _to_site3d rather than used.
        self.Zerr = r_z.swapaxes(1, 2) @ self.Zerr @ r_z

        # Tippers: rotation block sometimes missing (assume ZROT), and the
        # tippers themselves are sometimes absent entirely (left as NaN)
        try:
            t_rot_angles = _edi_block(
                dat, r"\s*>!\*\*\*\*TIPPER ROTATION ANGLES\*\*\*\*!\s*\n"
                r"\s*>TROT", nper)
        except AttributeError:
            t_rot_angles = z_rot_angles
        r_t = _rotation_matrices(t_rot_angles)

        self.T = np.full((nper, 2), np.nan, dtype=complex)
        self.Terr = np.full((nper, 2), np.nan)
        try:
            for comp, i in zip(("TX", "TY"), (0, 1)):
                real = _edi_block(dat, rf">{comp}R\.EXP\s+(?:ROT=TROT\s+)?", nper)
                imag = _edi_block(dat, rf">{comp}I\.EXP\s+(?:ROT=TROT\s+)?", nper)
                var = _edi_block(dat, rf">{comp}VAR\.EXP\s+(?:ROT=TROT\s+)?", nper)
                self.T[:, i] = real + 1j*imag
                self.Terr[:, i] = np.sqrt(var)
            self.T = np.einsum("nji,nj->ni", r_t, self.T)
            self.Terr = np.einsum("nji,nj->ni", r_t, self.Terr)
        except AttributeError:
            pass

    def impedance(self, angle=0.):
        """Impedance tensor and standard error rotated to ``angle`` degrees."""
        r = _rotation_matrices([angle])[0]
        return r @ self.Z @ r.T, r @ self.Zerr @ r.T

    def tippers(self, angle=0.):
        """Tippers (Tx, Ty) and standard errors rotated to ``angle`` degrees."""
        r = _rotation_matrices([angle])[0]
        return self.T @ r.T, self.Terr @ r.T


def _to_site3d(parsed, angle):
    """Build a Site3d from a parsed legacy file in bezpy conventions."""
    site = Site3d(parsed.station)
    site.latitude, site.longitude = parsed.coordinates

    z, err = parsed.impedance(angle=angle)
    order = np.argsort(parsed.periods)  # Site3d wants periods ascending
    site.periods = parsed.periods[order]
    z, err = z[order], err[order]
    site.Z = np.vstack([z[:, 0, 0], z[:, 0, 1], z[:, 1, 0], z[:, 1, 1]])
    with np.errstate(invalid="ignore"):
        z_var = err**2
    site.Z_var = np.vstack([z_var[:, 0, 0], z_var[:, 0, 1],
                            z_var[:, 1, 0], z_var[:, 1, 1]])
    if not np.all(np.isfinite(site.Z_var)) or np.any(site.Z_var <= 0):
        # Unusable variance information (absent or degenerate)
        site.Z_var = None
    site.calc_resistivity()
    return site


def read_zfile(filename, angle=0.):
    """Read an EMTF Z-file (.zss/.zrr/.zmm) into a :class:`Site3d`.

    :param filename: path to the Z-file
    :param angle: rotate the channels to this angle in degrees east of
        geographic north (default 0: geographic)
    :return: Site3d with periods ascending and Z as (xx, xy, yx, yy)
    """
    return _to_site3d(ZFile(filename), angle)


def read_edi(filename, angle=0.):
    """Read an EDI transfer-function file into a :class:`Site3d`.

    Any rotation recorded in the file is undone first, so ``angle=0``
    (default) returns geographic coordinates.

    :param filename: path to the EDI file
    :param angle: rotate to this angle in degrees east of geographic north
    :return: Site3d with periods ascending and Z as (xx, xy, yx, yy)
    """
    return _to_site3d(EDIFile(filename), angle)
