"""
Magnetotelluric datalogger classes.

"""
import datetime
import numpy as np
import pandas as pd
from scipy.signal import zpk2tf, filtfilt, bilinear_zpk

# Putting this here, so we don't have to reinstantiate a client
# every time we download waveforms
_IRIS_CLIENT = None


class DataLogger:
    """DataLogger class to enable transformations from samples to practical units."""

    def __init__(self):
        self.nimsid = None
        self.runlist = []
        self.runinfo = None
        self.samplingrate = None
        self.zpk = None
        self.timedelays = None

    def download_iris_waveforms(self, name, start_time, end_time):
        """Download waveforms from the IRIS client."""
        # The only reason we are putting a global in here is because
        # of how long it takes to create the Client, and with multiple
        # downloads from different sites, we really only need one client
        # to be stored somewhere in the code
        global _IRIS_CLIENT  # pylint: disable=global-statement
        if _IRIS_CLIENT is None:
            # pylint: disable=import-error,import-outside-toplevel
            from obspy.clients.fdsn import Client
            _IRIS_CLIENT = Client("IRIS")
        from obspy.core import UTCDateTime

        # Download the stream
        # The channels are
        # E: LQN/LQE
        # B: LFN/LFE/LFZ
        stream = _IRIS_CLIENT.get_waveforms("EM", name, "*", "*",
                                            UTCDateTime(start_time),
                                            UTCDateTime(end_time))
        # Convert the stream to a pandas DataFrame
        waveforms = _convert_stream_to_df(stream)
        # Channel conversion factors and renaming
        # The conversion factors were obtained from Anna Kelbert
        # and are standard for the IRIS database

        # Magnetic Field
        waveforms[["FE", "FN", "FZ"]] *= 0.01  # nT
        # Electric Field (assuming the dipole length as 100m)
        waveforms[["QN", "QE"]] *= 2.44141221047903e-05  # mV/km

        # Process the waveforms if there is a runlist object for this DataLogger object
        if self.runlist:
            waveforms = self._process_waveforms(waveforms)

        # Renaming
        waveforms.rename(columns={"FE": "BE", "FN": "BN", "FZ": "BZ",
                                  "QE": "EE", "QN": "EN"},
                         inplace=True)
        return waveforms

    def _process_waveforms(self, waveforms):
        """Process the waveforms with the DataLogger information."""
        # Correcting electric Field with actual length of dipole
        for runid in self.runlist:
            try:
                mask = ((waveforms.index > self.runinfo[runid]['Start']) &
                        (waveforms.index < self.runinfo[runid]['End']))
                waveforms["QN"].loc[mask] *= (100.0/self.runinfo[runid]['Ex'])  # mV/km
                waveforms["QE"].loc[mask] *= (100.0/self.runinfo[runid]['Ey'])  # mV/km

            # If there's no info about length, then use default length.
            except KeyError:
                pass

        if self.zpk is None:
            return waveforms

        # Filtering waveforms by NIMS system response
        for channel in waveforms:   # loop for FE, FN, FZ, QN, QE
            for filt in self.zpk[channel]:   # loop for filter
                zval = self.zpk[channel][filt]['z']
                pval = self.zpk[channel][filt]['p']
                kval = self.zpk[channel][filt]['k']

                # convert analog zpk to digital filter
                zval, pval, kval = bilinear_zpk(zval, pval, kval, self.samplingrate)
                b, a = zpk2tf(zval, pval, kval)

                waveforms[channel] = filtfilt(b, a, waveforms[channel].interpolate())
        return waveforms

    def add_run_info(self, runinfo, nimsid, samplingrate):
        """Add run information."""
        self.runinfo = runinfo
        self.nimsid = nimsid
        self.samplingrate = samplingrate

    def nim_system_response(self):
        """reads NIMS id and sampling rate and set parameters of Butterworth filter."""

        # get logger and hardware, backbone from sysid and sampling rate (Hz)
        logger, hardware, backbone = _get_logger_info(self.nimsid, self.samplingrate)

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
            raise ValueError('Unknown sampling rate, please check!')
        self.timedelays = timedelays

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


def _convert_stream_to_df(stream):
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


def _get_logger_info(sysid, samplingrate):
    """Returns logger and hardware, backbone from sysid and sampling rate (Hz)"""

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
        raise ValueError('Invalid system ID in NIMsysRsp')

    nim1 = int(sysid[0:j])
    # we know the following NIMS series
    nimlist1 = [2106, 2311, 2405, 2406, 2501, 2502, 2503, 2508, 2509, 2606, 2611,
                2612, 1303, 1305, 1105]

    if np.isnan(nim1):
        raise ValueError('NIMS ID ' + sysid + ' does not seem to be valid. Please correct.')
    if nim1 not in nimlist1:
        raise ValueError('We do not know NIMS series ' + str(nim1) + '. Check the system ID.')

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
        raise ValueError('NIMS ID ' + sysid + ' does not seem to be valid. Please correct.')
    elif nim2 not in nimlist2:
        raise ValueError('NIMS ID ' + sysid + ' does not seem to be valid. Please correct.')

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
            raise ValueError('A possible problem with the system ID detected. HP200 data' +
                             'logger has been inferred, but the system is not 2106-1/10.\n' +
                             'Please make sure ' + sysid + ' does not have an H character.')

    return logger, hardware, backbone
