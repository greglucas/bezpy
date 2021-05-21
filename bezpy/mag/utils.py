"""Magnetic field utility routines."""

__all__ = ["read_iaga", "detrend_polynomial", "filter_signal", "write_iaga_2002",
           "get_iaga_observatory", "get_iaga_observatory_codes"]

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

import pkg_resources
IAGA_PATH = pkg_resources.resource_filename('bezpy', 'mag/data') + "/"


def read_iaga_header(fname):
    """
    IAGA-2002 format defined here:
    https://www.ngdc.noaa.gov/IAGA/vdat/IAGA2002/iaga2002format.html
    The header record contains:
     - Format, Source of data, station name, iaga code, geodetic latitude,
       geodetic longitude, elevation, reported, sensor orientation
       digital sampling, data interval, data type
    """
    header_records = {"header_length": 0}

    with open(fname, 'r') as openfile:
        for newline in openfile:
            if newline[0] == ' ':
                header_records["header_length"] += 1
                label = newline[1:24].strip()
                description = newline[24:-2].strip()
                header_records[label.lower()] = description
            else:
                return header_records
    return header_records


def read_iaga(fname, return_xyzf=True, return_header=False):
    """Reads an IAGA-2002 data file and returns data in XYZF format."""
    # Get the header information
    header_records = read_iaga_header(fname)

    # This contains 4 letters stating the reported values
    #  - XYZF, HDZF, etc...
    if len(header_records['reported']) % 4 != 0:
        raise ValueError("The header record does not contain 4 values: {0}".format(
            header_records['reported']))

    record_length = len(header_records['reported']) // 4

    # This splits the string into 4 equal parts.
    # Some reportings from USGS can be 3 letters each, rather than 1
    # This keeps the last letter of each reported channel
    column_names = [x for x in header_records['reported'][record_length-1::record_length]]

    # Dealing with duplicate column names and appending the count if necessary
    seen_count = {}
    for i, col in enumerate(column_names):
        if col in seen_count:
            column_names[i] += str(seen_count[col])
            seen_count[col] += 1
        else:
            seen_count[col] = 1

    df = pd.read_csv(fname, header=header_records["header_length"],
                     delim_whitespace=True,
                     parse_dates=[[0, 1]], infer_datetime_format=True,
                     index_col=0, usecols=[0, 1, 3, 4, 5, 6],
                     na_values=[99999.90, 99999.0, 88888.80, 88888.00],
                     names=["Date", "Time"] + column_names)
    df.index.name = "Time"
    if (return_xyzf and "X" not in column_names and "Y" not in column_names):
        # Convert the data to XYZF format
        # Only convert HD
        if "H" not in column_names or "D" not in column_names:
            raise ValueError("Only have a converter for HDZF->XYZF\n" +
                             "Input file is: " + header_records['reported'])

        # IAGA-2002 D is reported in minutes of arc.
        df["X"] = df["H"] * np.cos(np.deg2rad(df["D"]/60.))
        df["Y"] = df["H"] * np.sin(np.deg2rad(df["D"]/60.))
        del df["H"], df["D"]
    
    if return_header:
        return df, header_records
    else:
        return df


def detrend_polynomial(data, deg=2):
    """Detrends the data with a polynomial of specified degree (default: 2)"""

    xvals = np.arange(len(data))
    poly = np.polyfit(xvals, data, deg=deg)
    # poly gives all the coefficients, polyval evaluates all these at the same xs
    return data - np.polyval(poly, xvals)


def filter_signal(data, sample_freq=1./60, lowcut=1e-4, highcut=1e-1, order=3):
    """A convenience method to apply butterworth filters to the data.

       data: array of samples at the given sampling frequency (no gaps)

       sample_freq: sampling frequency that data is given in (Hz)
                    [Default: 1/60 Hz (60 s sample period)]

       lowcut: low cutoff frequency (Hz)
               [Default: 1e-4 Hz, (10,000 s period)]

       highcut: high cutoff frequency (Hz)
                [Default: 1e-1 Hz, (10 s period)]

       order: Order of the butterworth filter to use
              [Default: 3]
    """

    nyquist = 0.5*sample_freq
    low = lowcut / nyquist
    high = highcut / nyquist

    # Create the butterworth filters
    # If the sample frequency is high enough, make it a band pass
    # otherwise, it will be a highpass filter
    # TODO: Might want to add in a low pass filter option?
    #       Or does the bandpass account for that automatically?
    if sample_freq > highcut:
        b, a = butter(order, [low, high], btype='band')
    else:
        b, a = butter(order, low, btype='highpass')

    # Apply the filter coefficients to the data and return
    return filtfilt(b, a, data)


IAGA_HEADER = "\n".join([
    " Format                 IAGA-2002                                    |",
    " Source of Data         Greg Lucas                                   |",
    " Station Name           Random                                       |",
    " IAGA CODE              RAN                                          |",
    " Geodetic Latitude      0.000                                        |",
    " Geodetic Longitude     0.000                                        |",
    " Elevation              0                                            |",
    " Reported               XYZF                                         |",
    " Sensor Orientation     HDZF                                         |",
    " Digital Sampling       0.01 second                                  |",
    " Data Interval Type     filtered 1-minute (00:15-01:45)              |",
    " Data Type              definitive                                   |",
    "DATE       TIME         DOY     Ex        Ey        RANZ      RANF   |\n"])


def write_iaga_2002(df, fname):
    """Write the dataframe out in IAGA-2002 format."""

    newdf = df.copy()
    col1, col2 = newdf.columns
    newdf["DOY"] = newdf.index.dayofyear
    col3 = "Ez"
    col4 = "Etot"
    newdf["Ez"] = np.nan
    newdf["Etot"] = np.nan
    # Move DOY to the front
    newdf = newdf[["DOY", col1, col2, col3, col4]]

    with open(fname, "w") as openfile:
        openfile.write(IAGA_HEADER)
        openfile.write(
            newdf.to_string(
                formatters={"DOY": "    {0:03d}   ".format,
                            col1: "{0:9.2f}".format,
                            col2: "{0:9.2f}".format,
                            col3: "{0:9.2f}".format(99999.0),
                            col4: "{0:9.2f}".format(99999.0)},
                na_rep=' 99999.00', header=False, index_names=False))


# Reading in IAGA codes and getting observatory information
# This file was downloaded from: http://www.intermagnet.org/imos/imotblobs-eng.php
# on 11/25/2017
# Format of the file:
# IAGA	Name	Country	Colatitute	East Longitude	Institute	GIN
_IAGA_SITES = {}
with open(IAGA_PATH + "intermagnet_observatories.dat", 'r') as f:
    f.readline()  # Read the header
    for line in f:
        elements = line.strip().split('\t')
        latitude = 90. - float(elements[3][:-1])
        longitude = float(elements[4][:-1])
        if longitude > 180:
            longitude -= 360
        _IAGA_SITES[elements[0][:3]] = {"name": elements[1], "country": elements[2],
                                        "latitude": latitude, "longitude": longitude,
                                        "institute": elements[5], "GIN": elements[6]}


def get_iaga_observatory_codes():
    """Get a list of all the IAGA obsevatory codes available."""
    return list(_IAGA_SITES.keys())


def get_iaga_observatory(iaga_code):
    """Get the IAGA observatory information from the code."""
    try:
        return _IAGA_SITES[iaga_code.upper()]
    except KeyError:
        raise ValueError("No observatory: ", iaga_code)
