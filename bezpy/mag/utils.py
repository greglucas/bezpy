"""Magnetic field utility routines."""

__all__ = ["read_iaga", "detrend_polynomial", "write_iaga_2002",
           "get_iaga_observatory", "get_iaga_observatory_codes"]

import pandas as pd
import numpy as np

import pkg_resources
IAGA_PATH = pkg_resources.resource_filename('bezpy', 'mag/data') + "/"

def read_iaga_header(fname):
    # IAGA-2002 format defined here:
    # https://www.ngdc.noaa.gov/IAGA/vdat/IAGA2002/iaga2002format.html
    # The header record contains:
    #  - Format, Source of data, station name, iaga code, geodetic latitude,
    #    geodetic longitude, elevation, reported, sensor orientation
    #    digital sampling, data interval, data type
    header_records = {"header_length": 0}

    with open(fname, 'r') as f:
        for line in f:
            if line[0] == ' ':
                header_records["header_length"] += 1
                label = line[1:24].strip()
                description = line[24:-2].strip()
                header_records[label.lower()] = description
            else:
                return header_records

def read_iaga(fname, return_xyzf=True):
    """Reads an IAGA-2002 data file and returns data in XYZF format."""
    # Get the header information
    header_records = read_iaga_header(fname)

    # This contains 4 letters stating the reported values
    #  - XYZF, HDZF, etc...
    if len(header_records['reported']) % 4 != 0:
        raise ValueError("The header record does not contain 4 values: {0}".format(
                          header_records['reported']))

    record_length = len(header_records['reported']) // 4
    # This keeps the last letter of each reported channel
    column_names = [x for x in header_records['reported'][record_length-1::record_length]]
    # This splits the string into 4 equal parts.
    # Some reportings from USGS can be 3 letters each, rather than 1
    #column_names = [header_records['reported'][i:i+record_length]
    #                for i in range(0, len(header_records['reported']), record_length)]

    df = pd.read_csv(fname, header=header_records["header_length"],
                    delim_whitespace=True,
                    parse_dates=[[0,1]], infer_datetime_format=True,
                    index_col=0, usecols=[0,1,3,4,5,6],
                    na_values=[99999.90, 99999.0, 88888.80, 88888.00],
                    names=["Date", "Time"] + column_names)
    df.index.name = "Time"
    if (return_xyzf and
        "X" not in column_names and
        "Y" not in column_names):
        # Convert the data to XYZF format
        # Only convert HD
        if "H" not in column_names or "D" not in column_names:
            raise ValueError("Only have a converter for HDZF->XYZF\n" +
                             "Input file is: " + header_records['reported'])

        # IAGA-2002 D is reported in minutes of arc.
        df["X"] = df["H"] * np.cos(np.deg2rad(df["D"]/60.))
        df["Y"] = df["H"] * np.sin(np.deg2rad(df["D"]/60.))
        del df["H"], df["D"]
    return df

def detrend_polynomial(ys, deg=2):
    xs = np.arange(len(ys))
    poly = np.polyfit(xs, ys, deg=deg)
    # poly gives all the coefficients, polyval evaluates all these at the same xs
    return ys - np.polyval(poly, xs)

IAGA_HEADER = "\n".join([\
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
    newdf = df.copy()
    col1, col2 = newdf.columns
    newdf["DOY"] = newdf.index.dayofyear
    col3 = "Ez"
    col4 = "Etot"
    newdf["Ez"] = np.nan
    newdf["Etot"] = np.nan
    # Move DOY to the front
    newdf = newdf[["DOY", col1, col2, col3, col4]]

    with open(fname, "w") as f:
        f.write(IAGA_HEADER)
        f.write(newdf.to_string(formatters={"DOY": lambda x: "    {0:03d}   ".format(x),
                       col1: lambda x: "{0:9.2f}".format(x),
                       col2: lambda x: "{0:9.2f}".format(x),
                       col3: lambda x: "{0:9.2f}".format(99999.0),
                       col4: lambda x: "{0:9.2f}".format(99999.0)},
                       na_rep=' 99999.00',
                       header=False, index_names=False))

# Reading in IAGA codes and getting observatory information
# This file was downloaded from: http://www.intermagnet.org/imos/imotblobs-eng.php
# on 11/25/2017
# Format of the file:
# IAGA	Name	Country	Colatitute	East Longitude	Institute	GIN

_IAGA_sites = {}

with open(IAGA_PATH + "intermagnet_observatories.dat", 'r') as f:
    f.readline() # Read the header
    for line in f:
        elements = line.strip().split('\t')
        latitude = 90. - float(elements[3][:-1])
        longitude = float(elements[4][:-1])
        if longitude > 180:
            longitude -= 360
        _IAGA_sites[elements[0][:3]] = {"name": elements[1], "country": elements[2],
                                       "latitude": latitude, "longitude": longitude,
                                       "institute": elements[5], "GIN": elements[6]}

def get_iaga_observatory_codes():
    return list(_IAGA_sites.keys())

def get_iaga_observatory(iaga_code):
    try:
        return _IAGA_sites[iaga_code.upper()]
    except:
        raise ValueError("No observatory: ", iaga_code)
