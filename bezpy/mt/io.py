"""Input/output functions for IRIS magnetotelluric data."""

__all__ = ["read_xml", "read_1d_usgs_profile", "get_1d_site"]

import glob
import datetime
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from . import Site3d
from . import Site1d

import pkg_resources
DATA_PATH_1d = pkg_resources.resource_filename('bezpy', 'mt/data_1d') + "/"

####################
# Helper functions
####################
def convert_float(s):
    try:
        return float(s)
    except:
        return None

def convert_int(s):
    try:
        return int(s)
    except:
        return None

def convert_datetime(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except:
        return None

def get_text(base, name):
    try:
        return base.find(name).text
    except:
        return None

def parse_data(data):
    # For storing periods in a dictionary
    periods = {}
    # Iterate through the periods
    for period in data:
        period_length = float(period.attrib['value'])
        periods[period_length] = {}
        df_dict = periods[period_length]

        for item in period:
            for data in item:
                column_name = item.tag + "_"
                if 'name' in data.attrib:
                    column_name += data.attrib['name']
                else:
                    column_name += data.attrib['output'] + "_" + data.attrib['input']

                if item.attrib['type'] == 'complex':
                    text = data.text.split()
                    column_value = float(text[0]) + float(text[1])*1j
                elif item.attrib['type'] == 'real':
                    column_value = float(data.text)
                else:
                    raise ValueError("Error parsing the item type:",
                                     item.attrib['type'],
                                     "\nShould be either \"real\" or \"complex\"")

                column_name = ''.join(i for i in column_name if not i.isdigit())
                df_dict[column_name.lower()] = column_value

    # Create a DataFrame for the data that was stored in the dictionary
    df = pd.DataFrame.from_dict(periods, orient='index')
    df.index.name = "period"

    return df



def read_xml(fname):
    """Read in an IRIS xml file and return a Site3d object"""
    root = ET.parse(fname).getroot()

    site_xml = root.find("Site")
    name = get_text(site_xml, "Id")
    # Creating the object
    site = Site3d(name)

    loc = site_xml.find("Location")
    site.latitude = convert_float(get_text(loc, "Latitude"))
    site.longitude = convert_float(get_text(loc, "Longitude"))
    site.elevation = convert_float(get_text(loc, "Elevation"))
    site.declination = convert_float(get_text(loc, "Declination"))

    site.start_time = convert_datetime(get_text(site_xml, "Start"))
    site.end_time = convert_datetime(get_text(site_xml, "End"))

    quality = site_xml.find("DataQualityNotes")
    site.rating = convert_int(get_text(quality, "Rating"))
    site.min_period = convert_float(get_text(quality, "GoodFromPeriod"))
    site.max_period = convert_float(get_text(quality, "GoodToPeriod"))

    site.quality_flag = convert_int(get_text(site_xml, "DataQualityWarnings/Flag"))

    site.sign_convention = -1 if "-" in get_text(root, "ProcessingInfo/SignConvention") else 1

    # Get all the data in a pandas dataframe
    site.data = parse_data(root.find("Data"))

    site.periods = np.array(site.data.index)
    site.Z = np.vstack([site.data['z_zxx'], site.data['z_zxy'],
                        site.data['z_zyx'], site.data['z_zyy']])

    # TODO: Change this from a default variance of ones
    # XXX: May need to eliminate the need for variance?

    try:
        site.Z_var = np.vstack([site.data['z.var_zxx'], site.data['z.var_zxy'],
                                site.data['z.var_zyx'], site.data['z.var_zyy']])
    except:
        # No variance in the data fields
        site.Z_var = None

    site.calc_resisitivity()
    return site

def read_1d_usgs_profile(fname):
    """Reads in a USGS conductivity profile.

    These are downloaded from:
    https://geomag.usgs.gov/conductivity/index.php

    Note that they are of a specific format, with thicknesses and conductivity
    listed. This should be adaptable to other 1d profiles from different locations
    with minor modifications to return a new 1d site object.
    """

    conductivities = []
    thicknesses = []
    # Just keep the last part of the file name
    profile_name = fname.strip(".txt").split("_")[-1]

    with open(fname, 'r') as f:
        for line in f:
            if line[0] == "*":
                continue
            # Moved past all comment lines
            # First line is supposed to be the number of layers
            num_layers = int(line.split()[0])
            f.readline() # Spaces between each set of points

            for i in range(num_layers):
                # Alternates conductivity/depth
                conductivities.append(float(f.readline().split()[0]))
                thicknesses.append(float(f.readline().split()[0]))
                f.readline()
            conductivities.append(float(f.readline().split()[0]))

            # Done with the file, so create the site and return it
            site = Site1d(name=profile_name,
                          thicknesses=thicknesses,
                          resistivities=[1./x for x in conductivities])
            return site

_sites1d = {}
for fname in glob.glob(DATA_PATH_1d + "earth_model_*.txt"):
    site = read_1d_usgs_profile(fname)
    _sites1d[site.name] = site
# Make AK-1 default to AK-1A
_sites1d["AK1"] = _sites1d["AK1A"]

def get_1d_site(name):
    # Test for dropping the "-" in the name as well
    newname = "".join(name.split("-"))
    if newname in _sites1d:
        return _sites1d[newname]

    raise ValueError("No 1d site profile with the name: " + name)
