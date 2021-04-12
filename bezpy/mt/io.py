"""Input/output functions for IRIS magnetotelluric data."""

__all__ = ["read_xml", "read_1d_usgs_profile", "get_1d_site"]

import glob
import datetime
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pkg_resources

from .site import Site1d, Site3d
from .datalogger import DataLogger

DATA_PATH_1D = pkg_resources.resource_filename('bezpy', 'mt/data_1d') + "/"


####################
# Helper functions
####################
def convert_float(s):
    """Converts values, handling bad strings."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def convert_int(s):
    """Converts values, handling bad strings."""
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def convert_datetime(s):
    """Converts values, handling bad strings."""
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except (ValueError, TypeError):
        return None


def get_text(base, name):
    """Gets the text from an xml element."""
    try:
        return base.find(name).text
    except AttributeError:
        return None


def parse_data(xml_data):
    """Parse data obtained from Anna's xml files."""
    # For storing periods in a dictionary
    periods = {}
    # Iterate through the periods
    for period in xml_data:
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

    xml_site = root.find("Site")
    name = get_text(xml_site, "Id")
    # Creating the object
    site = Site3d(name)
    # Store the parsed root xml element
    site.xml = root
    site.product_id = get_text(root, "ProductId")

    loc = xml_site.find("Location")
    site.latitude = convert_float(get_text(loc, "Latitude"))
    site.longitude = convert_float(get_text(loc, "Longitude"))
    site.elevation = convert_float(get_text(loc, "Elevation"))
    site.declination = convert_float(get_text(loc, "Declination"))

    site.start_time = convert_datetime(get_text(xml_site, "Start"))
    site.end_time = convert_datetime(get_text(xml_site, "End"))

    quality = xml_site.find("DataQualityNotes")
    site.rating = convert_int(get_text(quality, "Rating"))
    site.min_period = convert_float(get_text(quality, "GoodFromPeriod"))
    site.max_period = convert_float(get_text(quality, "GoodToPeriod"))

    site.quality_flag = convert_int(get_text(xml_site, "DataQualityWarnings/Flag"))

    site.sign_convention = -1 if "-" in get_text(root, "ProcessingInfo/SignConvention") else 1

    # Get all the data in a pandas dataframe
    site.data = parse_data(root.find("Data"))
    # Sort the index so periods are increasing
    site.data = site.data.sort_index()

    site.periods = np.array(site.data.index)
    site.Z = np.vstack([site.data['z_zxx'], site.data['z_zxy'],
                        site.data['z_zyx'], site.data['z_zyy']])
    try:
        site.Z_var = np.vstack([site.data['z.var_zxx'], site.data['z.var_zxy'],
                                site.data['z.var_zyx'], site.data['z.var_zyy']])
    except KeyError:
        # No variance in the data fields
        site.Z_var = None

    site.calc_resisitivity()

    site.datalogger = DataLogger()
    try:
        site.datalogger.runlist = get_text(xml_site, "RunList").split()
    except AttributeError:
        # No RunList means no data
        return site

    try:
        runinfo, nimsid, samplingrate = read_logger_info(site.xml)
        # No info about the nimsid from the logger read, so just return
        # without updating any information about it.
        if nimsid is None:
            return site
        site.datalogger.add_run_info(runinfo, nimsid, samplingrate)
        # Fill out the NIM System Response
        site.datalogger.nim_system_response()
    except ValueError:
        pass

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
            f.readline()  # Spaces between each set of points

            for _ in range(num_layers):
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


_SITES1D = {}
for temp_fname in glob.glob(DATA_PATH_1D + "earth_model_*.txt"):
    site1d = read_1d_usgs_profile(temp_fname)
    _SITES1D[site1d.name] = site1d
# Make AK-1 default to AK-1A
_SITES1D["AK1"] = _SITES1D["AK1A"]


def get_1d_site(name):
    """Returns the 1D site for the given name, if present."""
    # Test for dropping the "-" in the name as well
    newname = "".join(name.split("-"))
    if newname in _SITES1D:
        return _SITES1D[newname]

    raise ValueError("No 1d site profile with the name: " + name)


def read_logger_info(root):
    """Returns the run info, if present."""
    runinfo = {}
    nimsid = ""
    samplingrate = 1.0

    # Run through for each runid
    for field in root.findall("FieldNotes"):
        # runid of fieldnote
        runid = field.attrib['run']
        runinfo[runid] = {}

        try:
            nimsid = get_text(field.find('Instrument'), 'Id')
            samplingrate = convert_float(field.find('SamplingRate').text)
        except KeyError:
            pass

        # Run through E component
        for ecomp in field.findall("Dipole"):
            # Electric component name
            edir = ecomp.attrib['name']   # Ex or Ey
            # Electric dipole length
            runinfo[runid][edir] = convert_float(get_text(ecomp, "Length"))

        # set start and end datetime
        runinfo[runid]['Start'] = datetime.datetime.strptime(field.find('Start').text,
                                                             '%Y-%m-%dT%H:%M:%S')
        runinfo[runid]['End'] = datetime.datetime.strptime(field.find('End').text,
                                                           '%Y-%m-%dT%H:%M:%S')

    return runinfo, nimsid, samplingrate
