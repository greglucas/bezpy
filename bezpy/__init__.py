"""
bezpy: Python module for processing and analyzing magnetic field (B),
       electric field (E), and impedance (Z) data.
"""
from . import mag, mt, storms, tl

# The data-acquisition subpackage relies on optional dependencies (pooch,
# xarray, a netCDF engine). Only expose bezpy.data when they are installed so
# the base package stays importable without them. Users can always
# `import bezpy.data` directly to get the actionable "install bezpy[data]" error.
try:
    from . import data
except ImportError:
    data = None

__version__ = '0.2.0'
