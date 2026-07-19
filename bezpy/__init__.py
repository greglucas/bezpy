"""
bezpy: Python module for processing and analyzing magnetic field (B),
       electric field (E), and impedance (Z) data.
"""
from . import mag, mt, pipeline, plot, stats, storms, tl, validation
from .pipeline import Nowcast, calc_geoelectric_field, calc_voltages

# The data-acquisition subpackage relies on optional dependencies (pooch,
# xarray, a netCDF engine). Only expose bezpy.data when they are installed so
# the base package stays importable without them; accessing bezpy.data without
# them raises the actionable install hint below instead of an AttributeError.
try:
    from . import data
except ImportError as exc:
    _DATA_IMPORT_ERROR = str(exc)

    def __getattr__(name):
        if name == "data":
            raise ImportError(
                "bezpy.data requires optional dependencies; install them with "
                "`pip install 'bezpy[data]'` "
                f"(import failed with: {_DATA_IMPORT_ERROR})")
        raise AttributeError(f"module 'bezpy' has no attribute {name!r}")

__version__ = '0.5.0'
