"""
bezpy: Python module for processing and analyzing magnetic field (B),
       electric field (E), and impedance (Z) data.
"""
from . import mt
from . import tl
from . import mag

# Let users know if they're missing any of our hard dependencies
HARD_DEPENDENCIES = ("numpy", "pandas", "scipy", "shapely")
MISSING_DEPENDENCIES = []

for dependency in HARD_DEPENDENCIES:
    try:
        __import__(dependency)
    except ImportError as err:
        MISSING_DEPENDENCIES.append(dependency)

if MISSING_DEPENDENCIES:
    raise ImportError(
        "Missing required dependencies {0}".format(MISSING_DEPENDENCIES))
del HARD_DEPENDENCIES, MISSING_DEPENDENCIES, dependency
