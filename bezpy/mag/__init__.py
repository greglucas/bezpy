"""Magnetic field module, for working with magnetic fields and parsing observatory data."""
from .processing import combine_observatories, prepare, trim_pad  # noqa: F401
from .secs import SECSInterpolator  # noqa: F401
from .utils import *  # noqa: F403
