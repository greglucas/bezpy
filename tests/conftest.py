"""Shared pytest fixtures for the bezpy test suite."""

import pathlib

import numpy as np
import pytest

import bezpy

DATA_DIR = pathlib.Path(__file__).parent / "data"
SAMPLE_EMTF = DATA_DIR / "sample_emtf.xml"


@pytest.fixture
def sample_emtf_path():
    """Path to the bundled synthetic EMTF XML file."""
    return SAMPLE_EMTF


@pytest.fixture
def site3d(sample_emtf_path):
    """A Site3d parsed from the bundled EMTF XML fixture."""
    return bezpy.mt.read_xml(str(sample_emtf_path))


@pytest.fixture
def halfspace_site():
    """A 1D uniform half-space of 100 ohm-m (analytic reference)."""
    return bezpy.mt.Site1d("halfspace", thicknesses=[1e9], resistivities=[100.0])


@pytest.fixture
def rng():
    """A seeded random number generator for reproducible signals."""
    return np.random.default_rng(1234)
