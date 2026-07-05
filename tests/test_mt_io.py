"""Tests for bezpy.mt.io XML and 1D profile parsing."""
import re

import numpy as np
import pytest

import bezpy


def test_read_xml_basic_metadata(site3d):
    assert site3d.name == "TEST01"
    assert site3d.latitude == pytest.approx(40.0)
    assert site3d.longitude == pytest.approx(-105.0)
    # Network code parsed from the ExternalUrl (.../mda/EM/TEST01 -> EM).
    assert site3d.network_code == "EM"
    assert site3d.rating == 5
    assert site3d.min_period == pytest.approx(10.0)
    assert site3d.max_period == pytest.approx(10000.0)


def test_read_xml_impedance_shape(site3d):
    assert site3d.Z.shape == (4, len(site3d.periods))
    assert site3d.Z_var.shape == site3d.Z.shape
    # Periods are sorted ascending.
    assert np.all(np.diff(site3d.periods) > 0)


def test_read_xml_channel_orientation(site3d):
    assert set(site3d.channel_orientation) == {"Bx", "By", "Bz", "Ex", "Ey"}
    assert site3d.channel_orientation["By"] == pytest.approx(90.0)


def test_read_xml_missing_site_layout(tmp_path, sample_emtf_path):
    xml = sample_emtf_path.read_text()
    stripped = re.sub(r"<SiteLayout>.*?</SiteLayout>", "", xml, flags=re.S)
    fname = tmp_path / "no_layout.xml"
    fname.write_text(stripped)
    site = bezpy.mt.read_xml(str(fname))
    # Should parse without error and simply have no channel orientations.
    assert site.channel_orientation == {}


def test_read_xml_missing_sign_convention(tmp_path, sample_emtf_path):
    xml = sample_emtf_path.read_text()
    stripped = re.sub(r"<SignConvention>.*?</SignConvention>", "", xml, flags=re.S)
    fname = tmp_path / "no_sign.xml"
    fname.write_text(stripped)
    site = bezpy.mt.read_xml(str(fname))
    assert site.sign_convention in (-1, 1)


def test_read_xml_unknown_channel_is_skipped(tmp_path, sample_emtf_path):
    xml = sample_emtf_path.read_text()
    # Inject a remote-reference channel that has no mapping.
    injected = xml.replace(
        '<Magnetic name="Hx" orientation="0.000" x="0.000" y="0.000" z="0.000"/>',
        '<Magnetic name="Rx" orientation="0.000" x="0.000" y="0.000" z="0.000"/>'
        '<Magnetic name="Hx" orientation="0.000" x="0.000" y="0.000" z="0.000"/>',
        1,
    )
    fname = tmp_path / "extra_channel.xml"
    fname.write_text(injected)
    site = bezpy.mt.read_xml(str(fname))
    # The unknown "Rx" channel is skipped, known channels still parsed.
    assert "Bx" in site.channel_orientation


def test_read_xml_sign_convention_positive(site3d):
    # "exp(+ i\omega t)" has no minus sign -> +1.
    assert site3d.sign_convention == 1


def test_get_1d_site_known_profiles():
    site = bezpy.mt.get_1d_site("CP-1")
    assert site.resistivities.size > 0
    # The dash-less form resolves to the same profile.
    assert bezpy.mt.get_1d_site("CP1").name == site.name


def test_get_1d_site_unknown_raises():
    with pytest.raises(ValueError):
        bezpy.mt.get_1d_site("NOT_A_REAL_REGION")


def test_ak1_aliases_ak1a():
    assert bezpy.mt.get_1d_site("AK-1").name == bezpy.mt.get_1d_site("AK-1A").name
