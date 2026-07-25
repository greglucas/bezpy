"""Tests for bezpy.tl.network: the GIC nodal-admittance (Lehtinen-Pirjola) solve.

The EPRI-21 benchmark checks below (KCL conservation, symmetric-branch
equality, an isolated ungrounded winding carrying no current, and the
GSU-transformer cross-check against real PowerWorld-computed values) are
cross-validated in the session that added this module against
`lanl-ansi/PowerModelsGMD.jl`'s own EPRI-21 topology and verification data
-- see VALIDATION.md item 1.1 for the full derivation and the well-understood
factor-of-3 (single-line-equivalent total vs NERC TPL-007's per-phase
"effective GIC" reporting convention) that reconciles the two.
"""

import json
from importlib.resources import files

import numpy as np
import pytest
import shapely.geometry

from bezpy.tl import TransmissionLine
from bezpy.tl.kernels import apply_voltage_matrix, voltage_matrix
from bezpy.tl.network import Network, apply_gic_matrix, effective_gic, gic_matrix


def test_toy_series_loop():
    """Two substations, one line, one grounded winding each: a plain series
    loop hand-solvable with Ohm's law -- I = V / sum(R)."""
    substations = [{"id": "A", "resistance": 1.0}, {"id": "B", "resistance": 2.0}]
    transformers = [
        {
            "id": "TA",
            "type": "gsu",
            "hi_bus": 1,
            "lo_bus": 99,
            "substation": "A",
            "r_hi": 0.5,
            "r_lo": None,
        },
        {
            "id": "TB",
            "type": "gsu",
            "hi_bus": 2,
            "lo_bus": 98,
            "substation": "B",
            "r_hi": 1.5,
            "r_lo": None,
        },
    ]
    branches = [{"id": "L1", "from_bus": 1, "to_bus": 2, "resistance": 3.0}]
    net = Network(substations, transformers, branches)

    V_line = 100.0  # volts, driving bus1 -> bus2
    R_total = 3.0 + 0.5 + 1.5 + 1.0 + 2.0  # line + both windings + both groundings
    I_expected = V_line / R_total

    node_v = net.solve(np.array([V_line]))
    neutral = net.neutral_currents(node_v)
    # One continuous loop current: it enters the bus (out of ground) at one
    # substation and leaves the bus (into ground) at the other, so the two
    # transformers' (bus - substation) currents have opposite sign.
    np.testing.assert_allclose(neutral, [-I_expected, I_expected], rtol=1e-10)

    # KCL: exact by construction for a series loop
    Y, B = net.admittance()
    resid = Y @ node_v - (B @ np.array([V_line]))
    np.testing.assert_allclose(resid, 0.0, atol=1e-9)


def test_toy_ungrounded_gsu_carries_nothing():
    """A transformer with both windings ungrounded (r_hi=r_lo=None) is fully
    isolated from the DC network -- it must not even allocate a node."""
    substations = [{"id": "A", "resistance": 1.0}]
    transformers = [
        {
            "id": "T",
            "type": "gsu",
            "hi_bus": 1,
            "lo_bus": 2,
            "substation": None,
            "r_hi": None,
            "r_lo": None,
        }
    ]
    net = Network(substations, transformers, branches=[])
    assert net.n_nodes == 1  # just the one substation; buses 1,2 never referenced
    assert net.neutral_currents(np.array([0.0])) == 0.0


def test_invalid_transformer_type():
    with pytest.raises(ValueError, match="type must be one of"):
        Network(
            [{"id": "A", "resistance": 1.0}],
            [{"id": "T", "type": "delta-delta", "hi_bus": 1, "lo_bus": 2}],
            [],
        )


@pytest.fixture(scope="module")
def epri21():
    return Network.epri21_benchmark()


# Line-induced voltages (volts) for the EPRI-21 benchmark's own reference
# field, in `epri21.branches` order (see VALIDATION.md item 1.1); a fixed
# input vector shared by every test below that needs a driven solve.
EPRI21_LINE_V = np.array(
    [
        120.60003474069319,
        131.6938150979946,
        131.6938150979946,
        321.2612206582465,
        190.9859010213294,
        -20.137238063195745,
        160.17017349968367,
        -129.27518497259706,
        191.1051972595808,
        191.1051972595808,
        1.4850062354159403,
        -155.55611343458276,
        158.17375406876596,
        -93.15718079092285,
        169.82073202204822,
    ]
)


def test_epri21_structure(epri21):
    assert len(epri21.substations) == 8
    assert len(epri21.transformers) == 15
    assert len(epri21.branches) == 15
    types = {t["type"] for t in epri21.transformers}
    assert types == {"wye-wye", "gsu", "auto"}


def test_epri21_kcl_satisfied(epri21):
    node_v = epri21.solve(EPRI21_LINE_V)
    Y, B = epri21.admittance()
    resid = Y @ node_v - B @ EPRI21_LINE_V
    np.testing.assert_allclose(resid, 0.0, atol=1e-8)


def test_epri21_isolated_transformer_carries_nothing(epri21):
    """xf-1-2-0 is a wye-delta GSU (ungrounded) -- zero neutral current."""
    node_v = epri21.solve(EPRI21_LINE_V)
    neutral = epri21.neutral_currents(node_v)
    idx = next(i for i, t in enumerate(epri21.transformers) if t["id"] == "xf-1-2-0")
    assert neutral[idx] == 0.0


def test_epri21_symmetric_parallel_transformers_match(epri21):
    """Identical parallel transformers (same buses, same R, no EMF of their
    own) must carry exactly equal current -- a hard equality, not a
    tolerance, since it follows from Y's exact symmetry."""
    node_v = epri21.solve(EPRI21_LINE_V)
    neutral = epri21.neutral_currents(node_v)
    by_id = {t["id"]: i for i, t in enumerate(epri21.transformers)}
    for a, b in [
        ("xf-4-3-1", "xf-4-3-2"),
        ("xf-4-3-3", "xf-4-3-4"),
        ("xf-5-20-5", "xf-5-20-6"),
        ("xf-15-16-11", "xf-15-16-12"),
    ]:
        assert neutral[by_id[a]] == neutral[by_id[b]]


def test_epri21_matches_published_gsu_currents(epri21):
    """Cross-check against lanl-ansi/PowerModelsGMD.jl's PowerWorld-computed
    verification data for this exact topology (EpriCase_Mar2012_GICXformer_
    1Vkm.csv), for the six single-grounded-winding (GSU) transformers where
    no additional hi/lo combination formula is needed.

    That data reports GIC per phase; bezpy's Network solves the
    single-line-equivalent circuit (all three phases lumped), so the two
    differ by a consistent, verified factor of 3 (see VALIDATION.md 1.1).
    """
    node_v = epri21.solve(EPRI21_LINE_V)
    neutral = epri21.neutral_currents(node_v)
    by_id = {t["id"]: i for i, t in enumerate(epri21.transformers)}
    published = {  # GICXFIEffective1, Amps (per phase)
        "xf-6-7-7": 59.090,
        "xf-6-8-8": 59.090,
        "xf-12-13-9": 22.373,
        "xf-12-14-10": 22.373,
        "xf-17-18-13": 31.548,
        "xf-17-19-14": 31.548,
    }
    for tid, ref in published.items():
        mine = abs(neutral[by_id[tid]]) / 3.0
        assert mine == pytest.approx(ref, rel=0.005)


def _epri21_lines_and_sites():
    """TransmissionLines from the benchmark's own bus geography, tagged with
    metadata["id"] matching each branch, plus 4 corner sites spanning the
    network -- for exercising the real voltage_matrix -> gic_matrix pipeline."""
    net = Network.epri21_benchmark()
    with files("bezpy.tl").joinpath("data", "epri21_network.json").open() as f:
        raw = json.load(f)
    latlon = {b["id"]: (b["latitude"], b["longitude"]) for b in raw["buses"]}
    lines = []
    for b in net.branches:
        lat0, lon0 = latlon[b["from_bus"]]
        lat1, lon1 = latlon[b["to_bus"]]
        line = TransmissionLine(
            shapely.geometry.LineString([(lon0, lat0), (lon1, lat1)]), metadata={"id": b["id"]}
        )
        lines.append(line)

    lats = [ll[0] for ll in latlon.values()]
    lons = [ll[1] for ll in latlon.values()]
    pad = 1.0
    corners = np.array(
        [
            [min(lats) - pad, min(lons) - pad],
            [min(lats) - pad, max(lons) + pad],
            [max(lats) + pad, min(lons) - pad],
            [max(lats) + pad, max(lons) + pad],
        ]
    )
    return net, lines, corners


def test_gic_matrix_matches_direct_solve():
    """apply_gic_matrix(E, K) must exactly reproduce composing voltage_matrix
    and Network.solve/neutral_currents directly -- the linear-kernel
    factorization invariant every bezpy kernel relies on."""
    net, lines, site_xys = _epri21_lines_and_sites()
    for line in lines:
        line.set_delaunay_weights(site_xys)

    # A uniform 1000 mV/km (1 V/km) eastward field: identical at every
    # corner site, so Delaunay interpolation reproduces it exactly anywhere
    # inside the hull.
    E = np.array([[[0.0, 1000.0]] * len(site_xys)])  # (1 time, 4 sites, 2)

    K, valid = gic_matrix(net, lines, len(site_xys), how="delaunay")
    from_kernel = apply_gic_matrix(E, K, valid)[0]

    L, line_valid = voltage_matrix(lines, len(site_xys), how="delaunay")
    line_v = apply_voltage_matrix(E, L, line_valid)[0]
    node_v = net.solve(line_v)
    direct = net.neutral_currents(node_v)

    np.testing.assert_allclose(from_kernel, direct, rtol=1e-8, atol=1e-10)
    assert valid.all()  # every branch has a matching, fully-covered line


def test_gic_matrix_missing_line_raises():
    net = Network.epri21_benchmark()
    line = TransmissionLine(
        shapely.geometry.LineString([(-90, 33), (-89, 34)]), metadata={"id": "not-a-real-branch"}
    )
    with pytest.raises(ValueError, match="no matching line"):
        gic_matrix(net, [line], n_sites=2)


def test_effective_gic():
    np.testing.assert_allclose(effective_gic(10.0, -4.0, a=1.0, b=0.5), 8.0)
    np.testing.assert_allclose(effective_gic([10.0, -10.0], 0.0), [10.0, 10.0])


def test_network_rejects_nonpositive_resistance():
    with pytest.raises(ValueError, match="finite and positive"):
        Network(substations=[{"id": "S", "resistance": 0.0}], transformers=[], branches=[])


def test_network_rejects_ungrounded_conductive_component():
    with pytest.raises(ValueError, match="no grounded substation"):
        Network(
            substations=[],
            transformers=[],
            branches=[{"id": "L", "from_bus": "A", "to_bus": "B", "resistance": 1.0}],
        )


def test_network_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate branch"):
        Network(
            substations=[{"id": "S", "resistance": 1.0}],
            transformers=[],
            branches=[
                {"id": "L", "from_bus": "A", "to_bus": "B", "resistance": 1.0},
                {"id": "L", "from_bus": "B", "to_bus": "C", "resistance": 1.0},
            ],
        )
