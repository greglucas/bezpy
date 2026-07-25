"""Geomagnetically induced current (GIC): the DC-equivalent network solve.

Transmission lines and grounded transformer windings form a resistive DC
network driven by the line-integrated geoelectric field (the same quantity
:func:`bezpy.tl.kernels.voltage_matrix` already computes). This module
solves that network -- the Lehtinen & Pirjola (1985) problem, in the
nodal-admittance form shown equivalent to the original two-matrix
formulation by Boteler & Pirjola (2017) -- for the current through each
transformer's grounding connection, the quantity NERC TPL-007 assessments
and transformer heating studies actually need.

**Network topology.** Two kinds of node: one per substation (its local
grounding grid, connected to remote earth through a grounding resistance),
and one per bus (a transmission-line connection point at some voltage
level). Three kinds of edge:

* a transmission line, bus-to-bus, carrying the line's induced EMF;
* a two-winding transformer's grounded winding(s), bus-to-substation, no
  EMF of its own -- ``"wye-wye"`` (both windings grounded, two edges),
  ``"gsu"`` (a generator step-up; the grounded side only -- pass ``r_hi=None``
  or ``r_lo=None`` for the ungrounded side, or both ``None`` for a fully
  isolated delta/ungrounded-wye winding pair, which carries no GIC at all);
* an autotransformer's series winding (bus-to-bus) and common winding
  (bus-to-substation) -- ``"auto"``.

A transformer's *neutral current* -- the sum of its grounded windings'
currents (or the common winding's current alone, for an autotransformer,
since the series winding never touches ground) -- is what
:func:`gic_matrix` reports per transformer; use
:meth:`Network.winding_currents` for the individual hi/lo/series/common
breakdown (needed for the TPL-007 effective-GIC combination,
:func:`effective_gic`).

The network is linear in the driving line voltages exactly like the Level-1
voltage kernel, so it composes with :func:`bezpy.tl.kernels.voltage_matrix`
into one more build-once/apply-per-chunk kernel (see
:mod:`bezpy.tl.kernels` for the general pattern).

References: Lehtinen, M., & Pirjola, R. (1985). Currents produced in earthed
conductor networks by geomagnetically-induced electric fields. Annales
Geophysicae, 3(4), 479-484. Boteler, D. H., & Pirjola, R. J. (2017).
Modeling geomagnetically induced currents. Space Weather, 15(1), 258-276.
https://doi.org/10.1002/2016SW001499. Horton, R., Boteler, D., Overbye, T.
J., Pirjola, R., & Dugan, R. C. (2012). A test case for the calculation of
geomagnetically induced currents. IEEE Transactions on Power Delivery,
27(4), 2368-2373. https://doi.org/10.1109/TPWRD.2012.2206407.
"""

__all__ = ["Network", "gic_matrix", "apply_gic_matrix", "effective_gic"]

import json
from collections import Counter
from importlib.resources import files

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .kernels import voltage_matrix

_GROUNDED_PARTS = ("hi", "lo", "common")  # winding parts that reach a substation node
_TRANSFORMER_TYPES = ("wye-wye", "gsu", "auto")


def _records(items):
    """Accept a list of dicts, or anything with ``.to_dict("records")``
    (a pandas DataFrame), and return a list of plain dicts."""
    if hasattr(items, "to_dict"):
        return items.to_dict("records")
    return [dict(item) for item in items]


def _positive_resistance(value, label):
    """Return a finite positive resistance or raise an actionable error."""
    try:
        resistance = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} resistance must be a finite positive number") from None
    if not np.isfinite(resistance) or resistance <= 0:
        raise ValueError(f"{label} resistance must be finite and positive")
    return resistance


def _require_unique_ids(records, kind):
    """Validate required, unique record identifiers."""
    ids = [record.get("id") for record in records]
    if any(identifier is None for identifier in ids):
        raise ValueError(f"every {kind} must have a non-null 'id'")
    duplicates = sorted(
        (identifier for identifier, count in Counter(ids).items() if count > 1),
        key=str,
    )
    if duplicates:
        raise ValueError(f"duplicate {kind} id(s): {duplicates[:5]}")


class Network:
    """A DC-equivalent power network for GIC.

    :param substations: iterable of dicts (or a DataFrame) with ``id`` and
        ``resistance`` (grounding resistance to remote earth, Ohms)
    :param transformers: iterable of dicts (or a DataFrame), each with
        ``id``, ``hi_bus``, ``lo_bus``, ``type`` (``"wye-wye"``, ``"gsu"``,
        or ``"auto"``), ``substation`` (id; required unless every winding is
        ungrounded), and resistances in Ohms: ``r_hi``/``r_lo`` for
        ``"wye-wye"``/``"gsu"`` (either may be ``None`` for an ungrounded
        winding), or ``r_series``/``r_common`` for ``"auto"``
    :param branches: iterable of dicts (or a DataFrame), each a
        transmission line: ``id``, ``from_bus``, ``to_bus``, ``resistance``
        (Ohms). ``id`` should match a ``TransmissionLine``'s
        ``metadata["id"]`` for :func:`gic_matrix` to find its driving
        voltage.
    """

    def __init__(self, substations, transformers, branches):
        self.substations = _records(substations)
        self.transformers = _records(transformers)
        self.branches = _records(branches)
        _require_unique_ids(self.substations, "substation")
        _require_unique_ids(self.transformers, "transformer")
        _require_unique_ids(self.branches, "branch")
        substation_ids = {s["id"] for s in self.substations}
        for s in self.substations:
            s["resistance"] = _positive_resistance(s.get("resistance"), f"substation {s['id']!r}")
        for b in self.branches:
            for key in ("from_bus", "to_bus"):
                if b.get(key) is None:
                    raise ValueError(f"branch {b['id']!r} must have {key!r}")
            if b["from_bus"] == b["to_bus"]:
                raise ValueError(f"branch {b['id']!r} cannot connect a bus to itself")
            b["resistance"] = _positive_resistance(b.get("resistance"), f"branch {b['id']!r}")
        for t in self.transformers:
            if t.get("type") not in _TRANSFORMER_TYPES:
                raise ValueError(
                    f"transformer {t.get('id')!r}: type must be one of "
                    f"{_TRANSFORMER_TYPES}, got {t.get('type')!r}"
                )
            if t["type"] == "auto":
                resistance_keys = ("r_series", "r_common")
                missing = [key for key in resistance_keys if t.get(key) is None]
                if missing:
                    raise ValueError(f"autotransformer {t['id']!r} requires {', '.join(missing)}")
                for key in ("hi_bus", "lo_bus"):
                    if t.get(key) is None:
                        raise ValueError(f"autotransformer {t['id']!r} must have {key!r}")
            else:
                resistance_keys = ("r_hi", "r_lo")
                for resistance_key, bus_key in (("r_hi", "hi_bus"), ("r_lo", "lo_bus")):
                    if t.get(resistance_key) is not None and t.get(bus_key) is None:
                        raise ValueError(
                            f"transformer {t['id']!r} has {resistance_key} but no {bus_key}"
                        )
            grounded = False
            for key in resistance_keys:
                if t.get(key) is not None:
                    t[key] = _positive_resistance(t[key], f"transformer {t['id']!r} {key}")
                    if key != "r_series":
                        grounded = True
            if t["type"] == "auto" and not grounded:
                # r_common is required above in practice, but keep this
                # explicit so a future transformer type cannot bypass it.
                raise ValueError(f"autotransformer {t['id']!r} needs a grounded common winding")
            if grounded:
                if t.get("substation") not in substation_ids:
                    raise ValueError(
                        f"transformer {t['id']!r} references unknown or missing "
                        f"substation {t.get('substation')!r}"
                    )

        self._node_of = {}
        for s in self.substations:
            self._node("sub", s["id"])

        # Transmission-line edges, in self.branches order (this order is
        # the contract gic_matrix's site-driven voltages must line up with).
        self._branch_edges = [
            (
                self._node("bus", b["from_bus"]),
                self._node("bus", b["to_bus"]),
                float(b["resistance"]),
            )
            for b in self.branches
        ]

        # Grounded-winding edges: bus -> substation (or bus -> bus for an
        # autotransformer's series winding, which touches no substation).
        self._winding_edges = []  # dicts: t (transformer idx), part, i, j, r
        for ti, t in enumerate(self.transformers):
            sub = self._node("sub", t["substation"]) if t.get("substation") is not None else None
            if t["type"] in ("wye-wye", "gsu"):
                for part, bus_key, r_key in (("hi", "hi_bus", "r_hi"), ("lo", "lo_bus", "r_lo")):
                    r = t.get(r_key)
                    if r is not None:
                        self._winding_edges.append(
                            dict(
                                t=ti, part=part, i=self._node("bus", t[bus_key]), j=sub, r=float(r)
                            )
                        )
            else:  # "auto"
                self._winding_edges.append(
                    dict(
                        t=ti,
                        part="series",
                        i=self._node("bus", t["hi_bus"]),
                        j=self._node("bus", t["lo_bus"]),
                        r=float(t["r_series"]),
                    )
                )
                self._winding_edges.append(
                    dict(
                        t=ti,
                        part="common",
                        i=self._node("bus", t["lo_bus"]),
                        j=sub,
                        r=float(t["r_common"]),
                    )
                )
        self.n_nodes = len(self._node_of)
        self._validate_grounding()

    def _validate_grounding(self):
        """Every conductive component must have a path to remote earth."""
        adjacency = [set() for _ in range(self.n_nodes)]
        for i, j, _ in self._branch_edges:
            adjacency[i].add(j)
            adjacency[j].add(i)
        for winding in self._winding_edges:
            i, j = winding["i"], winding["j"]
            adjacency[i].add(j)
            adjacency[j].add(i)
        grounded = {self._node_of[("sub", s["id"])] for s in self.substations}
        unseen = set(range(self.n_nodes))
        while unseen:
            start = next(iter(unseen))
            stack = [start]
            component = set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                stack.extend(adjacency[node] - component)
            unseen -= component
            if component.isdisjoint(grounded):
                labels = [key for key, value in self._node_of.items() if value in component]
                raise ValueError(
                    "network has a conductive component with no grounded "
                    f"substation; first nodes: {labels[:5]}"
                )

    def _node(self, kind, num):
        key = (kind, num)
        if key not in self._node_of:
            self._node_of[key] = len(self._node_of)
        return self._node_of[key]

    def __repr__(self):
        return (
            f"Network({len(self.substations)} substations, "
            f"{len(self.transformers)} transformers, {len(self.branches)} branches)"
        )

    @classmethod
    def from_json(cls, path):
        """Load a network from the documented JSON format (see :meth:`to_json`)."""
        with open(path) as f:
            data = json.load(f)
        return cls(data["substations"], data["transformers"], data["branches"])

    def to_json(self, path):
        """Save ``{"substations": [...], "transformers": [...], "branches": [...]}``."""
        data = {
            "substations": self.substations,
            "transformers": self.transformers,
            "branches": self.branches,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def epri21_benchmark(cls):
        """The Horton et al. (2012) EPRI 21-bus GIC test system (see module docs).

        Ships as package data; topology only (no reference field) -- drive it
        with :meth:`solve` or :func:`gic_matrix` for a chosen E field. Node
        count and connectivity reproduce ``lanl-ansi/PowerModelsGMD.jl``'s
        ``epri21.m``, cross-checked against that project's own PowerWorld
        verification data (see ``docs/validation.md``, item 1.1).
        """
        path = files("bezpy.tl").joinpath("data", "epri21_network.json")
        with path.open() as f:
            data = json.load(f)
        return cls(data["substations"], data["transformers"], data["branches"])

    def admittance(self):
        """The static ``(Y, B)`` pair: nodal admittance and branch-EMF injection.

        ``Y`` (``n_nodes x n_nodes``, sparse) is built once per network and
        never depends on the driving field. ``B`` (``n_nodes x nbranches``,
        sparse) maps a vector of branch (line) EMFs to injected nodal
        currents: ``J = B @ v``.
        """
        rows, cols, vals = [], [], []

        def add(i, j, g):
            rows.extend((i, j, i, j))
            cols.extend((i, j, j, i))
            vals.extend((g, g, -g, -g))

        for i, j, r in self._branch_edges:
            add(i, j, 1.0 / r)
        for w in self._winding_edges:
            add(w["i"], w["j"], 1.0 / w["r"])
        for s in self.substations:
            n = self._node("sub", s["id"])
            rows.append(n)
            cols.append(n)
            vals.append(1.0 / float(s["resistance"]))
        Y = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(self.n_nodes, self.n_nodes))

        brows, bcols, bvals = [], [], []
        for k, (i, j, r) in enumerate(self._branch_edges):
            brows.extend((j, i))
            bcols.extend((k, k))
            bvals.extend((1.0 / r, -1.0 / r))
        B = scipy.sparse.csr_matrix(
            (bvals, (brows, bcols)), shape=(self.n_nodes, len(self._branch_edges))
        )
        return Y, B

    def solve(self, branch_voltages):
        """Node voltages for given branch (line) driving voltages.

        :param branch_voltages: volts, shape ``(nbranches,)`` or
            ``(ntimes, nbranches)``, aligned with :attr:`branches`
        :return: node voltages, ``(n_nodes,)`` or ``(ntimes, n_nodes)``
        """
        Y, B = self.admittance()
        v = np.asarray(branch_voltages, dtype=float)
        if v.ndim not in (1, 2) or v.shape[-1] != len(self.branches):
            raise ValueError(
                f"branch_voltages must have shape ({len(self.branches)},) or "
                f"(ntimes, {len(self.branches)}); got {v.shape}"
            )
        if not np.isfinite(v).all():
            raise ValueError("branch_voltages must contain only finite values")
        squeeze = v.ndim == 1
        v = np.atleast_2d(v)
        J = (B @ v.T).T  # (ntimes, n_nodes)
        lu = scipy.sparse.linalg.splu(Y.tocsc())
        V = lu.solve(J.T).T  # (ntimes, n_nodes)
        return V[0] if squeeze else V

    def winding_currents(self, node_voltages):
        """Per-winding currents (Amps) from solved node voltages.

        :param node_voltages: from :meth:`solve`, ``(n_nodes,)`` or ``(ntimes, n_nodes)``
        :return: dict keyed by ``(transformer_id, part)`` (``part`` one of
            ``"hi"``, ``"lo"``, ``"series"``, ``"common"``) to current arrays
            matching ``node_voltages``'s leading shape
        """
        V = np.asarray(node_voltages, dtype=float)
        if V.ndim not in (1, 2) or V.shape[-1] != self.n_nodes:
            raise ValueError(
                f"node_voltages must end with {self.n_nodes} network nodes; got {V.shape}"
            )
        axis = -1
        out = {}
        for w in self._winding_edges:
            tid = self.transformers[w["t"]]["id"]
            I = (np.take(V, w["i"], axis=axis) - np.take(V, w["j"], axis=axis)) / w["r"]
            out[(tid, w["part"])] = I
        return out

    def neutral_currents(self, node_voltages):
        """Per-transformer neutral (ground-bound) current, Amps.

        The sum of a transformer's grounded-winding currents (``hi``/``lo``
        for ``"wye-wye"``/``"gsu"``; just ``common`` for ``"auto"``, since
        the series winding never reaches ground) -- the current that
        actually flows to remote earth through that transformer, and the
        quantity :func:`gic_matrix` reports.

        :param node_voltages: from :meth:`solve`
        :return: array ``(ntransformers,)`` or ``(ntimes, ntransformers)``
        """
        winding = self.winding_currents(node_voltages)
        V = np.asarray(node_voltages, dtype=float)
        out_shape = V.shape[:-1] + (len(self.transformers),)
        out = np.zeros(out_shape)
        for w in self._winding_edges:
            if w["part"] in _GROUNDED_PARTS:
                tid = self.transformers[w["t"]]["id"]
                out[..., w["t"]] += winding[(tid, w["part"])]
        return out


def gic_matrix(network, lines, n_sites, how="delaunay"):
    """Precompute the site-E -> transformer-neutral-current kernel.

    Composes :meth:`Network.admittance` with
    :func:`bezpy.tl.kernels.voltage_matrix`, mirroring
    :func:`~bezpy.tl.kernels.voltage_matrix` / :func:`apply_voltage_matrix`
    -- the geometry and network topology are contracted out once; only a
    matmul remains per chunk (see :func:`apply_gic_matrix`).

    :param network: a :class:`Network`
    :param lines: sequence of ``TransmissionLine`` objects; each of
        ``network``'s branches is matched to one by
        ``line.metadata["id"] == branch["id"]``
    :param n_sites: number of E-field inputs (as in :func:`voltage_matrix`)
    :param how: E interpolation scheme, forwarded to
        :func:`~bezpy.tl.kernels.voltage_matrix`
    :return: ``(K, valid)``: ``K`` has shape ``(ntransformers, n_sites, 2)``
        (apply with :func:`apply_gic_matrix`); ``valid`` is ``(ntransformers,)``,
        False for a transformer whose neutral current depends on a line
        whose own interpolation weights are invalid (see
        :func:`~bezpy.tl.kernels.voltage_functional`)
    """
    line_index = {}
    for idx, line in enumerate(lines):
        lid = line.metadata.get("id")
        if lid is not None:
            if lid in line_index:
                raise ValueError(
                    f"multiple transmission lines have metadata id {lid!r}; "
                    "branch-to-geometry matching must be one-to-one"
                )
            line_index[lid] = idx
    missing = [b["id"] for b in network.branches if b["id"] not in line_index]
    if missing:
        raise ValueError(
            f"{len(missing)} network branch(es) have no matching line "
            f"(line.metadata['id']); first few: {missing[:5]}"
        )

    L, valid = voltage_matrix(lines, n_sites, how=how)

    Y, B_local = network.admittance()
    rows = np.arange(len(network.branches))
    cols = np.array([line_index[b["id"]] for b in network.branches])
    S = scipy.sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(network.branches), len(lines))
    )
    B = B_local @ S  # (n_nodes, nlines)

    lu = scipy.sparse.linalg.splu(Y.tocsc())
    X = lu.solve(B.toarray())  # (n_nodes, nlines)

    ntransformers = len(network.transformers)
    C = np.zeros((ntransformers, network.n_nodes))
    for w in network._winding_edges:  # pylint: disable=protected-access
        if w["part"] in _GROUNDED_PARTS:
            g = 1.0 / w["r"]
            C[w["t"], w["i"]] += g
            C[w["t"], w["j"]] -= g

    K_lines = C @ X  # (ntransformers, nlines): neutral current per unit line voltage
    K = np.einsum("tl,lsc->tsc", K_lines, L)

    invalid_lines = ~np.asarray(valid)
    valid_t = ~np.any((K_lines != 0) & invalid_lines[np.newaxis, :], axis=1)
    return K, valid_t


def apply_gic_matrix(E, K, valid=None):
    """Transformer neutral currents for all transformers at once.

    Mirrors :func:`~bezpy.tl.kernels.apply_voltage_matrix`.

    :param E: electric fields, shape ``(ntimes, n_sites, 2)``, mV/km
    :param K: from :func:`gic_matrix`
    :param valid: optional boolean mask; invalid transformers are set to NaN
    :return: neutral currents in Amps, shape ``(ntimes, ntransformers)``
    """
    E = np.asarray(E, dtype=float)
    K = np.asarray(K, dtype=float)
    if E.ndim != 3 or E.shape[-1] != 2:
        raise ValueError("E must have shape (ntimes, n_sites, 2)")
    if K.ndim != 3 or K.shape[2] != 2 or K.shape[1] != E.shape[1]:
        raise ValueError(f"K must have shape (ntransformers, {E.shape[1]}, 2); got {K.shape}")
    ntimes = E.shape[0]
    K_sparse = scipy.sparse.csr_matrix(K.reshape(K.shape[0], -1))
    I = E.reshape(ntimes, -1) @ K_sparse.T
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (K.shape[0],):
            raise ValueError(f"valid must have shape ({K.shape[0]},)")
        I[:, ~valid] = np.nan
    return I


def effective_gic(i_high, i_low, a=1.0, b=1.0):
    """TPL-007 effective GIC for a transformer: ``|a * I_high + b * I_low|``.

    The thermally-relevant combination of a two-winding transformer's hi-
    and low-side currents (:meth:`Network.winding_currents`); for a
    single-winding transformer (``"gsu"`` with only one grounded side),
    pass the other current as 0. ``a``/``b`` default to 1 (a plain sum);
    NERC TPL-007 guidance derives them from the transformer's turns ratio
    and configuration when that detail is available.

    :param i_high: hi-side winding current(s), Amps
    :param i_low: lo-side winding current(s), Amps
    :param a: weight on ``i_high``
    :param b: weight on ``i_low``
    :return: effective GIC, Amps, same shape as the (broadcast) inputs
    """
    return np.abs(a * np.asarray(i_high) + b * np.asarray(i_low))
