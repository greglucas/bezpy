# Network GIC: line voltages to transformer amperes

Line voltages are the quantity most published hazard studies stop at. What
utilities and NERC TPL-007 assessments actually need is **amperes through
each transformer neutral** — which requires solving the DC-equivalent
network of transmission lines, transformer windings, and substation
grounding: the Lehtinen & Pirjola (1985) problem.

`bezpy.tl.network.Network` solves it, and composes with the existing
Level-1 voltage kernel exactly the way `TelluricResponse` composes with the
impedance: `gic_matrix` contracts the network topology and line geometry
into one more build-once, apply-per-chunk kernel.

## The benchmark

bezpy ships the Horton et al. (2012) EPRI 21-bus GIC test system — the
standard published oracle for validating GIC software — as package data:

```python
import bezpy
from bezpy.tl.network import Network

net = Network.epri21_benchmark()
print(net)  # Network(8 substations, 15 transformers, 15 branches)
```

A `Network` is three things: **substations** (a grounding resistance to
remote earth), **transformers** (grounded winding(s) connecting a bus to
its substation — `"wye-wye"`, `"gsu"`, or `"auto"`; an ungrounded winding is
just `r_hi=None`/`r_lo=None`), and **branches** (transmission lines,
bus-to-bus, carrying the induced EMF). Build your own the same way:

```python
substations = [{"id": "SUB1", "resistance": 0.5}]     # Ohms to remote earth
transformers = [{"id": "T1", "type": "gsu", "hi_bus": "BUS1", "lo_bus": "GEN1",
                 "substation": "SUB1", "r_hi": 0.2, "r_lo": None}]  # delta LV: no path
branches = [{"id": "LINE1", "from_bus": "BUS1", "to_bus": "BUS2", "resistance": 3.5}]
net = Network(substations, transformers, branches)
net.to_json("my_network.json")           # Network.from_json(...) to reload
```

## Solving directly

Given line-driving voltages (from `calc_voltages`, or your own), `solve`
returns node voltages; `neutral_currents` reduces those to the one number
per transformer that matters for thermal assessment — the current actually
reaching ground through that transformer (both grounded windings summed,
or just the common winding for an autotransformer, since the series
winding never touches ground):

```python
import numpy as np

line_v = np.zeros(len(net.branches))     # volts, aligned with net.branches
node_v = net.solve(line_v)
neutral = net.neutral_currents(node_v)   # (ntransformers,) Amps
winding = net.winding_currents(node_v)   # {(transformer_id, "hi"/"lo"/...): current}
```

## The kernel: E straight to transformer amperes

Real hazard studies don't hand-supply `line_v` — they start from MT site E
fields, exactly like `voltage_matrix`. `gic_matrix` composes the two:

```python
from bezpy.tl.network import apply_gic_matrix, gic_matrix

# Each network branch is matched to a TransmissionLine by
# line.metadata["id"] == branch["id"]; build lines with that in mind:
lines = [bezpy.tl.TransmissionLine(geometry, metadata={"id": branch["id"]})
        for branch, geometry in zip(net.branches, line_geometries)]

K, valid = gic_matrix(net, lines, n_sites=len(sites), how="delaunay")
e = bezpy.calc_geoelectric_field(b_obs, sites, window=("2024-05-10", "2024-05-12"))
E = np.stack([e["Ex"].values, e["Ey"].values], axis=-1)
I = apply_gic_matrix(E, K, valid)        # (ntimes, ntransformers) Amps
```

`K` is built once (a sparse solve of the network's admittance matrix
against the line-EMF injection map, contracted with the voltage kernel);
every chunk after that is one matmul, the same economics as
`apply_voltage_matrix`.

## In the continuous scan

`scan(..., network=)` adds an `I` reducer group next to `E` and `V` —
`I_max`, `I_max_time`, and (with `reduce=("quantiles", "exceedances")`) the
same quantile/exceedance machinery, one row per transformer:

```python
result = bezpy.pipeline.scan(b_obs, sites, lines=lines, network=net,
                             chunk="1Y", reduce=("max", "exceedances"),
                             threshold={"E": 100., "V": 5000., "I": 50.})
```

`network=` requires `lines=` (GIC is driven by the same line-integrated E),
and every network branch must have a matching line by `metadata["id"]`.

## Effective GIC (TPL-007)

For a two-winding transformer, the thermally-relevant combination of hi-
and low-side current is `|a * I_high + b * I_low|`:

```python
from bezpy.tl.network import effective_gic

i_hi = winding[("T1", "hi")]
i_lo = winding[("T1", "lo")]
i_eff = effective_gic(i_hi, i_lo)   # a=b=1 by default; override per TPL-007 turns ratio
```

## Validation

The benchmark's topology, and `Network`'s solve on it, are cross-checked
against `lanl-ansi/PowerModelsGMD.jl`'s own PowerWorld-computed verification
data for this exact test system — see `VALIDATION.md` item 1.1 for the
full derivation, including the well-understood factor-of-3 that reconciles
bezpy's single-line-equivalent total current against NERC's per-phase
"effective GIC" reporting convention.

**Deliberately out of scope**: transformer thermal response models,
harmonics, reactive power loss, AC load flow. Real network data is
proprietary — bezpy ships the format, the solver, and the benchmark, not
utility data.
