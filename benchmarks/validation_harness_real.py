"""Real-storm method-comparison harness (docs/validation.md "Still to add"):
the reference-storm comparison and epsilon/tau tuning curves that
`benchmarks/validation_harness.py` cannot provide from synthetic truth alone
-- real observatory noise, real gaps, and real EMTF impedance bands.

Needs the local data archive (`~/.cache/bezpy`) populated for the CONUS
observatory network and the docs' example MT site bbox; see
`bezpy.data.download_observatory` / `download_mt_search` if it is not.

Run:  python benchmarks/validation_harness_real.py
"""
import warnings

import numpy as np
import pandas as pd

import bezpy
import bezpy.data as data
from bezpy.mag.processing import prepare
from bezpy.mag.secs import SECSInterpolator
from bezpy.validation import leave_one_out

# A 5-station CONUS network with data through both reference storms (used
# throughout docs/tests: docs/index.md, tests/test_pipeline.py-adjacent).
OBS = ["BOU", "FRD", "TUC", "NEW", "FRN"]

# (core storm window, wider load window) -- the wider window gives
# calc_geoelectric_field's pad="auto" (~24h per side, driven by the longest
# MT site period in-region) real data to pad with instead of truncating the
# storm itself.
STORMS = {
    "2024-05 Gannon": (("2024-05-10", "2024-05-12"),
                       ("2024-05-07", "2024-05-15")),
    "2003-10 Halloween": (("2003-10-29", "2003-11-01"),
                          ("2003-10-26", "2003-11-04")),
}

SITE_BBOX = (-110, 35, -100, 45)  # matches docs/index.md's example network


def _load(window):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return data.load_observatory_group(OBS, *window)


def _load_detrended(window):
    """Storm-window B with each station's per-window linear trend removed.

    SECS models only the ionospheric disturbance field, not each station's
    (very different) core-field baseline, so leave-one-out skill on *raw* B
    is dominated by a multi-thousand-nT per-station DC offset having nothing
    to do with interpolation quality. calc_geoelectric_field always runs
    prepare() (detrend="linear") before fitting; matching that here is what
    makes the RMSE below comparable to the synthetic score_fitting_methods
    table, where the source has no such baseline to begin with.
    """
    return prepare(_load(window), pad=0, detrend="linear", taper=None,
                   gaps="mask")


def fitting_method_table():
    """Real-storm analog of validation.score_fitting_methods (3.1): held-out
    observatory skill for each SECS fitting method, leave-one-out."""
    configs = {
        "lstsq (default)": {},
        "lstsq robust=huber": {"robust": "huber"},
        "kalman tau=600": {"method": "kalman", "tau": 600.},
        "kalman tau=3600": {"method": "kalman", "tau": 3600.},
    }
    rows = []
    for storm, (core, _) in STORMS.items():
        b_obs = _load_detrended(core)
        for name, kwargs in configs.items():
            df = leave_one_out(b_obs, interp=SECSInterpolator(**kwargs),
                               metrics=("rmse", "mae", "correlation"))
            rows.append({"storm": storm, "method": name,
                         "rmse_nT": df["rmse"].mean(),
                         "mae_nT": df["mae"].mean(),
                         "correlation": df["correlation"].mean()})
    return pd.DataFrame(rows).set_index(["storm", "method"])


def epsilon_tau_table():
    """epsilon (lstsq damping) and tau (Kalman correlation time) tuning
    curves via SECSInterpolator.loo_score on real storm data."""
    rows = []
    for storm, (core, _) in STORMS.items():
        b_obs = _load_detrended(core)
        base = SECSInterpolator()
        for epsilon in (0.01, 0.02, 0.05, 0.1, 0.2):
            score = base.loo_score(b_obs, metric="rmse", epsilon=epsilon)
            rows.append({"storm": storm, "param": "epsilon", "value": epsilon,
                         "rmse_nT": np.mean(list(score.values()))})
        for tau in (300., 600., 1800., 3600., 7200.):
            score = base.loo_score(b_obs, metric="rmse", method="kalman", tau=tau)
            rows.append({"storm": storm, "param": "tau (kalman)", "value": tau,
                         "rmse_nT": np.mean(list(score.values()))})
    return pd.DataFrame(rows).set_index(["storm", "param", "value"])


def extrapolate_table():
    """Real-storm analog of D6: does extrapolate="asymptotic" move peak |E|
    at real MT sites relative to the legacy "zeros" band-stop, and by how
    much."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sites = data.load_mt_sites(bbox=SITE_BBOX, min_rating=3)

    rows = []
    for storm, (core, wide) in STORMS.items():
        b_obs = _load(wide)
        start, end = pd.Timestamp(core[0]), pd.Timestamp(core[1])
        peaks = {}
        for option in ("zeros", "asymptotic"):
            e = bezpy.calc_geoelectric_field(b_obs, sites, window=(start, end),
                                             extrapolate=option)
            magnitude = np.hypot(e["Ex"].values, e["Ey"].values)
            peaks[option] = np.nanmax(magnitude, axis=0)
        ratio = peaks["asymptotic"] / peaks["zeros"]
        rows.append({"storm": storm, "n_sites": len(sites),
                     "ratio_median": np.median(ratio),
                     "ratio_p90": np.quantile(ratio, 0.9),
                     "ratio_max": np.max(ratio),
                     "ratio_min": np.min(ratio),
                     "n_nonfinite": int((~np.isfinite(ratio)).sum())})
    return pd.DataFrame(rows).set_index("storm")


def main():
    print("=== Real-storm SECS fitting method (leave-one-out skill) ===")
    print(fitting_method_table().to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print("=== Real-storm epsilon/tau tuning (mean LOO RMSE, nT) ===")
    print(epsilon_tau_table().to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print("=== Real-storm extrapolate zeros vs asymptotic "
          "(peak |E| ratio, asymptotic/zeros) ===")
    print(extrapolate_table().to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
