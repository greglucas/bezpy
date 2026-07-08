"""Method-comparison harness: scores every method option against synthetic
truth and prints the tables recorded in VALIDATION.md.

Run:  python benchmarks/validation_harness.py
"""
import numpy as np
import shapely.geometry

from bezpy.tl import TransmissionLine
from bezpy.validation import (
    score_extrapolation,
    score_interpolation_methods,
    synthetic_scenario,
)


def make_lines():
    """Short and long multi-vertex lines inside the site hull."""
    coords = [
        [(-106.5, 38.0), (-105.9, 38.4)],                                  # ~80 km
        [(-106.5, 38.0), (-105.0, 38.8), (-103.5, 39.6), (-102.0, 40.5)],  # ~500 km
        [(-107.5, 43.0), (-105.5, 42.0), (-103.5, 41.0), (-101.5, 40.0),
         (-99.5, 39.0)],                                                   # ~900 km
    ]
    return [TransmissionLine(shapely.geometry.LineString(c)) for c in coords]


def main():
    scenario = synthetic_scenario(seed=0, ntimes=8192)
    lines = make_lines()

    print("=== D7: E-interpolation scheme vs synthetic truth "
          "(relative RMS voltage error) ===")
    table = score_interpolation_methods(scenario, lines,
                                        how=("nn", "delaunay"),
                                        resample_km=(None, 5.0, 1.0))
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    print()
    print("=== D6: impedance band-edge handling vs synthetic truth "
          "(relative RMS E error) ===")
    for band in ((300., 7200.), (600., 3600.)):
        periods = np.logspace(np.log10(band[0]), np.log10(band[1]), 31)
        result = score_extrapolation(scenario, periods=periods)
        print(f"impedance band {band[0]:g}-{band[1]:g} s "
              f"(source band 120-28800 s):")
        print(result.to_string(float_format=lambda v: f"{v:.4f}"))
        print()


if __name__ == "__main__":
    main()
