"""Benchmark: current B->E->V pipeline vs precomputed kernels.

Scenario sized like one chunk of a multi-decade historical scan:
30 days of 1-minute data, 100 MT sites, 200 transmission lines.

Compares three ways to get line voltages from observatory B:
  A. current   : SiteCollection.convolve_fft -> per-line calc_voltages loop
  B. level 1   : SiteCollection.convolve_fft -> apply_voltage_matrix (L.E)
  C. level 2   : TelluricResponse.calc_voltages (tau(omega) * B(omega))

A "warm" timing repeats the call with the impedance/frequency caches
populated -- the relevant number when scanning many equal-length chunks.

Run:  python benchmarks/bench_kernels.py
"""
import time

import numpy as np
import shapely.geometry

from bezpy.mt.site import Site3d, SiteCollection
from bezpy.tl import TransmissionLine
from bezpy.tl.kernels import TelluricResponse, apply_voltage_matrix, voltage_matrix

RNG = np.random.default_rng(7)

NSITES = 100
NLINES = 200
NVERTS = 10
N0 = 43200          # 30 days of 1-minute data
DT = 60.0

LON0, LON1, LAT0, LAT1 = -110.0, -95.0, 32.0, 45.0


def make_site(i):
    site = Site3d(f"S{i:03d}")
    site.longitude = RNG.uniform(LON0, LON1)
    site.latitude = RNG.uniform(LAT0, LAT1)
    site.periods = np.logspace(0.85, 4.48, 31)   # ~7 s to 30,000 s, as in EMTF
    logt = np.log10(site.periods)
    phases = RNG.uniform(0, 2*np.pi, 4)
    amps = RNG.uniform(0.5, 2.0, 4)
    site.Z = np.array([a * (1 + 0.3*np.sin(logt + p)) *
                       np.exp(1j*(0.5 + 0.2*np.cos(logt + p)))
                       for a, p in zip(amps, phases)])
    site.Z_var = None
    return site


def make_line():
    # Random-walk line inside an interior box so it stays within the hull
    lon = RNG.uniform(LON0 + 2, LON1 - 2)
    lat = RNG.uniform(LAT0 + 2, LAT1 - 2)
    coords = [(lon, lat)]
    for _ in range(NVERTS - 1):
        lon += RNG.uniform(-0.3, 0.3)
        lat += RNG.uniform(-0.3, 0.3)
        coords.append((lon, lat))
    return TransmissionLine(shapely.geometry.LineString(coords))


def main():
    print(f"sites={NSITES} lines={NLINES} verts/line={NVERTS} "
          f"N0={N0} (30 d @ 1 min), N_fft=2^{int(np.log2(N0))+2}")

    sites = [make_site(i) for i in range(NSITES)]
    site_xys = np.array([[s.latitude, s.longitude] for s in sites])
    lines = [make_line() for _ in range(NLINES)]

    mag_x = np.cumsum(RNG.standard_normal(N0))   # red-noise-ish storm B
    mag_y = np.cumsum(RNG.standard_normal(N0))

    t0 = time.perf_counter()
    for line in lines:
        line.set_delaunay_weights(site_xys)
    t_weights = time.perf_counter() - t0
    print(f"\n[setup] delaunay weights for all lines : {t_weights:8.2f} s")

    collection = SiteCollection(sites)

    # --- A. current pipeline ------------------------------------------------
    def current(label):
        t0 = time.perf_counter()
        ex, ey = collection.convolve_fft(mag_x, mag_y, dt=DT)
        E = np.stack([ex.T, ey.T], axis=-1)
        t_e = time.perf_counter() - t0
        t0 = time.perf_counter()
        V = np.column_stack([line.calc_voltages(E, how="delaunay") for line in lines])
        t_v = time.perf_counter() - t0
        print(f"[A {label}] E fields: {t_e:8.2f} s   voltages: {t_v:8.2f} s   "
              f"total: {t_e+t_v:8.2f} s")
        return V

    V_a = current("cold")
    V_a = current("warm")

    # --- B. level 1: effective length vectors --------------------------------
    t0 = time.perf_counter()
    L, valid = voltage_matrix(lines, NSITES, how="delaunay")
    t_build = time.perf_counter() - t0
    print(f"\n[B build] voltage matrix L            : {t_build:8.2f} s")

    ex, ey = collection.convolve_fft(mag_x, mag_y, dt=DT)   # warm
    t0 = time.perf_counter()
    ex, ey = collection.convolve_fft(mag_x, mag_y, dt=DT)
    E = np.stack([ex.T, ey.T], axis=-1)
    t_e = time.perf_counter() - t0
    t0 = time.perf_counter()
    V_b = apply_voltage_matrix(E, L, valid)
    t_v = time.perf_counter() - t0
    print(f"[B warm ] E fields: {t_e:8.2f} s   voltages: {t_v:8.2f} s   "
          f"total: {t_e+t_v:8.2f} s")

    # --- C. level 2: telluric response functions -----------------------------
    t0 = time.perf_counter()
    tr = TelluricResponse(L, sites, valid=valid, periods=sites[0].periods)
    t_build = time.perf_counter() - t0
    print(f"\n[C build] tau at 31 master periods    : {t_build:8.2f} s")

    for label in ("cold", "warm"):
        t0 = time.perf_counter()
        V_c = tr.calc_voltages(mag_x, mag_y, dt=DT)
        print(f"[C {label}] B spectra -> voltages         : "
              f"{time.perf_counter()-t0:8.2f} s")

    # --- agreement ------------------------------------------------------------
    scale = np.sqrt(np.nanmean(V_a**2))
    print(f"\nrel RMS  B vs A: {np.sqrt(np.nanmean((V_b-V_a)**2))/scale:.2e}"
          f"   C vs A: {np.sqrt(np.nanmean((V_c-V_a)**2))/scale:.2e}")
    print(f"lines inside hull: {valid.sum()}/{NLINES}")


if __name__ == "__main__":
    main()
