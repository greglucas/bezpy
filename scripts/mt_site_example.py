"""Inspect a magnetotelluric site and turn B into E, end to end.

The successor to the old MT-example notebook: it loads MT transfer functions
from the ``bezpy.data`` cache (no hand-managed file paths, no Basemap), maps a
region by data quality, and then walks one site through the
three ways bezpy turns a magnetic field into an electric field:

* frequency domain -- ``site.convolve_fft`` (the workhorse the pipeline uses),
* the site's apparent-resistivity curve, raw and interpolated onto a dense
  frequency grid,
* time domain -- ``bezpy.mt.DTIR`` discrete impulse response, cross-checked
  against the frequency-domain impedance.

Run ``python mt_site_example.py --help`` for the options. With no arguments it
uses a well-rated cached site when available and otherwise downloads the
requested region from EarthScope SPUD. Pass ``--download`` to refresh first.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import bezpy
import bezpy.data


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("W", "S", "E", "N"),
        default=(-100, 35, -94, 41),
        help="region to map: west south east north (default: US plains)",
    )
    p.add_argument(
        "--min-rating",
        type=int,
        default=3,
        help="drop sites rated below this 1-5 quality (default: 3)",
    )
    p.add_argument(
        "--code", default=None, help="site code to inspect (default: the best-rated in bbox)"
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="refresh the region from EarthScope SPUD before loading",
    )
    p.add_argument("--outdir", default=".", help="where figures are written")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.download:
        # Search the SPUD archive for this region and download every match.
        bezpy.data.download_mt_search(
            bbox=tuple(args.bbox), min_rating=args.min_rating, verbose=True
        )

    sites = bezpy.data.load_mt_sites(bbox=tuple(args.bbox), min_rating=args.min_rating)
    if not sites:
        print("No matching MT sites are cached; downloading them from EarthScope SPUD.")
        bezpy.data.download_mt_search(
            bbox=tuple(args.bbox), min_rating=args.min_rating, verbose=True
        )
        sites = bezpy.data.load_mt_sites(bbox=tuple(args.bbox), min_rating=args.min_rating)
    if not sites:
        raise SystemExit("EarthScope SPUD returned no matching MT sites.")
    print(f"{len(sites)} MT sites in {tuple(args.bbox)} rated >= {args.min_rating}")

    # --- Map the region, colored by station quality ---------------------------
    latlons = np.array([[s.latitude, s.longitude] for s in sites])
    ratings = np.array([s.rating or 0 for s in sites])
    fig, ax = _quality_map(latlons, ratings, args.bbox)
    fig.savefig(outdir / "mt_sites_map.png", dpi=150, bbox_inches="tight")

    # --- Pick one site to inspect ---------------------------------------------
    if args.code is not None:
        site = next((s for s in sites if s.name == args.code), None)
        if site is None:
            raise SystemExit(f"{args.code} is not among the loaded sites")
    else:
        site = max(sites, key=lambda s: s.rating or 0)
    print(
        f"\nInspecting {site.name}: ({site.latitude:.2f}, {site.longitude:.2f}), "
        f"rating {site.rating}, {len(site.periods)} periods "
        f"{site.periods.min():.3g}-{site.periods.max():.3g} s"
    )

    # Apparent resistivity, measured and interpolated onto a dense grid. The
    # interpolation is the same log-period spline the convolution relies on.
    freqs = 1.0 / np.logspace(np.log10(5), np.log10(1e5), 500)
    site.plot_apparent_resistivity(interp_freqs=freqs)
    plt.gcf().savefig(outdir / f"{site.name}_resistivity.png", dpi=150, bbox_inches="tight")

    # --- B -> E, frequency domain vs. time domain -----------------------------
    # A single-polarization sinusoid makes the impedance's action obvious: a
    # north-south B drives E through Zxy/Zyy, an east-west B through Zxx/Zyx.
    ts = np.arange(10001)
    mag = np.sin(2 * np.pi * ts / 200.0)  # 200 s period, 1 s sampling
    zero = np.zeros_like(mag)

    dtir = bezpy.mt.DTIR(dt=1, nmin=-600, nmax=3600, model_space=True)
    dtir.calculate(site=site)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, (bx, by, driver) in zip(axes, [(mag, zero, "B_x"), (zero, mag, "B_y")]):
        ex_f, ey_f = site.convolve_fft(bx, by, dt=1)
        ex_t, ey_t = dtir.convolve(bx, by)
        ax.plot(ts, bx if driver == "B_x" else by, c="0.6", lw=1, label=f"${driver}$ (nT)")
        ax.plot(ts, ex_f, c="C0", label=r"$E_x$ FFT")
        ax.plot(ts, ey_f, c="C1", label=r"$E_y$ FFT")
        ax.plot(ts, ex_t, c="C0", ls="--", lw=1, label=r"$E_x$ DTIR")
        ax.plot(ts, ey_t, c="C1", ls="--", lw=1, label=r"$E_y$ DTIR")
        ax.set(ylabel="mV/km", title=f"{site.name}: response to ${driver}$")
        ax.legend(ncol=5, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(outdir / f"{site.name}_convolution.png", dpi=150, bbox_inches="tight")
    print(f"Figures written to {outdir}/")


def _quality_map(latlons, ratings, bbox):
    """Scatter the sites over a map, colored by rating. Cartopy if available."""
    fig, ax = plt.subplots(figsize=(9, 7))
    try:
        import cartopy.crs as ccrs

        ax = plt.axes(projection=ccrs.PlateCarree())
        bezpy.plot.add_map_features(ax)
        ax.set_extent([bbox[0], bbox[2], bbox[1], bbox[3]])
        transform = {"transform": ccrs.PlateCarree()}
    except ImportError:  # plain lon/lat axes as a fallback
        ax.set(
            xlim=(bbox[0], bbox[2]), ylim=(bbox[1], bbox[3]), xlabel="longitude", ylabel="latitude"
        )
        transform = {}
    sc = ax.scatter(
        latlons[:, 1],
        latlons[:, 0],
        c=ratings,
        cmap="viridis",
        vmin=1,
        vmax=5,
        s=45,
        edgecolor="k",
        linewidth=0.3,
        **transform,
    )
    fig.colorbar(sc, ax=ax, label="station quality (1-5)")
    ax.set_title("MT sites by data quality")
    return fig, ax


if __name__ == "__main__":
    main()
