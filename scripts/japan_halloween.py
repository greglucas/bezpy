"""The Halloween 2003 storm at the Japanese magnetic observatories.

A fully self-contained example: every input is fetched through ``bezpy.data``,
so it runs from a clean install with no hand-managed files. It

1. downloads the collocated magnetotelluric tensors for Kakioka (KAK),
   Memambetsu (MMB), and Kanoya (KNY) from the EarthScope SPUD archive with
   :func:`bezpy.data.download_mt_search` (the JMA survey of Fujii et al. 2015),
2. downloads the 28-31 October 2003 magnetic field for each from the BGS World
   Data Centre with ``source="wdc"``, and
3. models the induced geoelectric field with ``site.convolve_fft`` and plots B
   and E together.

The WDC source reaches far past this storm: Kakioka's record runs back to the
1910s (hourly from the 1950s-60s in the WDC), decades before the real-time
services -- fetch a multi-decade span the same way (``source="wdc"``) and feed
it to ``bezpy.pipeline.scan`` to turn that long record into hazard statistics.
"""

import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import bezpy
import bezpy.data

OBSERVATORIES = ("KAK", "MMB", "KNY")
START, END = "2003-10-28", "2003-10-31"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="wdc", choices=("wdc", "intermagnet"),
                   help="magnetic-field source (default: wdc)")
    p.add_argument("--outdir", default=".", help="where the figure is written")
    return p.parse_args()


def load_field(code, source):
    """Download and load a gap-free, detrended (Bx, By) for the storm window."""
    bezpy.data.download_observatory(code, START, END, source=source,
                                    level="definitive", sampling_period=60)
    df = bezpy.data.load_observatory(code, source=source, start=START, end=END,
                                     as_dataframe=True)
    df = df[["X", "Y"]].interpolate(limit_direction="both")
    t = np.arange(len(df))
    for col in ("X", "Y"):                     # remove the baseline (linear fit)
        df[col] -= np.polyval(np.polyfit(t, df[col], 1), t)
    return df


def main():
    args = parse_args()
    # Collocated MT tensors (the JMA survey: KAK, MMB, KNY). Downloaded once,
    # then cached; the first call also fetches the SPUD index.
    bezpy.data.download_mt_search(survey="JMA", verbose=True)

    fig, axes = plt.subplots(len(OBSERVATORIES), 1, figsize=(11, 9), sharex=True)
    for ax, code in zip(np.atleast_1d(axes), OBSERVATORIES):
        b = load_field(code, args.source)
        site = bezpy.data.load_mt_site(code, survey="JMA")
        ex, ey = site.convolve_fft(b["X"].to_numpy(), b["Y"].to_numpy(), dt=60)

        ax.plot(b.index, b["X"], c="k", lw=0.6, label=r"$B_x$")
        ax.plot(b.index, b["Y"], c="tab:red", lw=0.6, label=r"$B_y$")
        ax.set(ylabel="B (nT)", title=f"Halloween 2003 storm -- {code}")
        axe = ax.twinx()
        axe.plot(b.index, ex, c="tab:blue", lw=0.6, alpha=0.7, label=r"$E_x$")
        axe.plot(b.index, ey, c="tab:green", lw=0.6, alpha=0.7, label=r"$E_y$")
        axe.set_ylabel("modeled E (mV/km)")
        _merge_legends(ax, axe)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    out = f"{args.outdir}/japan_halloween_2003.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


def _merge_legends(ax_left, ax_right):
    handles = ax_left.get_legend_handles_labels()
    extra = ax_right.get_legend_handles_labels()
    ax_left.legend(handles[0] + extra[0], handles[1] + extra[1],
                   loc="upper right", ncol=4, fontsize=8)


if __name__ == "__main__":
    main()
