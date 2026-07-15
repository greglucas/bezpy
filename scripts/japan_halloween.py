"""The Halloween 2003 storm at the Japanese magnetic observatories.

The successor to the old Japan-Halloween notebook. It downloads the
28-31 October 2003 magnetic field at Kakioka (KAK), Memambetsu (MMB), and
Kanoya (KNY) straight from INTERMAGNET via ``bezpy.data`` -- no hand-managed
files -- and, when the collocated MT transfer function is available, models the
induced geoelectric field with ``site.convolve_fft`` and plots B and E side by
side.

Collocated MT tensors
---------------------
EarthScope's EMTF/SPUD archive carries an "INTERMAGNET" survey with transfer
functions estimated *at* several observatories, including KAK, MMB, and KNY
(``INTERMAGNET.<CODE>.2000-2011``). They are not part of the default
``bezpy.data`` cache and SPUD has no stable bounding-box search, so fetch them
once from their ds.iris.edu/spud/emtf pages and hand this script the files::

    python japan_halloween.py --mt-xml KAK=INTERMAGNET.KAK.2000-2011.xml \
                              --mt-xml MMB=INTERMAGNET.MMB.2000-2011.xml \
                              --mt-xml KNY=INTERMAGNET.KNY.2000-2011.xml

    # or, if you know the SPUD bundle id, cache it into bezpy.data first:
    #   bezpy.data.download_mt_spud(<bundle_id>)   # then --mt-xml is unneeded

Without a tensor a site still gets its B figure. Note the estimated impedances
bottom out near ~10 s, so 1-minute B (used here) is well inside their band; the
original notebook's 1-second measured-E comparison used Kakioka's own ``dgef``
electric-field product, which is a separate JMA distribution, not in bezpy.

A note on very long records
---------------------------
Kakioka's magnetic series runs back to 1913 and WDC Kyoto holds hourly means
from the 1950s -- a far longer baseline than the North American records, and an
appealing target for a decades-long ``bezpy.pipeline.scan``. bezpy.data does
not yet wire in a WDC/Kyoto magnetic source (only usgs/nrcan/intermagnet), so
that century-scale hazard example is a future addition, not something this
script can pull today.
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
    p.add_argument("--mt-xml", action="append", default=[], metavar="CODE=PATH",
                   help="collocated MT transfer function for an observatory "
                        "(repeatable); enables the modeled-E panel")
    p.add_argument("--outdir", default=".", help="where the figure is written")
    return p.parse_args()


def load_field(code):
    """Download and load a gap-free, detrended (Bx, By) for the storm window."""
    bezpy.data.download_observatory(code, START, END, source="intermagnet",
                                    level="definitive", sampling_period=60)
    df = bezpy.data.load_observatory(code, source="intermagnet",
                                     start=START, end=END, as_dataframe=True)
    df = df[["X", "Y"]].interpolate(limit_direction="both")
    t = np.arange(len(df))
    for col in ("X", "Y"):                     # remove the baseline (linear fit)
        df[col] -= np.polyval(np.polyfit(t, df[col], 1), t)
    return df


def load_tensor(code, mt_xml):
    """The collocated MT site for ``code``: the cache, then a --mt-xml path."""
    try:
        return bezpy.data.load_mt_site(code)
    except (FileNotFoundError, KeyError):
        pass
    if code in mt_xml:
        return bezpy.mt.read_xml(mt_xml[code])
    return None


def main():
    args = parse_args()
    mt_xml = dict(pair.split("=", 1) for pair in args.mt_xml)

    fig, axes = plt.subplots(len(OBSERVATORIES), 1, figsize=(11, 9), sharex=True)
    for ax, code in zip(np.atleast_1d(axes), OBSERVATORIES):
        b = load_field(code)
        ax.plot(b.index, b["X"], c="k", lw=0.6, label=r"$B_x$")
        ax.plot(b.index, b["Y"], c="tab:red", lw=0.6, label=r"$B_y$")
        ax.set(ylabel="B (nT)", title=f"Halloween 2003 storm -- {code}")

        site = load_tensor(code, mt_xml)
        if site is not None:
            ex, ey = site.convolve_fft(b["X"].to_numpy(), b["Y"].to_numpy(), dt=60)
            axe = ax.twinx()
            axe.plot(b.index, ex, c="tab:blue", lw=0.6, alpha=0.7, label=r"$E_x$")
            axe.plot(b.index, ey, c="tab:green", lw=0.6, alpha=0.7, label=r"$E_y$")
            axe.set_ylabel("modeled E (mV/km)")
            _merge_legends(ax, axe)
        else:
            ax.legend(loc="upper right", fontsize=8)
            print(f"{code}: no collocated MT tensor (B only) -- pass "
                  f"--mt-xml {code}=<path> to model E; see the module docstring.")

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
