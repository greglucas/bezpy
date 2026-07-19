"""Command-line entry points for bezpy: ``bezpy download`` / ``update`` /
``scan`` / ``nowcast``.

Thin argparse wrappers over the public API -- no business logic lives here;
see ``bezpy.data`` and ``bezpy.pipeline`` for the implementations. Imports of
``bezpy.data`` (needs the ``data`` extra) and pysecs (needs ``secs``) are
deferred into each subcommand's function body, not module level, so
``bezpy --help`` / ``bezpy download --help`` work without those installed --
the same lazy-optional-dependency posture as the rest of the package.
"""
import argparse
import sys
import time

__all__ = ["build_parser", "main"]


def _add_common_obs_args(p, level_default="variation"):
    p.add_argument("codes", nargs="+", metavar="CODE",
                   help="IAGA observatory code(s)")
    p.add_argument("--source", default="auto",
                   help="auto (default), usgs, nrcan, intermagnet, or wdc")
    p.add_argument("--level", default=level_default,
                   help=f"data quality level (default: {level_default})")
    p.add_argument("--sampling-period", type=float, default=60.,
                   help="seconds per sample (default: 60)")
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="bezpy",
        description="bezpy: B -> E -> V geoelectric hazard analysis.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser(
        "download", help="download an observatory interval into the local archive")
    _add_common_obs_args(p_download)
    p_download.add_argument("--start", required=True, help="UTC start (any datetime-like string)")
    p_download.add_argument("--end", required=True, help="UTC end (any datetime-like string)")
    p_download.set_defaults(func=cmd_download)

    p_update = sub.add_parser(
        "update", help="bring an observatory's local archive up to date")
    _add_common_obs_args(p_update)
    p_update.add_argument("--until", default=None,
                          help="UTC end (default: now)")
    p_update.add_argument("--default-days", type=int, default=3,
                          help="days to seed if nothing is stored yet (default: 3)")
    p_update.set_defaults(func=cmd_update)

    p_scan = sub.add_parser(
        "scan", help="continuous scan of a stored record into a hazard summary")
    p_scan.add_argument("--obs", nargs="+", required=True, metavar="CODE",
                        help="observatory codes to combine")
    p_scan.add_argument("--start", required=True)
    p_scan.add_argument("--end", required=True)
    p_scan.add_argument("--bbox", type=float, nargs=4, required=True,
                        metavar=("W", "S", "E", "N"), help="MT site region")
    p_scan.add_argument("--min-rating", type=int, default=3)
    p_scan.add_argument("--chunk", default="1Y",
                        help="chunk length, pandas period frequency (default: 1Y)")
    p_scan.add_argument("--reduce", nargs="+", default=["max"],
                        choices=["max", "exceedances", "quantiles"])
    p_scan.add_argument("--threshold", type=float, default=None,
                        help="exceedance level (mV/km), required for --reduce exceedances")
    p_scan.add_argument("--despike", action="store_true")
    p_scan.add_argument("--workers", type=int, default=None)
    p_scan.add_argument("--out", required=True, help="output netCDF path")
    p_scan.set_defaults(func=cmd_scan)

    p_nowcast = sub.add_parser(
        "nowcast", help="poll observatories and stream causal E (and V) to a CSV")
    p_nowcast.add_argument("--obs", nargs="+", required=True, metavar="CODE")
    p_nowcast.add_argument("--bbox", type=float, nargs=4, required=True,
                           metavar=("W", "S", "E", "N"), help="MT site region")
    p_nowcast.add_argument("--min-rating", type=int, default=3)
    p_nowcast.add_argument("--dt", type=float, default=60.)
    p_nowcast.add_argument("--nmin", default="auto")
    p_nowcast.add_argument("--nmax", default="auto")
    p_nowcast.add_argument("--decay-factor", type=float, default=1e-2)
    p_nowcast.add_argument("--poll-interval", type=float, default=60.,
                           help="seconds between polls (default: 60)")
    p_nowcast.add_argument("--lines", default=None,
                           help="optional path to a line geometry file (see "
                                "bezpy.tl.TransmissionLine); voltages are "
                                "added to the output when given")
    p_nowcast.add_argument("--out", required=True, help="output CSV path (appended to)")
    p_nowcast.add_argument("--once", action="store_true",
                           help="run a single poll/update/write cycle and exit "
                                "(for cron/systemd-timer-driven operation)")
    p_nowcast.set_defaults(func=cmd_nowcast)

    return parser


def cmd_download(args):
    import bezpy.data
    for code in args.codes:
        paths = bezpy.data.download_observatory(
            code, args.start, args.end, source=args.source, level=args.level,
            sampling_period=args.sampling_period, verbose=args.verbose)
        print(f"{code}: wrote {len(paths)} file(s)")


def cmd_update(args):
    import bezpy.data
    for code in args.codes:
        paths = bezpy.data.update_observatory(
            code, source=args.source, level=args.level, until=args.until,
            sampling_period=args.sampling_period, default_days=args.default_days,
            verbose=args.verbose)
        print(f"{code}: wrote {len(paths)} file(s)")


def cmd_scan(args):
    import bezpy.data
    import bezpy.pipeline

    if "exceedances" in args.reduce and args.threshold is None:
        raise SystemExit("--reduce exceedances requires --threshold")

    sites = bezpy.data.load_mt_sites(bbox=tuple(args.bbox), min_rating=args.min_rating)
    if not sites:
        raise SystemExit(f"no cached MT sites in bbox {tuple(args.bbox)}")
    b_obs = bezpy.data.load_observatory_group(args.obs, start=args.start, end=args.end)

    bezpy.pipeline.scan(
        b_obs, sites, chunk=args.chunk, reduce=tuple(args.reduce),
        threshold=args.threshold, despike_obs=args.despike,
        workers=args.workers, out_path=args.out)
    print(f"wrote {args.out}")


def _parse_nmin_nmax(value):
    return value if value == "auto" else int(value)


def _append_csv(path, result):
    """Append newly-finalized Nowcast rows to a wide-format CSV (header
    written once): time, <site>_Ex, <site>_Ey[, <line>_V, ...]."""
    import os

    import pandas as pd

    columns = {}
    for var, dim in (("Ex", "site"), ("Ey", "site"), ("V", "line")):
        if var not in result:
            continue
        values = result[var].values
        names = result[dim].values
        for i, name in enumerate(names):
            columns[f"{name}_{var}"] = values[:, i]
    df = pd.DataFrame(columns, index=pd.DatetimeIndex(result["time"].values, name="time"))
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header)


def cmd_nowcast(args):
    import bezpy
    import bezpy.data
    from bezpy.mag import SECSInterpolator

    sites = bezpy.data.load_mt_sites(bbox=tuple(args.bbox), min_rating=args.min_rating)
    if not sites:
        raise SystemExit(f"no cached MT sites in bbox {tuple(args.bbox)}")

    lines = None
    if args.lines is not None:
        import json

        import shapely.geometry

        from bezpy.tl import TransmissionLine
        with open(args.lines) as f:
            geoms = json.load(f)
        lines = [TransmissionLine(shapely.geometry.shape(g)) for g in geoms]

    nowcast = bezpy.pipeline.Nowcast(
        sites, SECSInterpolator(), dt=args.dt,
        nmin=_parse_nmin_nmax(args.nmin), nmax=_parse_nmin_nmax(args.nmax),
        decay_factor=args.decay_factor, lines=lines)

    last_end = None
    while True:
        for code in args.obs:
            bezpy.data.update_observatory(code, sampling_period=args.dt)
        b_obs = bezpy.data.load_observatory_group(args.obs, start=last_end)
        if b_obs.sizes.get("time", 0):
            result = nowcast.update(b_obs)
            if result.sizes.get("time", 0):
                _append_csv(args.out, result)
                last_end = result["time"].values[-1]
                print(f"wrote {result.sizes['time']} row(s) through {last_end}")
        if args.once:
            return
        time.sleep(args.poll_interval)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
