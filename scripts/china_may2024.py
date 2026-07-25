"""Inspect the open MT tensors from the May 2024 North China superstorm.

Wang (2025) archived the 22 EMTF transfer functions used for
``10.1029/2025SW004557`` at ``10.5281/zenodo.15107366``. This example
downloads only the small ``Impedance.zip`` file, verifies its repository
checksum, reads every ``.zrr`` member without extracting it, and runs bezpy's
physical impedance screen.

The record also contains ``May2024SuperStormdata.zip`` (about 211 MB) with
MATLAB magnetic/electric time series. It is deliberately not downloaded by
this quick example. The Z-files repeat one embedded coordinate for many
stations; use the per-station coordinates in that time-series archive before
performing spatial interpolation.
"""

import argparse
import collections

import bezpy
import bezpy.data

RECORD_ID = 15107366
FILENAME = "Impedance.zip"
DATA_DOI = "https://doi.org/10.5281/zenodo.15107366"
PAPER_DOI = "https://doi.org/10.1029/2025SW004557"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download the archive again even if the cached checksum is valid",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    archive = bezpy.data.download_zenodo_file(
        RECORD_ID,
        FILENAME,
        refresh=args.refresh,
    )
    sites = bezpy.mt.read_zfile_archive(archive)

    print(f"Paper: {PAPER_DOI}")
    print(f"Data:  {DATA_DOI}")
    print(f"Cache: {archive}")
    print(f"Read {len(sites)} transfer functions")
    print("\nstation                         latitude  longitude  periods  screen")
    print("------------------------------  --------  ---------  -------  ------")
    coordinates = collections.Counter()
    for site in sites.sites:
        coordinates[(site.latitude, site.longitude)] += 1
        flag = bezpy.mt.screen_impedance(site)["imp_flag"]
        print(
            f"{site.name:30s}  {site.latitude:8.3f}  {site.longitude:9.3f}"
            f"  {len(site.periods):7d}  {flag}"
        )

    repeated = [(coordinate, count) for coordinate, count in coordinates.items() if count > 1]
    if repeated:
        print("\nCoordinate warning:")
        for (latitude, longitude), count in sorted(repeated):
            print(f"  ({latitude:.3f}, {longitude:.3f}) occurs in {count} Z-files")
        print(
            "Use the coordinates stored with each station in "
            "May2024SuperStormdata.zip before any spatial calculation."
        )


if __name__ == "__main__":
    main()
