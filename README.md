# Bezpy

Bezpy is an open source library for analysis of magnetic (B), electric (E),
and impedance (Z) data within a geophysical framework. This library contains
routines for calculating the geoelectric field from the geomagnetic field in
multiple different ways.

## Features

- Geomagnetic to geoelectric field calculations
- Integration of the geoelectric field along transmission lines
- Built using established, fast, open source python libraries
    [Pandas](http://www.pandas.pydata.org/),
    [NumPy](http://www.numpy.org/),
    [SciPy](http://www.scipy.org/)

## Examples

Example notebooks can be found in [notebooks/](./notebooks/)

Example scripts for command line use can be found in [scripts/](./scripts/)

## Install

1. clone the git repository

    ```bash
    git clone https://github.com/greglucas/bezpy
    ```

2. Build and install the package

    ```bash
    python setup.py build
    python setup.py install
    ```

## License

The code is released under a BSD-3 license
[License described in LICENSE.md](./LICENSE.md)

## References

This package has been developed from different publications. Please consider citing the papers
that are relevant to the work you are doing if you are utilizing this code.

### Geoelectric field calculations

[doi:10.1002/2017GL076042](https://doi.org/10.1002/2017GL076042)

```bibtex
Love, J. J., Lucas, G. M., Kelbert, A., & Bedrosian, P. A. (2018).
Geoelectric hazard maps for the Mid‐Atlantic United States:
100 year extreme values and the 1989 magnetic storm.
Geophysical Research Letters, 44, doi:10.1002/2017GL076042.
```

### Transmission line integrations

[doi:10.1002/2017SW001779](https://doi.org/10.1002/2017SW001779)

```bibtex
Lucas, G. M., Love, J. J., & Kelbert, A. (2018). Calculation of voltages
in electric power transmission lines during historic geomagnetic storms:
An investigation using realistic earth impedances. Space Weather, 16,
181–195, doi:10.1002/2017SW001779.
```

### Time domain (DTIR)

[doi:10.1002/2017SW001594](https://doi.org/10.1002/2017SW001594)

```bibtex
Kelbert, A., C. C. Balch, A. Pulkkinen, G. D. Egbert,
J. J. Love, E. J. Rigler, and I. Fujii (2017),
Methodology for time-domain estimation of storm time geoelectric fields
using the 3-D magnetotelluric response tensors,
Space Weather, 15, 874–894, doi:10.1002/2017SW001594.
```

## Earthscope impedance database

[doi:10.17611/DP/EMTF.1](https://doi.org/10.17611/DP/EMTF.1)

```bibtex
Kelbert, A., G.D. Egbert and A. Schultz (2011),
IRIS DMC Data Services Products: EMTF, The Magnetotelluric Transfer Functions,
doi:10.17611/DP/EMTF.1.
```

## Problems/Questions

- [Report an issue using the GitHub issue tracker](http://github.com/greglucas/bezpy/issues)

## Additional Links

- [USGS Geomagnetism Program Home Page](http://geomag.usgs.gov/)
